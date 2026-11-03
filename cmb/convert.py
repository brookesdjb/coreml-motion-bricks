"""Convert NVIDIA's MotionBricks SOMA model to CoreML packages that run correctly on the Neural Engine.

    python -m cmb.convert                 # batch 1, 2, 4, 8 fp16 packages
    python -m cmb.convert --fp32          # also the fp32 batch-1 package
    python -m cmb.convert --batches 1     # just what the single-character demo needs

The source is NVIDIA's own PyTorch model (cmb/upstream_model.py: their exporter code and released checkpoints,
the model their ONNX files are exported from), unmodified. Per batch size B:
  1. capture: make_fx(functionalize(model)) on real recorded inputs, with plain-matmul attention and two
     capture-time shims for NVIDIA's runtime checks and one_hot.
  2. rewrite: FX rewrites (cmb/fx_rewrites.py) that keep the maths but avoid what coremltools or the Neural
     Engine get wrong; the rewritten graph must match eager PyTorch.
  3. convert: torch.export -> coremltools ML Program, fp16 except the geometry ops (NVIDIA's own precision rule
     in tools/onnx/precision.py: networks fp16, geometry fp32); above batch 4 also tensors under 128 elements.
  4. check, before saving: some work must be placed on the Neural Engine (one failed piece makes CoreML drop the
     whole program to the CPU), and the package must match NVIDIA's fp32 PyTorch on every recorded input.
"""
import argparse
import contextlib
import shutil
import sys
import time

import numpy as np

from .paths import BATCH_SIZES, COREML_DIR, FEEDS, SOURCE_ONNX, coreml_package, coreml_package_fp32

INPUT_NAMES = ["global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
               "local_poses", "has_local_poses", "allowed_pred_num_tokens"]
OUTPUT_NAMES = ["pred_global_motions", "pred_num_tokens"]  # the names NVIDIA's engine contract uses

# Op types kept in fp32: NVIDIA's rule (tools/onnx/precision.py) keeps geometry and integration in fp32.
FP32_OPS = {"upsample_nearest_neighbor", "upsample_bilinear", "resize", "sin", "cos", "atan", "atan2", "acos",
            "asin", "cumsum", "sqrt", "rsqrt", "real_div", "inverse", "reduce_l2_norm", "reduce_sum_square", "pow",
            "exp", "log", "sign", "floor", "round"}
SMALL = 128  # above batch 4, ops whose largest output is smaller than this stay fp32 (CoreML runs them on the CPU)

# Checked on every recorded input before a package is saved: mean |difference| over the generations that chose
# the same clip length as NVIDIA's model (NVIDIA's own fp16 ONNX model scores ~0.006), and how many may choose a
# different one. The model picks the length by ArgMax; on a near-tie fp16 rounding can pick the neighbouring,
# equally valid length, whose frames are not comparable.
MAX_MEAN_DIFF = 0.012
MAX_OTHER_LENGTH = 1 / 40


def load_feeds(batch):
    """A batch of `batch` different real recorded generations (build/feeds.npz)."""
    z = np.load(FEEDS)
    n = int(z["n"])
    return {k: np.concatenate([z[f"{k}_{(i * 7) % n}"] for i in range(batch)]) for k in INPUT_NAMES}


def all_feeds():
    z = np.load(FEEDS)
    return [{k: z[f"{k}_{i}"] for k in INPUT_NAMES} for i in range(int(z["n"]))]


# --- 1. capture -----------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _capture_shims():
    """NVIDIA's model code calls torch through the module alias `t`. While capturing:
    * t.all(...) -> True: input-validation asserts that would otherwise turn into data-dependent graph breaks;
    * one_hot as a comparison: aten.one_hot has no coremltools converter."""
    import types

    import torch

    def one_hot(x, num_classes):
        return (x.long().unsqueeze(-1) == torch.arange(num_classes, device=x.device)).long()

    functional = types.ModuleType("functional_shim")
    functional.__dict__.update(torch.nn.functional.__dict__)
    functional.one_hot = one_hot
    nn = types.ModuleType("nn_shim")
    nn.__dict__.update(torch.nn.__dict__)
    nn.functional = functional
    shim = types.ModuleType("torch_capture_shim")
    shim.__dict__.update(torch.__dict__)
    shim.all = lambda *a, **k: True
    shim.nn = nn
    modules = [sys.modules[n] for n in ("motionbricks_exporter.models.soma.inference",
                                        "motionbricks_exporter.models.features")]
    saved = [m.t for m in modules]
    for m in modules:
        m.t = shim
    try:
        yield
    finally:
        for m, t in zip(modules, saved):
            m.t = t


def capture(model, feeds):
    """FX graph of the model on these inputs, functionalized, with attention as plain matmul/softmax."""
    import torch
    from torch.func import functionalize
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.nn.attention import SDPBackend, sdpa_kernel
    args = tuple(torch.from_numpy(feeds[n]) for n in INPUT_NAMES)
    with torch.no_grad(), _capture_shims(), sdpa_kernel(SDPBackend.MATH):
        return make_fx(functionalize(model), tracing_mode="real")(*args), args


# --- 3. convert -----------------------------------------------------------------------------------------------

_patched = False


def _patch_coremltools():
    """coremltools' `int`/`bool` casts fail on a one-element constant that isn't already that type (idempotent)."""
    global _patched
    if _patched:
        return
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

    def fixed_cast(py_type, original):
        def convert(context, node):
            x = ops._get_inputs(context, node, expected=1)[0]
            if x.can_be_folded_to_const() and np.size(x.val) == 1 and not isinstance(x.val, py_type):
                context.add(mb.const(val=py_type(np.asarray(x.val).reshape(-1)[0].item()), name=node.name),
                            node.name)
            else:
                original(context, node)
        return convert

    names = _TORCH_OPS_REGISTRY.name_to_func_mapping
    for name, py_type in (("int", int), ("bool", bool)):
        names[name] = fixed_cast(py_type, names[name])
    _patched = True


def _precision(batch, fp32):
    import coremltools as ct
    if fp32:
        return ct.precision.FLOAT32

    def fp16(op):
        if op.op_type in FP32_OPS:
            return False
        if batch <= 4:
            return True
        sizes = []
        for o in op.outputs:
            shape = getattr(o, "shape", None)
            if shape is None or any(not isinstance(d, int) for d in shape):
                return True
            sizes.append(int(np.prod(shape)) if len(shape) else 1)
        return max(sizes, default=0) >= SMALL
    return ct.transform.FP16ComputePrecision(op_selector=fp16)


def to_coreml(model, batch, fp32=False):
    """Capture, rewrite and convert the model at this batch size; returns the in-memory MLModel."""
    import coremltools as ct
    import torch
    from torch._decomp import get_decompositions

    from . import fx_rewrites
    _patch_coremltools()
    torch.backends.mha.set_fastpath_enabled(False)  # the fused fast path isn't capturable
    feeds = load_feeds(batch)
    gm, args = capture(model, feeds)
    with torch.no_grad():
        ref = model(*args)[0].numpy()
        counts = fx_rewrites.apply(gm, batch)
        diff = float(np.abs(gm(*args)[0].numpy() - ref).max())
    print("  rewrites:", ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    if diff > 1e-3:
        raise RuntimeError(f"rewritten graph differs from NVIDIA's model by {diff:.2g}; a rewrite is wrong")
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for n, name in zip(placeholders, INPUT_NAMES, strict=True):  # the package's input names
        n.name = n.target = name
    gm.recompile()
    ep = torch.export.export(gm, args)
    table = torch.export.default_decompositions()
    table.update(get_decompositions([torch.ops.aten.linalg_cross, torch.ops.aten._safe_softmax]))
    for op in list(table):  # keep nearest upsampling whole: coremltools' upsample op runs on the Neural Engine
        if "upsample" in str(op):
            table.pop(op)
    ep = ep.run_decompositions(table)
    return ct.convert(ep, convert_to="mlprogram", outputs=[ct.TensorType(name=n) for n in OUTPUT_NAMES],
                      compute_precision=_precision(batch, fp32), compute_units=ct.ComputeUnit.ALL,
                      minimum_deployment_target=ct.target.macOS14)


# --- 4. check -------------------------------------------------------------------------------------------------

def neural_engine_ops(package):
    """Number of ops CoreML places on the Neural Engine (0 means it dropped the whole program to the CPU)."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    m = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    path = m.get_compiled_model_path()
    try:
        plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        n = 0
        for op in plan.model_structure.program.functions["main"].block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            n += u is not None and "NeuralEngine" in type(u.preferred_compute_device).__name__
        return n
    finally:
        del m
        shutil.rmtree(path, ignore_errors=True)


def accuracy(package, batch, feeds, refs, units):
    """Over every recorded input (in batches of `batch`): mean |difference| to the reference over the valid
    frames, for all generations and for those that chose the same clip length; and how many did."""
    import coremltools as ct
    m = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
    path = m.get_compiled_model_path()
    errs, same = [], []
    try:
        for s in range(0, len(feeds) - batch + 1, batch):
            idx = range(s, s + batch)
            # inputs by name, as the engine feeds them
            out = m.predict({k: np.concatenate([feeds[i][k] for i in idx]) for k in INPUT_NAMES})
            motion, tokens = out["pred_global_motions"].reshape(batch, 64, 366), np.ravel(out["pred_num_tokens"])
            for j, i in enumerate(idx):
                rm, rt = refs[i]
                n = 4 * int(rt[0])
                same.append(int(tokens[j]) == int(rt[0]))
                errs.append(np.abs(motion[j, :n] - rm[0, :n]).mean())
    finally:
        del m
        shutil.rmtree(path, ignore_errors=True)
    errs, same = np.array(errs), np.array(same)
    return float(errs.mean()), float(errs[same].mean()), int(same.sum()), len(errs)


def reference_outputs(model, feeds):
    """NVIDIA's fp32 PyTorch outputs for every recorded input."""
    import torch
    torch.backends.mha.set_fastpath_enabled(False)  # the same attention path as the captured graph
    with torch.no_grad():
        return [[t.numpy() for t in model(*(torch.from_numpy(f[k]) for k in INPUT_NAMES))] for f in feeds]


def build(model, batch, out, feeds, refs, fp32=False):
    """Convert, check, and only then put the package at `out`."""
    t = time.perf_counter()
    mlm = to_coreml(model, batch, fp32)
    tmp = out.with_name(f".{out.name}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    mlm.save(str(tmp))
    compiled = mlm.get_compiled_model_path()  # don't leave a ~360 MB copy in $TMPDIR
    del mlm
    shutil.rmtree(compiled, ignore_errors=True)
    try:
        units = "ALL" if fp32 else "CPU_AND_NE"
        if not fp32:
            ne = neural_engine_ops(tmp)
            if ne == 0:
                raise RuntimeError("CoreML placed nothing on the Neural Engine (an ANE compile failed; see "
                                   "experiments/direct/mil_window.py bisect)")
        mean, mean_same, same, n = accuracy(tmp, batch, feeds, refs, units)
        print(f"  {units}: mean|d| vs NVIDIA PyTorch {mean:.4f} ({mean_same:.4f} where the clip length matches), "
              f"same clip length {same}/{n}{'' if fp32 else f', {ne} ops on the Neural Engine'} "
              f"({time.perf_counter() - t:.0f} s)", flush=True)
        if mean_same > MAX_MEAN_DIFF or n - same > max(1, int(n * MAX_OTHER_LENGTH)):
            raise RuntimeError(f"batch {batch} package does not match NVIDIA's model")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


# --- recorded inputs ------------------------------------------------------------------------------------------

def capture_feeds(dst=FEEDS):
    """Record real network inputs from NVIDIA's runtime (walk, turn, run, both gym jumps) using ONNX Runtime and
    NVIDIA's published ONNX model. They are what the CoreML models are captured and checked on."""
    from .sim import Session
    s = Session(backend="ort")
    s.engine.recorded_feeds = []
    dt = 1 / 30
    for i in range(int(9 / dt)):
        t = i * dt
        move = (0, 1) if t < 3 else (1, 0) if t < 6 else (0, 0)
        s.step(dt, move=move, facing=move if move != (0, 0) else (0, 1), run=3 < t < 6)
    for p in s.primitives[:2]:
        goal = np.array(p["def"].transform.p[:2])
        for _ in range(30 * 15):
            d = goal - s.character.character_ground_position[:2]
            h = d / max(np.linalg.norm(d), 1e-6)
            s.step(dt, move=tuple(h), facing=tuple(h))
    feeds = s.engine.recorded_feeds
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **{f"{k}_{i}": v for i, f in enumerate(feeds) for k, v in f.items()}, n=len(feeds))
    return len(feeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=list(BATCH_SIZES))
    ap.add_argument("--fp32", action="store_true", help="also build the fp32 batch-1 package")
    args = ap.parse_args()
    t0 = time.perf_counter()
    from .upstream_model import build_model, download_checkpoints
    if not FEEDS.exists():
        if not SOURCE_ONNX.exists() or SOURCE_ONNX.stat().st_size < 1_000_000:
            raise SystemExit(f"{SOURCE_ONNX} is missing or a Git LFS pointer; run ./setup.sh first")
        print("recording real inputs from NVIDIA's runtime ...", flush=True)
        print(f"  {capture_feeds()} generations recorded", flush=True)
    print("NVIDIA checkpoints (~800 MB on first run) ...", flush=True)
    download_checkpoints()
    model = build_model()
    feeds = all_feeds()
    refs = reference_outputs(model, feeds)
    jobs = [(b, coreml_package(b), False) for b in args.batches]
    if args.fp32:
        jobs.append((1, coreml_package_fp32(), True))
    for batch, out, fp32 in jobs:
        print(f"converting batch {batch}{' fp32' if fp32 else ''} -> {out.relative_to(COREML_DIR.parent.parent)} ...",
              flush=True)
        build(model, batch, out, feeds, refs, fp32)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min; packages in {COREML_DIR}", flush=True)


if __name__ == "__main__":
    main()
