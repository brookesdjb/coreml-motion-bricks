"""Convert NVIDIA's MotionBricks SOMA ONNX model to CoreML packages that run correctly on the Neural Engine.

    python -m cmb.convert                 # batch 1, 2, 4, 8 fp16 packages (+ ORT graph, fp32 reference)
    python -m cmb.convert --fp32          # also the fp32 batch-1 package
    python -m cmb.convert --batches 1     # just what the single-character demo needs

Pipeline, per batch size B:
  1. simplify: pin the batch to B and constant-fold (onnxsim).
  2. upcast:   rewrite the mixed fp16/fp32 graph as pure fp32. ONNX Runtime's CPU kernels quietly compute much
               of the fp16 graph in fp32; PyTorch and CoreML honour its explicit fp16 casts, and the root-motion
               decode divides by small numbers, so the fp16 rounding showed up as foot sliding.
  3. onnx2torch -> torch.jit.trace -> coremltools, with the fixes below, traced on real recorded inputs.

Fixes (each found by comparing against ONNX Runtime; see docs/findings.md):
  * onnx2torch 1.5.15 CumSum treats a missing `reverse` attribute as 1 (ONNX default is 0): the heading was
    integrated backwards. Re-registered with the right default.
  * Missing converters: ScatterElements, OneHot, LpNormalization; empty optional inputs; negative CumSum axes;
    torch op names coremltools doesn't know (greater, broadcast_to, resolve_conj, ...).
  * Neural Engine at batch > 4 gave different results per batch slot for identical inputs. Three workarounds:
    scatter written as a compare-and-select mask; Conv1d folded into a Conv2d that sees batch 1; ops on tensors
    under 256 elements kept in fp32 (CoreML then runs those few on the CPU).
"""
import argparse
import shutil
import time

import numpy as np

from .paths import (BATCH_SIZES, COREML_DIR, FEEDS, ONNX_DIR, REFERENCE_ONNX, SOURCE_ONNX, STATIC_ONNX,
                    coreml_package, coreml_package_fp32)

INPUT_NAMES = ["global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
               "local_poses", "has_local_poses", "allowed_pred_num_tokens"]


# --- 1. simplify ----------------------------------------------------------------------------------------------

def simplify(batch, dst=None):
    """Pin the batch size and constant-fold the source graph."""
    import onnx
    import onnxsim
    dst = dst or ONNX_DIR / f"soma_fp16_b{batch}.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shapes = {"global_root_values": [batch, 8, 5], "has_global_root_values": [batch, 8],
              "local_root_values": [batch, 8, 4], "has_local_root_values": [batch, 8],
              "local_poses": [batch, 8, 267], "has_local_poses": [batch, 8], "allowed_pred_num_tokens": [batch, 11]}
    model, ok = onnxsim.simplify(onnx.load(str(SOURCE_ONNX)), overwrite_input_shapes=shapes)
    if not ok:
        raise RuntimeError("onnxsim could not validate the simplified graph")
    onnx.save(model, str(dst))
    return dst


# --- 2. upcast ------------------------------------------------------------------------------------------------

def upcast(src, dst):
    """Rewrite as pure fp32: fp16 weights become fp32 (lossless) and casts to fp16 become casts to fp32."""
    import onnx
    import onnx.numpy_helper as nh
    from onnx import TensorProto as T
    m = onnx.load(str(src))
    g = m.graph
    for i, init in enumerate(g.initializer):
        if init.data_type == T.FLOAT16:
            g.initializer[i].CopyFrom(nh.from_array(nh.to_array(init).astype(np.float32), init.name))
    for n in g.node:
        for a in n.attribute:
            if n.op_type == "Cast" and a.name == "to" and a.i == T.FLOAT16:
                a.i = T.FLOAT
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == T.FLOAT16:
                a.t.CopyFrom(nh.from_array(nh.to_array(a.t).astype(np.float32), a.t.name))
    del g.value_info[:]  # re-inferred below
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return dst


# --- 3. ONNX -> PyTorch -> CoreML ------------------------------------------------------------------------------

_registered = False


def _register_fixes():
    """Converter fixes and additions for onnx2torch and coremltools (idempotent)."""
    global _registered
    if _registered:
        return
    import torch
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
    from onnx2torch.node_converters.cumsum import OnnxCumSum
    from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
    from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node

    def register(op_type, versions, make_module):
        def convert(node, graph):
            return OperationConverterResult(torch_module=make_module(node), onnx_mapping=onnx_mapping_from_node(node))
        for v in versions:
            _CONVERTER_REGISTRY[OperationDescription(domain="", operation_type=op_type, version=v)] = convert

    class ScatterElements(torch.nn.Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, data, indices, updates):
            axis = self.axis % data.dim()
            if indices.shape[axis] != 1:
                return torch.scatter(data, self.axis, indices.long(), updates)
            # One write per position along `axis` (every scatter in this model): a compare-and-select mask
            # instead of scatter_along_axis, which the Neural Engine got wrong above batch 4.
            n = data.shape[axis]
            shape = [1] * data.dim()
            shape[axis] = n
            mask = torch.arange(n, device=data.device).reshape(shape) == (indices.long() % n)
            return torch.where(mask, updates.expand_as(data) if updates.shape != data.shape else updates, data)

    class OneHot(torch.nn.Module):  # axis=-1, values = [off, on]; depth is constant after simplification
        def forward(self, indices, depth, values):
            n = int(depth.reshape(-1)[0])
            hot = ((indices.long() % n).unsqueeze(-1) == torch.arange(n, device=indices.device)).to(values.dtype)
            return hot * (values[1] - values[0]) + values[0]

    class LpNormalization(torch.nn.Module):  # onnxsim emits it at batch > 1
        def __init__(self, axis, p):
            super().__init__()
            self.axis, self.p = axis, p

        def forward(self, x):
            return x / torch.linalg.vector_norm(x, ord=self.p, dim=self.axis, keepdim=True)

    register("ScatterElements", (16, 18), lambda n: ScatterElements(n.attributes.get("axis", 0)))
    register("OneHot", (11,), lambda n: OneHot())
    register("LpNormalization", (1,), lambda n: LpNormalization(n.attributes.get("axis", -1), n.attributes.get("p", 2)))
    # onnx2torch 1.5.15 bug: `reverse` defaults to 1 when absent (ONNX default 0).
    register("CumSum", (11, 14), lambda n: OnnxCumSum(bool(n.attributes.get("exclusive", 0)),
                                                       bool(n.attributes.get("reverse", 0))))

    names = _TORCH_OPS_REGISTRY.name_to_func_mapping
    for long, short in {"greater": "gt", "less": "lt", "greater_equal": "ge", "less_equal": "le",
                        "not_equal": "ne", "broadcast_to": "expand", "Int": "int"}.items():
        if long not in names and short in names:
            names[long] = names[short]
    for name in ("resolve_conj", "resolve_neg"):  # no-ops for real tensors
        if name not in names:
            def identity(context, node):
                context.add(_get_inputs(context, node, expected=1)[0], torch_name=node.name)
            identity.__name__ = name
            register_torch_op(identity)
    _registered = True


def load_feeds(batch):
    """A batch of `batch` different real recorded generations (build/feeds.npz)."""
    z = np.load(FEEDS)
    n = int(z["n"])
    return {k: np.concatenate([z[f"{k}_{(i * 7) % n}"] for i in range(batch)]) for k in INPUT_NAMES}


def torch_model(src_onnx, batch):
    """onnx2torch model of `src_onnx` with every fix applied (also used by tools/bisect_batch.py)."""
    import onnx
    import onnx.numpy_helper as nh
    import onnx2torch
    import torch
    _register_fixes()
    g = onnx.load(str(src_onnx))
    for n in g.graph.node:  # drop trailing empty optional inputs (e.g. Clip with only min)
        while n.input and n.input[-1] == "":
            n.input.pop()
    # coremltools rejects negative CumSum axes: rewrite them as positive using inferred ranks.
    g = onnx.shape_inference.infer_shapes(g)
    ranks = {v.name: len(v.type.tensor_type.shape.dim) for v in list(g.graph.value_info) + list(g.graph.input)}
    consts = {i.name: nh.to_array(i) for i in g.graph.initializer}
    for n in g.graph.node:
        if n.op_type == "CumSum" and int(consts[n.input[1]]) < 0:
            g.graph.initializer.append(nh.from_array(
                np.array(int(consts[n.input[1]]) + ranks[n.input[0]], dtype=np.int64), n.input[1] + "_pos"))
            n.input[1] += "_pos"
    tm = onnx2torch.convert(g).eval()
    if batch > 4:
        _fold_conv1d(tm)
    return tm


def _fold_conv1d(tm):
    """Compute each Conv1d as a Conv2d over (1, C, B, L) with a 1-tall kernel: same maths, but the Neural Engine
    sees batch 1 (above batch 4 it mixed up slots)."""
    import torch

    class BatchFoldedConv1d(torch.nn.Module):
        def __init__(self, conv):
            super().__init__()
            self.conv = torch.nn.Conv2d(conv.in_channels, conv.out_channels, (1, conv.kernel_size[0]),
                                        stride=(1, conv.stride[0]), padding=(0, conv.padding[0]),
                                        dilation=(1, conv.dilation[0]), groups=conv.groups,
                                        bias=conv.bias is not None)
            with torch.no_grad():
                self.conv.weight.copy_(conv.weight.unsqueeze(2))
                if conv.bias is not None:
                    self.conv.bias.copy_(conv.bias)

        def forward(self, x):  # (B, C, L)
            return self.conv(x.permute(1, 0, 2).unsqueeze(0)).squeeze(0).permute(1, 0, 2)

    for name, mod in list(tm.named_modules()):
        if isinstance(mod, torch.nn.Conv1d) and mod.padding_mode == "zeros":
            parent = tm.get_submodule(name.rsplit(".", 1)[0]) if "." in name else tm
            setattr(parent, name.rsplit(".", 1)[-1], BatchFoldedConv1d(mod))


def to_coreml(src_fp32_onnx, batch, out, fp32=False):
    """Trace the fixed torch model on real inputs and convert to a CoreML ML Program."""
    import coremltools as ct
    import onnxruntime as ort
    import torch
    feeds = load_feeds(batch)
    tm = torch_model(src_fp32_onnx, batch)
    args = tuple(torch.from_numpy(feeds[n]) for n in INPUT_NAMES)
    ref = ort.InferenceSession(str(src_fp32_onnx), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    with torch.no_grad():
        per_slot = np.abs(tm(*args)[0].numpy() - ref).reshape(batch, -1).max(axis=1)
        # A converter bug corrupts every slot. A single slot can legitimately differ: the model picks tokens by
        # ArgMax, and on a near-tie, float rounding in the batched maths can pick a different (valid) token.
        if np.median(per_slot) > 1e-3:
            raise RuntimeError(f"torch model differs from ONNX Runtime (per slot {per_slot.round(4)}); "
                               "a converter fix is missing")
        traced = torch.jit.trace(tm, args, check_trace=False)
    if fp32:
        precision = ct.precision.FLOAT32
    elif batch > 4:
        def fp16_unless_small(op):  # tiny root-conditioning tensors stay fp32 (and so on the CPU)
            sizes = []
            for o in op.outputs:
                shape = getattr(o, "shape", None)
                if shape is None or any(not isinstance(d, int) for d in shape):
                    return True
                sizes.append(int(np.prod(shape)) if len(shape) else 1)
            return max(sizes, default=0) >= 256
        precision = ct.transform.FP16ComputePrecision(op_selector=fp16_unless_small)
    else:
        precision = ct.precision.FLOAT16
    mlm = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=feeds[n].shape, dtype=feeds[n].dtype)
                                     for n in INPUT_NAMES],
                     convert_to="mlprogram", compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
                     minimum_deployment_target=ct.target.macOS14)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    mlm.save(str(out))
    compiled = mlm.get_compiled_model_path()  # don't leave a ~360 MB copy in $TMPDIR
    del mlm
    shutil.rmtree(compiled, ignore_errors=True)
    return out


# --- recorded inputs ------------------------------------------------------------------------------------------

def capture_feeds(dst=FEEDS):
    """Record real network inputs from NVIDIA's runtime (walk, turn, run, both gym jumps) using ONNX Runtime.
    They are used to trace the CoreML models and by tools/check_accuracy.py."""
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
    np.savez_compressed(dst, **{f"{k}_{i}": v for i, f in enumerate(feeds) for k, v in f.items()}, n=len(feeds))
    return len(feeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=list(BATCH_SIZES))
    ap.add_argument("--fp32", action="store_true", help="also build the fp32 batch-1 package")
    ap.add_argument("--keep-onnx", action="store_true", help="keep intermediate ONNX graphs (~0.7 GB each)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if not SOURCE_ONNX.exists() or SOURCE_ONNX.stat().st_size < 1_000_000:
        raise SystemExit(f"{SOURCE_ONNX} is missing or a Git LFS pointer; run ./setup.sh first")
    if not STATIC_ONNX.exists():
        print("simplifying batch-1 graph (ONNX Runtime backend) ...", flush=True)
        simplify(1, STATIC_ONNX)
    if not REFERENCE_ONNX.exists():
        print("writing fp32 reference graph ...", flush=True)
        upcast(STATIC_ONNX, REFERENCE_ONNX)
    if not FEEDS.exists():
        print("recording real inputs from NVIDIA's runtime ...", flush=True)
        print(f"  {capture_feeds()} generations recorded", flush=True)
    jobs = [(b, coreml_package(b), False) for b in args.batches]
    if args.fp32:
        jobs.append((1, coreml_package_fp32(), True))
    for batch, out, fp32 in jobs:
        print(f"converting batch {batch}{' fp32' if fp32 else ''} -> {out.relative_to(COREML_DIR.parent.parent)} ...",
              flush=True)
        fp32_src = REFERENCE_ONNX if batch == 1 else upcast(simplify(batch), ONNX_DIR / f"soma_fp32_b{batch}.onnx")
        to_coreml(fp32_src, batch, out, fp32=fp32)
        if batch != 1 and not args.keep_onnx:
            for p in (ONNX_DIR / f"soma_fp16_b{batch}.onnx", ONNX_DIR / f"soma_fp32_b{batch}.onnx"):
                p.unlink(missing_ok=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min; packages in {COREML_DIR}", flush=True)


if __name__ == "__main__":
    main()
