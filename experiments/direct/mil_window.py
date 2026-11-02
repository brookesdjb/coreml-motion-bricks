"""Cut a converted .mlpackage down to the ops in [i, j) of its MIL main block and ask CoreML where they would run.

Used to find the one Neural Engine segment whose compile fails (system log: ANECompiler "Fail to call
ZinComputeMutableProgramAddObjects"), which makes CoreML drop the whole program to the CPU.

  python mil_window.py <pkg> list                 # print ops with index
  python mil_window.py <pkg> <i> <j>              # NE op count for window [i, j)
  python mil_window.py <pkg> bisect               # smallest window [i, j) that still has 0 NE ops
"""
import sys, collections, shutil, os
import numpy as np
import coremltools as ct
from coremltools.models import utils
from coremltools.models.utils import load_spec, _try_get_weights_dir_path
from coremltools.converters.mil.frontend.milproto.load import load as milproto_load
from coremltools.converters.mil.converter import mil_convert
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.models.compute_plan import MLComputePlan

PKG = sys.argv[1]
spec = load_spec(PKG); wdir = _try_get_weights_dir_path(PKG)

def fresh():
    prog = milproto_load(spec, spec.specificationVersion, wdir)
    f = prog.functions["main"]; f.operations = list(f.operations)
    return prog

def window(i, j):
    prog = fresh()
    ops = prog.functions["main"].operations
    if j < len(ops):
        b = prog.functions["main"]
        outs = utils._get_first_chunk_outputs(b, j)
        extra = set(filter(None, os.environ.get("EXTRA", "").split(",")))  # also expose these vars
        outs = list(outs) + [o.outputs[0] for o in ops[:j] if o.outputs and o.outputs[0].name in extra and o.outputs[0] not in outs]
        for v in list(b.outputs):          # drop the old outputs (clear() alone leaves stale bookkeeping)
            v.consuming_blocks.remove(b)
        b.outputs.clear(); b.set_outputs(outs)
        PASS_REGISTRY["common::dead_code_elimination"](prog)
    if i > 0:
        # the second-chunk builder makes every value crossing op i an input; find op i (or the next op that
        # survived pruning) by name, since pruning shifted the indices
        names = [o.name for o in ops]
        now = {o.name: k for k, o in enumerate(prog.functions["main"].operations)}
        k = next(now[n] for n in names[i:] if n in now)
        prog = utils._make_second_chunk_prog(prog, k)
    return prog

def ne_count(i, j, keep=None):
    prog = window(i, j)
    n_ops = len(prog.functions["main"].operations)
    m = mil_convert(prog, convert_to="mlprogram", convert_from="milinternal",
                    specification_version=spec.specificationVersion, compute_units=ct.ComputeUnit.CPU_AND_NE,
                    pass_pipeline=ct.PassPipeline.EMPTY, skip_model_load=True)
    if keep:
        m.save(keep)
    tmp = f"{os.environ.get('TMPDIR', '/tmp')}/mil_window.mlpackage"
    shutil.rmtree(tmp, ignore_errors=True); m.save(tmp)
    cpath = ct.utils.compile_model(tmp)
    plan = MLComputePlan.load_from_path(cpath, compute_units=ct.ComputeUnit.CPU_AND_NE)
    devs, segs, last = collections.Counter(), collections.Counter(), None
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is not None:
            dev = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
            devs[dev] += 1
            if op.operator_name != "const":
                segs[dev + "_segments"] += dev != last; last = dev
    shutil.rmtree(cpath, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
    return devs.get("NeuralEngine", 0), sum(devs.values()), {**devs, **segs}

if __name__ == "__main__":
    ops = fresh().functions["main"].operations
    N = len(ops)
    if sys.argv[2] == "list":
        lo, hi = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, N)
        for k in range(lo, hi):
            o = ops[k]
            ins = {n: (v.name if hasattr(v, "name") else [x.name for x in v]) for n, v in o.inputs.items()
                   if not (hasattr(v, "op") and v.op is not None and v.op.op_type == "const")}
            print(k, o.op_type, o.outputs[0].name, [(str(x.dtype).split(".")[-1].replace("'>", ""), x.shape) for x in o.outputs], ins)
    elif sys.argv[2] == "bisect":
        # end: smallest j with [0, j) bad; then start: largest i with [i, j) bad. "bad" = ops on NE == 0
        # while a sizeable number of ops are non-const.
        def bad(i, j):
            for k in range(8):  # some cut points make an invalid program; nudge the cut
                try:
                    ne, tot, d = ne_count(i, j); break
                except Exception as e:
                    print(f"  [{i}, {j}) invalid cut: {str(e)[:60]}", flush=True)
                    if i: i += 1
                    else: j += 1
            print(f"  [{i}, {j}) ops {tot} {d}", flush=True)
            return ne == 0 and tot > 3
        lo, hi = int(os.environ.get("JLO", 0)), int(os.environ.get("JHI", N))
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if bad(0, mid): hi = mid
            else: lo = mid
        j = hi
        lo, hi = 0, j - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if bad(mid, j): lo = mid
            else: hi = mid
        print(f"\nsmallest failing window: [{lo}, {j})  ({ops[lo].op_type} .. {ops[j-1].op_type} {ops[j-1].outputs[0].name})")
    elif sys.argv[2] != "diverge":
        i, j = int(sys.argv[2]), int(sys.argv[3])
        print(ne_count(i, j, keep=sys.argv[4] if len(sys.argv) > 4 else None))

REF = np.load(os.environ["REF"]) if os.environ.get("REF") else None

def diverge(i, j, feed):
    """Max |GPU - CPU| / (|CPU| max + 1e-3) over the float outputs of window [0, j)."""
    prog = window(0, j)
    m = mil_convert(prog, convert_to="mlprogram", convert_from="milinternal",
                    specification_version=spec.specificationVersion, compute_units=ct.ComputeUnit.CPU_ONLY,
                    pass_pipeline=ct.PassPipeline.EMPTY, skip_model_load=True)
    tmp = f"{os.environ.get('TMPDIR', '/tmp')}/mil_window_d.mlpackage"
    shutil.rmtree(tmp, ignore_errors=True); m.save(tmp)
    outs = {}
    if REF is not None:  # compare one device against fp32 PyTorch values of the same graph nodes
        mm = ct.models.MLModel(tmp, compute_units=getattr(ct.ComputeUnit, os.environ.get("VS", "CPU_AND_GPU")))
        names = {i.name for i in mm.get_spec().description.input}
        o = mm.predict({k: v for k, v in feed.items() if k in names})
        p = mm.get_compiled_model_path(); del mm; shutil.rmtree(p, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
        worst = (0.0, None)
        for k, y in o.items():
            base = k.replace("_cast_fp16", "").replace("_to_fp16", "").replace("_to_fp32", "")
            if base not in REF.files: continue
            x = REF[base].astype(np.float64)
            if x.shape != np.shape(y) and x.size == np.size(y): y = np.reshape(y, x.shape)
            if x.shape != np.shape(y): continue
            r = float(np.abs(x - np.asarray(y, np.float64)).max() / (np.abs(x).max() + 1e-3))
            worst = max(worst, (r, k), key=lambda t: t[0])
        return worst
    for units in ("CPU_ONLY", os.environ.get("VS", "CPU_AND_GPU")):
        mm = ct.models.MLModel(tmp, compute_units=getattr(ct.ComputeUnit, units))
        names = {i.name for i in mm.get_spec().description.input}
        outs[units] = mm.predict({k: v for k, v in feed.items() if k in names})
        p = mm.get_compiled_model_path(); del mm; shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    a, b = outs.values()
    worst = (0.0, None)
    for k in a:
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        if x.size == 0: continue
        r = float(np.abs(x - y).max() / (np.abs(x).max() + 1e-3))
        worst = max(worst, (r, k), key=lambda t: t[0])
    return worst

if __name__ == "__main__" and sys.argv[2] == "diverge":
    sys.path.insert(0, ".")
    from cmb.convert import INPUT_NAMES, load_feeds
    B = int(sys.argv[3]); f = load_feeds(B)
    names = [i.name for i in spec.description.input]
    feed = {nm: f[k] for nm, k in zip(names, INPUT_NAMES)}
    lo, hi = int(os.environ.get("JLO", 1)), int(os.environ.get("JHI", N))
    TH = float(os.environ.get("TH", 0.05))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            r, k = diverge(0, mid, feed)
        except Exception as e:
            print(f"  [0, {mid}) invalid: {str(e)[:80]}", flush=True); lo = mid; continue
        print(f"  [0, {mid}) worst rel diff {r:.4f} at {k}", flush=True)
        if r > TH: hi = mid
        else: lo = mid
    print(f"\nfirst diverging cut: {hi} ({ops[hi-1].op_type} {ops[hi-1].outputs[0].name})")
