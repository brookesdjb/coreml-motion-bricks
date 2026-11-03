import sys, time, shutil, collections; sys.path.insert(0, "."); sys.path.insert(0, "experiments/direct")
import os
import numpy as np, torch, coremltools as ct
import ctfix, capshim, ipfix  # noqa
torch.backends.mha.set_fastpath_enabled(False)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize
from torch._decomp import get_decompositions
from cmb.upstream_model import build_model
from cmb.convert import INPUT_NAMES, load_feeds
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
m = build_model(); f = load_feeds(B)
args = tuple(torch.from_numpy(f[k]) for k in INPUT_NAMES)
with torch.no_grad():
    ref = m(*args)[0].numpy()
    capshim.install()
    SKIP = os.environ.get('SKIP', '').split(',')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):  # plain matmul/softmax attention, not a CPU-specific fused kernel
        gm = make_fx(functionalize(m), tracing_mode="real")(*args)
    print('rewrote index_put:', ipfix.rewrite_index_put(gm), 'int scatter:', ipfix.rewrite_int_scatter(gm), 'scatter.value:', ipfix.rewrite_scatter_value(gm), 'cumsum axes:', ipfix.positive_cumsum_axes(gm), 'finite masks:', ipfix.finite_masks(gm) if os.environ.get('FINITE_MASKS') else 0, 'fills/sort:', ipfix.static_fills_and_sort(gm) if 'fills_sort' not in SKIP else 0, 'scatter->mask:', ipfix.scatters_as_masks(gm) if 'scatter_mask' not in SKIP else 0, 'gather->onehot:', ipfix.gathers_as_onehot(gm) if 'gather_onehot' not in SKIP else 0, 'conv1d batch fold:', ipfix.fold_conv1d_batch(gm) if os.environ.get('FOLD_CONV') else 0, 'upsample->repeat:', ipfix.nearest_upsample_as_repeat(gm) if os.environ.get('REPEAT_UPSAMPLE') else 0, 'slice/select_scatter->cat:', ipfix.scatters_as_cat(gm) if os.environ.get('SCATTER_CAT') else 0, 'safe_softmax->float:', ipfix.safe_softmax_as_float(gm) if os.environ.get('FLOAT_SAFE_SOFTMAX') else 0, '; fx matches eager:', float(np.abs(gm(*args)[0].numpy() - ref).max()))
    ep = torch.export.export(gm, args)
    extra = [torch.ops.aten.linalg_cross, torch.ops.aten._safe_softmax]
    import os
    if os.environ.get("MINIMAL_DECOMP"):
        table = get_decompositions(extra)  # keep attention / layer_norm as whole ops for coremltools
    else:
        table = torch.export.default_decompositions(); table.update(get_decompositions(extra))
        # Keep nearest upsampling whole (coremltools' upsample op runs on the Neural Engine); decomposed it
        # becomes _unsafe_index gathers.
        for op in list(table):
            if "upsample" in str(op):
                table.pop(op)
    ep = ep.run_decompositions(table)
    if os.environ.get("DUMP_REF"):  # fp32 value of every float node of the converted graph, by node name
        vals = {}
        class Rec(torch.fx.Interpreter):
            def run_node(self, n):
                r = super().run_node(n)
                if torch.is_tensor(r) and r.is_floating_point() and r.numel() > 1:
                    vals[n.name] = r.numpy().copy()
                return r
        Rec(ep.module()).run(*args)
        np.savez(os.environ["DUMP_REF"], **vals); print("dumped", len(vals), "reference values")
t = time.time()
try:
    SMALL = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    def fp16_unless_small(op):
        sizes = []
        for o in op.outputs:
            shape = getattr(o, "shape", None)
            if shape is None or any(not isinstance(d, int) for d in shape):
                return True
            sizes.append(int(np.prod(shape)) if len(shape) else 1)
        return max(sizes, default=0) >= SMALL
    # NVIDIA's own fp16 rule (tools/onnx/precision.py): networks in fp16 except LayerNorm/Softmax/Resize;
    # geometry and integration in fp32. Approximated by op type, plus tiny tensors.
    SENSITIVE = set(os.environ.get("FP32_OPS", "").split(",")) | {"upsample_nearest_neighbor", "upsample_bilinear", "resize",
                 "sin", "cos", "atan", "atan2", "acos", "asin", "cumsum", "sqrt", "rsqrt", "real_div", "inverse",
                 "reduce_l2_norm", "reduce_sum_square", "pow", "exp", "log", "sign", "floor", "round"}
    seen = {}
    def nvidia_like(op):
        seen[op.op_type] = seen.get(op.op_type, 0) + 1
        if op.op_type in SENSITIVE:
            return False
        return fp16_unless_small(op) if SMALL else True
    precision = ct.transform.FP16ComputePrecision(op_selector=nvidia_like)
    mlm = ct.convert(ep, convert_to="mlprogram", compute_precision=precision, compute_units=ct.ComputeUnit.ALL,
                     minimum_deployment_target=getattr(ct.target, os.environ.get("TARGET", "macOS14")))
except Exception as e:
    print("CONVERT FAILED:", repr(e)[:500]); raise SystemExit
print(f"converted in {time.time()-t:.0f}s; fp32 op types kept:", sorted(k for k in seen if k in SENSITIVE))
OUT = os.environ.get("OUT", f"build/coreml/direct_b{B}.mlpackage")
mlm.save(OUT)
feed = {i.name: f[n] for i, n in zip(mlm.get_spec().description.input, INPUT_NAMES)}
out = next(v for v in mlm.predict(feed).values() if np.size(v) > 100 * B)
print("CoreML (ALL) vs PyTorch on sample:", float(np.abs(out.reshape(ref.shape) - ref).mean()))
p = mlm.get_compiled_model_path(); del mlm; shutil.rmtree(p, ignore_errors=True)
