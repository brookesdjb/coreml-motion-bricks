"""Find the first node in the direct graph whose inclusion makes CoreML put nothing on the Neural Engine."""
import sys, os, shutil, collections; sys.path.insert(0, "."); sys.path.insert(0, "experiments/direct")
import numpy as np, torch, coremltools as ct
import ctfix, capshim, ipfix  # noqa
torch.backends.mha.set_fastpath_enabled(False)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch._decomp import get_decompositions
from coremltools.models.compute_plan import MLComputePlan
from cmb.upstream_model import build_model
from cmb.convert import INPUT_NAMES, load_feeds
m = build_model(); f = load_feeds(1)
args = tuple(torch.from_numpy(f[k]) for k in INPUT_NAMES)
with torch.no_grad():
    capshim.install()
    with sdpa_kernel(SDPBackend.MATH):
        gm = make_fx(functionalize(m), tracing_mode="real")(*args)
    ipfix.rewrite_index_put(gm); ipfix.rewrite_int_scatter(gm); ipfix.rewrite_scatter_value(gm)
    ipfix.positive_cumsum_axes(gm); ipfix.static_fills_and_sort(gm)
    ipfix.scatters_as_masks(gm); ipfix.gathers_as_onehot(gm)
vals = {}
from torch.fx import Interpreter
class Rec(Interpreter):
    def run_node(self, n):
        r = super().run_node(n); vals[n.name] = r; return r
with torch.no_grad(): Rec(gm).run(*args)
cands = [n for n in gm.graph.nodes if n.op == "call_function" and torch.is_tensor(vals.get(n.name))
         and vals[n.name].is_floating_point() and vals[n.name].numel() > 1]
table = torch.export.default_decompositions(); table.update(get_decompositions([torch.ops.aten.linalg_cross]))

def prefix(cut):
    cuts = cut if isinstance(cut, (list, tuple)) else [cut]
    names = {c.name for c in cuts}
    g, env = torch.fx.Graph(), {}
    for n in gm.graph.nodes:
        if n.op == "output": continue
        env[n.name] = g.node_copy(n, lambda x: env[x.name])
        names.discard(n.name)
        if not names: break
    g.output(tuple(env[c.name] for c in cuts) if len(cuts) > 1 else env[cuts[0].name])
    p = torch.fx.GraphModule(gm, g); p.graph.eliminate_dead_code(); p.recompile(); return p

def ne_share(cut):
    with torch.no_grad():
        ep = torch.export.export(prefix(cut), args).run_decompositions(table)
    mlm = ct.convert(ep, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                     compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.macOS14)
    path = mlm.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    devs = collections.Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is not None: devs[type(u.preferred_compute_device).__name__] += 1
    del mlm; shutil.rmtree(path, ignore_errors=True)
    return devs.get("MLNeuralEngineComputeDevice", 0), sum(devs.values())

lo, hi = (int(os.environ["LO"]), int(os.environ["HI"])) if os.environ.get("LO") else (0, 1)
cache = {}
def verdict(i):
    """'bad' = big program with nothing on the Neural Engine; 'good' = some on the NE; 'small' = pruned too small."""
    if i not in cache:
        ne, tot = ne_share(cands[i])
        cache[i] = "good" if ne else ("bad" if tot > 500 else "small")
        print(f"  node {i:4d} {cands[i].name:<36} {str(cands[i].target)[:34]:<34} NE ops {ne}/{tot} -> {cache[i]}", flush=True)
    return cache[i]
while os.environ.get("LO") and hi - lo > 1:
    mid = (lo + hi) // 2
    v = verdict(mid)
    j = mid
    while v == "small" and j < hi - 1:   # step forward to the next prefix big enough to judge
        j += 1; v = verdict(j)
    if v == "bad": hi = j
    else: lo = j
n = cands[hi] if os.environ.get("LO") else None
if n is not None:
    print("\nfirst node that makes the whole program CPU-only:", n.name, n.target)
if n is not None: print("args:", [(a.name, tuple(vals[a.name].shape), str(vals[a.name].dtype)) if hasattr(a, "name") and torch.is_tensor(vals.get(a.name)) else a for a in n.args])
if n is not None: print("output:", tuple(vals[n.name].shape), vals[n.name].dtype)

if os.environ.get("SHOW"):
    byname = {x.name: x for x in gm.graph.nodes}
    def show(x, d=0):
        if not hasattr(x, "name") or d > 7: return
        v = vals.get(x.name)
        info = f"{tuple(v.shape)} {str(v.dtype).replace('torch.','')}" if torch.is_tensor(v) else ""
        if torch.is_tensor(v) and v.numel() <= 8: info += f" = {v.flatten().tolist()}"
        print("  " * d + f"{x.name} [{str(x.target)[:45]}] {info}")
        for a in x.all_input_nodes: show(a, d + 1)
    show(byname[os.environ["SHOW"]])
    raise SystemExit

suspects = os.environ.get("SUSPECTS")
if suspects:
    byname = {x.name: x for x in gm.graph.nodes}
    for name in suspects.split(","):
        if "+" in name:
            ne, tot = ne_share([byname[x] for x in name.split("+")])
            print(f"  output={name:<40} NE ops {ne}/{tot}", flush=True); continue
        if name not in byname:
            print(f"  output={name:<22} (not in graph after rewrites)", flush=True); continue
        ne, tot = ne_share(byname[name])
        print(f"  output={name:<22} {str(byname[name].target)[:40]:<40} NE ops {ne}/{tot}", flush=True)
