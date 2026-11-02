import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/direct")
import os, numpy as np, torch, coremltools as ct, time
torch.backends.mha.set_fastpath_enabled(False)
from cmb.upstream_model import build_model
from cmb.convert import INPUT_NAMES
from cmb.paths import FEEDS
m = build_model()
z = np.load(FEEDS); n = int(z["n"])
feeds = [{k: z[f"{k}_{i}"] for k in INPUT_NAMES} for i in range(n)]
with torch.no_grad():
    refs = [[t.numpy() for t in m(*(torch.from_numpy(f[k]) for k in INPUT_NAMES))] for f in feeds]
for B in map(int, sys.argv[1:]):
    for units in os.environ.get("UNITS", "CPU_AND_NE,CPU_AND_GPU").split(","):
        mm = ct.models.MLModel(os.environ.get("PKG", "build/coreml/direct_b{B}.mlpackage").format(B=B), compute_units=getattr(ct.ComputeUnit, units))
        names = [i.name for i in mm.get_spec().description.input]
        errs, same, ts = [], 0, []
        for s in range(0, n - B + 1, B):
            idx = list(range(s, s + B))
            feed = {nm: np.concatenate([feeds[i][k] for i in idx]) for nm, k in zip(names, INPUT_NAMES)}
            t = time.perf_counter(); o = mm.predict(feed); ts.append(time.perf_counter() - t)
            mot = next(v for v in o.values() if np.size(v) > 100 * B).reshape(B, 64, 366)
            tok = np.ravel(next(v for v in o.values() if np.size(v) == B))
            for j, i in enumerate(idx):
                rm, rt = refs[i]; k = 4 * int(rt[0]); same += int(tok[j]) == int(rt[0])
                errs.append(np.abs(mot[j, :k] - rm[0, :k]).mean())
        print(f"direct batch {B} {units:<12} mean|d| vs NVIDIA PyTorch {np.mean(errs):.4f}  same length {same}/{len(errs)}  {np.median(ts[1:])*1000:.1f} ms/call", flush=True)
