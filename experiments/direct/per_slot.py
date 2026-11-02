"""Per-batch-slot error of a batched package vs NVIDIA PyTorch: which slots go wrong, on which compute units."""
import sys, os; sys.path.insert(0, "."); sys.path.insert(0, "experiments/direct")
import numpy as np, torch, coremltools as ct
torch.backends.mha.set_fastpath_enabled(False)
from cmb.upstream_model import build_model
from cmb.convert import INPUT_NAMES, load_feeds
pkg, B = sys.argv[1], int(sys.argv[2])
m = build_model(); f = load_feeds(B)
same = os.environ.get("SAME")  # every slot gets slot 0's input
if same: f = {k: np.repeat(v[:1], B, 0) for k, v in f.items()}
with torch.no_grad():
    refs = [[t.numpy() for t in m(*(torch.from_numpy(f[k][i:i+1]) for k in INPUT_NAMES))] for i in range(B)]
for units in os.environ.get("UNITS", "CPU_ONLY,CPU_AND_GPU,CPU_AND_NE").split(","):
    mm = ct.models.MLModel(pkg, compute_units=getattr(ct.ComputeUnit, units))
    names = [i.name for i in mm.get_spec().description.input]
    o = mm.predict({nm: f[k] for nm, k in zip(names, INPUT_NAMES)})
    mot = next(v for v in o.values() if np.size(v) > 100 * B).reshape(B, 64, 366)
    tok = np.ravel(next(v for v in o.values() if np.size(v) == B))
    print(f"{units:<12}", " ".join(f"{np.abs(mot[j, :4*int(refs[j][1][0])] - refs[j][0][0, :4*int(refs[j][1][0])]).mean():.3f}/{int(tok[j])}:{int(refs[j][1][0])}" for j in range(B)))
