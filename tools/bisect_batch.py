"""Find the first operation that gives different results per batch slot on the Neural Engine.

With the same input in every batch slot, every slot must produce the same output. This cuts the converted
torch graph after a chosen node, converts that prefix to CoreML (fp16), runs it on the Neural Engine and
measures the spread between slots; a binary search over the node order finds where the slots diverge.
This is how the batch > 4 workarounds in cmb/convert.py were found.

    python -m cmb.convert --batches 8 --keep-onnx       # needs build/onnx/soma_fp32_b8.onnx
    python tools/bisect_batch.py --batch 8 [--threshold 0.001] [--lo N --hi M] [--cuts NodeA,NodeB]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.fx import Interpreter  # noqa: E402

from cmb.convert import INPUT_NAMES, torch_model  # noqa: E402
from cmb.paths import FEEDS, ONNX_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.02, help="relative slot spread that counts as broken")
    ap.add_argument("--lo", type=int)
    ap.add_argument("--hi", type=int)
    ap.add_argument("--cuts", help="comma-separated node names to test directly instead of bisecting")
    args = ap.parse_args()
    import coremltools as ct

    src = ONNX_DIR / f"soma_fp32_b{args.batch}.onnx"
    if not src.exists():
        raise SystemExit(f"{src} missing: run python -m cmb.convert --batches {args.batch} --keep-onnx")
    tm = torch_model(src, args.batch)
    z = np.load(FEEDS)
    same = {k: np.concatenate([z[f"{k}_3"]] * args.batch) for k in INPUT_NAMES}  # identical input in every slot
    inputs = tuple(torch.from_numpy(same[n]) for n in INPUT_NAMES)

    vals = {}

    class Record(Interpreter):
        def run_node(self, n):
            r = super().run_node(n)
            vals[n.name] = r
            return r
    with torch.no_grad():
        Record(tm).run(*inputs)
    nodes = [n for n in tm.graph.nodes if n.op == "call_module" and torch.is_tensor(vals.get(n.name))
             and vals[n.name].is_floating_point() and vals[n.name].dim() >= 1
             and vals[n.name].shape[0] == args.batch and vals[n.name].numel() > args.batch]

    def prefix(cut):
        g, env = torch.fx.Graph(), {}
        for n in tm.graph.nodes:
            if n.op == "output":
                continue
            env[n.name] = g.node_copy(n, lambda x: env[x.name])
            if n is cut:
                break
        g.output(env[cut.name])
        gm = torch.fx.GraphModule(tm, g)  # dead-code removal needs the owning module
        gm.graph.eliminate_dead_code()
        gm.recompile()
        return gm.eval()

    def spread(cut):
        with torch.no_grad():
            traced = torch.jit.trace(prefix(cut), inputs, check_trace=False)
        mlm = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=same[n].shape, dtype=same[n].dtype)
                                         for n in INPUT_NAMES],
                         convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                         compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.macOS14)
        out = np.asarray(next(iter(mlm.predict(same).values())), dtype=np.float64)
        compiled = mlm.get_compiled_model_path()
        del mlm
        shutil.rmtree(compiled, ignore_errors=True)  # don't leave a copy per conversion in $TMPDIR
        return float(np.abs(out - out[0:1]).max() / (np.abs(out[0]).max() + 1e-6))

    if args.cuts:
        by_name = {str(n.target): n for n in tm.graph.nodes} | {n.name: n for n in tm.graph.nodes}
        for c in args.cuts.split(","):
            print(f"{c:<45} slot spread {spread(by_name[c]):.4f}", flush=True)
        return
    lo = args.lo if args.lo is not None else 0
    hi = args.hi if args.hi is not None else len(nodes) - 1
    cache = {}

    def bad(i):
        if i not in cache:
            s = spread(nodes[i])
            cache[i] = s > args.threshold
            print(f"node {i:4d} {str(nodes[i].target):<55} slot spread {s:.4f} {'BAD' if cache[i] else 'ok'}",
                  flush=True)
        return cache[i]

    print(f"{len(nodes)} candidate cut points", flush=True)
    if not bad(hi):
        print("no divergence at the upper bound")
        return
    while lo < hi:
        mid = (lo + hi) // 2
        if bad(mid):
            hi = mid
        else:
            lo = mid + 1
    print(f"\nfirst divergent node: {nodes[lo].target}  output {tuple(vals[nodes[lo].name].shape)}")


if __name__ == "__main__":
    main()
