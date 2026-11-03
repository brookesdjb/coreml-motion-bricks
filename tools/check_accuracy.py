"""Check every converted CoreML package against NVIDIA's fp32 PyTorch model on real recorded inputs.

    python tools/check_accuracy.py            # all batch sizes, Neural Engine + GPU
    python tools/check_accuracy.py --units CPU_AND_NE

For each package: mean absolute difference over the valid frames (NVIDIA's own fp16 ONNX model scores ~0.006
here), whether the model chose the same clip length, and per-call time. `python -m cmb.convert` runs the
same check on the Neural Engine before it saves a package.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from cmb.convert import INPUT_NAMES, all_feeds, reference_outputs  # noqa: E402
from cmb.engine import load_coreml, split_outputs  # noqa: E402
from cmb.paths import BATCH_SIZES, coreml_package, coreml_package_fp32  # noqa: E402
from cmb.upstream_model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", default=["CPU_AND_NE", "CPU_AND_GPU"])
    args = ap.parse_args()
    feeds = all_feeds()
    refs = reference_outputs(build_model(), feeds)
    packages = [(b, coreml_package(b)) for b in BATCH_SIZES] + [(1, coreml_package_fp32())]
    print(f"{len(feeds)} recorded generations; reference: NVIDIA's PyTorch model (fp32)")
    for batch, pkg in packages:
        if not pkg.exists():
            continue
        for units in args.units:
            m = load_coreml(pkg, units)
            errs, same, times = [], 0, []
            for s in range(0, len(feeds) - batch + 1, batch):
                idx = list(range(s, s + batch))
                t = time.perf_counter()
                motion, tokens = split_outputs(m.predict({k: np.concatenate([feeds[i][k] for i in idx])
                                                          for k in INPUT_NAMES}), batch)
                times.append(time.perf_counter() - t)
                for j, i in enumerate(idx):
                    rm, rt = refs[i]
                    same += int(tokens[j]) == int(rt[0])
                    n = 4 * int(rt[0])
                    errs.append(np.abs(motion[j, :n] - rm[0, :n]).mean())
            ok = np.mean(errs) < 0.02 and same >= 0.95 * len(errs)
            print(f"{pkg.name:<26} {units:<12} mean|d| {np.mean(errs):.4f}  same length {same}/{len(errs)}  "
                  f"{np.median(times[1:]) * 1000:5.1f} ms/call  {'OK' if ok else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    main()
