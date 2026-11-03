"""Export the recorded network inputs and fp32 reference outputs for the Swift sample apps.

    python tools/export_swift_feeds.py                       # build/feeds.npz -> build/swift_feeds/
    python tools/export_swift_feeds.py --build /path/to/build --out /tmp/feeds

Writes one raw little-endian file per tensor (float32 for float inputs and outputs, int32 for the
masks and token counts) plus manifest.json describing names, dtypes and shapes. The reference
outputs come from the pure fp32 ONNX graph, as in tools/check_accuracy.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from cmb.convert import INPUT_NAMES  # noqa: E402
from cmb.paths import BUILD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=BUILD, help="build/ folder holding feeds.npz and onnx/")
    ap.add_argument("--out", type=Path, default=None, help="default: <build>/swift_feeds")
    args = ap.parse_args()
    out = args.out or args.build / "swift_feeds"
    out.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    z = np.load(args.build / "feeds.npz")
    ref = ort.InferenceSession(str(args.build / "onnx" / "soma_fp32_b1.onnx"), providers=["CPUExecutionProvider"])
    samples = []
    for i in range(int(z["n"])):
        feed = {k: z[f"{k}_{i}"] for k in INPUT_NAMES}
        motion, tokens = ref.run(None, feed)
        tensors = {**feed, "ref_pred_global_motions": motion.astype(np.float32),
                   "ref_pred_num_tokens": np.asarray(tokens).astype(np.int32)}
        entry = {}
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.int32 if arr.dtype.kind in "iu" else np.float32)
            fname = f"{i:03d}_{name}.bin"
            arr.astype(arr.dtype.newbyteorder("<")).tofile(out / fname)
            entry[name] = {"file": fname, "dtype": "int32" if arr.dtype == np.int32 else "float32",
                           "shape": list(arr.shape)}
        samples.append(entry)
    (out / "manifest.json").write_text(json.dumps({"inputs": INPUT_NAMES, "samples": samples}, indent=1))
    print(f"wrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
