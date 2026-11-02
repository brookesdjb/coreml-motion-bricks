"""Run one of NVIDIA's MotionBricks example scripts, unmodified, on a Mac.

    python tools/run_upstream_example.py generate_once
    python tools/run_upstream_example.py locomotion_loop

The scripts in third_party/motionbricks/docs/examples ask for a CUDA device and NVIDIA's TensorRT engine.
This runner maps Warp's CUDA device to the CPU device and NVIDIA's MotionBricksSOMAEngine to cmb's
MacSOMAEngine (CoreML on the Neural Engine), then executes the script as NVIDIA wrote it. Only the SOMA
(human) model is supported; the Unitree G1 examples are not.
"""
import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmb import _compat  # noqa: E402,F401
from cmb.engine import MacSOMAEngine  # noqa: E402
from cmb.paths import UPSTREAM  # noqa: E402

import warp as wp  # noqa: E402
import motionbricks.model.soma.engine as soma_engine  # noqa: E402

EXAMPLES = UPSTREAM / "docs" / "examples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", choices=sorted(p.stem for p in EXAMPLES.glob("*.py") if not p.stem.startswith("_")))
    ap.add_argument("--backend", default="CPU_AND_NE", help="MacSOMAEngine backend (default: Neural Engine)")
    args, rest = ap.parse_known_args()
    if "unitree_g1" in rest:
        raise SystemExit("The Unitree G1 model isn't converted; only SOMA examples run on the Mac.")

    set_device = wp.set_device
    wp.set_device = lambda device=None: set_device("cpu" if str(device).startswith("cuda") else device)

    class UpstreamSignatureEngine(MacSOMAEngine):
        """Accepts NVIDIA's constructor arguments (ONNX path, TensorRT options) and ignores the TensorRT ones."""

        def __init__(self, onnx_path=None, *, model_to_runtime=wp.transform_identity(), **_):
            super().__init__(args.backend, model_to_runtime=model_to_runtime)
    soma_engine.MotionBricksSOMAEngine = UpstreamSignatureEngine  # what docs/examples/_rig.py imports

    sys.path.insert(0, str(EXAMPLES))
    sys.argv = [str(EXAMPLES / f"{args.example}.py"), *rest]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
