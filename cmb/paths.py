"""Where things live: the pinned upstream checkout, and everything this repo generates (under build/)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party" / "motionbricks"  # NVlabs/motionbricks, pinned as a git submodule
BUILD = REPO / "build"                            # generated models, compiled models, logs (git-ignored)

# NVIDIA's published fp16 ONNX model: the "ort" reference backend, which also records the inputs below
SOURCE_ONNX = UPSTREAM / "motionbricks" / "assets" / "onnx" / "soma" / "mmm_v6_soma_fp16.onnx"
COREML_DIR = BUILD / "coreml"
COMPILED_DIR = BUILD / "compiled"
LOG_DIR = BUILD / "logs"
FEEDS = BUILD / "feeds.npz"  # real network inputs recorded from NVIDIA's runtime (cmb.convert, first run)

BATCH_SIZES = (1, 2, 4, 8)


def coreml_package(batch: int) -> Path:
    return COREML_DIR / f"soma_b{batch}.mlpackage"


def coreml_package_fp32() -> Path:
    return COREML_DIR / "soma_b1_fp32.mlpackage"
