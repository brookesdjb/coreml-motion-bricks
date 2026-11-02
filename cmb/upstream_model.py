"""NVIDIA's own PyTorch SOMA model, built with their exporter code from the released checkpoints.

NVIDIA's ONNX files are exported from exactly this model (third_party/motionbricks/tools/onnx). Converting it
to CoreML directly avoids the ONNX round trip and any ONNX-to-PyTorch converter.

    from cmb.upstream_model import download_checkpoints, build_model
    download_checkpoints()           # ~800 MB from huggingface.co/nvidia/MotionBricks, checksum-verified
    model = build_model()            # NVIDIA's BatchedSOMA, fp32, eval mode
"""
import importlib.util
import sys

from .paths import BUILD, UPSTREAM

EXPORTER = UPSTREAM / "tools" / "onnx"            # NVIDIA's exporter package (models + checkpoint loading)
CHECKPOINTS = BUILD / "checkpoints" / "MotionBricks" / "soma"
HF_REPO = "nvidia/MotionBricks"


def exporter():
    """Import NVIDIA's exporter package. Its folder is named `onnx`, which would shadow the real onnx package
    on sys.path, so it is loaded under another name."""
    name = "motionbricks_exporter"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EXPORTER / "__init__.py",
                                                      submodule_search_locations=[str(EXPORTER)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _release():
    exporter()
    from motionbricks_exporter.checksums import CHECKPOINTS as RELEASES
    return RELEASES["soma"]


def download_checkpoints():
    """Fetch the SOMA checkpoint files NVIDIA's exporter needs (and nothing else) from Hugging Face."""
    from huggingface_hub import hf_hub_download
    release = _release()
    for entry in release["files"]:
        target = CHECKPOINTS / release["root_directory"] / entry["path"]
        if target.exists() and target.stat().st_size == entry["bytes"]:
            continue
        hf_hub_download(HF_REPO, f"soma/{release['root_directory']}/{entry['path']}",
                        local_dir=str(CHECKPOINTS.parent.parent / "MotionBricks"))
    return CHECKPOINTS


def build_model():
    """NVIDIA's BatchedSOMA (fp32) with the released weights; checksums are verified by NVIDIA's loader."""
    exporter()
    from motionbricks_exporter.checkpoints import checkpoint_root, load_sources
    from motionbricks_exporter.models.soma import build_soma
    release = _release()
    sources = load_sources(checkpoint_root(CHECKPOINTS, release), release)
    model = build_soma(sources)
    del sources
    return model.eval()
