#!/bin/sh
# One-time setup: Python environment, the pinned NVIDIA MotionBricks checkout, and the model/clip files it
# needs from Git LFS (~0.5 GB). Then run:  .venv/bin/python -m cmb.convert
set -e
cd "$(dirname "$0")"

[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "Apple Silicon Mac required (CoreML / Neural Engine)."; exit 1; }
command -v git-lfs >/dev/null || { echo "Git LFS required: brew install git-lfs"; exit 1; }
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || { echo "Python 3.12 required (brew install python@3.12), or set PYTHON=..."; exit 1; }

echo "== NVIDIA MotionBricks (pinned submodule)"
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init third_party/motionbricks
# Only what the SOMA runtime and the conversion use: the fp16 ONNX model, the SOMA skeleton, its motion
# clips (locomotion/idle/interaction targets) and smart primitives. Skips the G1 robot and docs media.
git -C third_party/motionbricks lfs pull --include="motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx,motionbricks/assets/human/soma_base_skel_minimal_no_root.usd,assets/motions/soma/**"

echo "== Python environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo
echo "Done. Next: .venv/bin/python -m cmb.convert    (about 10-15 minutes, ~3 GB in build/)"
