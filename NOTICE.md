# Notices

**coreml-motion-bricks** — Copyright 2026 David Brookes. Licensed under the Apache License 2.0 (see `LICENSE`).

This project is an independent port and is not affiliated with or endorsed by NVIDIA.

## NVIDIA MotionBricks

This project runs and converts **NVIDIA MotionBricks** (https://github.com/NVlabs/motionbricks), included as a
git submodule in `third_party/motionbricks` at a pinned commit and not modified.

* MotionBricks source code: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0; documentation and
  assets CC-BY-4.0. See `third_party/motionbricks/LICENSE`.
* MotionBricks models (the ONNX files the conversion reads): licensed by NVIDIA Corporation under the
  [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
  See `third_party/motionbricks/motionbricks/assets/onnx/LICENSE.md`.

This repository does **not** redistribute the models or any converted models. `python -m cmb.convert` builds
CoreML models on your machine from your own copy of NVIDIA's ONNX files. If you redistribute converted models,
the NVIDIA Open Model Agreement applies to them: include a copy of the agreement and retain NVIDIA's notices.
