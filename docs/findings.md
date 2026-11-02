# Findings

What it took to run NVIDIA MotionBricks (SOMA model) accurately on Apple's Neural Engine, what we measured,
and what didn't work. Measurements are from an Apple M2 Pro (6 performance + 4 efficiency cores, 16 GB),
macOS 26, with the versions in `requirements.txt`.

## 1. Running without TensorRT

MotionBricks' runtime ships with a TensorRT engine and a Newton/pyglet viewer. Only the network call needs
replacing: the pose encode/decode is written in NVIDIA Warp, which runs on the CPU device on a Mac.
`cmb.engine.MacSOMAEngine` subclasses NVIDIA's `MotionBricksSOMAEngine`, skips its TensorRT constructor, and
sends the network inputs to CoreML or ONNX Runtime. Import-time stubs for TensorRT and the viewer packages are
in `cmb/_compat.py`.

## 2. Conversion bugs and fixes

The route is ONNX → PyTorch (onnx2torch) → `torch.jit.trace` → CoreML ML Program. Every step was validated
against ONNX Runtime on 84 real network inputs recorded from NVIDIA's runtime (walking, turning, running,
jumps).

| Problem | Symptom | Fix |
|---|---|---|
| **onnx2torch 1.5.15 `CumSum` treats a missing `reverse` attribute as 1** (ONNX default 0) | Heading integrated backwards; characters' feet slid ~5× more than with ONNX Runtime | Re-register CumSum with the correct default. Found by comparing every intermediate tensor: CumSum's input matched to 3e-6, its output was off by 1.2 |
| Mixed fp16/fp32 graph | ONNX Runtime CPU silently computes much of it in fp32; PyTorch/CoreML honour the fp16 casts, and the root decode divides by small numbers | Rewrite the graph as pure fp32 before converting (lossless); let CoreML choose fp16 compute itself |
| Missing / unsupported converters | Conversion errors | ScatterElements, OneHot, LpNormalization converters; drop empty optional inputs; make CumSum axes positive; alias torch op names coremltools lacks (`greater`, `broadcast_to`, `resolve_conj`, …) |

After the fixes, the fp32 CoreML model matches the fp32 reference exactly (mean |Δ| 0.0000); the fp16 models
are as close as NVIDIA's own fp16 ONNX model (≈0.008–0.011).

## 3. Neural Engine bugs at batch > 4

Batch 1, 2 and 4 are accurate on the Neural Engine. At batch 8, **identical inputs in every slot gave
different outputs per slot** (repeating every 4 slots) on the Neural Engine, while the GPU was correct.
`tools/bisect_batch.py` cuts the graph after a node, converts that prefix, and checks slot agreement, to
binary-search the first bad operation. Three separate problems:

1. **Scatter** (writing each character's end pose into the trajectory): CoreML's `scatter_along_axis`
   mixed slots. Rewritten as a compare-and-select mask (`where(arange == index, update, data)`).
2. **Tiny tensors** (8×1×4 root-conditioning values): two inputs that each matched across slots produced a
   sum that didn't. Ops on tensors under 256 elements are kept in fp32, which CoreML runs on the CPU.
3. **Conv1d**: folded into a Conv2d over (1, C, B, L) with a 1-tall kernel, so the Neural Engine sees batch 1.
   This also doubled the convolutions' speed.

## 4. Performance (single model calls)

| Backend | ms per generation (batch 1) |
|---|---|
| ONNX Runtime, CPU | 27–30 |
| CoreML, CPU + Neural Engine | **~11** |
| CoreML, CPU + GPU | ~13 |
| CoreML, let CoreML choose | ~14–16 (it puts 86% on the GPU) |

With **CPU + Neural Engine, ~99% of the estimated work runs on the Neural Engine** (CoreML compute plan).
Throughput with calls queued in one batch-API submission:

| Model | characters/s |
|---|---|
| batch 1 | ~90 |
| batch 4 × 8 queued | ~370 |
| batch 8 × 8 queued | **~700** |
| batch 16 × 8 queued | ~800 (accurate, but needs 128 requests waiting) |

Running Neural Engine and GPU models side by side was *slower* in total than the Neural Engine alone.

## 5. Interactive use

* NVIDIA's default input smoothing (0.6 s speed half-life) feels sluggish for a player; a "responsive"
  preset (`cmb.sim.RESPONSIVE`) halves start/turn latency without hurting foot contact. Key → visible motion
  is still ~0.4 s and a clip is at least 0.8 s, so this suits NPCs better than a player character.
* CoreML's `predict` holds the Python GIL (a busy Python thread drops to ~20% speed while it runs), so a
  background thread cannot overlap inference with simulation.

## 6. Crowds

Each NPC is a full NVIDIA SOMA character. Per process, NPC steps run on threads and a batcher sends all pending
generation requests to the Neural Engine together; NPCs are split over worker processes.

| Configuration (15 Hz) | Real-time capacity |
|---|---|
| 1 process, batch 4 | ~45 NPCs |
| 4 processes, batch 4 | ~185 |
| 6 processes, batch 8 | ~200 |
| **6 processes, batch 8, calm profile** | **~300** (300 NPCs at 1.01× real time) |

The **calm crowd profile** (`cmb.crowd.apply_calm_profile`) asks for ~40% fewer clips (0.94 vs 1.56 per NPC
per second) at the same foot-slide: longer clips (12–16 tokens), targets placed further ahead, regeneration
only on ~40° / 50% changes, and NPCs that hold a heading instead of re-aiming every tick.

In the browser demo the crowd's bone transforms are computed in a vertex shader from a joint-position
texture, so the page costs ~0.03 ms of JavaScript per frame.

### What didn't help

* **More worker processes**: 6 was best; 8 and 10 were slower (more processes queueing for one Neural Engine).
* **A shared Neural Engine service process** pooling every worker's requests: slower (0.66× vs 0.89× at
  200 NPCs). Workers aren't in lock-step, so pooling mostly added latency and cross-process copies.
* **Look-ahead generation** (prefetch each NPC's next clip from the poses its current clip will reach, on the
  service process, install it at the transition): worked and kept motion quality, but only ~half of clips are
  predictable, and the extra per-NPC work outweighed the saved waiting (0.52–0.58× vs 1.03× at 200 NPCs).

### Ideas not yet tried

* Reusing generated clips across NPCs doing near-identical things, with live generation only for NPCs near
  the camera or doing something distinctive (level of detail).
* Vectorising NVIDIA's per-NPC Python runtime across all NPCs.

## 7. Operational gotchas

* `coremltools.models.MLModel(path)` compiles a ~360 MB copy into `$TMPDIR` per load and only deletes it on a
  clean exit; killed processes leaked 34 GB. `cmb.engine.load_coreml` compiles once into `build/compiled/`.
* CoreML frees prediction inputs later on its own queue without the GIL; if that drops the last reference to
  a numpy array, Python segfaults. Keep recent inputs referenced (see `cmb.crowd.Batcher`).
* Multiprocessing on macOS uses `spawn`; workers should stagger their start so they don't all compile Warp
  kernels at once.
