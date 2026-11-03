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

## 2. Conversion route

`cmb/convert.py` converts **NVIDIA's own PyTorch model** (built by their exporter code from the released
checkpoints, the model their ONNX files are exported from) directly: `make_fx(functionalize(model))` →
FX rewrites (`cmb/fx_rewrites.py`) → `torch.export` → CoreML ML Program. NVIDIA's code is not modified; two
shims are applied only while capturing (`t.all` → True for input-validation asserts, `one_hot` as a
comparison). Every package is checked against NVIDIA's fp32 PyTorch model on 84 real network inputs recorded
from NVIDIA's runtime (walking, turning, running, jumps) before it is saved. The full investigation, with the
tools used to find each problem, is in `experiments/direct/RESULTS.md`.

| Problem | Symptom | Fix |
|---|---|---|
| One Neural Engine piece fails to compile (`ANECompiler … ZinComputeMutableProgramAddObjects`) | CoreML drops the **whole** program to the CPU, not just that piece | Remove what splits the graph into many NE/CPU pieces, and the patterns the compiler rejects (rows below and in section 3) |
| `aten._safe_softmax` (PyTorch's MATH attention) decomposes into int `fill_like` and bool `any`/`logical_not` | Every attention layer split into NE / CPU / NE pieces; one of them failed to compile | Float-only `softmax(x) · amax(x > −5e3)`, with −inf mask fills made a finite −1e4 |
| Scatters: coremltools lowers in-place slice writes to `scatter` on the flattened tensor, and builds one-position scatter masks from int ops it tiles to full size | Large ops on the CPU in the motion-assembly tail (batch 8: 24 → 16 ms per call once fixed) | Full-tensor slice writes are just their source, others a `cat`; one-position scatters a float position mask and a mul/add blend |
| gathers, `sort`, `*_like` fills | NE compile failures | One-hot select-and-sum; full-length `topk`; constant-shape `full` |
| coremltools: negative `cumsum` axes, `int`/`bool` casts of one-element constants, scalar scatter values typed int32 | Conversion errors | Positive axes; patched casts; tensor scatter values |
| Geometry in fp16 | Root-motion decode divides by small numbers: foot sliding | NVIDIA's own precision rule (their `tools/onnx/precision.py`): networks fp16, geometry ops fp32 |

The fp16 packages score 0.009–0.010 mean |Δ| against NVIDIA's fp32 PyTorch, about as close as NVIDIA's own
fp16 ONNX model (0.006); the fp32 package matches it exactly.

### Earlier route: ONNX → onnx2torch (git tag `onnx-route`)

Until this repo converted the PyTorch model directly, it converted NVIDIA's ONNX file through onnx2torch.
Bugs found on that route, for anyone still on it:

| Problem | Symptom | Fix |
|---|---|---|
| **onnx2torch 1.5.15 `CumSum` treats a missing `reverse` attribute as 1** (ONNX default 0) | Heading integrated backwards; characters' feet slid ~5× more than with ONNX Runtime | Re-register CumSum with the correct default. Found by comparing every intermediate tensor: CumSum's input matched to 3e-6, its output was off by 1.2 |
| Mixed fp16/fp32 graph | ONNX Runtime CPU silently computes much of it in fp32; PyTorch/CoreML honour the fp16 casts, and the root decode divides by small numbers | Rewrite the graph as pure fp32 before converting (lossless); let CoreML choose fp16 compute itself |
| Missing / unsupported converters | Conversion errors | ScatterElements, OneHot, LpNormalization converters; drop empty optional inputs; make CumSum axes positive; alias torch op names coremltools lacks (`greater`, `broadcast_to`, `resolve_conj`, …) |

## 3. Neural Engine bugs by batch size

Each batch size above 1 hit its own Neural Engine compiler bug; the GPU was correct throughout. Found by
cutting the converted program at the MIL level and checking where the Neural Engine departs from fp32 PyTorch,
or where it loses the Neural Engine (`experiments/direct/mil_window.py diverge` / `bisect`).

1. **Batch 2, wrong values.** The token embedding's gather → reshape (B·16, 160) → linear was miscomputed
   (0.24 mean |Δ|, the same in every slot). Exposing the gather's output made it correct. Fix: the embedding
   as a one-hot times the table. (This also made batches 1 and 8 slightly more accurate.)
2. **Batch 4, everything on the CPU.** `nn.MultiheadAttention` flattens its output seq-first, as
   (T·B, H·D) rows, into the output projection; the compiler rejected that transpose → reshape → linear.
   Fix: the projection computed batch-first, so its transpose cancels against the one back to batch-first.
3. **Batch 8, wrong values.** Ops on tiny tensors (the 8×8 `has_*` masks, (8,8,4) root values) were
   miscomputed. Ops whose largest output has under 128 elements are kept in fp32, which CoreML runs on the
   CPU (64 still failed). Conv1d is folded into a Conv2d over (1, C, B, L) with a 1-tall kernel, so the
   Neural Engine sees batch 1; that was the batch-8 speed fix. Both apply above batch 4 only: at batch 1 the
   small-tensor rule costs accuracy (0.051).

## 4. Performance (single model calls)

| Backend | ms per generation (batch 1) |
|---|---|
| ONNX Runtime, CPU | 28–30 |
| CoreML, CPU + Neural Engine | **~10** |
| CoreML, CPU + GPU | ~13 |
| CoreML, let CoreML choose | ~14–16 (it puts 86% on the GPU) |

With **CPU + Neural Engine, ~99% of the estimated work runs on the Neural Engine** (CoreML compute plan).
Throughput with calls queued in one batch-API submission:

| Model | characters/s |
|---|---|
| batch 1 | ~110 |
| batch 1 × 8 queued | ~130 |
| batch 4 × 8 queued | ~360 |
| batch 8 × 8 queued | **~700** |
| batch 16 × 8 queued | ~800 (measured on the ONNX route; accurate, but needs 128 requests waiting) |

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
| 6 processes, batch 8, NVIDIA default settings | ~200 (200 NPCs at 1.05× real time) |
| **6 processes, batch 8, calm profile** | **~300** (300 NPCs at 1.03× real time) |

The **calm crowd profile** (`cmb.crowd.apply_calm_profile`) asks for ~40% fewer clips (0.96–0.98 vs 1.58 per NPC
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
