# Swift samples: MotionBricks SOMA on CoreML, natively

Two thin SwiftUI apps (macOS and iOS) over one shared Swift package. They load the batch-1 SOMA CoreML
model, replay the 84 network inputs recorded from NVIDIA's runtime, and report accuracy against the fp32
reference, time per call, and where CoreML places the work (Neural Engine / GPU / CPU).

## Layout

```
swift/
  MotionBricksKit/               shared Swift package (all the real code)
    Sources/MotionBricksKit/
      SOMAModel.swift            load .mlpackage/.mlmodelc with selectable compute units; typed predict + timing
      FeedSet.swift              reads the exported feeds (manifest.json + raw .bin tensors)
      Benchmark.swift            replay all feeds, mean |Δ| on valid frames, same-length count, median/p90 ms
      Placement.swift            per-op device preference and NE cost share via MLComputePlan
      BenchmarkView.swift        the shared SwiftUI screen both apps show
    Sources/mbbench/main.swift   command-line harness (same code path, no UI)
  Apps/Shared/MotionBricksApp.swift   the only app source; one #if os(macOS) block for locating files
  scripts/stage_ios_resources.sh      copies model + feeds into Apps/iOS/Resources for bundling
  project.yml                         XcodeGen spec -> MotionBricksSamples.xcodeproj (both app targets)
```

The model contract follows `cmb/convert.py` (`INPUT_NAMES`) and `cmb/engine.py` (`split_outputs`): seven
inputs (float32 values, int32 masks/token allowance), outputs picked by size — motion `[1, 64, 366]` (fp16
from the Neural Engine model) and `pred_num_tokens` `[1]`. Accuracy is computed as in
`tools/check_accuracy.py`: mean |Δ| over the first `4 × reference tokens` frames, OK if < 0.02 and ≥ 95 %
of clips choose the same length. Minimum OS: macOS 15 / iOS 18 (needed for `Float16` `MLShapedArray`).

## 1. Export the feeds (once)

Needs `build/feeds.npz` and `build/onnx/soma_fp32_b1.onnx` from the Python pipeline (`setup.sh` / convert).

```bash
.venv/bin/python tools/export_swift_feeds.py --out swift/Feeds
```

Writes 84 samples × 9 tensors (~11 MB) plus `manifest.json`. Use `--build /path/to/build` to read another checkout's `build/`.

## 2. Mac

```bash
brew install xcodegen   # once
cd swift && xcodegen generate && open MotionBricksSamples.xcodeproj
```

Run the **MotionBricksMac** scheme. It looks for the model at `MB_MODEL`, else `build/coreml/soma_b1.mlpackage`
or `swift/Models/soma_b1.mlpackage` in the repo it was built from; feeds at `MB_FEEDS`, else `swift/Feeds` or
`build/swift_feeds`. "Choose…" buttons pick anything else. Pick compute units, press **Run**.
Set `MB_AUTORUN=1` to start a run on launch and print the summary to stdout.

Headless, same code:

```bash
cd swift/MotionBricksKit && swift run -c release mbbench ../../build/coreml/soma_b1.mlpackage ../Feeds ne gpu cpu
```

## 3. iPhone

The simulator has no Neural Engine (and CoreML there runs on the CPU), so its numbers mean nothing;
NE numbers need a real device.

1. Stage the model and feeds into the app bundle:
   `swift/scripts/stage_ios_resources.sh /path/to/build/coreml/soma_b1.mlpackage swift/Feeds`
2. `cd swift && xcodegen generate && open MotionBricksSamples.xcodeproj`
3. Select the **MotionBricksiOS** target → Signing & Capabilities → choose your Team, and change the
   bundle id (`com.example.motionbricks.ios`) to one you own.
4. Plug in the iPhone (A12 or newer, iOS 18+), trust the Mac, enable Developer Mode
   (Settings → Privacy & Security → Developer Mode), pick it as the run destination.
5. Use the **Release** configuration for timing (Product → Scheme → Edit Scheme → Run → Build Configuration),
   then ⌘R. Xcode compiles the `.mlpackage` to `soma_b1.mlmodelc` at build time; the first run then
   specialises it for the NE (several seconds, shown as "Load + compile").
6. Press **Run 84 recorded inputs** with "CPU + Neural Engine", then compare "CPU + GPU".

Simulator compile check: `xcodebuild -scheme MotionBricksiOS -destination 'generic/platform=iOS Simulator' build`.

## Results so far (M2 Pro, macOS 27, Xcode 27.1, `build/coreml/soma_b1.mlpackage` from the ONNX route)

| Compute units | mean \|Δ\| | same length | median ms/call | p90 | placement (MLComputePlan) |
|---|---|---|---|---|---|
| CPU + Neural Engine (Mac app) | 0.0092 | 84/84 | 9.93 | 10.65 | 1570/2413 ops on NE, 843 CPU; **98.8 % of est. cost on NE** |
| CPU + Neural Engine (mbbench) | 0.0092 | 84/84 | 10.61 | 11.12 | same |
| CPU + GPU (mbbench) | 0.0100 | 84/84 | 12.84 | 13.57 | 2169 GPU, 244 CPU |
| CPU only (mbbench) | 0.0502 | 84/84 | 10.88 | 11.24 | 2413 CPU — over the 0.02 bar, so flagged MISMATCH |

These match the Python/coremltools numbers (~9–11 ms, ~0.009). The CPU ops on the NE plan are mostly
cheap shape/cast ops (1.2 % of estimated cost). The CPU-only accuracy drop is worth a look if CPU fallback
ever matters. iOS: builds for the simulator; **no device numbers yet.**

## Not done / next steps

- **iPhone numbers** on a real device (steps above).
- **Playback view (stretch, not started):** a shared RealityKit view drawing joints as spheres/lines needs
  the 366-channel motion decoded into joint positions, which is NVIDIA's pose decode, below.
- **Porting NVIDIA's Python runtime to Swift** would involve:
  - the *controller* (`MotionBricksSOMAEngine` and its controller in `third_party/motionbricks`): building the
    seven inputs each step from the character state, keyframe/root targets and masks, choosing
    `allowed_pred_num_tokens`, and stitching generated clips into a continuous stream;
  - the *Warp kernels* that decode `pred_global_motions` (root + per-joint features) into a SOMA skeleton
    pose and apply the model↔runtime transform: rewrite them as Swift/SIMD or Metal compute kernels;
  - skeleton/rig data (joint hierarchy, rest pose) exported from the Python assets as JSON;
  - a parity harness like this one, comparing decoded poses with the Python runtime frame by frame.
- Godot/Unreal plugins can reuse `MotionBricksKit` via a C shim, or call CoreML from Objective-C++.
- Check NVIDIA's model-weight licence before any commercial use.
