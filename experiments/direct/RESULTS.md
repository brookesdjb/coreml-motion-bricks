# Direct route on the Neural Engine: results

Direct route: NVIDIA's PyTorch SOMA model (`cmb/upstream_model.py`, the released checkpoints) → FX capture → `torch.export` → coremltools.
No ONNX and no onnx2torch. Machine: M2 Pro 16 GB, macOS 26, coremltools 9.0, torch 2.7.0.

## Outcome

| | old ONNX route (`soma_bN`) | direct route (`direct_bN`) |
|---|---|---|
| **batch 1** NE accuracy (mean\|Δ\| vs NVIDIA PyTorch, 84 inputs) | ~0.009 | **0.0096**, same length 84/84 |
| batch 1 placement (`placement.py`) | NE 1173 ops / 99% cost, 8 NE segments | NE 1543 ops / 98% cost, 8 NE segments |
| batch 1, 1 call (`bench_direct.py`) | 9.0 ms, 111 chars/s | **9.0–9.3 ms**, 107–111 chars/s |
| batch 1, 8 calls queued | 7.8 ms, 128–130 chars/s | **7.7 ms, 129 chars/s** |
| **batch 8** NE accuracy (80 inputs) | — | **0.0104**, same length 80/80 |
| batch 8 placement | NE 1333 ops / 96%, 14 NE segments | NE 1782 ops / 95%, 10 NE segments |
| batch 8, 1 call | 21.9 ms, 365 chars/s | 24.5 ms, 326 chars/s |
| batch 8, 8 calls queued | 11.7 ms, 684 chars/s | 12.5 ms, 640 chars/s |

- Batch 1 meets the target: same speed and same accuracy as the old route.
- Batch 8 is accurate. It is about 12% slower for a single call and 6% slower with 8 calls queued.
- GPU, for reference: batch 1 is 0.0100 at 12–13 ms; batch 8 is 0.0070 at 25–31 ms.
- All timings come from the same machine, measured back to back.
- `eval_direct.py` reports higher ms/call than `bench_direct.py` because it times `predict()` with dict conversion and no warm queue.
- The old-route figures were re-measured in this run with `bench_direct.py`, which uses the same method as `tools/bench_neural_engine.py`.

## What was wrong, and what fixed it

### 1. Everything on the CPU (batch 1 and 8): `aten._safe_softmax`

**Symptom.** With every op in fp16, CoreML placed all ~2,450 ops on the CPU.

**Cause, from the system log.**
- During model load, `ANECompilerService` compiles the program's NE pieces one at a time.
- About 37 pieces succeeded. Then one failed:
  `ANECompiler: error: Fail to call ZinComputeMutableProgramAddObjects(.)` → `aned: Compilation failed`.
- After that single failure, CoreML gave up on the Neural Engine for the **whole** program, not just that piece.
- To see this, run `/usr/bin/log show --start … --predicate 'process CONTAINS "ANE" OR process CONTAINS "aned"'` while running `placement.py`.

**Why the graph was cut into so many pieces.**
- PyTorch's MATH attention ends in `aten._safe_softmax`: softmax, but 0 for rows that are fully masked.
- Its decomposition is an **int** `fill_like` plus `logical_not` / `any` (bool → int `reduce_max` → bool) / `select`.
- None of these can run on the NE, so each of the 24 attention layers was split into NE / CPU / NE pieces.
- `mil_window.py bisect` cut the converted program at the MIL level and found where it flips:
  - the prefix ending just before op 4644 runs with **27 NE segments** (1200 NE ops);
  - adding the next op (the 23rd attention's output `linear`) puts the whole thing on the CPU.

**It is not a fixed segment cap.** A batch-8 variant with fp32 `layer_norm` ran 55 NE segments. One particular piece fails to compile depending on how the graph is cut. That is why the earlier session's rewrites (sort → topk, static fills, scatter/gather) appeared to help or hurt: they moved the cuts around.

**Fix (`ipfix.safe_softmax_as_float`).**
- `softmax(x) * amax(x > -5e3)`: float-only and the same values.
- Used together with `finite_masks`, so masked scores are −1e4 and never −inf; that way a fully masked row gives 0, not NaN.
- The int and bool islands disappear: batch 1 goes to **8 NE segments, the same as the old route**.

### 2. Wrong results at batch 8 (NE and GPU): small tensors and batched Conv1d

**Symptom.**
- With fix 1 alone, batch 8 was wrong: NE 0.30 with every slot predicting 14 tokens; GPU 0.22. CPU was right.
- It was deterministic. With the same input in every slot, every slot was wrong by the same amount.

**Diagnosis.**
- `mil_window.py diverge` bisects device output against fp32 PyTorch values of the *same* graph nodes; `DUMP_REF` dumps those values.
- Note: CPU fp16 is not a usable reference. It drifts about 10× more than the GPU or NE (0.05 vs 0.01 end to end).
- On the NE, the first wrong value is `(local_root_values − mean)/std` → reshape (8,8,4) → (64,4), right after the model input.
  - Slots 1–7 read as zeros.
  - Slot 0's last channel is wrong.
- In other words, ops on tiny tensors are miscomputed at batch 8.
- This is the same problem the old route documents ("Neural Engine at batch > 4 gave different results per batch slot").

**Fix: the old route's two workarounds, ported as FX rewrites.**
- Ops whose largest output has fewer than **128** elements are kept fp32. CoreML runs those on the CPU (`direct_try.py <B> 128`).
  - Threshold sweep: 0 and 64 are still wrong (0.30); 128 and 256 are right.
  - So the culprits are 64–127-element tensors, e.g. the (8,8) `has_*` masks.
- `ipfix.fold_conv1d_batch`: each Conv1d is computed as a Conv2d over (1, C, B, L), so the NE sees one image.
  - This is the speed fix: batch 8 went from 29.7 to 24.4 ms per call, and from 452 to 640 chars/s queued.
- `ipfix.nearest_upsample_as_repeat`: the decomposed `upsample_nearest1d` gather becomes expand + reshape. Speed is unchanged, but it removes 4 gathers; optional.

**Batch 1 does not want these.** With threshold 128, batch 1 on the NE drops to 0.051. So the recipe depends on batch size, as the old route's does: these workarounds apply only above batch 4.

## Reproduce

Run from the repo root with `.venv/bin/python`. Packages land in `build/coreml/direct_b{1,8}.mlpackage`.

```bash
# batch 1
FP32_OPS="" FINITE_MASKS=1 FLOAT_SAFE_SOFTMAX=1 .venv/bin/python experiments/direct/direct_try.py 1
# batch 8 (batch > 4): also fold Conv1d batches, repeat-upsample, small tensors (<128 elements) in fp32
FP32_OPS="" FINITE_MASKS=1 FLOAT_SAFE_SOFTMAX=1 FOLD_CONV=1 REPEAT_UPSAMPLE=1 .venv/bin/python experiments/direct/direct_try.py 8 128

.venv/bin/python experiments/direct/placement.py build/coreml/direct_b1.mlpackage
.venv/bin/python experiments/direct/eval_direct.py 1 8
.venv/bin/python experiments/direct/bench_direct.py build/coreml/direct_b1.mlpackage:1 build/coreml/direct_b8.mlpackage:8 build/coreml/soma_b1.mlpackage:1 build/coreml/soma_b8.mlpackage:8
```

**Rewrite set, in the order `direct_try.py` applies it.**

`ipfix` FX rewrites:
- `rewrite_index_put`, `rewrite_int_scatter`, `rewrite_scatter_value`, `positive_cumsum_axes`
- `finite_masks` (**required** by the next rewrite)
- `static_fills_and_sort`, `scatters_as_masks`, `gathers_as_onehot`
- `fold_conv1d_batch` and `nearest_upsample_as_repeat`, at batch > 4 only
- **`safe_softmax_as_float`**

Other parts:
- Capture runs under `capshim` (`t.all` → True, `one_hot` as a comparison), with `sdpa_kernel(MATH)`.
- `ctfix` patches coremltools' int/bool cast.
- Export uses the default decompositions plus `linalg_cross`, keeping `upsample` whole.
- fp16 everywhere except the geometry op types listed in `direct_try.py` (sin/cos/atan/cumsum/real_div/…), plus the small-tensor rule at batch > 4.
- `TARGET=macOS14`.

**Diagnostic tools added.**
- `mil_window.py`
  - `<pkg> list`
  - `<pkg> <i> <j>`: NE op count for an op window
  - `<pkg> bisect`: smallest prefix that loses the NE
  - `<pkg> diverge <B>` with `REF=` / `VS=`: first op where a device departs from fp32 PyTorch
- `per_slot.py`: per-slot error; `SAME=1` gives every slot the same input.
- `bench_direct.py`: throughput, as `tools/bench_neural_engine.py` measures it.
- `placement.py`: now also prints the NE/CPU segment count, and the top ops by cost with `TOP=n`.

## Proposal: `cmb/convert.py` on the direct route

1. **Capture.** `cmb/upstream_model.build_model()` → `make_fx(functionalize(model))` under `sdpa_kernel(MATH)` and `capshim`.
   - The shims live in `cmb/` and are applied at capture time only, so NVIDIA's code stays unmodified.
2. **Rewrite.** Move `ipfix.py` to `cmb/fx_rewrites.py` as a list of `(name, fn, applies(batch))`.
   - Assert the FX graph still matches eager PyTorch (≤ 1e-4) after rewriting, as `direct_try.py` does now.
   - Batch > 4 adds `fold_conv1d_batch`, `nearest_upsample_as_repeat` and the < 128-element fp32 rule.
3. **Convert.** `torch.export` + decompositions → `ct.convert(..., compute_precision=FP16ComputePrecision(op_selector))`.
   - The selector is the geometry list plus the small-tensor rule.
   - This is NVIDIA's own `tools/onnx/precision.py` idea: networks fp16, geometry fp32.
4. **Check before saving, per batch size.**
   - Run `placement.py`'s check: fail if NE ops == 0. That catches any regression of the whole-program fallback.
   - Run the 84-input accuracy check against `build_model()`: mean|Δ| ≤ 0.012, same length ≥ 83/84.
5. **Remove** the ONNX / onnxsim / onnx2torch steps, their patches, and the onnx2torch dependency. Keep the old route on a tag for comparison until batch 8 matches its speed.

**Open item: batch 8 is 6–12% slower.**
- The direct graph has about 2× the reshapes and 5× the casts of the onnxsim-simplified graph.
- Most of these come from integer index and geometry paths between the networks.
- Next steps:
  - a MIL cleanup pass (cast/reshape folding),
  - or move the post-network geometry into one fp32 tail, e.g. by FX provenance (`record_module_stack`).
- Batches 2 and 4 are not yet built or measured on this route.
