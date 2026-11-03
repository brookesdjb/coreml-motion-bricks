# Direct route on the Neural Engine: results

Direct route: NVIDIA's PyTorch SOMA model (`cmb/upstream_model.py`, the released checkpoints) → FX capture → `torch.export` → coremltools.
No ONNX and no onnx2torch. Machine: M2 Pro 16 GB, macOS 26, coremltools 9.0, torch 2.7.0.

## Outcome

| | old ONNX route (`soma_bN`) | direct route (`direct_bN`) |
|---|---|---|
| **batch 1** NE accuracy (mean\|Δ\| vs NVIDIA PyTorch, 84 inputs) | 0.0091 | **0.0093**, same length 84/84 |
| batch 1 placement (`placement.py`) | NE 1173 ops / 99% cost, 8 NE segments | NE 1570 ops / 99% cost, 8 NE segments |
| batch 1, 1 call (`bench_direct.py`) | 8.9 ms, 113 chars/s | **9.0 ms, 112 chars/s** |
| batch 1, 8 calls queued | 7.5 ms, 133 chars/s | **7.6 ms, 132 chars/s** |
| **batch 2** NE accuracy (84 inputs) | 0.0091 | **0.0093**, same length 84/84 |
| batch 2 placement | — | NE 1618 ops / 98%, 8 NE segments |
| batch 2, 1 call | 12.2 ms, 164 chars/s | **10.6 ms, 188 chars/s** |
| batch 2, 8 calls queued | 8.4 ms, 238 chars/s | **8.6 ms, 232 chars/s** |
| **batch 4** NE accuracy (84 inputs) | 0.0091 | **0.0093**, same length 84/84 |
| batch 4 placement | — | NE 1618 ops / 95%, 8 NE segments |
| batch 4, 1 call | 17.4 ms, 230 chars/s | **14.3 ms, 281 chars/s** |
| batch 4, 8 calls queued | 11.4 ms, 352 chars/s | **11.0 ms, 362 chars/s** |
| **batch 8** NE accuracy (80 inputs) | 0.0099 | **0.0101**, same length 80/80 |
| batch 8 placement | NE 1333 ops / 96%, 14 NE segments | NE 1793 ops / 95%, 9 NE segments |
| batch 8, 1 call | 21.9 ms, 365 chars/s | **16.0 ms, 499 chars/s** |
| batch 8, 8 calls queued | 11.6 ms, 690 chars/s | **11.5 ms, 697 chars/s** |

- Every batch size meets the target: as accurate as the old route, and as fast or faster. A single call is level at batch 1 and faster at batch 2 (13%), 4 (18%) and 8 (27%); queued throughput is level (within 3%) at every size.
- Old-route accuracy is the README's figure; its timings were re-measured back to back with the direct ones, on the same machine.
- GPU, for reference (`eval_direct.py`): 0.0100 / 0.0078 / 0.0077 / 0.0073 at batch 1 / 2 / 4 / 8.
- `eval_direct.py` reports higher ms/call than `bench_direct.py` because it times `predict()` with dict conversion and no warm queue.
- `bench_direct.py` measures the same way as `tools/bench_neural_engine.py`.

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

### 3. Batch-8 speed: integer index paths and coremltools' scatter/select lowering

With fixes 1 and 2, batch 8 was accurate but took 24.5 ms per call (old route 21.9).

**Finding where the time went.**
- `mil_window.py <pkg> time 8 <cuts>` times prefixes of the program on the NE.
- `cpu_big.py` lists CPU-placed ops with large outputs.
- Together they showed two big pieces of work on the CPU.

**a) The last context frames written into the (8,64,512) token buffer** (`scatters_as_masks`).
- The position mask was built from an int index tiled to (8,1,512), with an int `floor_div` modulo.
- coremltools also tiles every `select` input to the full shape.
- So the whole chain, including the large `tile`/`equal`/`select`, ran on the CPU.
- Fix: `_scatter_one_as_float_mask` (`FLOAT_SCATTER_MASK=1`):
  - the position test is done in float, which is exact for positions ≤ 64;
  - it compares against the un-repeated (B,1,1) index;
  - it writes `x·(1−m) + src·m` instead of `where`, which broadcasts as plain mul/add.
- The blend is what mattered: **22.4 → 16.1 ms per call**.

**b) `aten.slice_scatter` / `select_scatter`** (functionalized in-place slice writes).
- coremltools lowers these to `scatter` on the flattened tensor: at batch 8, six (187,392,) scatters on the CPU, in the motion-assembly tail.
- 12 of the 15 overwrite the whole tensor, so they are just their source.
- Fix: `ipfix.scatters_as_cat` (`SCATTER_CAT=1`): the source for full-tensor writes, otherwise a `cat` of the untouched pieces and the source.
- Result: 23.5 → 22.4 ms.

**Both fixes help batch 1 a little too** (9.3 → 8.8 ms), so they are part of the recipe at every batch size.

### 4. Batches 2 and 4: two more Neural Engine compiler bugs

With the batch-1 recipe, batch 2 stayed on the NE (98%) but was wrong: 0.24, with the GPU at 0.0075. Batch 4 went entirely to the CPU (0.052, 83 ms). The batch > 4 workarounds (small-tensor fp32, Conv1d folding) changed neither.

**a) Batch 2, wrong values: the token embedding** (`embeddings_as_onehot`, `ONEHOT_EMBED=1`).
- Not slot-dependent: with the same input in both slots, both were wrong.
- `mil_window.py diverge` found the first wrong value at `addmm_41`, the second layer of the MLP after the token embedding:
  gather (2,16,5,32) → reshape (32,160) → linear → leaky_relu → reshape → linear.
- Exposing the gather output (or the reshape after it) as an extra output made the whole chain correct. Exposing the first linear's output made the NE fail at run time ("IOSurface smaller than the model expects"). So the NE compiler mishandles that fused gather → reshape → linear chain at batch 2.
- Fix: `aten.embedding` as a one-hot (float compare) times the table, so there is no gather.
- It helps at every size: batch 1 0.0097 → 0.0093, batch 8 0.0104 → 0.0101, and GPU batch 8 0.059 → 0.007 (the open item from fix 3a).

**b) Batch 4, everything on the CPU: attention output projections** (`attn_out_proj_batch_first`, `ATTN_BATCH_FIRST=1`).
- `mil_window.py bisect` found the flip at an attention output projection: `nn.MultiheadAttention` turns its (B,H,T,D) output into seq-first (T,B,H,D), flattens it to (T·B, H·D) rows, and runs the 2-D linear. Adding that linear made the ANE compile fail, which drops the whole program to the CPU.
- Keeping a leading dim on the linear did not help; coremltools folds the reshape back.
- Fix: compute the projection batch-first, (B,T,H·D) @ W + b, then transpose to (T,B,E). That transpose cancels against MHA's transpose back to batch-first, so the NE sees the same layout as at batch 1.
- Batch 4: CPU-only 83 ms → NE 95%, 14.3 ms, 0.0093. No change at the other sizes beyond noise, so it is applied everywhere.

## Reproduce

Run from the repo root with `.venv/bin/python`. Packages land in `build/coreml/direct_b{1,2,4,8}.mlpackage`.

```bash
# batch 1; the same for 2 and 4 with that argument
FP32_OPS="" FINITE_MASKS=1 FLOAT_SAFE_SOFTMAX=1 FLOAT_SCATTER_MASK=1 SCATTER_CAT=1 ONEHOT_EMBED=1 ATTN_BATCH_FIRST=1 .venv/bin/python experiments/direct/direct_try.py 1
# batch 8 (batch > 4): also fold Conv1d batches, repeat-upsample, small tensors (<128 elements) in fp32
FP32_OPS="" FINITE_MASKS=1 FLOAT_SAFE_SOFTMAX=1 FLOAT_SCATTER_MASK=1 SCATTER_CAT=1 ONEHOT_EMBED=1 ATTN_BATCH_FIRST=1 FOLD_CONV=1 REPEAT_UPSAMPLE=1 .venv/bin/python experiments/direct/direct_try.py 8 128

.venv/bin/python experiments/direct/placement.py build/coreml/direct_b1.mlpackage
.venv/bin/python experiments/direct/eval_direct.py 1 2 4 8
.venv/bin/python experiments/direct/bench_direct.py build/coreml/direct_b1.mlpackage:1 build/coreml/soma_b1.mlpackage:1 build/coreml/direct_b2.mlpackage:2 build/coreml/soma_b2.mlpackage:2 build/coreml/direct_b4.mlpackage:4 build/coreml/soma_b4.mlpackage:4 build/coreml/direct_b8.mlpackage:8 build/coreml/soma_b8.mlpackage:8
```

**Rewrite set, in the order `direct_try.py` applies it.**

`ipfix` FX rewrites:
- `rewrite_index_put`, `rewrite_int_scatter`, `rewrite_scatter_value`, `positive_cumsum_axes`
- `finite_masks` (**required** by the next rewrite)
- `static_fills_and_sort`
- `scatters_as_masks`, with **`FLOAT_SCATTER_MASK=1`** (float position test, blend instead of `where`)
- `gathers_as_onehot`
- `fold_conv1d_batch` and `nearest_upsample_as_repeat`, at batch > 4 only
- **`scatters_as_cat`** (`slice_scatter` / `select_scatter`)
- **`safe_softmax_as_float`**
- **`embeddings_as_onehot`**
- **`attn_out_proj_batch_first`**

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
  - `<pkg> time <B> <cuts>`: NE ms per call of program prefixes. Rough, since the cut's outputs are copied too.
- `cpu_big.py <pkg> [min elements]`: CPU-placed ops with large outputs.
- `per_slot.py`: per-slot error; `SAME=1` gives every slot the same input.
- `bench_direct.py`: throughput, as `tools/bench_neural_engine.py` measures it.
- `placement.py`: now also prints the NE/CPU segment count, and the top ops by cost with `TOP=n`.

## Proposal: `cmb/convert.py` on the direct route

1. **Capture.** `cmb/upstream_model.build_model()` → `make_fx(functionalize(model))` under `sdpa_kernel(MATH)` and `capshim`.
   - The shims live in `cmb/` and are applied at capture time only, so NVIDIA's code stays unmodified.
2. **Rewrite.** Move `ipfix.py` to `cmb/fx_rewrites.py` as a list of `(name, fn, applies(batch))`.
   - Assert the FX graph still matches eager PyTorch (≤ 1e-4) after rewriting, as `direct_try.py` does now.
   - Make `FLOAT_SCATTER_MASK`'s version the only `scatters_as_masks`.
   - Every size gets the same set; there is no batch-2 or batch-4 special case.
   - Batch > 4 adds `fold_conv1d_batch`, `nearest_upsample_as_repeat` and the < 128-element fp32 rule.
3. **Convert.** `torch.export` + decompositions → `ct.convert(..., compute_precision=FP16ComputePrecision(op_selector))`.
   - The selector is the geometry list plus the small-tensor rule.
   - This is NVIDIA's own `tools/onnx/precision.py` idea: networks fp16, geometry fp32.
4. **Check before saving, per batch size.**
   - Run `placement.py`'s check: fail if NE ops == 0. That catches any regression of the whole-program fallback.
   - Run the 84-input accuracy check against `build_model()`: mean|Δ| ≤ 0.012, same length ≥ 83/84.
5. **Remove** the ONNX / onnxsim / onnx2torch steps, their patches, and the onnx2torch dependency. Keep the old route on a tag for comparison.

**Open items.**
- **CPU work still left:** the FK and geometry on the inputs and outputs, (B,·,30,3) in fp32 by design, and the tail's one-hot `index_put` blends. That is fine for now; it's the next place to look if more speed is needed.
