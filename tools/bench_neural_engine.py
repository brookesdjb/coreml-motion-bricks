"""Neural Engine benchmarks: where CoreML places the work, and throughput with batched + queued calls.

    python tools/bench_neural_engine.py
"""
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from cmb.convert import INPUT_NAMES  # noqa: E402
from cmb.engine import load_coreml  # noqa: E402
from cmb.paths import BATCH_SIZES, COMPILED_DIR, FEEDS, coreml_package  # noqa: E402


def placement(units):
    """Share of CoreML's estimated cost per device for the batch-1 model."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    load_coreml(coreml_package(1), units)  # make sure the compiled copy exists
    plan = MLComputePlan.load_from_path(str(COMPILED_DIR / "soma_b1.mlmodelc"),
                                        compute_units=getattr(ct.ComputeUnit, units))
    cost = collections.Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        dev = type(usage.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
        c = plan.get_estimated_cost_for_mlprogram_operation(op)
        cost[dev] += c.weight if c else 0
    total = sum(cost.values())
    return ", ".join(f"{k} {v / total:.0%}" for k, v in cost.most_common())


def main():
    z = np.load(FEEDS)
    feeds = [{k: z[f"{k}_{i}"] for k in INPUT_NAMES} for i in range(int(z["n"]))]
    for units in ("CPU_AND_NE", "ALL", "CPU_AND_GPU"):
        print(f"placement {units:<12} {placement(units)}", flush=True)
    for batch in BATCH_SIZES:
        if not coreml_package(batch).exists():
            continue
        m = load_coreml(coreml_package(batch), "CPU_AND_NE")
        chunks = [{k: np.concatenate([feeds[(c * batch + j) % len(feeds)][k] for j in range(batch)])
                   for k in INPUT_NAMES} for c in range(8)]
        for queued in (1, 8):
            work = chunks[:queued]
            run = (lambda: m.predict(work)) if queued > 1 else (lambda: m.predict(work[0]))
            run()
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 3:
                run()
                n += 1
            dt = (time.perf_counter() - t0) / n
            print(f"batch {batch} x {queued} queued: {queued * batch / dt:5.0f} characters/s "
                  f"({dt / queued * 1000:5.1f} ms per call)", flush=True)


if __name__ == "__main__":
    main()
