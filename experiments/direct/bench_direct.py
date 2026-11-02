"""Throughput as tools/bench_neural_engine.py measures it (1 call, and 8 calls queued in one predict), for any
packages, side by side.   python experiments/direct/bench_direct.py <pkg>:<batch> ..."""
import sys, os, shutil, tempfile, time; sys.path.insert(0, ".")
import numpy as np, coremltools as ct
from cmb.convert import INPUT_NAMES
from cmb.paths import FEEDS
z = np.load(FEEDS); feeds = [{k: z[f"{k}_{i}"] for k in INPUT_NAMES} for i in range(int(z["n"]))]
tmpdir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
try:
    for arg in sys.argv[1:]:
        pkg, batch = arg.rsplit(":", 1); batch = int(batch)
        compiled = os.path.join(tmpdir, "m.mlmodelc"); shutil.rmtree(compiled, ignore_errors=True)
        ct.utils.compile_model(pkg, compiled)
        m = ct.models.CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        names = [i.name for i in ct.utils.load_spec(pkg).description.input]
        chunks = [{nm: np.concatenate([feeds[(c * batch + j) % len(feeds)][k] for j in range(batch)])
                   for nm, k in zip(names, INPUT_NAMES)} for c in range(8)]
        for queued in (1, 8):
            work = chunks[:queued]
            run = (lambda: m.predict(work)) if queued > 1 else (lambda: m.predict(work[0]))
            run(); n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 5:
                run(); n += 1
            dt = (time.perf_counter() - t0) / n
            print(f"{pkg:<45} batch {batch} x {queued} queued: {queued * batch / dt:5.0f} characters/s "
                  f"({dt / queued * 1000:5.1f} ms per call)", flush=True)
        del m
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
