"""MacSOMAEngine: NVIDIA's SOMA engine with TensorRT swapped for CoreML (or ONNX Runtime).

The pose encoding/decoding stays NVIDIA's own Warp kernels, which run on the CPU device; only the network call
changes. Backends:

* "CPU_AND_NE" (default), "CPU_AND_GPU", "CPU_ONLY", "ALL": the fp16 CoreML model with those compute units.
  CPU_AND_NE puts ~99% of the work on the Neural Engine and leaves the GPU free.
* "FP32:<units>", e.g. "FP32:ALL": the fp32 CoreML model (exact against the reference, a little slower).
* "ort": ONNX Runtime on the CPU, running NVIDIA's published fp16 ONNX model.
* a callable taking the input feeds dict and returning (motion, tokens): used by cmb.crowd's batcher.
"""
import os
import shutil
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import _compat  # noqa: F401  (must precede motionbricks imports)
from .paths import COMPILED_DIR, SOURCE_ONNX, coreml_package, coreml_package_fp32

import warp as wp  # noqa: E402
from motionbricks.model.config import MotionBricksInferenceConfig  # noqa: E402
from motionbricks.model.soma.engine import MotionBricksSOMAEngine  # noqa: E402


def load_coreml(package, units="CPU_AND_NE"):
    """Load a CoreML package via a compiled copy kept in build/compiled/.

    MLModel(package) compiles a fresh ~360 MB copy into $TMPDIR on every load and only deletes it on a clean
    exit, so crashed or killed processes leak them (we once found 34 GB). Compiling once avoids that and
    loads faster. Safe when several processes load the same model at once."""
    import coremltools as ct
    package = Path(package)
    if not package.exists():
        raise FileNotFoundError(f"{package} not found; run `python -m cmb.convert` first")
    compiled = COMPILED_DIR / (package.stem + ".mlmodelc")
    if not compiled.exists() or compiled.stat().st_mtime < package.stat().st_mtime:
        COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = COMPILED_DIR / f".{package.stem}.{os.getpid()}.mlmodelc"
        shutil.rmtree(tmp, ignore_errors=True)
        ct.utils.compile_model(str(package), str(tmp))
        try:
            shutil.rmtree(compiled, ignore_errors=True)
            os.rename(tmp, compiled)
        except OSError:  # another process won the race; use its copy
            shutil.rmtree(tmp, ignore_errors=True)
    return ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, units))


def split_outputs(out, batch=1):
    """CoreML output names are generated; pick the motion and token-count tensors by size."""
    motion = next(v for v in out.values() if np.size(v) > 100 * batch).reshape(batch, 64, 366)
    tokens = np.ravel(next(v for v in out.values() if np.size(v) == batch))
    return motion, tokens


class _Timer:
    """Wall-clock stand-in for NVIDIA's GenerationTimer, which needs CUDA events."""

    def __init__(self):
        self.milliseconds = {}

    @contextmanager
    def stage(self, name):
        t = time.perf_counter()
        yield
        self.milliseconds[name] = (time.perf_counter() - t) * 1000

    def read(self):
        pass


class MacSOMAEngine(MotionBricksSOMAEngine):
    """MotionBricksSOMAEngine with the TensorRT plan replaced by a CoreML / ONNX Runtime backend."""

    def __init__(self, backend="CPU_AND_NE", *, model_to_runtime=wp.transform_identity()):
        # Deliberately not calling super().__init__: it builds a TensorRT plan. This sets up the same state.
        self.backend = backend
        self.recorded_feeds = None  # set to a list to capture every network input
        self._closed = False
        self._stream = None
        self._cuda_graph = None
        specs = self.contract.tensor_specs
        self._tensor_specs = {s.name: s for s in specs}
        self._buffers = {s.name: wp.zeros(s.shape, dtype=wp.dtype_from_numpy(s.dtype), device="cpu") for s in specs}
        self._inputs = [s.name for s in specs if s.mode.value == "input"]
        if callable(backend):
            self._run, self.backend = backend, "external"
        elif backend == "ort":
            import onnxruntime as ort
            sess = ort.InferenceSession(str(SOURCE_ONNX), providers=["CPUExecutionProvider"])
            self._run = lambda feeds: sess.run(["pred_global_motions", "pred_num_tokens"], feeds)
        else:
            package, units = ((coreml_package_fp32(), backend[5:]) if backend.startswith("FP32:")
                              else (coreml_package(1), backend))
            model = load_coreml(package, units)
            # CoreML frees its inputs later on its own queue without the GIL; if that drops the last reference
            # to a numpy array, Python segfaults. Holding recent inputs makes Python free them instead.
            keepalive = deque(maxlen=64)

            def run(feeds):
                keepalive.append(feeds)
                motion, tokens = split_outputs(model.predict(feeds))
                return motion, tokens
            self._run = run
        self._use_cuda_graph = False
        self._model_to_runtime = wp.transform(model_to_runtime)
        self._runtime_to_model = wp.transform_inverse(self._model_to_runtime)
        self.timer = _Timer()
        self._alignment = wp.empty(1, dtype=wp.transform, device="cpu")
        self._generated_frame_count = wp.empty(1, dtype=wp.int32, device="cpu")
        self._initialize_pose_layout()

    @property
    def device(self):
        return wp.get_device("cpu")

    @property
    def stream(self):
        return None

    def enqueue_async(self):
        feeds = {n: self._buffers[n].numpy() for n in self._inputs}
        if self.recorded_feeds is not None:
            self.recorded_feeds.append({k: v.copy() for k, v in feeds.items()})
        motion, tokens = self._run(feeds)
        self._buffers["pred_global_motions"].assign(np.asarray(motion, dtype=np.float32).reshape(1, 64, 366))
        self._buffers["pred_num_tokens"].assign(np.asarray(tokens, dtype=np.int32).reshape(1))

    def synchronize(self):
        pass

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["MacSOMAEngine", "MotionBricksInferenceConfig", "load_coreml", "split_outputs"]
