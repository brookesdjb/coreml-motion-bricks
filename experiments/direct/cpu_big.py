"""CPU-placed ops (under CPU_AND_NE) with large outputs: the work that keeps a package off the Neural Engine."""
import sys, ast, re, collections, shutil, subprocess
import numpy as np, coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
pkg, minsize = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4096
listing = subprocess.run([sys.executable, "experiments/direct/mil_window.py", pkg, "list"], capture_output=True, text=True).stdout
shapes = {}
for line in listing.splitlines():
    m = re.match(r"(\d+) (\S+) (\S+) (\[.*?\]) ", line)
    if m:
        shapes[m.group(3)] = [s for _, s in ast.literal_eval(m.group(4))]
mm = ct.models.MLModel(pkg); path = mm.get_compiled_model_path()
plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
tot = collections.Counter()
for k, op in enumerate(plan.model_structure.program.functions["main"].block.operations):
    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if u is None or "CPU" not in type(u.preferred_compute_device).__name__: continue
    shp = shapes.get(op.outputs[0].name, [()])[0]
    n = int(np.prod(shp)) if shp else 1
    tot["elements"] += n
    if n >= minsize:
        print(k, op.operator_name.split(".")[-1], op.outputs[0].name, shp)
print("total CPU output elements", tot["elements"])
del mm; shutil.rmtree(path, ignore_errors=True)
