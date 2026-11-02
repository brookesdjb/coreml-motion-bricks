import collections, coremltools as ct, shutil, sys
from coremltools.models.compute_plan import MLComputePlan
m = ct.models.MLModel(sys.argv[1]); path = m.get_compiled_model_path()
plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
ops, cost = collections.Counter(), collections.Counter()
top = []
segs, last = collections.Counter(), None
for op in plan.model_structure.program.functions["main"].block.operations:
    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if u is None: continue
    dev = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
    c = plan.get_estimated_cost_for_mlprogram_operation(op); ops[dev] += 1; top.append((c.weight if c else 0, dev, op.operator_name, [o.name for o in op.outputs][:1])); segs[dev] += dev != last and op.operator_name != "const"; last = dev if op.operator_name != "const" else last; cost[dev] += c.weight if c else 0
tot = sum(cost.values()) or 1
import os
for w, d, t, o in sorted(top, reverse=True)[:int(os.environ.get('TOP', 0))]:
    print(f'  {w/tot:6.2%} {d:<12} {t:<28} {o}')
print("placement", dict(ops), {k: f"{v/tot:.0%}" for k, v in cost.items()}, "segments", dict(segs))
shutil.rmtree(path, ignore_errors=True)
