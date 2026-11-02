import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch import ops as _ops
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

def _fixed_cast(py_type, original):
    def convert(context, node):
        x = _ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and np.size(x.val) == 1 and not isinstance(x.val, py_type):
            context.add(mb.const(val=py_type(np.asarray(x.val).reshape(-1)[0].item()), name=node.name), node.name)
        else:
            original(context, node)
    return convert

for name, py_type in (("int", int), ("bool", bool)):
    original = _TORCH_OPS_REGISTRY.name_to_func_mapping[name]
    _TORCH_OPS_REGISTRY.name_to_func_mapping[name] = _fixed_cast(py_type, original)
