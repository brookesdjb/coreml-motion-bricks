import sys, types, torch
def install():
    def one_hot(x, num_classes):
        return (x.long().unsqueeze(-1) == torch.arange(num_classes, device=x.device)).long()
    functional = types.ModuleType("functional_shim"); functional.__dict__.update(torch.nn.functional.__dict__)
    functional.one_hot = one_hot
    nn = types.ModuleType("nn_shim"); nn.__dict__.update(torch.nn.__dict__); nn.functional = functional
    shim = types.ModuleType("torch_capture_shim"); shim.__dict__.update(torch.__dict__)
    shim.all = lambda *a, **k: True
    shim.nn = nn
    for name in ("motionbricks_exporter.models.soma.inference", "motionbricks_exporter.models.features"):
        sys.modules[name].t = shim
