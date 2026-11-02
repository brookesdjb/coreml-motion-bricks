"""Import-time shims that let NVIDIA's MotionBricks code run on a Mac without its GPU/viewer stack.

Everything here is a workaround for the pinned upstream commit (see README). Import this module before any
`motionbricks` or `app` import; cmb modules do it for you.

* tensorrt: motionbricks.inference imports it at module level. cmb.engine.MacSOMAEngine replaces the TensorRT
  path entirely, so a stub module is enough.
* newton, pyglet, imgui_bundle, trimesh, OpenGL: imported at module level by the demo app's rig/renderer
  modules. The headless runtime never calls them, so they (and every submodule) resolve to MagicMocks.
* sys.path: the upstream checkout is not an installed package.
"""
import importlib.abc
import importlib.machinery
import sys
from unittest.mock import MagicMock

from .paths import UPSTREAM

sys.modules.setdefault("tensorrt", MagicMock(name="tensorrt"))


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve MagicMock modules for viewer-only packages and all their submodules."""
    PACKAGES = ("newton", "pyglet", "imgui_bundle", "trimesh", "OpenGL")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.PACKAGES:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        mod = MagicMock(name=spec.name)
        mod.__path__, mod.__spec__ = [], spec
        return mod

    def exec_module(self, module):
        pass


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
