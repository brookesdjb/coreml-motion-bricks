"""Headless version of NVIDIA's Scene Preview character loop: the SOMA character runtime, smart primitives
(jump, vault, duck, sit) and auto-interaction, driven by MacSOMAEngine instead of TensorRT. No viewer."""
import numpy as np

from . import _compat  # noqa: F401  (must precede motionbricks / app imports)
from .engine import MacSOMAEngine
from .paths import UPSTREAM

import warp as wp  # noqa: E402
from motionbricks.io.usd import load_skeleton_from_usd  # noqa: E402
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file  # noqa: E402
from motionbricks.model.contract import ModelRig  # noqa: E402
from motionbricks.runtime.smart_primitive import InteractionBox, compute_interaction_box_overlaps  # noqa: E402
from motionbricks.utils.pose_utils import compute_world_poses  # noqa: E402
from app.runtime.common.clip_registry import ClipRegistry  # noqa: E402
from app.runtime.common.io.scene_document import load_scene_document  # noqa: E402
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool  # noqa: E402
from app.runtime.common.motion import LocomotionCommand, LocomotionSettings  # noqa: E402
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402
from app.runtime.common.smart_primitive_behaviors import behavior_types  # noqa: E402
from app.runtime.soma.character import SOMACharacter  # noqa: E402

DEFAULT_SCENE = UPSTREAM / "assets" / "scenes" / "soma_gym.scene.json"
MAX_DT = 0.1

# NVIDIA's defaults smooth input heavily (speed half-life 0.6 s up / 0.4 s down) for a weighty feel.
# This preset reacts roughly twice as fast; regeneration thresholds are unchanged.
RESPONSIVE = LocomotionSettings(facing_halflife=0.08, velocity_direction_halflife=0.1,
                                speed_acceleration_halflife=0.25, speed_deceleration_halflife=0.18)


class SharedAssets:
    """Scene, skeleton and clip/smart-primitive pool, loaded once and shared by every character."""

    def __init__(self, scene_path=DEFAULT_SCENE):
        wp.init()
        wp.set_device("cpu")
        scene, _, _ = load_scene_document(scene_path)
        self.scene = scene.flatten()
        self.skeleton = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"),
                                               SOMA_SKELETON_PRIM_PATH)
        self.to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())  # = app rig.model_to_runtime(SOMA)
        self.pool = SmartPrimitiveAssetPool(ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton),
                                            self.skeleton.reference_local_transforms, self.to_runtime)
        for d in self.scene.smart_primitives:
            self.pool.load(d.reference)


class Session:
    """One character in a loaded scene: its smart primitives and the per-frame step.

    Pass shared/engine/start to run many characters on one SharedAssets (see crowd.py)."""

    def __init__(self, scene_path=DEFAULT_SCENE, backend="CPU_AND_NE", settings=None, *, shared=None, engine=None,
                 start=None):
        shared = shared or SharedAssets(scene_path)
        self.scene = scene = shared.scene
        skeleton = shared.skeleton
        to_runtime = shared.to_runtime
        self.skeleton = skeleton
        self.engine = engine or MacSOMAEngine(backend, model_to_runtime=to_runtime)
        pool = shared.pool
        self.character = SOMACharacter(self.engine, asset_pool=pool,
                                       initial_transform=start if start is not None else scene.character_start.transform)
        self.machine = self.character.state_machine
        if settings is not None:
            self.set_settings(settings)

        self.primitives, offset = [], 0
        for d in scene.smart_primitives:
            rt = pool.instantiate(d.reference, behavior_types()[d.behavior])
            rt.set_node_world_transform(0, d.transform)
            self.machine.interaction_state.prepare(self.character, rt)
            self.primitives.append({"def": d, "rt": rt, "offset": offset, "count": rt.interaction_box_count})
            offset += rt.interaction_box_count
        self.boxes = wp.empty(max(1, offset), dtype=InteractionBox)
        self.overlaps = wp.empty(max(1, offset), dtype=wp.int32)
        for p in self.primitives:
            if p["count"]:
                p["rt"].get_world_interaction_boxes(self.boxes[p["offset"]:p["offset"] + p["count"]])
        self.box_count = offset
        self.active, self.blocked, self.auto_interact = None, set(), True
        self._world = wp.empty_like(self.character.current_pose)
        self.parents = skeleton.parent_indices.numpy().tolist()
        self.joint_names = [skeleton.joint_name(i) for i in range(skeleton.num_joints)]

    def set_settings(self, settings):
        """Swap the movement smoothing on the character and every locomotion state."""
        self.character.movement.settings = settings
        for state in vars(self.machine).values():
            for st in (state if isinstance(state, (list, tuple)) else (state,)):
                if hasattr(st, "settings") and isinstance(getattr(st, "settings"), LocomotionSettings):
                    st.settings = settings

    @property
    def mode_names(self):
        return [m.name for m in self.machine.style_modes] if hasattr(self.machine, "style_modes") else []

    def step(self, dt, move=(0.0, 0.0), facing=(0.0, 1.0), run=False, mode=None, interact=False):
        """Advance one frame. move/facing are world-space XY ground vectors (Z up)."""
        dt = min(max(dt, 1e-6), MAX_DT)
        mv = np.array([move[0], move[1], 0.0], dtype=np.float32)
        fc = np.array([facing[0], facing[1], 0.0], dtype=np.float32)
        fc /= max(np.linalg.norm(fc), 1e-6)
        self.character.step(dt, LocomotionCommand(facing_direction=fc, movement=mv if np.linalg.norm(mv) > 1e-3 else None,
                                                  use_max_speed=run, requested_mode=mode))
        self._update_interactions(interact)

    def _update_interactions(self, pressed):
        nearest = None
        if self.box_count:
            if self.active is not None and self.active["count"]:
                self.active["rt"].get_world_interaction_boxes(
                    self.boxes[self.active["offset"]:self.active["offset"] + self.active["count"]])
            compute_interaction_box_overlaps(self.boxes[:self.box_count], self.character.probe_box,
                                             self.overlaps[:self.box_count], None)
            hits = self.overlaps.numpy()
            pos = self.character.character_ground_position
            for p in self.primitives:
                p["hit"] = bool(np.any(hits[p["offset"]:p["offset"] + p["count"]]))
            cands = [p for p in self.primitives if p["hit"]]
            nearest = min(cands, key=lambda p: float(np.sum((np.array(p["def"].transform.p) - pos) ** 2)), default=None)
        m = self.machine
        if self.active is not None and not m.is_interacting:
            self.blocked.add(self.active["def"].id)
            self.active = None
        for p in self.primitives:
            if not p.get("hit"):
                self.blocked.discard(p["def"].id)
        if m.is_interacting:
            if pressed:
                m.interact()
        elif nearest is not None and (pressed or (self.auto_interact and nearest["def"].id not in self.blocked)):
            m.start_interaction(self.character, nearest["rt"])
            self.active = nearest

    def world_joints(self):
        """World-space joint positions of the pose being shown, shape (joints, 3), Z up."""
        compute_world_poses(self.character.current_pose.reshape((1, -1)), self.skeleton.parent_indices,
                            self._world.reshape((1, -1)), wp.transform_identity())
        return self._world.numpy()[:, :3]

    def scene_description(self):
        cubes = [{"p": list(map(float, c.transform.p)), "q": list(map(float, c.transform.q)),
                  "size": [float(c.width), float(c.depth), float(c.height)], "color": list(map(float, c.color))}
                 for c in self.scene.solid_cubes]
        prims = [{"name": p["def"].name, "p": list(map(float, p["def"].transform.p)),
                  "q": list(map(float, p["def"].transform.q))} for p in self.primitives]
        return {"cubes": cubes, "primitives": prims, "parents": self.parents, "joints": self.joint_names}
