"""The crowd demo's street: 60 m along +Y, buildings either side, rows of NVIDIA's gym obstacles (jump box,
jump wall, duck bar) turned to face the walking direction. Generated into build/scenes/ from the upstream gym
scene, with smart-primitive references made absolute so the file can live outside the upstream tree."""
import copy
import json
import math

from .paths import BUILD, UPSTREAM

GYM = UPSTREAM / "assets" / "scenes" / "soma_gym.scene.json"
STREET_START, STREET_END, LANE_HALF_WIDTH = -2.0, 62.0, 4.0
LENGTH, HALF_WIDTH = 60.0, 6.0
# (gym template, y, [x positions]). Gym primitives face +X at -90 deg, so 0 deg faces up the street.
ROWS = [("SmallJumpSP", 12, [-3.0, 0.0, 3.0]), ("MidJumpSP", 22, [-2.5, 2.5]), ("AvoidanceSP", 32, [-3.0, 0.0, 3.0]),
        ("SmallJumpSP", 42, [-1.5, 1.5]), ("MidJumpSP", 50, [0.0])]


def _yaw(deg):
    r = math.radians(deg) / 2
    return [0.0, 0.0, math.sin(r), math.cos(r)]


def street_scene(path=BUILD / "scenes" / "street.scene.json"):
    """Write the street scene if needed and return its path."""
    if path.exists() and path.stat().st_mtime >= GYM.stat().st_mtime and path.stat().st_mtime >= _SELF_MTIME:
        return path
    gym = json.loads(GYM.read_text())
    templates = {p["name"]: p for p in gym["smart_primitives"]}
    children = {}
    for c in gym["solid_cubes"]:
        children.setdefault(c["parent_id"], []).append(c)
    scene = {k: copy.deepcopy(gym[k]) for k in ("format", "schema_version", "rig", "editor")}
    scene["name"] = "street"
    scene["character_start"] = {"id": 1, "transform": {"translation": [0.0, 2.0, 0.0], "rotation_xyzw": _yaw(0)},
                                "euler_degrees": [0.0, 0.0, 0.0]}
    scene["smart_primitives"], scene["solid_cubes"] = [], []
    next_id = 100
    for name, y, xs in ROWS:
        for x in xs:
            p = copy.deepcopy(templates[name])
            old_id, p["id"] = p["id"], next_id
            p["name"] = f"{name}_{next_id}"
            p["reference"] = str((GYM.parent / p["reference"]).resolve())
            p["transform"] = {"translation": [x, y, 0.0], "rotation_xyzw": _yaw(0)}
            p["euler_degrees"] = [0.0, 0.0, 0.0]
            scene["smart_primitives"].append(p)
            for c in children.get(old_id, []):
                c = copy.deepcopy(c)
                next_id += 1
                c["id"], c["parent_id"] = next_id, p["id"]
                scene["solid_cubes"].append(c)
            next_id += 1
    for side in (-1, 1):  # buildings
        for i, y in enumerate(range(-4, int(LENGTH) + 8, 8)):
            h = 6.0 + 5.0 * ((i * 7 + (side > 0) * 3) % 4)
            scene["solid_cubes"].append({
                "id": next_id, "name": f"Building_{next_id}", "parent_id": None,
                "transform": {"translation": [side * (HALF_WIDTH + 3.0), float(y), h / 2], "rotation_xyzw": _yaw(0)},
                "euler_degrees": [0.0, 0.0, 0.0],
                "dimensions": {"width": 6.0, "depth": 7.6, "height": h},
                "color": [0.32 + 0.05 * (i % 3), 0.34 + 0.04 * ((i + 1) % 3), 0.40 + 0.05 * (i % 2)]})
            next_id += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{id(scene)}.tmp")
    tmp.write_text(json.dumps(scene, indent=1))
    tmp.replace(path)  # atomic: several worker processes may call this at once
    return path


_SELF_MTIME = __import__("pathlib").Path(__file__).stat().st_mtime
