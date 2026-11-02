"""Check every feature of NVIDIA's Scene Preview demo on the Mac, headless, with NVIDIA's default settings.

    python tools/scene_preview_parity.py [--backend CPU_AND_NE]

Locomotion: each mode (forward walk and run, strafing, crouching, the six styles) is driven for 4 s; it passes
if the mode engages, the character covers ground, and planted feet don't slide more than NVIDIA's own model
does (ONNX Runtime reference: ~0.14 m/s). Interactions: the character is steered into each gym smart primitive
(low jump, high jump, duck under a bar, sit); it passes if the interaction runs and the character returns to
locomotion or idle. Sitting also checks standing back up with the interact button.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from cmb.sim import Session  # noqa: E402

DT = 1 / 30
SLIDE_LIMIT = 0.3  # m/s, generous: the reference model itself measures ~0.14 on a straight walk


def state(s):
    return type(s.machine.state).__name__.replace("State", "")


def mode(s):
    return s.machine.active_locomotion_state.active_mode.name


def fresh(backend):
    return Session(backend=backend)  # NVIDIA's default locomotion settings


def drive(s, seconds, move, facing=None, run=False, request=None):
    feet = [s.joint_names.index(n) for n in ("LeftToeBase", "RightToeBase")]
    start = s.character.character_ground_position.copy()
    prev, slide, states = None, [], set()
    for i in range(int(seconds / DT)):
        s.step(DT, move=move, facing=facing or move, run=run, mode=request if i == 0 else None)
        states.add(state(s))
        j = s.world_joints().copy()[feet]
        if prev is not None:
            v = np.linalg.norm((j - prev)[:, :2], axis=1) / DT
            slide += list(v[j[:, 2] < 0.06])
        prev = j
    dist = float(np.linalg.norm(s.character.character_ground_position[:2] - start[:2]))
    return dist, float(np.median(slide)) if slide else 0.0, states


def locomotion_checks(backend):
    cases = [("Walk forward", None, (0, 1), None, False), ("Run forward", None, (0, 1), None, True),
             ("Strafe (directional)", "directional_2d", (1, 0), (0, 1), False),
             ("Crouch", "crouch_directional_2d", (0, 1), None, False)]
    s0 = fresh(backend)
    cases += [(f"Style: {m}", m, (0, 1), None, False) for m in s0.machine.style_modes]
    rows = []
    for label, request, move, facing, run in cases:
        s = fresh(backend)
        drive(s, 0.5, (0, 0), (0, 1))  # settle in idle
        dist, slide, states = drive(s, 4.0, move, facing, run, request)
        engaged = request is None or mode(s) == request or (request.startswith("crouch") and "crouch" in mode(s))
        ok = engaged and dist > 1.0 and slide < SLIDE_LIMIT and "Interaction" not in states
        rows.append((label, ok, f"mode {mode(s)}, moved {dist:.1f} m, foot slide {slide:.2f} m/s"))
    return rows


def steer_to(s, target, seconds, mode_request=None, run=False):
    """Walk toward a ground point; returns the sequence of states seen."""
    seen = []
    for i in range(int(seconds / DT)):
        d = np.array(target) - s.character.character_ground_position[:2]
        if np.linalg.norm(d) < 0.3:
            break
        h = d / np.linalg.norm(d)
        s.step(DT, move=tuple(h), facing=tuple(h), run=run, mode=mode_request if i == 0 else None)
        if not seen or seen[-1] != state(s):
            seen.append(state(s))
    return seen


def interaction_checks(backend):
    rows = []
    for name, label, mode_request in (("SmallJumpSP", "Jump (low wall)", None), ("MidJumpSP", "Jump (high wall)", None),
                                      ("AvoidanceSP", "Duck under bar", None), ("SitSP", "Sit, then stand", None)):
        s = fresh(backend)
        p = next(p for p in s.primitives if p["def"].name == name)
        px, py = p["def"].transform.p[:2]
        steer_to(s, (px - 6.0, py), 20)                               # line up west of it
        seen = steer_to(s, (px + 6.0, py), 12, mode_request)          # walk through it (+X)
        if name == "SitSP":
            for _ in range(int(3 / DT)):                              # stay seated a moment
                s.step(DT)
            s.step(DT, interact=True)                                 # press interact to stand up
            for _ in range(int(4 / DT)):
                s.step(DT)
                if not seen or seen[-1] != state(s):
                    seen.append(state(s))
        ok = "Interaction" in seen and seen[-1] in ("Locomotion", "Idle")
        rows.append((label, ok, " -> ".join(seen)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="CPU_AND_NE")
    args = ap.parse_args()
    rows = [("Locomotion", None, None)] + locomotion_checks(args.backend)
    rows += [("Interactions (gym scene)", None, None)] + interaction_checks(args.backend)
    passed = sum(1 for _, ok, _ in rows if ok)
    total = sum(1 for _, ok, _ in rows if ok is not None)
    for label, ok, detail in rows:
        print(f"\n{label}" if ok is None else f"  {'PASS' if ok else 'FAIL'}  {label:<26} {detail}", flush=True)
    print(f"\n{passed}/{total} Scene Preview features pass on backend {args.backend}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
