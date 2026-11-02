"""Motion quality: drive one character through the same walk / turn / stop with each backend and compare.

    python tools/motion_quality.py [--image docs/images/walk_comparison.png]

Reports planted-foot slide (toe speed while within 6 cm of the ground; lower is better), the mean distance
between each backend's joints and ONNX Runtime's over the run, and the time per generation; optionally draws
the walks side by side as stick figures.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from cmb.sim import Session  # noqa: E402

DT = 1 / 30
BACKENDS = [("ort", "ONNX Runtime, CPU (reference)"), ("CPU_AND_NE", "CoreML, Neural Engine"),
            ("CPU_AND_GPU", "CoreML, GPU")]


def drive(backend):
    s = Session(backend=backend)
    feet = [s.joint_names.index(n) for n in ("LeftToeBase", "RightToeBase")]
    frames, slide, gen_ms = [], [], []
    for i in range(int(9 / DT)):
        t = i * DT
        move = (0, 1) if t < 4 else (-1, 0) if t < 7 else (0, 0)  # walk, turn left, stop
        s.step(DT, move=move, facing=move if move != (0, 0) else (-1, 0))
        if "inference" in s.engine.timer.milliseconds:
            gen_ms.append(s.engine.timer.milliseconds.pop("inference"))
        j = s.world_joints().copy()
        if frames:
            f, fp = j[feet], frames[-1][feet]
            v = np.linalg.norm((f - fp)[:, :2], axis=1) / DT
            slide += list(v[f[:, 2] < 0.06])
        frames.append(j)
    return np.array(frames), float(np.median(slide)), float(np.median(gen_ms[1:])), s.parents


def draw(runs, parents, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, series = "#0b0b0b", "#52514e", "#2a78d6"
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs)), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (label, frames) in zip(axes, runs):
        ax.set_facecolor("#fcfcfb")
        picks = np.linspace(int(1 / DT), int(4 / DT) - 1, 9).astype(int)  # steady straight walk, side view
        for f in picks:
            p = frames[f]
            for j, par in enumerate(parents):
                if par >= 0:
                    ax.plot([p[j, 1], p[par, 1]], [p[j, 2], p[par, 2]], color=series, lw=1.4,
                            solid_capstyle="round")
        ax.set_title(label, loc="left", fontsize=10, color=ink)
        ax.set_aspect("equal")
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d9d8d4")
        ax.set_ylabel("height (m)", fontsize=8, color=muted)
    axes[-1].set_xlabel("forward (m)", fontsize=8, color=muted)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    args = ap.parse_args()
    runs, ref = [], None
    print(f"{'backend':<32} {'foot slide':>11} {'mean joint distance to ORT':>27} {'ms/generation':>14}")
    for backend, label in BACKENDS:
        frames, slide, ms, parents = drive(backend)
        ref = frames if ref is None else ref
        dist = np.linalg.norm(frames - ref, axis=-1).mean()
        print(f"{label:<32} {slide:8.2f} m/s {dist * 100:22.1f} cm {ms:13.1f}", flush=True)
        runs.append((label, frames))
    if args.image:
        draw(runs, parents, args.image)
        print("saved", args.image)


if __name__ == "__main__":
    main()
