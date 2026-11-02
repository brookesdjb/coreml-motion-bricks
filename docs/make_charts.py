"""Draw the README charts from measured results (numbers from build/logs/, M2 Pro; see docs/findings.md).

    .venv/bin/python docs/make_charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "images"
SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def hbar(path, title, subtitle, labels, values, unit, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(labels) + 1.3))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, height=0.56, color=SERIES)
    top = max(values)
    for yi, v in zip(y, values):
        ax.text(v + top * 0.015, yi, fmt.format(v) + f" {unit}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.set_xlim(0, top * 1.22)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK, y=0.98)
    ax.set_title(subtitle, loc="left", fontsize=8.5, color=MUTED, pad=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# build/logs/motion_quality.txt: median time per generation, one character
hbar("generation_time.png", "Time per generation (one character)", "Lower is better. Median over a 9 s walk/turn/stop.",
     ["ONNX Runtime, CPU", "CoreML, GPU", "CoreML, Neural Engine"], [26.1, 12.8, 9.7], "ms", "{:.1f}")
# build/logs/bench_ne.txt: batch-B model, 8 calls queued per batch-API submission
hbar("ne_throughput.png", "Neural Engine throughput by batch size",
     "Characters generated per second, 8 calls queued per submission. Higher is better.",
     ["batch 1", "batch 2", "batch 4", "batch 8"], [132, 239, 352, 692], "/s")
print("charts written to", OUT)
