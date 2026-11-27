#!/usr/bin/env python3
# fig_levels.py -- Figure: the three dispatch levels of the instruction family. A product of a matrix with R
# rows of n elements is drawn as R rows of groups of eight elements. One coloured block is the work of one
# dispatched instruction: one group (XMACC, XMACC_S), one row (XMACC_N, XMACC_N16, XMACC_V) or the whole
# product (XMATMUL). The figure is a schematic and carries no measured number.
# Usage (from the repository root): python3 training/fig_levels.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R, GROUPS = 4, 6
BLUE, LIGHT = "#0072B2", "#CFE5F2"
panels = [("(a) group level", "XMACC, XMACC_S", r"$R \cdot \lceil n/8 \rceil$ dispatches", "group"),
          ("(b) row level", "XMACC_N, XMACC_N16, XMACC_V", r"$R$ dispatches", "row"),
          ("(c) matrix level", "XMATMUL", "1 dispatch", "matrix")]
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75))
for ax, (title, names, count, level) in zip(axes, panels):
    for r in range(R):
        for g in range(GROUPS):
            ax.add_patch(Rectangle((g, R - 1 - r), 1, 0.8, facecolor=LIGHT, edgecolor="white", linewidth=0.8))
    pad = 0.06
    if level == "group":
        for r in range(R):
            for g in range(GROUPS): ax.add_patch(Rectangle((g + pad, R - 1 - r + pad), 1 - 2 * pad, 0.8 - 2 * pad, fill=False, edgecolor=BLUE, linewidth=1.0))
    elif level == "row":
        for r in range(R): ax.add_patch(Rectangle((pad, R - 1 - r + pad), GROUPS - 2 * pad, 0.8 - 2 * pad, fill=False, edgecolor=BLUE, linewidth=1.0))
    else:
        ax.add_patch(Rectangle((-0.08, -0.08), GROUPS + 0.16, R - 0.2 + 0.16, fill=False, edgecolor=BLUE, linewidth=1.0))
    ax.set_xlim(-0.3, GROUPS + 0.3); ax.set_ylim(-1.75, R + 0.75); ax.axis("off")
    ax.text(GROUPS / 2, R + 0.3, title, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(GROUPS / 2, -0.45, names, ha="center", va="top", fontsize=7, family="monospace")
    ax.text(GROUPS / 2, -1.15, count, ha="center", va="top", fontsize=7.5)
axes[0].annotate("", xy=(1, R - 0.1), xytext=(0, R - 0.1), arrowprops=dict(arrowstyle="<->", linewidth=0.5, shrinkA=0, shrinkB=0))
axes[0].text(0.5, R + 0.02, "8 elements", ha="center", va="bottom", fontsize=6)
axes[0].text(-0.2, (R - 0.2) / 2, "R rows", ha="right", va="center", fontsize=6.5, rotation=90)
fig.tight_layout(pad=0.2, w_pad=1.0); fig.savefig("outputs/fig_levels.pdf"); fig.savefig("/tmp/fig_levels.png", dpi=150)
print("wrote outputs/fig_levels.pdf")
