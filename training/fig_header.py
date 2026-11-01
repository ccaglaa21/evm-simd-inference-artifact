#!/usr/bin/env python3
# fig_header.py -- two schematic figures: the MatrixHeader descriptor (fig_header.pdf) and the two-layer
# memory check (fig_safety.pdf). No measured number is drawn; the only constant is the schedule price of
# XTRANSPOSE (10 gas). Usage: python3 fig_header.py   (writes the two PDFs into the current directory)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HDR_F, HDR_E = "#D6E4F0", "#365C82"      # header cells
DAT_F, DAT_E = "#F5ECD6", "#8C6E28"      # data lanes
ACC = "#C85028"                          # accent (swap arrows, halt)
GREY = "#555555"
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif"})

def cell(ax, x, y, w, h, text, fc, ec, fs=7, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def matrix(ax, x, y, rows, name, colw=6.5, rowh=6.5, gap=9):
    """draw name = [ rows ] with text cells; rows is a list of lists of mathtext strings"""
    n, m = len(rows), len(rows[0])
    h = n * rowh
    ax.text(x, y, name + " =", ha="left", va="center", fontsize=7)
    x0 = x + gap
    ax.plot([x0, x0 - 1.2, x0 - 1.2, x0], [y + h / 2, y + h / 2, y - h / 2, y - h / 2], color="black", lw=0.7)
    x1 = x0 + m * colw
    ax.plot([x1, x1 + 1.2, x1 + 1.2, x1], [y + h / 2, y + h / 2, y - h / 2, y - h / 2], color="black", lw=0.7)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            ax.text(x0 + colw * (j + 0.5), y + h / 2 - rowh * (i + 0.5), t, ha="center", va="center", fontsize=7)
    return x1 + 2

# ---------------------------------------------------------------- Figure 1: MatrixHeader
fig, ax = plt.subplots(figsize=(3.35, 3.1))
ax.set_xlim(0, 100); ax.set_ylim(-1, 100); ax.axis("off")
CW = 16      # header cell width; the eight data lanes share the same total width (4 x 16 = 8 x 8)
X0 = 26

def header_row(y, fields):
    x = X0
    for t in fields:
        cell(ax, x, y, CW, 7, t, HDR_F, HDR_E); x += CW

# block 1: the structure
ax.text(0, 97, "Example of the MatrixHeader structure", fontsize=7, weight="bold")
ax.text(X0 - 2, 87.5, "MatrixHeader:", ha="right", va="center", color=GREY)
header_row(84, ["rows: 2", "cols: 4", "$s_r$: 4", "$s_c$: 1"])
ax.text(X0 + 4 * CW + 2, 87.5, "32 byte", va="center", color=GREY)
ax.text(X0 - 2, 76.5, "data:", ha="right", va="center", color=GREY)
for i in range(8):
    cell(ax, X0 + 8 * i, 73, 8, 7, f"$a_{i}$", DAT_F, DAT_E)
ax.text(X0 + 4 * CW + 2, 76.5, "16-bit", va="center", color=GREY)
matrix(ax, 0, 61, [["$a_0$","$a_1$","$a_2$","$a_3$"],["$a_4$","$a_5$","$a_6$","$a_7$"]], "$A$")
ax.text(46, 61, r"element $(i,j)$ is lane $i \cdot s_r + j \cdot s_c$", fontsize=6.5, va="center", color=GREY)

# block 2: the transpose (same gap above as between the other blocks)
ax.text(0, 45, "XTRANSPOSE operation, 10 gas, zero-copy", fontsize=7, weight="bold")
ax.text(X0 - 2, 30.5, "MatrixHeader:", ha="right", va="center", color=GREY)
header_row(27, ["rows: 4", "cols: 2", "$s_r$: 1", "$s_c$: 4"])
for x0, x1 in ((X0 + 0.5 * CW, X0 + 1.5 * CW), (X0 + 2.5 * CW, X0 + 3.5 * CW)):
    ax.add_patch(FancyArrowPatch((x0, 35), (x1, 35), connectionstyle="arc3,rad=-0.5",
                                 arrowstyle="<->,head_length=3,head_width=2", color=ACC, lw=1.0))
matrix(ax, 0, 11, [["$a_0$","$a_4$"],["$a_1$","$a_5$"],["$a_2$","$a_6$"],["$a_3$","$a_7$"]], "$A^{\\top}$", rowh=5.5, gap=12)
ax.text(36, 11, "same eight lanes, read as the transpose", va="center", color=GREY)

fig.savefig("fig_header.pdf", bbox_inches="tight", pad_inches=0.02)
print("written fig_header.pdf")

# ---------------------------------------------------------------- Figure 2: two-layer memory check
fig, ax = plt.subplots(figsize=(3.35, 2.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.8, linespacing=1.4)

def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->,head_length=3,head_width=2", color="black", lw=0.8))

box(1, 66, 40, 30, "operands from the stack\n$\\mathit{off}$, $r$, $s$ and the header", HDR_F, HDR_E)
box(53, 66, 46, 30, "highest byte touched\n$\\mathit{off}+32+2\\,\\alpha(r,s{+}7)$, in 256 bits", HDR_F, HDR_E)
box(32, 14, 36, 30, "standard memory check\nexpansion charged", HDR_F, HDR_E)
box(1, 14, 26, 30, "instruction body\nreads and writes", DAT_F, DAT_E)
box(73, 14, 26, 30, "out of gas\nno access", "#F6DCD3", ACC)
arrow((41, 81), (53, 81))
ax.plot([76, 76, 50, 50], [66, 55, 55, 44], color="black", lw=0.8)
arrow((50, 46), (50, 44))
arrow((32, 29), (27, 29)); ax.text(29.5, 36, "in budget", ha="center", fontsize=5.6, color=GREY)
arrow((68, 29), (73, 29)); ax.text(70.5, 36, "over budget", ha="center", fontsize=5.6, color=GREY)
ax.text(70.5, 49, "or $\\geq 2^{32}$", ha="center", fontsize=5.6, color=GREY)
ax.text(1, 4, "second layer, XGATHER only: an index outside the data matrix reads as 0", fontsize=5.8, color=GREY, va="center")
fig.savefig("fig_safety.pdf", bbox_inches="tight", pad_inches=0.02)
print("written fig_safety.pdf")
