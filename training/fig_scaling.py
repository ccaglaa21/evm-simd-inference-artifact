#!/usr/bin/env python3
# fig_scaling.py -- Figure: row-length scaling of the row-level instruction on two architectures. For every row
# length it draws the ratio of the packed scalar program (Path S) to the row-level program (Path N): the time
# ratio on the vector build and on the scalar fallback build of the x86-64 host and of the AArch64 runner
# (median of three runs, with a bar from the smallest to the largest run), and the ratio of the computational
# gas, which is the same on every build and machine. Every number is read from outputs/scaling_summary.txt,
# which training/scaling_summary.py writes from the twelve run files; nothing is typed in.
# Usage (from the repository root): python3 training/scaling_summary.py && python3 training/fig_scaling.py
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = open("outputs/scaling_summary.txt").read()
assert "ALL CHECKS PASSED" in text and "FAILED" not in text


def section(title, until):
    a = text.index(title); b = text.index(until, a)
    return text[a:b]


# compute gas ratio S/N per row length
gas = re.findall(r"^\s*(\d+) \|\s*\d+ \|\s*\d+ \|\s*\d+ \|\s*([\d.]+) \|", section("COMPUTE GAS", "TIME RATIO, packed"), re.M)
I = np.array([int(g[0]) for g in gas]); G = np.array([float(g[1]) for g in gas])

# time ratio S/N: four columns of "median [min, max]"
cell = r"\s*([\d.]+) \[([\d.]+), ([\d.]+)\]"
rows = re.findall(r"^\s*(\d+) \|" + cell + r" \|" + cell + r" \|" + cell + r" \|" + cell, section("TIME RATIO, packed", "TIME RATIO, Path S over the sixteen"), re.M)
assert [int(r[0]) for r in rows] == list(I)
T = np.array([[float(x) for x in r[1:]] for r in rows]).reshape(len(rows), 4, 3)   # [length, series, (median, min, max)]
print("row lengths =", list(I)); print("gas ratio =", list(G)); print("time ratio medians at the last length =", list(T[-1, :, 0]))

series = [("x86-64, AVX2", "#0072B2", "o", "-"), ("x86-64, scalar", "#0072B2", "o", (0, (3, 2))),
          ("AArch64, NEON", "#D55E00", "s", "-"), ("AArch64, scalar", "#D55E00", "s", (0, (3, 2)))]

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(3.35, 2.35))
for k, (name, color, marker, style) in enumerate(series):
    med, lo, hi = T[:, k, 0], T[:, k, 1], T[:, k, 2]
    ax.errorbar(I, med, yerr=[med - lo, hi - med], color=color, linestyle=style, linewidth=1.1, marker=marker,
                markersize=3.0, markerfacecolor="white", markeredgewidth=0.9, elinewidth=0.7, capsize=1.5, label=name)
ax.plot(I, G, color="#333333", linestyle=(0, (1, 1.5)), linewidth=1.1, marker="^", markersize=3.0,
        markerfacecolor="white", markeredgewidth=0.9, label="computational gas")
ax.set_xscale("log", base=2); ax.set_xticks(I[::2]); ax.set_xticklabels([str(v) for v in I[::2]])
ax.set_ylim(0, max(T[:, :, 2].max(), G.max()) * 1.08)
ax.set_xlabel("row length (elements per neuron)"); ax.set_ylabel("ratio Path S / Path N")
ax.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=2.6, borderaxespad=0.2, labelspacing=0.25)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(color="#dddddd", linewidth=0.4); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig("outputs/fig_scaling.pdf", bbox_inches="tight", pad_inches=0.06)
print("written outputs/fig_scaling.pdf")
