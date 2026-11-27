#!/usr/bin/env python3
# fig_schedule.py -- Figure: covering gas of the fixed-work instructions on the four builds and their prices.
# The covering gas of an instruction on a build is its measured time divided by the reference rate of the
# build (ns per gas of the reference step), that is, the gas that pays for its time. For every instruction
# the figure draws four bars (x86-64 AVX2, AArch64 NEON, x86-64 scalar fallback, AArch64 scalar fallback) and
# a black line at the price. For XMACC_S the bar is the larger of its two layouts (the hostile view). Every
# number is read from outputs/price_calibration.txt, which training/price_calibration.py writes from the
# twelve price_bench files; nothing is typed in.
# Usage (from the repository root): python3 training/price_calibration.py && python3 training/fig_schedule.py
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = open("outputs/price_calibration.txt").read()
assert "CALIBRATION COMPLETE" in text
need = {}; price = {}; order = []
for m in re.finditer(r"^\s*(X\w+) fixed price[^|]*\|(.*)\| price\s+(\d+)\s*$", text, re.M):
    name = m.group(1); g = [float(x) for x in re.findall(r"([\d.]+) gas", m.group(2))]; assert len(g) == 4
    if name not in need: order.append(name); need[name] = g
    else: need[name] = [max(a, b) for a, b in zip(need[name], g)]
    price[name] = int(m.group(3))
order.sort(key=lambda n: price[n])
for n in order: print("%-10s covering gas %s price %d" % (n, need[n], price[n])); assert max(need[n]) <= price[n]

# column in the file (0 x86 avx2, 1 x86 scalar, 2 arm64 neon, 3 arm64 scalar), label, colour
series = [(0, "x86-64, AVX2", "#1F4E8C", None), (1, "x86-64, scalar fallback", "#1F4E8C", "////"),
          (2, "AArch64, NEON", "#A3202F", None), (3, "AArch64, scalar fallback", "#A3202F", "////")]
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.5, "hatch.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(7.0, 2.9)); x = np.arange(len(order)); w = 0.19
for k, (col, label, color, hatch) in enumerate(series):
    ax.bar(x + (k - 1.5) * w, [need[n][col] for n in order], w, label=label, color=color if hatch is None else "white",
           edgecolor=color, hatch=hatch, linewidth=0.6)
for i, n in enumerate(order):
    ax.plot([i - 2.2 * w, i + 2.2 * w], [price[n]] * 2, color="black", linewidth=1.1, label="price" if i == 0 else None)
    ax.text(i, price[n] * 1.06, str(price[n]), ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_ylim(2, 130); ax.set_yticks([2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7, family="monospace", rotation=30, ha="right", rotation_mode="anchor")
ax.set_ylabel("covering gas\n(measured time / reference rate)"); ax.tick_params(width=0.5, length=2.5)
ax.grid(axis="y", linewidth=0.3, color="0.85"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(ncol=5, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.15), columnspacing=1.2, handlelength=1.4)
fig.tight_layout(pad=0.3); fig.savefig("outputs/fig_schedule.pdf")
print("wrote outputs/fig_schedule.pdf")
