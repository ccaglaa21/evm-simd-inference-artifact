#!/usr/bin/env python3
# fig_cold_cost.py -- Figure: cold single-image inference cost of five contracts.
# Every number is read from the output files of run_all.sh; nothing is typed in.
# Usage (from the repository root): python3 training/fig_cold_cost.py
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def grab(filename, pattern):
    text = open("outputs/" + filename).read()
    return int(re.search(pattern, text).group(1))

bars = [
    ("standard EVM,\nstorage",   grab("mnist_e2e_cb.txt",    r"C\s+cold\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)"), "#D55E00"),
    ("XMACC,\nstorage",    grab("mnist_e2e_cb.txt",    r"B\s+cold\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)"), "#E69F00"),
    ("XMACC_N,\nstorage",  grab("mnist_rowmacc.txt",   r"\[N= 1\] N-deployable cold/warm = (\d+)/"),              "#F0E442"),
    ("XMACC_N, code,\n8 lanes",     grab("mnist_codecopy.txt",  r"\[N= 1\] cold/warm = (\d+)/"),                           "#009E73"),
    ("XMACC_N, code,\n16 lanes",    grab("mnist_codecopy16.txt", r"\[N= 1\] cold/warm = (\d+)/"),                          "#7FD1AE"),
]
cap = grab("mnist_codecopy.txt", r"EIP-7825 cap (\d+)")
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(3.35, 2.05))          # 85 mm wide, one column
y = list(range(len(bars)))[::-1]                      # first contract on top
ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62, linewidth=0)
for yi, (name, gas, _) in zip(y, bars):
    ax.text(gas * 1.12, yi, f"{gas:,}", va="center", ha="left", fontsize=8)
ax.axvline(cap, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8)
ax.annotate("per-transaction\ngas cap (EIP-7825)\n16,777,216", xy=(cap, 0.5), xytext=(cap * 3, 0.5), fontsize=7, color="#555555", va="center", ha="left", arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6})
ax.set_xscale("log")
ax.set_xlim(1e5, 1e9)
ax.set_ylim(-0.9, len(bars) - 0.4)
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars])
ax.set_xlabel("cold single-image inference cost (gas)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#dddddd", linewidth=0.4)
ax.set_axisbelow(True)
fig.savefig("outputs/fig_cold_cost.pdf", bbox_inches="tight", pad_inches=0.02)
print("written outputs/fig_cold_cost.pdf")

