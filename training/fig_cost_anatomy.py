#!/usr/bin/env python3
# fig_cost_anatomy.py -- stacked bars of the cold single-image cost (Table 4): weight read, pure computation
# and memory for Path B and Path C, read from outputs/mnist_e2e_cb.txt. Nothing is typed in.
# Usage (from the repository root): python3 training/fig_cost_anatomy.py  -> outputs/fig_cost_anatomy.pdf
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = open("outputs/mnist_e2e_cb.txt").read()
assert "ALL CHECKS PASSED" in text or "END-TO-END C/B" in text
rows = {}
for name in ("C", "B"):
    m = re.search(r"^\s*" + name + r"\s+cold\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)", text, re.M)
    rows[name] = [int(x) for x in m.groups()]        # sload, memory, compute, total
print("C cold:", rows["C"], "| B cold:", rows["B"])

COLS = {"weight read": "#222222", "pure computation": "#C0392B", "memory": "#E67E22"}
plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "axes.linewidth": 0.5})
fig, ax = plt.subplots(figsize=(3.35, 1.55))
labels = ["Path C\nbyte extraction", "Path B\nfixed-width SIMD"]
order = [("weight read", 0), ("pure computation", 2), ("memory", 1)]
bottoms = [0.0, 0.0]
small = {"C": [], "B": []}
for comp, idx in order:
    vals = [rows["C"][idx] / 1e6, rows["B"][idx] / 1e6]
    ax.barh([0, 1], vals, left=bottoms, color=COLS[comp], height=0.5, label=comp, edgecolor="white", linewidth=0.4)
    for i, v in enumerate(vals):
        name = "C" if i == 0 else "B"
        share = 100.0 * rows[name][idx] / rows[name][3]
        if share >= 10:
            ax.text(bottoms[i] + v / 2, i, f"{share:.1f}%", ha="center", va="center", fontsize=6, color="white")
        else:
            small[name].append(f"{comp} {share:.1f}%")
    bottoms = [bottoms[0] + vals[0], bottoms[1] + vals[1]]
for i, name in enumerate(("C", "B")):
    tail = f"{rows[name][3] / 1e6:.1f} M gas"
    if small[name]:
        tail += "\n" + ", ".join(small[name])
    ax.text(bottoms[i] + 0.4, i, tail, va="center", fontsize=5.8, color="#333333", linespacing=1.3)
wr = rows["B"][0] / 1e6
ax.axvline(wr, color="#888888", linestyle=(0, (3, 2)), linewidth=0.6, zorder=0)
ax.text(wr, 1.42, f"weight read ends at {wr:.1f}", ha="center", va="bottom", fontsize=5.5, color="#555555")
ax.set_ylim(-0.45, 1.75)
ax.set_xticks(range(0, 45, 5)); ax.set_xticks(range(0, 45, 1), minor=True)
ax.tick_params(axis="x", which="minor", length=1.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
ax.set_xlabel("gas of one cold inference (millions)")
ax.set_xlim(0, max(bottoms) * 1.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower left", fontsize=6, frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.0))
fig.tight_layout(pad=0.3)
fig.savefig("outputs/fig_cost_anatomy.pdf", bbox_inches="tight", pad_inches=0.02)
print("written outputs/fig_cost_anatomy.pdf")
