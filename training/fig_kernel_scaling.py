#!/usr/bin/env python3
# fig_kernel_scaling.py -- Figure: pure computational gas of the A^T v kernel at ten sizes (points) with the
# closed-form polynomials (lines), and the ratio C/B with its asymptote. Every number is read from
# outputs/atv_closed_form.txt; nothing is typed in.
# Usage (from the repository root): python3 training/fig_kernel_scaling.py
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = open("outputs/atv_closed_form.txt").read()

# rows of the "PURE COMPUTE PER K" table:  K | pure compute C | pure compute B | C/B | ...
rows = re.findall(r"^\s*(\d+) \|\s*(\d+) \|\s*(\d+) \|\s*([\d.]+) \|", text, re.M)
K = np.array([int(r[0]) for r in rows]); C = np.array([int(r[1]) for r in rows])
B = np.array([int(r[2]) for r in rows]); R = np.array([float(r[3]) for r in rows])

# closed-form coefficients and the asymptote, as printed by the program
aC, bC, cC = [float(x) for x in re.search(r"C \(scalar\): OG\(K\) = ([\d.]+)\*K\^2 \+ ([\d.]+)\*K \+ ([\d.]+)", text).groups()]
aB, bB, cB = [float(x) for x in re.search(r"B \(SIMD\): OG\(K\) = ([\d.]+)\*K\^2 \+ ([\d.]+)\*K \+ ([\d.]+)", text).groups()]
asym = float(re.search(r"ASYMPTOTIC PURE-COMPUTE C/B \(K->inf\) = aC/aB = [\d.]+/[\d.]+ = ([\d.]+)x", text).group(1))
assert "ALL CHECKS PASSED" in text
print("K =", list(K)); print("ratios =", list(R)); print("asymptote =", asym)

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.linewidth": 0.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.35, 1.95), gridspec_kw={"width_ratios": [1.15, 1]})
Ks = np.logspace(np.log10(K[0]), np.log10(K[-1]), 200)
ax1.plot(Ks, aC * Ks**2 + bC * Ks + cC, color="#D55E00", linewidth=1.2)             # Path C, closed form
ax1.plot(Ks, aB * Ks**2 + bB * Ks + cB, color="#009E73", linewidth=1.2)             # Path B, closed form
ax1.plot(K, C, "o", color="#D55E00", markersize=3.2, markerfacecolor="white", markeredgewidth=0.9)
ax1.plot(K, B, "s", color="#009E73", markersize=3.0, markerfacecolor="white", markeredgewidth=0.9)
ax1.set_xscale("log", base=2); ax1.set_yscale("log")
ax1.set_xlabel("matrix size $K$ ($A$ is $K \\times K$)"); ax1.set_ylabel("computational gas")
ax1.plot([], [], "-o", color="#D55E00", markersize=3.2, markerfacecolor="white", label="Path C")
ax1.plot([], [], "-s", color="#009E73", markersize=3.0, markerfacecolor="white", label="Path B")
ax1.legend(loc="lower right", fontsize=7, frameon=False, handlelength=1.8, borderaxespad=0.3)
ax2.plot(K, R, "-o", color="#333333", linewidth=1.0, markersize=3.0, markerfacecolor="white", markeredgewidth=0.9)
ax2.axhline(asym, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8)
ax2.text(K[-1] * 1.1, asym * 0.72, f"asymptote {asym:.2f}", va="top", ha="right", fontsize=7, color="#555555")
ax2.set_xscale("log", base=2); ax2.set_ylim(0, asym * 1.15)
ax2.set_xlabel("matrix size $K$"); ax2.set_ylabel("gas ratio C / B")
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#dddddd", linewidth=0.4); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig("outputs/fig_kernel_scaling.pdf", bbox_inches="tight", pad_inches=0.06)
print("written outputs/fig_kernel_scaling.pdf")
