#!/usr/bin/env python3
# price_calibration.py -- Price of XMACC_V from the measured times of four builds.
# Reads outputs/price_calibration_runs/{x86,arm64}_run{1,2,3}_gram_{avx2|neon,scalar}.txt. These are the runs of
# gram_views made while XMACC_V still had its provisional price (14 gas per strided group of eight elements,
# 11 gas per unit-stride group, no charge per call). Writes outputs/price_calibration.txt.
#
# Method. Path S uses standard instructions only, so its nanoseconds per gas is the reference. For a path that
# uses XMACC_V, the time-neutral charge of one call is the gas that one call would have to cost for the whole
# path to take the nanoseconds per gas of Path S:
#     charge per call = ((t_path / t_S) * computeS - (compute_path - gas charged to XMACC_V)) / calls
# with the median times of the three runs. The charge is computed for the five row lengths R = 16 ... 256 at
# C = 32 (2 to 32 groups of eight elements per call) and fitted by least squares to
#     charge per call = base + per_group * groups.
# This is done for the strided rows (Path V) and for the unit-stride rows (Path K) on each build. The price is
# the smallest whole number that is not below the fitted value of any build.
# Every number is read from the output files; nothing is typed in.
# Usage (from the repository root): python3 training/price_calibration.py
import math
import re
import statistics

BUILDS = [("x86", "avx2", "x86-64, AVX2 build"), ("x86", "scalar", "x86-64, scalar fallback build"),
          ("arm64", "neon", "AArch64, NEON build"), ("arm64", "scalar", "AArch64, scalar fallback build")]
RUNS = (1, 2, 3)
C_FIT = 32
R_FIT = (16, 32, 64, 128, 256)
OLD_STRIDED, OLD_UNIT = 14, 11


def read(plat, build, run):
    return open(f"outputs/price_calibration_runs/{plat}_run{run}_gram_{build}.txt").read()


def gas_rows(text):
    a = text.index("COMPUTE GAS = total gas"); b = text.index("LARGEST FEATURE")
    g = {}
    for m in re.finditer(r"^\s*(\d+) \|\s*(\d+) \|\s*\d+ \|\s*(\d+) \|\s*(\d+) \|\s*(\d+) \|", text[a:b], re.M):
        g[(int(m[1]), int(m[2]))] = (int(m[3]), int(m[4]), int(m[5]))
    return g


def time_rows(text):
    a = text.index("\nTIME (median"); b = text.index("\n\n", a)
    t = {}
    for m in re.finditer(r"^\s*(\d+) \|\s*(\d+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|", text[a:b], re.M):
        t[(int(m[1]), int(m[2]))] = (float(m[3]), float(m[4]), float(m[5]))
    return t


def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    c1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    c0 = my - c1 * mx
    worst = max(abs(c0 + c1 * x - y) / y for x, y in zip(xs, ys))
    return c0, c1, worst


def main():
    out = []
    w = out.append
    w("PRICE OF XMACC_V FROM THE MEASURED TIMES OF FOUR BUILDS")
    w("Source: outputs/price_calibration_runs/ (gram_views with the provisional price: 14 per strided group, 11 per unit-stride group, no charge per call)")
    texts = {(p, b, r): read(p, b, r) for p, b, _ in BUILDS for r in RUNS}
    G = gas_rows(next(iter(texts.values())))
    same = all(gas_rows(t) == G for t in texts.values())
    w(f"Gas tables identical in all {len(texts)} files: {'yes' if same else 'NO'}")
    w(f"Fit: C = {C_FIT}, R = {', '.join(str(r) for r in R_FIT)} (groups per call: {', '.join(str((r + 7) // 8) for r in R_FIT)}), median times of three runs")
    w("")
    w("TIME-NEUTRAL CHARGE OF ONE CALL (gas), by row length")
    w("                         build | rows    |" + "".join(f" R={r:<6d}|" for r in R_FIT))
    res = {}
    for p, b, label in BUILDS:
        T = [time_rows(texts[(p, b, r)]) for r in RUNS]
        xs, yv, yk = [], [], []
        for R in R_FIT:
            k = (R, C_FIT); cV, cK, cS = G[k]; grp = (R + 7) // 8; calls = C_FIT * C_FIT
            tv, tk, ts = (statistics.median(t[k][i] for t in T) for i in range(3))
            xs.append(grp)
            yv.append(((tv / ts) * cS - (cV - OLD_STRIDED * grp * calls)) / calls)
            yk.append(((tk / ts) * cS - (cK - OLD_UNIT * grp * calls)) / calls)
        res[(p, b)] = (fit(xs, yv), fit(xs, yk))
        w(f"{label:>30} | strided |" + "".join(f" {y:7.1f} |" for y in yv))
        w(f"{label:>30} | unit    |" + "".join(f" {y:7.1f} |" for y in yk))
    w("")
    w("LEAST-SQUARES FIT: charge per call = base + per_group * groups")
    w("                         build | strided: base | per group | largest deviation | unit: base | per group | largest deviation")
    for p, b, label in BUILDS:
        (v0, v1, vd), (k0, k1, kd) = res[(p, b)]
        w(f"{label:>30} | {v0:13.2f} | {v1:9.2f} | {100 * vd:16.1f}% | {k0:10.2f} | {k1:9.2f} | {100 * kd:16.1f}%")
    base = math.ceil(max(max(r[0][0], r[1][0]) for r in res.values()))
    strided = math.ceil(max(r[0][1] for r in res.values()))
    unit = math.ceil(max(r[1][1] for r in res.values()))
    w("")
    w("PRICE (smallest whole numbers not below the fitted value of any build)")
    w(f"charge per call:                   {base} gas")
    w(f"per group of eight, strided row:   {strided} gas")
    w(f"per group of eight, unit stride:   {unit} gas")
    w("")
    w("*** CALIBRATION COMPLETE ***" if same else "*** INSPECT: gas tables differ ***")
    open("outputs/price_calibration.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


main()
