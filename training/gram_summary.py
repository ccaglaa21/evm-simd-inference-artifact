#!/usr/bin/env python3
# gram_summary.py -- Summary of the Gram-matrix runs (gram_views) on the builds of the two architectures.
# Reads outputs/scaling_runs/{x86,arm64}_run{1,2,3}_gram_{avx2|neon,scalar}.txt (three runs on the vector build
# and on the scalar fallback build of each machine) and writes outputs/gram_summary.txt. A platform whose files
# are not present is left out and named. For every shape it reports the median, the smallest and the largest
# value of the three runs. It also checks that the gas tables and the check lines of all files are identical.
# Every number is read from the output files; nothing is typed in.
# Usage (from the repository root): python3 training/gram_summary.py
import os
import re
import statistics

PLATFORMS = [("x86", "avx2", "x86-64, AVX2 build"), ("x86", "scalar", "x86-64, scalar fallback build"),
             ("arm64", "neon", "AArch64, NEON build"), ("arm64", "scalar", "AArch64, scalar fallback build")]
RUNS = (1, 2, 3)


def path(plat, build, run):
    return f"outputs/scaling_runs/{plat}_run{run}_gram_{build}.txt"


def det_part(text):
    out, skip = [], False
    for l in text.splitlines():
        if l.startswith("TIME"):
            skip = True
        if l == "":
            skip = False
        if not skip and "med=" not in l:
            out.append(l)
    return "\n".join(out)


def time_rows(text):
    # random data: R | C | V ms | K ms | S ms | S/V | S/K | K/V | faster | ns V | ns K | ns S | strided price | unit price
    rows = {}
    a = text.index("\nTIME (median"); b = text.index("\n\n", a) if "\n\n" in text[a:] else len(text)
    for m in re.finditer(r"^\s*(\d+) \|\s*(\d+) \|" + r"\s*([\d.]+) \|" * 6 + r"\s*\w \|" + r"\s*([\d.]+) \|" * 4 + r"\s*(-?[\d.]+)\s*$", text[a:b], re.M):
        g = m.groups(); v = [float(x) for x in g[2:]]
        rows[("random", int(g[0]), int(g[1]))] = dict(V=v[0], K=v[1], S=v[2], nsV=v[6], nsK=v[7], nsS=v[8], pV=v[9], pK=v[10])
    # image data: data | R | C | V ms | K ms | S ms | S/V | S/K | K/V
    a = text.index("\nTIME ON REAL IMAGE DATA")
    for m in re.finditer(r"^\s*([\w-]+) \|\s*(\d+) \|\s*(\d+) \|" + r"\s*([\d.]+) \|" * 5 + r"\s*([\d.]+)\s*$", text[a:], re.M):
        g = m.groups()
        rows[(g[0], int(g[1]), int(g[2]))] = dict(V=float(g[3]), K=float(g[4]), S=float(g[5]))
    return rows


def med3(vals):
    return statistics.median(vals), min(vals), max(vals)


def fmt(vals, nd=2):
    m, lo, hi = med3(vals)
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    have = [p for p in PLATFORMS if all(os.path.exists(path(p[0], p[1], r)) for r in RUNS)]
    missing = [p for p in PLATFORMS if p not in have]
    texts = {(p[0], p[1], r): open(path(p[0], p[1], r)).read() for p in have for r in RUNS}
    out = []
    w = out.append
    w("GRAM MATRIX G = A^T A WITH VIEWS: SUMMARY OF THREE RUNS PER BUILD")
    w("Source: outputs/scaling_runs/{x86,arm64}_run{1,2,3}_gram_{avx2|neon,scalar}.txt (gram_views)")
    w("Builds present: " + "; ".join(p[2] for p in have))
    if missing:
        w("Builds NOT present (left out): " + "; ".join(p[2] for p in missing))
    first = det_part(next(iter(texts.values())))
    same = all(det_part(t) == first for t in texts.values())
    allpass = all("ALL CHECKS PASSED" in t for t in texts.values())
    w(f"Gas tables and check lines identical in all {len(texts)} files: {'yes' if same else 'NO'}")
    w(f"Every file ends with ALL CHECKS PASSED: {'yes' if allpass else 'NO'}")
    w("")
    w("GAS (identical on every build), copied from the first file")
    t0 = next(iter(texts.values()))
    a = t0.index("COMPUTE GAS = total gas"); b = t0.index("\nTIME (median")
    w(t0[a:b])
    a = t0.index("COMPUTE GAS (identical on every build) and trace"); b = t0.index("\nTIME ON REAL IMAGE DATA")
    w(t0[a:b])
    rows = {k: time_rows(t) for k, t in texts.items()}
    keys = list(next(iter(rows.values())).keys())
    for plat, build, label in have:
        w("")
        w(f"TIME, {label}: median [smallest, largest] of three runs")
        w("          data |    R |   C | S/V time            | S/K time            | K/V time            | ns/MAC V            | ns/MAC K            | ns/MAC S            | faster")
        for k in keys:
            rr = [rows[(plat, build, r)][k] for r in RUNS]
            mac = k[1] * k[2] * k[2]
            sv = [x["S"] / x["V"] for x in rr]; sk = [x["S"] / x["K"] for x in rr]; kv = [x["K"] / x["V"] for x in rr]
            nv = [x["V"] * 1e6 / mac for x in rr]; nk = [x["K"] * 1e6 / mac for x in rr]; ns = [x["S"] * 1e6 / mac for x in rr]
            w(f"{k[0]:>14} | {k[1]:4d} | {k[2]:3d} | {fmt(sv):<19} | {fmt(sk):<19} | {fmt(kv):<19} | {fmt(nv):<19} | {fmt(nk):<19} | {fmt(ns):<19} | {'V' if statistics.median(kv) > 1 else 'K'}")
        w(f"TIME-NEUTRAL PRICES PER GROUP OF EIGHT, {label} (the price at which the path has the nanoseconds per gas of Path S; provisional prices: 14 strided, 11 unit)")
        w("    R |   C | strided (Path V)        | unit stride (Path K)")
        for k in keys:
            if k[0] != "random":
                continue
            rr = [rows[(plat, build, r)][k] for r in RUNS]
            w(f"{k[1]:5d} | {k[2]:3d} | {fmt([x['pV'] for x in rr]):<23} | {fmt([x['pK'] for x in rr])}")
    w("")
    w("*** SUMMARY COMPLETE ***" if same and allpass else "*** INSPECT: files differ or a check failed ***")
    open("outputs/gram_summary.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


main()
