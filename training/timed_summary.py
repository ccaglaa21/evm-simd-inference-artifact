#!/usr/bin/env python3
# timed_summary.py -- One summary of the two timed experiments over the four builds.
# Reads outputs/scaling_runs/<machine>_run<1,2,3>_gram_<build>.txt (gram_views) and
# outputs/scaling_runs/<machine>_run<1,2,3>_<build>.txt (rowmacc_scaling) and writes outputs/timed_summary.txt.
# Gas is read from run 1 of the first build after a test that all twelve files carry the same gas table.
# A time is the median of the three runs of the build (the COMPUTE time of the program: analysis and the
# copy-and-return program are already subtracted by the program). Ratios are formed from these medians.
# Every number is read from the output files; nothing is typed in.
# Usage (from the repository root): python3 training/timed_summary.py
import statistics

D = "outputs/scaling_runs"
BUILDS = [("x86", "avx2", "x86-64 AVX2"), ("x86", "scalar", "x86-64 scalar"), ("arm64", "neon", "AArch64 NEON"), ("arm64", "scalar", "AArch64 scalar")]
PROG = [("GRAM MATRIX (gram_views): G = A^T A, R rows, C features", "gram_", "COMPUTE GAS = ", 3, ["V", "K", "MV", "MK", "S"], "compute S"),
        ("DENSE LAYER (rowmacc_scaling): 128 neurons x I inputs", "", "COMPUTE GAS = ", 1, ["S", "B", "N", "N16", "M"], "compute S")]


def tab(path, head):
    L = open(path).read().split("\n"); i = next(n for n, x in enumerate(L) if x.startswith(head)); hdr = [h.strip() for h in L[i + 1].split(" | ")]; rows = []
    for x in L[i + 2:]:
        f = [y.strip() for y in x.split(" | ")]
        if len(f) < len(hdr): break
        rows.append(dict(zip(hdr, f)))
    return rows


def main():
    out = []; w = out.append
    w("SUMMARY OF THE TIMED EXPERIMENTS ON FOUR BUILDS (times: medians of three runs, ms of computation)")
    for title, mid, ghead, nk, paths, _ in PROG:
        f = lambda p, b, r: f"{D}/{p}_run{r}_{mid}{b}.txt"
        gas = [tab(f(p, b, r), ghead) for p, b, _ in BUILDS for r in (1, 2, 3)]; same = all(g == gas[0] for g in gas); G = gas[0]; keys = list(G[0].keys())[:nk]
        w(""); w("=" * 20 + " " + title + " " + "=" * 20)
        w("Compute gas, the same table in all twelve files: %s" % ("yes" if same else "NO"))
        others = [p for p in paths if p != "S"]
        w(" | ".join("%13s" % k for k in keys) + " | " + " | ".join("%11s" % ("gas " + p) for p in paths) + " | " + " | ".join("%6s" % ("S/" + p) for p in others))
        gr = {p: [] for p in others}
        for row in G:
            g = {p: int(row["compute " + p]) for p in paths}
            for p in others: gr[p].append(g["S"] / g[p])
            w(" | ".join("%13s" % row[k] for k in keys) + " | " + " | ".join("%11d" % g[p] for p in paths) + " | " + " | ".join("%6.2f" % (g["S"] / g[p]) for p in others))
        w("Range of the gas ratio over the %d rows: " % len(G) + ", ".join("S/%s %.2f to %.2f" % (p, min(gr[p]), max(gr[p])) for p in others))
        for p, b, label in BUILDS:
            T = [tab(f(p, b, r), "TIME, COMPUTE") for r in (1, 2, 3)]; tr = {q: [] for q in others}
            w(""); w("Time of the computation, %s" % label)
            w(" | ".join("%13s" % k for k in keys) + " | " + " | ".join("%10s" % ("ms " + q) for q in paths) + " | " + " | ".join("%6s" % ("S/" + q) for q in others) + " | spread")
            for i, row in enumerate(T[0]):
                runs = {q: [float(t[i][q]) for t in T] for q in paths}; m = {q: statistics.median(runs[q]) for q in paths}
                sp = max((max(v) - min(v)) / statistics.median(v) for v in runs.values() if statistics.median(v) > 0)
                for q in others: tr[q].append(m["S"] / m[q] if m[q] > 0 else float("nan"))
                w(" | ".join("%13s" % row[k] for k in keys) + " | " + " | ".join("%10.4f" % m[q] for q in paths) + " | " + " | ".join("%6.2f" % (m["S"] / m[q] if m[q] > 0 else float("nan")) for q in others) + " | %5.1f%%" % (100 * sp))
            w("Range of the time ratio, %s: " % label + ", ".join("S/%s %.2f to %.2f" % (q, min(tr[q]), max(tr[q])) for q in others))
    w(""); w("spread = largest (max - min) / median of the three runs among the paths of the row")
    w("*** SUMMARY COMPLETE ***")
    text = "\n".join(out) + "\n"; open("outputs/timed_summary.txt", "w").write(text); print(text, end="")


main()
