#!/usr/bin/env python3
# scaling_summary.py -- Summary of the row-length scaling runs on two processor architectures.
# Reads the twelve files outputs/scaling_runs/{x86,arm64}_run{1,2,3}_{avx2|neon,scalar}.txt (three runs of
# rowmacc_scaling on the vector build and on the scalar fallback build of each machine) and writes
# outputs/scaling_summary.txt. For every row length it reports the median, the smallest and the largest value
# of the three runs. It also checks that the gas tables and the check lines of the twelve files are identical.
# Every number is read from the output files; nothing is typed in.
# Usage (from the repository root): python3 training/scaling_summary.py
import re
import statistics

PLATFORMS = [("x86", "avx2", "x86-64, AVX2 build"), ("x86", "scalar", "x86-64, scalar fallback build"),
             ("arm64", "neon", "AArch64, NEON build"), ("arm64", "scalar", "AArch64, scalar fallback build")]
RUNS = (1, 2, 3)


def read(plat, build, run):
    return open(f"outputs/scaling_runs/{plat}_run{run}_{build}.txt").read()


def gas_part(text):
    a = text.index("TOTAL GAS"); b = text.index("\nTIME")
    checks = "\n".join(l for l in text.splitlines() if re.search(r"bit-exact|harness|CHECK", l))
    return text[a:b] + "\n" + checks


def time_rows(text):
    # rows of the TIME table: I | S ms | B ms | N ms | N16 ms | S/N | B/N | S/N16 | ns/MAC S | B | N | N16
    part = text[text.index("\nTIME"):]
    rows = {}
    for m in re.finditer(r"^\s*(\d+) \|" + r"\s*([\d.]+) \|" * 10 + r"\s*([\d.]+)\s*$", part, re.M):
        v = [float(x) for x in m.groups()[1:]]
        rows[int(m.group(1))] = dict(S=v[0], B=v[1], N=v[2], N16=v[3], S_N=v[4], B_N=v[5], S_N16=v[6],
                                     nsS=v[7], nsB=v[8], nsN=v[9], nsN16=v[10])
    return rows


def compute_rows(text):
    # rows of the COMPUTE GAS table: I | harness 8 | harness 16 | compute S | B | N | N16 | S/N | B/N | S/N16 | gas/MAC N | N16
    a = text.index("COMPUTE GAS"); b = text.index("\nTIME")
    rows = []
    for m in re.finditer(r"^\s*(\d+) \|\s*\d+ \|\s*\d+ \|\s*(\d+) \|\s*(\d+) \|\s*(\d+) \|\s*(\d+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.]+)\s*$", text[a:b], re.M):
        g = m.groups()
        rows.append((int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), g[5], g[6], g[7], g[8], g[9]))
    return rows


def mmm(values):
    return statistics.median(values), min(values), max(values)


out = []
texts = {(p, b, r): read(p, b, r) for p, b, _ in PLATFORMS for r in RUNS}
for key, t in texts.items():
    assert "ALL CHECKS PASSED" in t and "FAILED" not in t, key

ref = gas_part(texts[("x86", "avx2", 1)])
same = all(gas_part(t) == ref for t in texts.values())
out.append("ROW-LENGTH SCALING, SUMMARY OF THREE RUNS PER BUILD ON TWO ARCHITECTURES")
out.append("source: outputs/scaling_runs/*.txt (12 files), written by experiments/rowmacc_scaling.cpp")
out.append("")
out.append(f"[CHECK] gas tables and check lines identical in the {len(texts)} files: {'PASSED' if same else 'FAILED'}")
out.append("")
out.append("COMPUTE GAS (total gas minus harness gas), identical on every build and machine")
out.append("     I |    compute S |    compute B |    compute N |  S/N |  B/N | N gas/MAC")
crow = compute_rows(texts[("x86", "avx2", 1)])
for (I, cS, cB, cN, cN16, sn, bn, sn16, gm, gm16) in crow:
    out.append(f"{I:6d} | {cS:12d} | {cB:12d} | {cN:12d} | {sn:>4} | {bn:>4} | {gm:>9}")
out.append("")

LEN = [r[0] for r in crow]
tabs = {(p, b): [time_rows(texts[(p, b, r)]) for r in RUNS] for p, b, _ in PLATFORMS}
for (p, b), runs in tabs.items():
    for rows in runs:
        assert sorted(rows) == sorted(LEN), (p, b)

out.append("TIME RATIO, packed scalar program (Path S) over the row-level program (Path N): median [min, max] of three runs")
out.append("     I |        x86-64 AVX2 |      x86-64 scalar |       AArch64 NEON |     AArch64 scalar")
for I in LEN:
    cells = []
    for p, b, _ in PLATFORMS:
        md, lo, hi = mmm([rows[I]["S_N"] for rows in tabs[(p, b)]])
        cells.append(f"{md:5.2f} [{lo:4.2f}, {hi:4.2f}]")
    out.append(f"{I:6d} | " + " | ".join(cells))
out.append("")

out.append("TIME RATIO, Path S over the sixteen-lane row-level program (Path N16): median [min, max] of three runs")
out.append("     I |        x86-64 AVX2 |      x86-64 scalar |       AArch64 NEON |     AArch64 scalar")
for I in LEN:
    cells = []
    for p, b, _ in PLATFORMS:
        md, lo, hi = mmm([rows[I]["S_N16"] for rows in tabs[(p, b)]])
        cells.append(f"{md:5.2f} [{lo:4.2f}, {hi:4.2f}]")
    out.append(f"{I:6d} | " + " | ".join(cells))
out.append("")

out.append("NANOSECONDS PER MULTIPLY-ACCUMULATE of Path N: median [min, max] of three runs")
out.append("     I |        x86-64 AVX2 |      x86-64 scalar |       AArch64 NEON |     AArch64 scalar")
for I in LEN:
    cells = []
    for p, b, _ in PLATFORMS:
        md, lo, hi = mmm([rows[I]["nsN"] for rows in tabs[(p, b)]])
        cells.append(f"{md:5.2f} [{lo:4.2f}, {hi:4.2f}]")
    out.append(f"{I:6d} | " + " | ".join(cells))
out.append("")

out.append("VECTOR BUILD OVER SCALAR BUILD, time of Path N on the scalar build divided by the time on the vector build")
out.append("(run r of the scalar build paired with run r of the vector build): median [min, max] of three runs")
out.append("     I |        x86-64 AVX2 |       AArch64 NEON")
for I in LEN:
    cells = []
    for p, v in (("x86", "avx2"), ("arm64", "neon")):
        ratios = [tabs[(p, "scalar")][k][I]["N"] / tabs[(p, v)][k][I]["N"] for k in range(len(RUNS))]
        md, lo, hi = mmm(ratios)
        cells.append(f"{md:5.2f} [{lo:4.2f}, {hi:4.2f}]")
    out.append(f"{I:6d} | " + " | ".join(cells))
out.append("")

out.append("LAYER-1 SHAPE (128 neurons x 784 inputs), the figures quoted in the paper")
for p, b, name in PLATFORMS:
    sn = mmm([rows[784]["S_N"] for rows in tabs[(p, b)]])
    sn16 = mmm([rows[784]["S_N16"] for rows in tabs[(p, b)]])
    ns = mmm([rows[784]["nsN"] for rows in tabs[(p, b)]])
    out.append(f"  {name:31s}: S/N = {sn[0]:.2f} [{sn[1]:.2f}, {sn[2]:.2f}] | S/N16 = {sn16[0]:.2f} [{sn16[1]:.2f}, {sn16[2]:.2f}] | ns per MAC of N = {ns[0]:.2f} [{ns[1]:.2f}, {ns[2]:.2f}]")
c784 = [r for r in crow if r[0] == 784][0]
out.append(f"  compute gas, every build and machine: S = {c784[1]}, N = {c784[3]}, S/N = {c784[5]}")
out.append("")

# Median times and time-neutral prices at the layer-1 shape. The derivation is the one of training/price_neutral.py,
# applied to each of the three runs: pure computation = total gas - memory expansion - CALLDATACOPY copy cost, and the
# time-neutral price p* solves pure_S / (other + p* x groups) = time_S / time_X for X = N (XMACC_N) and X = B (XMACC).
def memgas(nbytes):
    w = (nbytes + 31) // 32
    return 3 * w + (w * w) // 512

t1 = texts[("x86", "avx2", 1)]
gas = {n: int(re.search(r"---- I = 784 ----.*?\[" + n + r"\] bit-exact[^\n]*?gas (\d+)", t1, re.S).group(1)) for n in ("S", "B", "N")}
NEU, G = 128, 784 // 8
groups = NEU * G
MEND = G * 32 + NEU * G * 32 + NEU * 32
copy = (3 + 3 * G) + (3 + 3 * NEU * G)
pureS = gas["S"] - memgas(MEND + 32) - copy
pureB = gas["B"] - memgas(MEND) - copy
pureN = gas["N"] - memgas(MEND) - copy
otherN = pureN - 11 * groups
otherB = pureB - 11 * groups
out.append("LAYER-1 SHAPE, MEDIAN TIMES AND TIME-NEUTRAL PRICES (derivation of training/price_neutral.py, applied to each run)")
out.append(f"  pure computation gas (total - memory expansion - copy cost): S {pureS} | B {pureB} | N {pureN} | groups {groups}")
out.append("  median [min, max] of three runs")
for p, b, name in PLATFORMS:
    runs = tabs[(p, b)]
    ms = {k: mmm([rows[784][k] for rows in runs]) for k in ("S", "B", "N", "N16")}
    pN = mmm([(pureS / (rows[784]["S"] / rows[784]["N"]) - otherN) / groups for rows in runs])
    pB = mmm([(pureS / (rows[784]["S"] / rows[784]["B"]) - otherB) / groups for rows in runs])
    bn = mmm([rows[784]["B_N"] for rows in runs])
    out.append(f"  {name}")
    out.append("    time per execution in ms: " + " | ".join(f"{k} {ms[k][0]:.3f} [{ms[k][1]:.3f}, {ms[k][2]:.3f}]" for k in ("S", "B", "N", "N16")))
    out.append(f"    time ratio B/N = {bn[0]:.2f} [{bn[1]:.2f}, {bn[2]:.2f}]")
    out.append(f"    time-neutral price of XMACC_N = {pN[0]:.2f} [{pN[1]:.2f}, {pN[2]:.2f}] gas per group (provisional 11)")
    out.append(f"    time-neutral price of XMACC   = {pB[0]:.2f} [{pB[1]:.2f}, {pB[2]:.2f}] gas per instruction (provisional 11)")
out.append("")
out.append("*** ALL CHECKS PASSED ***" if same else "*** A CHECK FAILED -> the numbers are not valid ***")

open("outputs/scaling_summary.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
raise SystemExit(0 if same else 1)
