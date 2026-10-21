#!/usr/bin/env python3
# price_neutral.py -- time-neutral price of the row-level instruction XMACC_N, derived from the
# wall-clock measurement of rowmacc_wallclock (AVX2 build and scalar-fallback build).
#
# rowmacc_wallclock times three straight-line programs that compute the same 128 x 784 layer:
# S (packed-multiplication scalar, standard instructions only), B (fixed-width XMACC) and N
# (row-level XMACC_N, one instruction per neuron). The time-neutral price of XMACC_N is the price
# p* at which the gas ratio of the pure computation of S and N equals their measured time ratio:
#     compute_S / (other_N + p* * groups) = time_S / time_N
# where compute_X = measured gas of X - memory expansion - CALLDATACOPY copy cost (the parts that
# are the same work in every program and do not depend on the instruction price), groups is the
# number of XMACC_N groups executed (128 neurons x 98 groups) and other_N is the gas of N's
# instructions other than XMACC_N. The same derivation is applied to XMACC (program B, one XMACC per
# group). The memory map of rowmacc_wallclock is recomputed here from the printed shape.
# Pure Python, no third-party package. Reads $OUT/rowmacc_wallclock.txt and
# $OUT/fallback/rowmacc_wallclock.txt (OUT defaults to /out).
import os, re, sys

OUT = os.environ.get("OUT", "/out")
PRICE = 11   # provisional price of XMACC_N per group and of XMACC per instruction

def parse(path):
    t = open(path).read()
    m = re.search(r"(\d+) neurons x (\d+) inputs", t)
    neu, inp = int(m.group(1)), int(m.group(2))
    d = {"neu": neu, "inp": inp}
    for n in "SBN":
        d[n + "gas"] = int(re.search(r"\[" + n + r"\] bit-exact.*?gas (\d+)", t).group(1))
        d[n + "t"] = float(re.search(r"\[" + n + r"\] med=([\d.]+) s", t).group(1))
    if "ALL CHECKS PASSED" not in t:
        print("checks of", path, "did not pass; no derivation"); sys.exit(1)
    return d

def memgas(nbytes):
    w = (nbytes + 31) // 32
    return 3 * w + (w * w) // 512

a = parse(os.path.join(OUT, "rowmacc_wallclock.txt"))
f = parse(os.path.join(OUT, "fallback", "rowmacc_wallclock.txt"))
assert (a["neu"], a["inp"]) == (f["neu"], f["inp"]) and a["Sgas"] == f["Sgas"] and a["Ngas"] == f["Ngas"] and a["Bgas"] == f["Bgas"]
NEU, I = a["neu"], a["inp"]
G = I // 8
groups = NEU * G
# memory map of rowmacc_wallclock: inputs (G words), weights (NEU*G words), outputs (NEU words);
# program S uses one more scratch word (its correction word)
MEND = G * 32 + NEU * G * 32 + NEU * 32
mem_BN = memgas(MEND)
mem_S = memgas(MEND + 32)
copy = (3 + 3 * G) + (3 + 3 * NEU * G)          # two CALLDATACOPY copy costs (3 gas per word each, plus 3 static)
compS = a["Sgas"] - mem_S - copy
compB = a["Bgas"] - mem_BN - copy
compN = a["Ngas"] - mem_BN - copy
otherN = compN - PRICE * groups
otherB = compB - PRICE * groups

print(f"rowmacc_wallclock: {NEU} neurons x {I} inputs, {groups} groups of 8 (multiply-accumulates: {NEU * I})")
print(f"memory expansion gas {mem_BN} (S: {mem_S}), CALLDATACOPY copy cost {copy}")
print(f"pure computation gas: S {compS} | B {compB} (XMACC {PRICE} x {groups} = {PRICE * groups}, other {otherB}) | N {compN} (XMACC_N {PRICE} x {groups} = {PRICE * groups}, other {otherN})")
print(f"gas ratios of the pure computation at the provisional price {PRICE}: S/N {compS / compN:.2f} | S/B {compS / compB:.2f}")
print()
for name, d in (("AVX2 build", a), ("scalar-fallback build", f)):
    rN = d["St"] / d["Nt"]
    pN = (compS / rN - otherN) / groups
    rB = d["St"] / d["Bt"]
    pB = (compS / rB - otherB) / groups
    print(f"{name}: median times S {d['St'] * 1e3:.3f} ms, B {d['Bt'] * 1e3:.3f} ms, N {d['Nt'] * 1e3:.3f} ms")
    print(f"  time ratio S/N = {rN:.2f} -> time-neutral price of XMACC_N = {pN:.2f} gas per group (provisional {PRICE})")
    if pB > 0:
        print(f"  time ratio S/B = {rB:.2f} -> time-neutral price of XMACC = {pB:.2f} gas per instruction (provisional {PRICE})")
    else:
        print(f"  time ratio S/B = {rB:.2f} -> no non-negative time-neutral price of XMACC exists: the {otherB / groups:.2f} gas of loads and pushes per group already exceed the time-neutral budget of {compS / rB / groups:.2f} gas per group")
print()
print("Reading: a time-neutral price below the provisional price means that the provisional price overcharges the instruction")
print("relative to its measured execution time, so the gas gains reported at the provisional price understate the time advantage.")
