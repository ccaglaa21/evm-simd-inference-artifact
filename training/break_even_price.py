#!/usr/bin/env python3
# break_even_price.py -- break-even price of the row-level instructions XMACC_N and XMACC_N16.
#
# The row-level instructions are priced provisionally (11 gas per 8-element group, 22 gas per 16-element
# group, carried over from the fixed-width XMACC). This script asks the converse question of
# price_neutral.py: how high could that price rise before the row-level path loses its gas advantage over
# the packed-multiplication scalar baseline (Path S, standard instructions only)? For a comparison in which
# the scalar side costs gas_S and the row-level side costs other_N + p * groups, the break-even price is
#     p* = (gas_S - other_N) / groups,
# where other_N is the gas of the row-level program outside its XMACC_N instructions and groups is the number
# of executed groups. Below p* the row-level path is cheaper. Three comparisons are evaluated, all from
# measured outputs: the 128 x 784 kernel of rowmacc_wallclock, the full MNIST network with unrolled code,
# and the deployable looped contracts. The number of groups of the full network is derived from the layer
# sizes in deep_meta.txt (the program reads the file in the current directory, like the other Python
# stages). Pure Python, no third-party package. Reads $OUT (default /out).
import os, re, sys

OUT = os.environ.get("OUT", "/out")

def read(name):
    return open(os.path.join(OUT, name)).read()

def num(pattern, text, path):
    m = re.search(pattern, text)
    if not m:
        print(f"pattern not found in {path}: {pattern}"); sys.exit(1)
    return int(m.group(1).replace(",", ""))

def memgas(nbytes):
    w = (nbytes + 31) // 32
    return 3 * w + (w * w) // 512

for name in ("rowmacc_wallclock.txt", "mnist_rowmacc.txt", "mnist_loops.txt", "mnist_codecopy16.txt", "mnist_swar_baseline.txt"):
    if "ALL CHECKS PASSED" not in read(name):
        print("checks of", name, "did not pass; no derivation"); sys.exit(1)

# layer sizes -> executed groups of the full network (one group = one XMACC_N over 8 elements)
meta = [int(t) for t in open("deep_meta.txt").read().split()]
H1, H2 = meta[3], meta[4]
I, O = 784, 10
groups8 = H1 * (I // 8) + H2 * (H1 // 8) + O * (H2 // 8)
groups16 = H1 * (I // 16) + H2 * (H1 // 16) + O * (H2 // 16)
print(f"network {I}->{H1}->{H2}->{O}: {groups8} groups of 8 elements (XMACC_N), {groups16} groups of 16 elements (XMACC_N16)")

rows = []   # (label, gas of the scalar side, other_N, groups, provisional price, unit)

# 1. kernel of rowmacc_wallclock (same decomposition as price_neutral.py)
t = read("rowmacc_wallclock.txt")
neu, inp = (int(x) for x in re.search(r"(\d+) neurons x (\d+) inputs", t).groups())
Sgas = num(r"\[S\] bit-exact.*?gas (\d+)", t, "rowmacc_wallclock.txt")
Ngas = num(r"\[N\] bit-exact.*?gas (\d+)", t, "rowmacc_wallclock.txt")
N16gas = num(r"\[N16\] bit-exact.*?gas (\d+)", t, "rowmacc_wallclock.txt")
G = inp // 8; kg8 = neu * G; G16 = inp // 16; kg16 = neu * G16
mem_BN = memgas(G * 32 + neu * G * 32 + neu * 32); mem_S = memgas(G * 32 + neu * G * 32 + neu * 32 + 32)
copy = (3 + 3 * G) + (3 + 3 * neu * G)
mem16 = memgas(G16 * 32 + neu * G16 * 32 + neu * 32); copy16 = (3 + 3 * G16) + (3 + 3 * neu * G16)
compS = Sgas - mem_S - copy; compN = Ngas - mem_BN - copy; compN16 = N16gas - mem16 - copy16
P8, P16 = 11, 22
rows.append((f"kernel {neu} x {inp}, unrolled, pure computation", compS, compN - P8 * kg8, kg8, P8, "8-element group"))
rows.append((f"kernel {neu} x {inp}, unrolled, pure computation, 16 lanes", compS, compN16 - P16 * kg16, kg16, P16, "16-element group"))

# 2. full network, unrolled single-image pure computation (mnist_rowmacc.txt)
t = read("mnist_rowmacc.txt")
p8 = num(r"XMACC_N priced at (\d+) gas per group", t, "mnist_rowmacc.txt")
N_unr = num(r"N-unrolled pure compute \(single image\): (\d+)", t, "mnist_rowmacc.txt")
S_unr = num(r"Path S ([\d,]+)", t, "mnist_rowmacc.txt")
N_dep = num(r"N-deployable marginal per image \(memory-stripped\): (\d+)", t, "mnist_rowmacc.txt")
assert S_unr == num(r"S  cold\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+(\d+)", read("mnist_swar_baseline.txt"), "mnist_swar_baseline.txt")
rows.append(("full network, unrolled, single-image pure computation (Table 7 row e)", S_unr, N_unr - p8 * groups8, groups8, p8, "8-element group"))

# 3. deployable contracts, marginal cost per image (mnist_loops.txt and mnist_codecopy16.txt)
t = read("mnist_loops.txt")
S2 = num(r"S2 \(packed scalar\) (\d+)", t, "mnist_loops.txt")
rows.append(("deployable, loop per neuron, marginal cost per image (Table 7 row f)", S2, N_dep - p8 * groups8, groups8, p8, "8-element group"))
t = read("mnist_codecopy16.txt")
p16 = num(r"XMACC_N16 priced at (\d+) gas per 16-element group", t, "mnist_codecopy16.txt")
N16_dep = num(r"marginal per image (\d+) \| code", t, "mnist_codecopy16.txt")
rows.append(("deployable, 16 lanes, code-as-storage, marginal cost per image", S2, N16_dep - p16 * groups16, groups16, p16, "16-element group"))

print()
print("break-even price p* = (gas of the scalar side - gas of the row-level side outside XMACC_N) / groups")
print(f"{'comparison':<72} {'scalar':>9} {'other_N':>8} {'groups':>7} {'p*':>7} {'prov.':>5} {'p*/prov.':>8}")
for label, s, other, g, p, unit in rows:
    pstar = (s - other) / g
    print(f"{label:<72} {s:>9} {other:>8} {g:>7} {pstar:>7.2f} {p:>5} {pstar / p:>8.2f}   gas per {unit}")

print()
tn = re.findall(r"time-neutral price of XMACC_N = ([\d.]+) gas per group", read("price_neutral.txt")) if os.path.exists(os.path.join(OUT, "price_neutral.txt")) else []
print("Reading: the row-level path stays cheaper than the packed scalar baseline up to p* gas per group. The provisional")
print("price could be raised by the factor in the last column before the two paths cost the same gas.")
if tn:
    print(f"The time-neutral prices of XMACC_N in price_neutral.txt, {' and '.join(tn)} gas per group on the two builds, lie below")
    print(f"the provisional price, so the admissible interval for a price that neither undercharges the instruction in time")
    print(f"nor removes its gas advantage runs from {max(float(x) for x in tn):.2f} (slowest build) to {min((r[1] - r[2]) / r[3] for r in rows if r[5].startswith('8')):.2f} gas per group (smallest p* of the 8-lane comparisons).")
