#!/usr/bin/env python3
# precompile_bound.py -- analytic cost bound of a hypothetical dot-product precompile against the row-level
# instruction, from the gas schedule (no execution).
#
# The precompile alternative of the paper is not implemented. This script bounds its gas from the schedule:
# one CALL per neuron (the precompile address is always warm, 100 gas, EIP-2929), the seven stack operands of
# the CALL (7 x PUSH, 3 gas each), the precompile's own price modelled like the identity precompile
# (base 15 gas plus 3 gas per 32-byte word of input, EIP-150 schedule), and an unknown per-element price p
# per multiply-accumulate. The input of one neuron is its input row and its weight row (two words per group
# of eight 16-bit elements), passed through memory; CALL charges no copy cost for its argument region beyond
# memory expansion, which both designs pay alike and which is therefore left out on both sides. The return
# value (one 32-bit score) is written to memory by the CALL. Overheads inside the precompile (argument
# decoding, result encoding) are not charged, so the bound favours the precompile.
# The row-level side is the part of Path N that the precompile would replace: the XMACC_N instructions at the
# price read from mnist_rowmacc.txt (one per neuron, 11 gas per group of eight) and their three stack operands
# (3 x PUSH). Everything else of a neuron (bias word, lane sum, activation, packing) is the same work in both
# designs and is left out on both sides, as is the weight loader. Reads $OUT/mnist_rowmacc.txt and
# deep_meta.txt; the network's XMACC_N count and the ratio against Path N's whole pure computation are
# printed for reference.
import os, re, sys

OUT = os.environ.get("OUT", "/out")
t = open(os.path.join(OUT, "mnist_rowmacc.txt")).read()
if "ALL CHECKS PASSED" not in t:
    print("mnist_rowmacc checks did not pass; no derivation"); sys.exit(1)
m = re.search(r"N-unrolled pure compute \(single image\): (\d+)\s+\(loader (\d+),", t)
N_unr, loader = int(m.group(1)), int(m.group(2))
price = int(re.search(r"XMACC_N priced at (\d+) gas per group", t).group(1))

meta = [int(x) for x in open("deep_meta.txt").read().split()]
H1, H2 = meta[3], meta[4]
I, O = 784, 10
layers = [(I, H1), (H1, H2), (H2, O)]
neurons = sum(n for _, n in layers)
macs = sum(i * n for i, n in layers)
words = sum(n * 2 * (i // 8) for i, n in layers)      # input row + weight row, one word per group of 8

groups = sum(n * (i // 8) for i, n in layers)
CALL_WARM, PUSH, BASE, PER_WORD = 100, 3, 15, 3
rowlevel = price * groups + neurons * 3 * PUSH
fixed = neurons * (CALL_WARM + 7 * PUSH + BASE) + PER_WORD * words
print(f"network {I}->{H1}->{H2}->{O}: {neurons} neurons (one CALL each), {macs} multiply-accumulates, {groups} groups of 8, {words} input words")
print(f"row-level side (the part a precompile would replace): {neurons} XMACC_N at {price} gas per group = {price * groups} + 3 PUSH per neuron {neurons * 3 * PUSH} = {rowlevel} gas")
print(f"  (for reference, the whole pure computation of Path N per image is {N_unr} gas, {N_unr - loader} without the loader)")
print(f"precompile side, fixed part: {neurons} x (CALL {CALL_WARM} + 7 PUSH {7 * PUSH} + base {BASE}) + {PER_WORD} x {words} words = {fixed} gas")
print()
print(f"{'per-element price p (gas per MAC)':<36} {'precompile total':>16} {'ratio to row-level':>19}")
for p in (0.0, 0.25, 0.5, 1.0, price / 8, 2.0):
    total = fixed + p * macs
    print(f"{p:<36.3f} {total:>16.0f} {total / rowlevel:>19.2f}")
pstar = (rowlevel - fixed) / macs
print()
print(f"break-even per-element price of the precompile: p* = ({rowlevel} - {fixed}) / {macs} = {pstar:.3f} gas per multiply-accumulate")
print(f"(the row-level instruction charges {price} gas per group of 8, {price / 8:.3f} gas per multiply-accumulate; at that price the precompile costs {(fixed + price / 8 * macs) / rowlevel:.2f} times the row-level side)")
print("Reading: the precompile is cheaper only when its per-element price is below p*; the fixed call and")
print("input-word charges of one call per neuron are what the row-level instruction avoids.")
