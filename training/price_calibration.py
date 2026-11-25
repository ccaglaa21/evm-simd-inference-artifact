#!/usr/bin/env python3
# price_calibration.py -- Gas schedule of the added instructions from the measured times of four builds.
# Reads outputs/price_calibration_runs/{x86,arm64}_run{1,2,3}_price_{avx2|neon,scalar}.txt (price_bench, three
# runs on each of the four builds) and writes outputs/price_calibration.txt. The interpreter carries the
# constants that this script prints.
#
# RULE. On a build, the reference rate r is the time per gas of the scalar multiply-accumulate step
# (PUSH, MLOAD, PUSH, MLOAD, MUL, ADD, 20 gas), the smallest of its three program lengths. An instruction
# that takes time t on that build must cost at least t / r gas (its covering gas). The time of an instruction
# includes the pushes of its operands. Times are medians of the three runs.
#   Fixed work.      price = smallest whole number that is not below the covering gas of any build (for XMACC_S
#                    of both layouts, the transposed view and the hostile view).
#   Size-dependent.  XMACC_N, XMACC_N16, XMACC_V:  gas = CALL + GROUP * groups of eight elements
#                    XMATMUL:                      gas = CALL + ENTRY * entries + GROUP * entries * groups
#                    XGATHER:                      gas = CALL + INDEX * indices
#     For each instruction, among all whole-number constants for which the gas of every measured unit-stride
#     case covers its time on every build, the script takes the ones with the smallest mean ratio of charged
#     gas to covering gas. The charge per strided element of XMACC_V and per packed element of XMATMUL is then
#     the smallest whole number that also covers every strided case of the instruction (transposed views and
#     the hostile view).
# Every number is read from the output files; nothing is typed in.
# Usage (from the repository root): python3 training/price_calibration.py
import math
import os
import statistics
import sys

BUILDS = [("x86", "avx2", "x86-64, AVX2 build"), ("x86", "scalar", "x86-64, scalar fallback build"),
          ("arm64", "neon", "AArch64, NEON build"), ("arm64", "scalar", "AArch64, scalar fallback build")]
RUNS = (1, 2, 3)
FAMILY = ("XMACC_N", "XMACC_N16", "XMACC_V", "XMATMUL")
DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/price_calibration_runs"


def table(path):
    t = open(path).read(); a = t.index("TIME OF ONE REPETITION"); b = t.index("\n\n", a); rows = {}
    for ln in t[a:b].split("\n")[2:]:
        f = [x.strip() for x in ln.split(" | ")]
        rows[(f[0], f[1], int(f[2]))] = dict(groups=int(f[4]), entries=int(f[5]), strided=int(f[6]), ns=float(f[12]))
    return rows


def main():
    out = []; w = out.append
    builds = [b for b in BUILDS if all(os.path.exists(f"{DIR}/{b[0]}_run{r}_price_{b[1]}.txt") for r in RUNS)]
    w("GAS SCHEDULE OF THE ADDED INSTRUCTIONS FROM THE MEASURED TIMES OF %d BUILDS" % len(builds))
    w(f"Source: {DIR}/<machine>_run<1,2,3>_price_<build>.txt (price_bench), medians of three runs")
    T = {}; C = None; same = True
    for p, b, _ in builds:
        runs = [table(f"{DIR}/{p}_run{r}_price_{b}.txt") for r in RUNS]
        if C is None: C = runs[0]
        same = same and all(list(run.keys()) == list(C.keys()) for run in runs)
        T[(p, b)] = {k: statistics.median(run[k]["ns"] for run in runs) for k in C}
    w("Cases: %d, the same list in all %d files: %s" % (len(C), len(builds) * len(RUNS), "yes" if same else "NO"))
    r = {}
    w("")
    w("REFERENCE RATE r (ns per gas): the scalar multiply-accumulate step at three program lengths, and once more at the end of the run")
    w("%32s | %9s | %9s | %9s | %9s | %10s | end / start (4096 steps)" % ("build", "512 steps", "4096", "32768", "r (min)", "end of run"))
    for p, b, label in builds:
        t = T[(p, b)]; st = [t[k] / 20 for k in C if k[0] == "REF_STEP" and "end of run" not in k[1]]; en = [t[k] / 20 for k in C if k[0] == "REF_STEP" and "end of run" in k[1]][0]
        r[(p, b)] = min(st)
        w("%32s | %9.3f | %9.3f | %9.3f | %9.3f | %10.3f | %.3f" % (label, st[0], st[1], st[2], r[(p, b)], en, en / st[1]))
    w("STANDARD INSTRUCTIONS FOR CONTEXT (ns per gas at their own gas, pushes of the operands included)")
    for k in [k for k in C if k[0] in ("REF_OP", "REF_WALK")]:
        gas = {"DUP, ADD": 6, "DUP, MUL": 8, "DUP, DUP, DUP, MULMOD, POP": 19, "PUSH, PUSH, KECCAK256 of one word, POP": 44}.get(k[1], 8)
        w("%52s |" % k[1] + "".join(" %s %s %6.3f |" % (p, b, T[(p, b)][k] / gas) for p, b, _ in builds))
    need = lambda k: max(T[(p, b)][k] / r[(p, b)] for p, b, _ in builds)        # gas that covers case k on every build
    w("")
    w("FIXED-WORK INSTRUCTIONS: time of one repetition (ns) and covering gas on each build, and the price")
    fixed = {}
    for k in C:
        if k[1].startswith("fixed price"): fixed[k[0]] = max(fixed.get(k[0], 0), math.ceil(need(k)))
    for k in C:
        if not k[1].startswith("fixed price"): continue
        w("%10s %-26s |" % (k[0], k[1]) + "".join(" %s %s %6.2f ns %6.2f gas |" % (p, b, T[(p, b)][k], T[(p, b)][k] / r[(p, b)]) for p, b, _ in builds) + " price %3d" % fixed[k[0]])
    w("")
    w("SIZE-DEPENDENT INSTRUCTIONS")
    sched = {}; charged = {}
    for i in FAMILY:
        unit = [k for k in C if k[0] == i and C[k]["strided"] == 0]; best = None
        for G in range(1, 65):
            for E in (range(0, 65) if i == "XMATMUL" else [0]):
                call = max(0, math.ceil(max(need(k) - G * C[k]["groups"] - E * C[k]["entries"] for k in unit)))
                obj = statistics.mean((call + G * C[k]["groups"] + E * C[k]["entries"]) / need(k) for k in unit)
                if best is None or obj < best[0] - 1e-12: best = (obj, call, E, G)
        obj, call, E, G = best
        model = lambda k, call=call, E=E, G=G: call + G * C[k]["groups"] + E * C[k]["entries"]
        st = [k for k in C if k[0] == i and C[k]["strided"] > 0]
        S = max(0, math.ceil(max((need(k) - model(k)) / C[k]["strided"] for k in st))) if st else 0
        sched[i] = (call, E, G, S)
        for k in unit + st: charged[k] = model(k) + S * C[k]["strided"]
        if i == "XMACC_N16": w("%-9s | CALL %3d | per word of sixteen elements %2d (two groups of eight at %d) | unit-stride cases %2d, mean ratio of charged gas to covering gas %.3f" % (i, call, 2 * G, G, len(unit), obj))
        elif i == "XMATMUL": w("%-9s | CALL %3d | ENTRY %d | GROUP %2d | per packed element %d | unit-stride cases %2d, mean ratio of charged gas to covering gas %.3f" % (i, call, E, G, S, len(unit), obj))
        elif i == "XMACC_V": w("%-9s | CALL %3d | GROUP %2d | per strided element %d | unit-stride cases %2d, mean ratio of charged gas to covering gas %.3f" % (i, call, G, S, len(unit), obj))
        else: w("%-9s | CALL %3d | GROUP %2d | unit-stride cases %2d, mean ratio of charged gas to covering gas %.3f" % (i, call, G, len(unit), obj))
        for k in st: w("            covering gas per strided element above the unit-stride part: %-52s n=%-5d %7.3f" % (k[1], k[2], (need(k) - model(k)) / C[k]["strided"]))
    gk = [k for k in C if k[0] == "XGATHER"]; bestg = None
    for gi in range(1, 33):
        gc = max(0, math.ceil(max(need(k) - gi * k[2] for k in gk))); o = statistics.mean((gc + gi * k[2]) / need(k) for k in gk)
        if bestg is None or o < bestg[0] - 1e-12: bestg = (o, gc, gi)
    _, GC, GI = bestg
    for k in gk: charged[k] = GC + GI * k[2]
    for k in C:
        if k[1].startswith("fixed price"): charged[k] = fixed[k[0]]
    w("%-9s | CALL %3d | per index %d" % ("XGATHER", GC, GI))
    w("")
    w("COVERAGE: time of a case divided by the time that its gas covers at the reference rate of the build (at most 1 means covered)")
    worst = 0
    for p, b, label in builds:
        vals = [(T[(p, b)][k] / (g * r[(p, b)]), k) for k, g in charged.items()]
        m = max(vals); worst = max(worst, m[0])
        w("%32s | cases %d | largest %.3f (%s, %s, n=%d) | median %.3f" % (label, len(vals), m[0], *m[1], statistics.median(v for v, _ in vals)))
    w("")
    w("*** CALIBRATION COMPLETE: every measured case is covered on every build ***" if same and worst <= 1.0 + 1e-12 else "*** INSPECT: a case is not covered or the case lists differ ***")
    text = "\n".join(out) + "\n"
    if len(sys.argv) <= 1: open("outputs/price_calibration.txt", "w").write(text)
    print(text, end="")


main()
