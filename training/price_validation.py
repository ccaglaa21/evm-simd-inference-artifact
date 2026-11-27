#!/usr/bin/env python3
# price_validation.py -- Tests the gas schedule on a second, independent measurement session.
# The schedule in outputs/price_calibration.txt was derived from the files in outputs/price_calibration_runs.
# This script reads that schedule and the files in outputs/price_validation_runs (price_bench, three runs on
# each of the four builds, measured after the schedule was fixed) and writes outputs/price_validation.txt.
# For every case it prints the time of the case divided by the time that its gas covers at the reference
# rate of the build in the validation session. A value above 1 means that the gas of the case does not cover
# its time in this session. Nothing is fitted here and no price is changed.
# Usage (from the repository root): python3 training/price_validation.py
import re
import statistics
import price_calibration as pc

CAL = "outputs/price_calibration.txt"
DIR = "outputs/price_validation_runs"


def schedule():
    t = open(CAL).read(); fx = {}; sc = {}
    for m in re.finditer(r"^\s*(X\w+) fixed price.*price\s+(\d+)\s*$", t, re.M): fx[m.group(1)] = int(m.group(2))
    g = lambda name, pat: [int(x) for x in re.search(r"^%s\s*\| %s" % (name, pat), t, re.M).groups()]
    c, G = g("XMACC_N", r"CALL\s+(\d+) \| GROUP\s+(\d+)"); sc["XMACC_N"] = (c, 0, G, 0)
    c, G = g("XMACC_N16", r"CALL\s+(\d+) \| per word of sixteen elements\s+\d+ \(two groups of eight at (\d+)\)"); sc["XMACC_N16"] = (c, 0, G, 0)
    c, G, S = g("XMACC_V", r"CALL\s+(\d+) \| GROUP\s+(\d+) \| per strided element (\d+)"); sc["XMACC_V"] = (c, 0, G, S)
    c, E, G, S = g("XMATMUL", r"CALL\s+(\d+) \| ENTRY (\d+) \| GROUP\s+(\d+) \| per packed element (\d+)"); sc["XMATMUL"] = (c, E, G, S)
    gc, gi = g("XGATHER", r"CALL\s+(\d+) \| per index (\d+)")
    return fx, sc, gc, gi


def main():
    fx, sc, gc, gi = schedule(); out = []; w = out.append
    w("GAS SCHEDULE TESTED ON A SECOND MEASUREMENT SESSION")
    w(f"Schedule: {CAL}. Times: {DIR}/<machine>_run<1,2,3>_price_<build>.txt, medians of three runs")
    w("Fixed prices: " + ", ".join("%s %d" % kv for kv in fx.items()))
    w("XMACC_N %d + %d per group | XMACC_N16 %d + %d per word of sixteen | XMACC_V %d + %d per group + %d per strided element"
      % (sc["XMACC_N"][0], sc["XMACC_N"][2], sc["XMACC_N16"][0], 2 * sc["XMACC_N16"][2], sc["XMACC_V"][0], sc["XMACC_V"][2], sc["XMACC_V"][3]))
    w("XMATMUL %d + %d per entry + %d per entry and group + %d per packed element | XGATHER %d + %d per index" % (*sc["XMATMUL"], gc, gi))
    w("")
    allv = []
    for p, b, label in pc.BUILDS:
        runs = [pc.table(f"{DIR}/{p}_run{r}_price_{b}.txt") for r in pc.RUNS]; C = runs[0]
        T = {k: statistics.median(run[k]["ns"] for run in runs) for k in C}
        r = min(T[k] / 20 for k in C if k[0] == "REF_STEP" and "end of run" not in k[1]); vals = []
        for k in C:
            if k[1].startswith("fixed price"): gas = fx[k[0]]
            elif k[0] in sc: c, E, G, S = sc[k[0]]; gas = c + G * C[k]["groups"] + E * C[k]["entries"] + S * C[k]["strided"]
            elif k[0] == "XGATHER": gas = gc + gi * k[2]
            else: continue
            vals.append((T[k] / (gas * r), label.strip(), k, gas, T[k]))
        allv += vals; over = [v for v in vals if v[0] > 1.0]
        w("%32s | reference rate %.3f ns per gas | cases %d | above 1: %d | largest %.3f | median %.3f"
          % (label, r, len(vals), len(over), max(vals)[0], statistics.median(v[0] for v in vals)))
    over = sorted([v for v in allv if v[0] > 1.0], reverse=True)
    w("")
    w("CASES ABOVE 1 (time of the case / time that its gas covers)")
    for v in over: w("  %.3f | %-30s | %s, %s, n=%d | gas %d | %.2f ns" % (v[0], v[1], *v[2], v[3], v[4]))
    w("")
    w("TOTAL: %d cases on %d builds, %d covered, %d above 1, largest %.3f" % (len(allv), len(pc.BUILDS), len(allv) - len(over), len(over), max(allv)[0]))
    w("*** VALIDATION COMPLETE ***")
    text = "\n".join(out) + "\n"; open("outputs/price_validation.txt", "w").write(text); print(text, end="")


if __name__ == "__main__":
    main()
