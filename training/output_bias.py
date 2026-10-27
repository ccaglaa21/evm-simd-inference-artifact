#!/usr/bin/env python3
# output_bias.py -- effect of the omitted output-layer bias on the 500 canonical images of each dataset.
#
# The contracts compute the output scores as z3 = a2 W3q without the bias b3 of the floating-point model
# (the argmax is taken over z3). This script runs the same integer pipeline in Python, once as the contracts
# do it (b3 omitted) and once with b3 added at the scale of the products a2 W3q, which is
#     255 S^3 / 2^(SHIFT1+SHIFT2)      (a2 carries 255 S^2 / 2^(SHIFT1+SHIFT2), W3q carries S),
# and reports, for both, the agreement with the float prediction of deep_ref.txt / fashion_ref.txt (the same
# references as the C++ accuracy programs), the label accuracy, and the largest |score| reached, against the
# 32-bit limit of the accumulators. b3 is read from deep_b3.txt / fashion_b3.txt, written by the training
# scripts at that scale. Reads the files of data/ from the current directory; uses numpy when available.
import sys
try:
    import numpy as np
except ImportError:
    np = None

def read_ints(path):
    with open(path) as f:
        return [int(t) for t in f.read().split()]

def analyse(prefix, batchfile, reffile, label):
    S, SH1, SH2, H1, H2 = read_ints(prefix + "_meta.txt")[:5]
    I = 784
    W1 = read_ints(prefix + "_W1.txt"); W2 = read_ints(prefix + "_W2.txt"); W3 = read_ints(prefix + "_W3.txt")
    b1 = read_ints(prefix + "_b1.txt"); b2 = read_ints(prefix + "_b2.txt"); b3 = read_ints(prefix + "_b3.txt")
    toks = open(batchfile).read().split()
    p = 0; Nb = int(toks[p]); p += 2
    imgs = []
    for im in range(min(Nb, 500)):
        p += 2 + 10
        imgs.append([int(t) for t in toks[p:p + I]]); p += I
    n_img = len(imgs)
    rt = open(reffile).read().split()
    labels = [int(rt[1 + 2 * k]) for k in range(n_img)]; pfloat = [int(rt[2 + 2 * k]) for k in range(n_img)]
    scale3 = 255 * S * S * S // (1 << (SH1 + SH2))
    print(f"=== {label} ({prefix}_*): S={S} SHIFT1={SH1} SHIFT2={SH2} layers {I}->{H1}->{H2}->10, images={n_img} ===")
    print(f"  b3 range [{min(b3)}, {max(b3)}] at the scale 255 S^3 / 2^(SHIFT1+SHIFT2) = {scale3}")
    for name, use_b3 in (("b3 omitted (as the contracts compute)", False), ("b3 added at the consistent scale", True)):
        match = acc = 0; zmax = 0; changed = 0
        for k in range(n_img):
            x = imgs[k]
            if np is not None:
                xa = np.array(x, dtype=np.int64)
                z1 = xa @ np.array(W1, dtype=np.int64).reshape(I, H1) + np.array(b1, dtype=np.int64)
                a1 = np.minimum(np.maximum(z1, 0) >> SH1, 32767)
                z2 = a1 @ np.array(W2, dtype=np.int64).reshape(H1, H2) + np.array(b2, dtype=np.int64)
                a2 = np.minimum(np.maximum(z2, 0) >> SH2, 32767)
                z3 = a2 @ np.array(W3, dtype=np.int64).reshape(H2, 10)
                z3b = z3 + np.array(b3, dtype=np.int64)
                zz = z3b if use_b3 else z3
                pred = int(np.argmax(zz)); zmax = max(zmax, int(np.max(np.abs(zz))))
                changed += int(np.argmax(z3) != np.argmax(z3b))
            else:
                z1 = [b1[n] + sum(x[i] * W1[i * H1 + n] for i in range(I)) for n in range(H1)]
                a1 = [min(max(v, 0) >> SH1, 32767) for v in z1]
                z2 = [b2[n] + sum(a1[i] * W2[i * H2 + n] for i in range(H1)) for n in range(H2)]
                a2 = [min(max(v, 0) >> SH2, 32767) for v in z2]
                z3 = [sum(a2[i] * W3[i * 10 + c] for i in range(H2)) for c in range(10)]
                z3b = [z3[c] + b3[c] for c in range(10)]
                zz = z3b if use_b3 else z3
                am = lambda v: max(range(10), key=lambda c: (v[c], -c))
                pred = am(zz); zmax = max(zmax, max(abs(v) for v in zz)); changed += int(am(z3) != am(z3b))
            match += (pred == pfloat[k]); acc += (pred == labels[k])
        print(f"  {name}:")
        print(f"    agreement with the float prediction {match}/{n_img}, label accuracy {acc}/{n_img}, "
              f"largest |score| {zmax} ({100.0 * zmax / 2147483647:.2f}% of 2^31-1)")
    print(f"  predictions changed by adding b3: {changed} of {n_img}")
    print()

analyse("deep", "mnist_batch.txt", "deep_ref.txt", "MNIST")
analyse("fashion", "fashion_batch.txt", "fashion_ref.txt", "Fashion-MNIST")
