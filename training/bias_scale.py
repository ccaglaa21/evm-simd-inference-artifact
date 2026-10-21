#!/usr/bin/env python3
# bias_scale.py -- scale check of the hidden-layer-2 bias of the integer networks, and per-layer
# saturation counts, on the 500 canonical images of each dataset.
#
# The integer pipeline of the contracts is  z1 = x W1q + b1i,  a1 = clip(relu(z1) >> SHIFT1),
# z2 = a1 W2q + b2i,  a2 = clip(relu(z2) >> SHIFT2),  z3 = a2 W3q,  with x in 0..255 and
# W*q = round(W * S).  Since a1 ~ a1_float * 255 S / 2^SHIFT1 and W2q ~ W2 * S, the product
# a1 W2q carries the scale 255 S^2 / 2^SHIFT1, so the bias of layer 2 must carry the same scale:
#     b2i (consistent) = round(b2 * 255 S^2 / 2^SHIFT1).
# The training scripts shipped with this artifact wrote  b2i = round(b2 * 255 S / 2^SHIFT1), a factor
# S smaller, which makes the layer-2 bias of the evaluated networks almost zero. This script measures
# the effect on the canonical images: predictions with the shipped b2i, with b2i * S (the consistent
# scale, reconstructed from the shipped integers) and with b2i = 0, each compared with the float
# prediction stored in the batch file and with the label. It also counts, per layer, the activations
# clipped at 32767 and the zero activations, which the paper reports as saturation.
# Reads the files of data/ from the current directory; uses numpy when available.
import sys
try:
    import numpy as np
except ImportError:
    np = None

def read_ints(path):
    with open(path) as f:
        return [int(t) for t in f.read().split()]

def analyse(prefix, batchfile, label):
    S, SH1, SH2, H1, H2 = read_ints(prefix + "_meta.txt")[:5]
    I = 784
    W1 = read_ints(prefix + "_W1.txt"); W2 = read_ints(prefix + "_W2.txt"); W3 = read_ints(prefix + "_W3.txt")
    b1 = read_ints(prefix + "_b1.txt"); b2 = read_ints(prefix + "_b2.txt")
    toks = open(batchfile).read().split()
    p = 0; Nb = int(toks[p]); p += 2
    imgs, labels, pfloat = [], [], []
    for im in range(min(Nb, 500)):
        labels.append(int(toks[p])); pfloat.append(int(toks[p + 1])); p += 2 + 10
        imgs.append([int(t) for t in toks[p:p + I]]); p += I
    n_img = len(imgs)
    variants = [("b2 as shipped (scale 255 S / 2^SHIFT1)", 1), ("b2 x S (consistent scale 255 S^2 / 2^SHIFT1)", S), ("b2 = 0", 0)]
    print(f"=== {label} ({prefix}_*): S={S} SHIFT1={SH1} SHIFT2={SH2} layers {I}->{H1}->{H2}->10, images={n_img} ===")
    print(f"  shipped b2 range [{min(b2)}, {max(b2)}]; a1 W2q products reach the scale 255 S^2 / 2^SHIFT1, so the shipped bias is S = {S} times too small")
    for name, mult in variants:
        b2v = [v * mult for v in b2]
        match = acc = 0; sat1 = sat2 = zero1 = zero2 = 0; z2mag = []
        for k in range(n_img):
            x = imgs[k]
            if np is not None:
                xa = np.array(x, dtype=np.int64)
                z1 = xa @ np.array(W1, dtype=np.int64).reshape(I, H1) + np.array(b1, dtype=np.int64)
                a1 = np.minimum(np.maximum(z1, 0) >> SH1, 32767)
                z2 = a1 @ np.array(W2, dtype=np.int64).reshape(H1, H2) + np.array(b2v, dtype=np.int64)
                a2 = np.minimum(np.maximum(z2, 0) >> SH2, 32767)
                z3 = a2 @ np.array(W3, dtype=np.int64).reshape(H2, 10)
                pred = int(np.argmax(z3)); sat1 += int(np.sum((np.maximum(z1, 0) >> SH1) > 32767)); sat2 += int(np.sum((np.maximum(z2, 0) >> SH2) > 32767))
                zero1 += int(np.sum(a1 == 0)); zero2 += int(np.sum(a2 == 0)); z2mag.append(int(np.median(np.abs(z2))))
            else:
                z1 = [b1[n] + sum(x[i] * W1[i * H1 + n] for i in range(I)) for n in range(H1)]
                a1 = [min(max(v, 0) >> SH1, 32767) for v in z1]
                z2 = [b2v[n] + sum(a1[i] * W2[i * H2 + n] for i in range(H1)) for n in range(H2)]
                a2 = [min(max(v, 0) >> SH2, 32767) for v in z2]
                z3 = [sum(a2[i] * W3[i * 10 + c] for i in range(H2)) for c in range(10)]
                pred = max(range(10), key=lambda c: (z3[c], -c)); sat1 += sum(1 for v in z1 if (max(v, 0) >> SH1) > 32767); sat2 += sum(1 for v in z2 if (max(v, 0) >> SH2) > 32767)
                zero1 += sum(1 for v in a1 if v == 0); zero2 += sum(1 for v in a2 if v == 0); z2mag.append(sorted(abs(v) for v in z2)[H2 // 2])
            match += (pred == pfloat[k]); acc += (pred == labels[k])
        print(f"  {name}:")
        print(f"    agreement with the float prediction {match}/{n_img}, label accuracy {acc}/{n_img}")
        print(f"    saturation: layer 1 {sat1} of {n_img * H1} activations clipped at 32767, {zero1} zero | layer 2 {sat2} of {n_img * H2} clipped, {zero2} zero | median |z2| {sorted(z2mag)[len(z2mag) // 2]}, median |b2 term| {sorted(abs(v) for v in b2v)[H2 // 2]}")
    print()

analyse("deep", "mnist_batch.txt", "MNIST")
analyse("fashion", "fashion_batch.txt", "Fashion-MNIST")
