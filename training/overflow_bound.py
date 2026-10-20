#!/usr/bin/env python3
# overflow_bound.py -- 32-bit accumulator analysis for the canonical integer networks.
#
# For every layer of the MNIST (deep_*) and Fashion-MNIST (fashion_*) networks this script reports
#   (a) the worst-case magnitude of the layer accumulator over the whole admissible input range
#       (layer 1: pixels 0..255; layers 2 and 3: clipped activations 0..32767), computed from the
#       absolute weight sums, and
#   (b) the largest magnitude actually reached on the 500 canonical images with the exact integer
#       pipeline of the on-chain contract (bias, ReLU, right shift, clip to 0..32767).
# Both are compared with the signed 32-bit limit 2^31 - 1 used by the accumulation instructions.
# Pure Python, no third-party package; reads the files of data/ from the current directory.

import sys

LIM = 2**31 - 1

def read_ints(path):
    with open(path) as f:
        return [int(t) for t in f.read().split()]

def analyse(prefix, batchfile, label):
    S, SH1, SH2, H1, H2 = read_ints(prefix + "_meta.txt")[:5]
    I = 784
    W1 = read_ints(prefix + "_W1.txt"); W2 = read_ints(prefix + "_W2.txt"); W3 = read_ints(prefix + "_W3.txt")
    b1 = read_ints(prefix + "_b1.txt"); b2 = read_ints(prefix + "_b2.txt")
    assert len(W1) == I * H1 and len(W2) == H1 * H2 and len(W3) == H2 * 10

    # (a) worst-case bounds: sum of |w| times the largest admissible input, plus |bias|
    def bound(W, rows, cols, xmax, bias):
        best = 0
        for n in range(cols):
            s = sum(abs(W[i * cols + n]) for i in range(rows)) * xmax + (abs(bias[n]) if bias else 0)
            best = max(best, s)
        return best
    B1 = bound(W1, I, H1, 255, b1)
    B2 = bound(W2, H1, H2, 32767, b2)
    B3 = bound(W3, H2, 10, 32767, None)

    # (b) observed maxima on the 500 canonical images, exact integer pipeline
    toks = open(batchfile).read().split()
    p = 0
    Nb = int(toks[p]); p += 1; p += 1  # Nb, wsc
    def rss(v, sh):
        if v < 0: v = 0
        v >>= sh
        return 32767 if v > 32767 else v
    obs = [0, 0, 0]
    n_img = min(Nb, 500)
    for im in range(n_img):
        p += 2 + 10  # label, float prediction, 10 scores
        img = [int(t) for t in toks[p:p + I]]; p += I
        z1 = [b1[n] + sum(img[i] * W1[i * H1 + n] for i in range(I)) for n in range(H1)]
        obs[0] = max(obs[0], max(abs(v) for v in z1))
        a1 = [rss(v, SH1) for v in z1]
        z2 = [b2[n] + sum(a1[i] * W2[i * H2 + n] for i in range(H1)) for n in range(H2)]
        obs[1] = max(obs[1], max(abs(v) for v in z2))
        a2 = [rss(v, SH2) for v in z2]
        z3 = [sum(a2[i] * W3[i * 10 + c] for i in range(H2)) for c in range(10)]
        obs[2] = max(obs[2], max(abs(v) for v in z3))

    print(f"=== {label} ({prefix}_*): S={S} SHIFT1={SH1} SHIFT2={SH2} layers {I}->{H1}->{H2}->10, images={n_img} ===")
    print(f"  weight ranges: W1 [{min(W1)},{max(W1)}]  W2 [{min(W2)},{max(W2)}]  W3 [{min(W3)},{max(W3)}]")
    for k, (B, name, xr) in enumerate([(B1, "layer 1", "pixels 0..255"), (B2, "layer 2", "inputs 0..32767"), (B3, "layer 3", "inputs 0..32767")]):
        verdict = "cannot overflow (bound below 2^31-1)" if B <= LIM else "MAY overflow for some admissible input (bound above 2^31-1)"
        print(f"  {name}: worst-case |accumulator| <= {B}  over {xr}  -> {verdict}")
        print(f"  {name}: observed max |accumulator| on the canonical images = {obs[k]}  ({obs[k] / LIM * 100:.2f}% of 2^31-1)")
    print()

analyse("deep", "mnist_batch.txt", "MNIST")
analyse("fashion", "fashion_batch.txt", "Fashion-MNIST")
print("limit 2^31-1 =", LIM)
