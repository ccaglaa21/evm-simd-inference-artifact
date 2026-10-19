import numpy as np

# --- meta + weights + biases (the SAME files as the EVM) ---
S,S1,S2,H1,H2 = [int(x) for x in open('deep_meta.txt').read().split()]
I=784
W1=np.array([int(x) for x in open('deep_W1.txt').read().split()],dtype=np.int64).reshape(I,H1)
W2=np.array([int(x) for x in open('deep_W2.txt').read().split()],dtype=np.int64).reshape(H1,H2)
W3=np.array([int(x) for x in open('deep_W3.txt').read().split()],dtype=np.int64).reshape(H2,10)
b1=np.array([int(x) for x in open('deep_b1.txt').read().split()],dtype=np.int64)
b2=np.array([int(x) for x in open('deep_b2.txt').read().split()],dtype=np.int64)
print(f"meta: S={S} SHIFT1={S1} SHIFT2={S2} H1={H1} H2={H2}")
print(f"W1{W1.shape} W2{W2.shape} W3{W3.shape}  max|W| per layer: {np.abs(W1).max()},{np.abs(W2).max()},{np.abs(W3).max()}")

# --- 500 images + labels (mnist_batch.txt: Nb wsc, then per image: tlb pfo 10float 784pixel) ---
toks=open('mnist_batch.txt').read().split()
N=500; p=2  # first two tokens are the header
imgs=np.zeros((N,I),dtype=np.int64); labels=np.zeros(N,dtype=np.int64)
for im in range(N):
    labels[im]=int(toks[p]); p+=1      # tlb = true label
    p+=1                                # skip pfo (float prediction)
    p+=10                               # skip 10 floats
    for i in range(I): imgs[im][i]=int(float(toks[p])); p+=1
print(f"mnist_batch: tokens consumed={p}, in file={len(toks)} (should be equal/close)")

# --- the SAME integer pipeline as the EVM ---
def infer(A,B,C):
    z1=imgs@A+b1; h1=np.clip(np.maximum(0,z1)>>S1,0,32767)
    z2=h1@B+b2;   h2=np.clip(np.maximum(0,z2)>>S2,0,32767)
    z3=h2@C
    return np.argmax(z3,axis=1)

# --- CHECK: int16 -> reproduce the EVM's 98% + compare with deep_ref ---
pred16=infer(W1,W2,W3)
acc16=(pred16==labels).sum()
rf=open('deep_ref.txt').read().split(); refpred=np.array([int(rf[1+2*i+1]) for i in range(N)])
match_ref=(pred16==refpred).sum()
print(f"\n[CHECK] int16 accuracy = {acc16}/500 = {100*acc16/N:.1f}%  (must EQUAL the EVM: 490/500=98%)")
print(f"[CHECK] int16 == deep_ref predictions: {match_ref}/500  (500 means the pipeline is EXACTLY right)")

# --- int8: round every layer to int8, scale back to the same scale (pipeline unchanged) ---
def int8_roundtrip(W):
    m=np.abs(W).max()
    if m==0: return W.copy()
    W8=np.clip(np.round(W*127.0/m),-127,127)       # int8 representation (256 levels)
    return np.round(W8*m/127.0).astype(np.int64)   # back to the same int16 scale
W1q,W2q,W3q=int8_roundtrip(W1),int8_roundtrip(W2),int8_roundtrip(W3)
pred8=infer(W1q,W2q,W3q)
acc8=(pred8==labels).sum(); fid=(pred8==pred16).sum()
print(f"\n=== int8 RESULT ===")
print(f"int8 accuracy            = {acc8}/500 = {100*acc8/N:.1f}%")
print(f"int8 vs int16 fidelity   = {fid}/500 = {100*fid/N:.1f}%  (how many predictions unchanged)")
print(f"ACCURACY LOSS (int16->int8) = {100*(acc16-acc8)/N:.1f} points")
