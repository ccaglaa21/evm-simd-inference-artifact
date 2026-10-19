import numpy as np

S,S1,S2,H1,H2 = [int(x) for x in open('fashion_meta.txt').read().split()]
I=784
W1=np.array([int(x) for x in open('fashion_W1.txt').read().split()],dtype=np.int64).reshape(I,H1)
W2=np.array([int(x) for x in open('fashion_W2.txt').read().split()],dtype=np.int64).reshape(H1,H2)
W3=np.array([int(x) for x in open('fashion_W3.txt').read().split()],dtype=np.int64).reshape(H2,10)
b1=np.array([int(x) for x in open('fashion_b1.txt').read().split()],dtype=np.int64)
b2=np.array([int(x) for x in open('fashion_b2.txt').read().split()],dtype=np.int64)
print(f"meta: S={S} SHIFT1={S1} SHIFT2={S2} H1={H1} H2={H2}")
print(f"max|W| per layer: {np.abs(W1).max()},{np.abs(W2).max()},{np.abs(W3).max()}")

# fashion_batch: Nb wsc, then per image: tlb pfo 10float 784pixel  (the SAME parse as fashion_onchain_acc)
toks=open('fashion_batch.txt').read().split()
N=500; p=2
imgs=np.zeros((N,I),dtype=np.int64); labels=np.zeros(N,dtype=np.int64)
for im in range(N):
    labels[im]=int(toks[p]); p+=1     # tlb
    p+=1                               # skip pfo
    p+=10                              # skip 10 floats
    for i in range(I): imgs[im][i]=int(float(toks[p])); p+=1
print(f"batch: tokens consumed={p}, in file={len(toks)}")

# the SAME pipeline as the EVM (NO bias, NO ReLU in the output layer)
def infer(A,B,C):
    z1=imgs@A+b1; h1=np.clip(np.maximum(0,z1)>>S1,0,32767)
    z2=h1@B+b2;   h2=np.clip(np.maximum(0,z2)>>S2,0,32767)
    z3=h2@C
    return np.argmax(z3,axis=1)

pred16=infer(W1,W2,W3); acc16=(pred16==labels).sum()
rf=open('fashion_ref.txt').read().split(); refpred=np.array([int(rf[1+2*i+1]) for i in range(N)])
match_ref=(pred16==refpred).sum()
print(f"\n[CHECK] int16 accuracy = {acc16}/500 = {100*acc16/N:.1f}%  (must EQUAL the EVM Fashion: 434/500=86.8%)")
print(f"[CHECK] int16 == fashion_ref predictions: {match_ref}/500  (500 means the pipeline is EXACTLY right)")

def int8_roundtrip(W):
    m=np.abs(W).max()
    if m==0: return W.copy()
    W8=np.clip(np.round(W*127.0/m),-127,127)
    return np.round(W8*m/127.0).astype(np.int64)
W1q,W2q,W3q=int8_roundtrip(W1),int8_roundtrip(W2),int8_roundtrip(W3)
pred8=infer(W1q,W2q,W3q); acc8=(pred8==labels).sum(); fid=(pred8==pred16).sum()
print(f"\n=== int8 RESULT (Fashion) ===")
print(f"int8 accuracy            = {acc8}/500 = {100*acc8/N:.1f}%")
print(f"int8 vs int16 fidelity   = {fid}/500 = {100*fid/N:.1f}%")
print(f"ACCURACY LOSS (int16->int8) = {100*(acc16-acc8)/N:.1f} points")
