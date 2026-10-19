import numpy as np
S,S1,S2,H1,H2=[int(x) for x in open('fashion_meta.txt').read().split()]; I=784
W1=np.array([int(x) for x in open('fashion_W1.txt').read().split()],dtype=np.int64).reshape(I,H1)
W2=np.array([int(x) for x in open('fashion_W2.txt').read().split()],dtype=np.int64).reshape(H1,H2)
W3=np.array([int(x) for x in open('fashion_W3.txt').read().split()],dtype=np.int64).reshape(H2,10)
b1=np.array([int(x) for x in open('fashion_b1.txt').read().split()],dtype=np.int64)
b2=np.array([int(x) for x in open('fashion_b2.txt').read().split()],dtype=np.int64)
toks=open('fashion_batch.txt').read().split(); N=500; p=2
imgs=np.zeros((N,I),dtype=np.int64); labels=np.zeros(N,dtype=np.int64)
for im in range(N):
    labels[im]=int(toks[p]); p+=2+10
    for i in range(I): imgs[im][i]=int(float(toks[p-12+2+i]))  # safe: recompute
# the offset trick above may be confusing -> re-read cleanly:
p=2; imgs=np.zeros((N,I),dtype=np.int64); labels=np.zeros(N,dtype=np.int64)
for im in range(N):
    labels[im]=int(toks[p]); p+=1; p+=1; p+=10
    for i in range(I): imgs[im][i]=int(float(toks[p])); p+=1

z1=imgs@W1+b1; h1=np.clip(np.maximum(0,z1)>>S1,0,32767)
z2=h1@W2+b2;   h2=np.clip(np.maximum(0,z2)>>S2,0,32767)
z3=h2@W3                      # (N,10) scores
pred=np.argmax(z3,axis=1)
rf=open('fashion_ref.txt').read().split(); refpred=np.array([int(rf[1+2*i+1]) for i in range(N)])

mismatch=np.where(pred!=refpred)[0]
print(f"total mismatches: {len(mismatch)}/500")
# for each mismatch: gap between the top-2 scores (small = ambiguous image = harmless)
srt=np.sort(z3,axis=1)
gap=srt[:,-1]-srt[:,-2]        # best - second (in score units)
gap_scale=gap/np.maximum(np.abs(srt[:,-1]),1)*100  # closeness as a percentage
print("\nTop-2 score gap on the mismatched images (small=ambiguous):")
for i in mismatch[:15]:
    print(f"  img {i:3d}: mine={pred[i]} ref={refpred[i]} true={labels[i]} | top2_gap={gap[i]:>12d} ({gap_scale[i]:.2f}%)")
print(f"\nMEAN percent-gap on mismatches: {gap_scale[mismatch].mean():.2f}%")
print(f"MEAN percent-gap on ALL images : {gap_scale.mean():.2f}%")
print(f"-> if the mismatch gap << the overall gap: mismatches occur on ambiguous images (HARMLESS)")
