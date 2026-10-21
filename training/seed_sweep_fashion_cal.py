# seed_sweep_fashion_cal.py — CALIBRATION EXPERIMENT: single-test-image
# calibration (CANONICAL) vs training-set-100-image-max calibration (NEW).
# Read-only. CHECK: the canonical method at seed 42 gives SHIFT 6/13 + 434/443/466 EXACTLY.
# Question: does the Fashion loss range (0.05-2.41 points) narrow with the NEW method?
import numpy as np, struct, time
DATA='fashion_data'
def read_images(path):
    with open(path,'rb') as f:
        m,n,r,c=struct.unpack('>IIII',f.read(16))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r*c)
def read_labels(path):
    with open(path,'rb') as f:
        m,n=struct.unpack('>II',f.read(8)); return np.frombuffer(f.read(),dtype=np.uint8)
Xtr_raw=read_images(f'{DATA}/train-images-idx3-ubyte'); Xtr=Xtr_raw.astype(np.float64)/255.0
ytr=read_labels(f'{DATA}/train-labels-idx1-ubyte')
Xte_raw=read_images(f'{DATA}/t10k-images-idx3-ubyte'); Xte=Xte_raw.astype(np.float64)/255.0
yte=read_labels(f'{DATA}/t10k-labels-idx1-ubyte')
def oh(y,k=10):
    o=np.zeros((len(y),k));o[np.arange(len(y)),y]=1;return o
Ytr=oh(ytr)
def relu(x):return np.maximum(0,x)

def train(seed):
    np.random.seed(seed)
    H1,H2=128,64
    W1=np.random.randn(784,H1)*0.05;b1=np.zeros(H1)
    W2=np.random.randn(H1,H2)*0.05;b2=np.zeros(H2)
    W3=np.random.randn(H2,10)*0.05;b3=np.zeros(10)
    lr=0.1;batch=128
    for e in range(25):
        idx=np.random.permutation(len(Xtr))
        for i in range(0,len(Xtr),batch):
            bi=idx[i:i+batch];xb,yb=Xtr[bi],Ytr[bi]
            z1=xb@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
            z3-=z3.max(1,keepdims=True);p=np.exp(z3);p/=p.sum(1,keepdims=True)
            dz3=(p-yb)/len(bi);dW3=a2.T@dz3;db3=dz3.sum(0)
            da2=dz3@W3.T;dz2=da2*(z2>0);dW2=a1.T@dz2;db2=dz2.sum(0)
            da1=dz2@W2.T;dz1=da1*(z1>0);dW1=xb.T@dz1;db1=dz1.sum(0)
            W1-=lr*dW1;b1-=lr*db1;W2-=lr*dW2;b2-=lr*db2;W3-=lr*dW3;b3-=lr*db3
    return W1,b1,W2,b2,W3,b3

def quantize(fp, calib_imgs):
    # calib_imgs: RAW image(s) to use for calibration (int64, [n,784])
    W1,b1,W2,b2,W3,b3=fp; S=4096
    W1q=np.round(W1*S).astype(np.int64);W2q=np.round(W2*S).astype(np.int64);W3q=np.round(W3*S).astype(np.int64)
    b1i=np.round(b1*255*S).astype(np.int64)
    z1s=calib_imgs@W1q+b1i; a1s=np.maximum(0,z1s)
    SHIFT1=0
    while a1s.max()>>SHIFT1>32767: SHIFT1+=1
    a1_=np.clip(a1s>>SHIFT1,0,32767)
    # layer-2 bias scale: a1_small*W2q carries 255*S^2/2^SHIFT1, so the bias carries the same scale
    b2i=np.round(b2*(255*S*S/(2**SHIFT1))).astype(np.int64)
    z2s=a1_@W2q+b2i; a2s=np.maximum(0,z2s)
    SHIFT2=0
    while a2s.max()>>SHIFT2>32767: SHIFT2+=1
    return (W1q,b1i,W2q,b2i,W3q),(S,SHIFT1,SHIFT2)

def evalq(fp,qp,shifts):
    W1,b1,W2,b2,W3,b3=fp; W1q,b1i,W2q,b2i,W3q=qp; S,S1,S2=shifts
    z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
    pf=np.argmax(z3,axis=1)
    imgs=Xte_raw.astype(np.int64)
    q1=np.clip(np.maximum(0,imgs@W1q+b1i)>>S1,0,32767)
    q2=np.clip(np.maximum(0,q1@W2q+b2i)>>S2,0,32767)
    pq=np.argmax(q2@W3q,axis=1)
    n=len(yte)
    return (int((pf==yte).sum()), int((pq==yte).sum()), int((pq==pf).sum()),
            int((pf[:500]==yte[:500]).sum()), int((pq[:500]==yte[:500]).sum()), int((pq[:500]==pf[:500]).sum()))

seeds=[42,7,123,2024,31337]
print("seed | CANONICAL SH/loss(points,10k) | NEW SH/loss | agree10k C->N")
rowsK=[];rowsY=[]
for sd in seeds:
    t0=time.time()
    fp=train(sd)
    # CANONICAL: a single TEST image (Xte_raw[0]) — train_deep_fashion verbatim
    qpK,shK=quantize(fp, Xte_raw[0].astype(np.int64)[None,:])
    fK=evalq(fp,qpK,shK)
    # NEW: the first 100 images of the TRAINING set (via the max activation)
    qpY,shY=quantize(fp, Xtr_raw[:100].astype(np.int64))
    fY=evalq(fp,qpY,shY)
    kK=(fK[0]-fK[1])/100.0; kY=(fY[0]-fY[1])/100.0
    rowsK.append((sd,shK[1],shK[2],fK)); rowsY.append((sd,shY[1],shY[2],fY))
    print(f"{sd:5d} | {shK[1]}/{shK[2]}  {kK:+.2f} | {shY[1]}/{shY[2]}  {kY:+.2f} | {fK[2]} -> {fY[2]}   ({time.time()-t0:.0f}s)", flush=True)
    if sd==42:
        gate = shK==(4096,6,13) and fK[4]==434 and fK[3]==443 and fK[5]==466
        print(f"  [CHECK seed42 CANONICAL] SHIFT 6/13 + 434/443/466: {'PASSED' if gate else 'MISMATCH — STOP, report'}", flush=True)

lossK=np.array([ (r[3][0]-r[3][1])/100.0 for r in rowsK])
lossY=np.array([ (r[3][0]-r[3][1])/100.0 for r in rowsY])
print(f"\nLOSS (10k, points): CANONICAL range [{lossK.min():.2f},{lossK.max():.2f}] mean {lossK.mean():.2f}±{lossK.std():.2f}")
print(f"                    NEW       range [{lossY.min():.2f},{lossY.max():.2f}] mean {lossY.mean():.2f}±{lossY.std():.2f}")
print("(The NEW method calibrates from TRAINING data -> the test leakage is also removed.)")
