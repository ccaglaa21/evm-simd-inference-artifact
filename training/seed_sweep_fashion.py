# seed_sweep_fashion.py — SEED DIVERSITY (Fashion-MNIST), train_deep_fashion.py VERBATIM.
# Read-only: fashion_*.txt is NOT WRITTEN. CHECK: seed 42 -> the fashion_meta SHIFTs
# + canonical q500=434, f500=443, agree500=466 must be reproduced.
# NOTE: this is NOT an ON-CHAIN MEASUREMENT; it is the statistics of the quantization scheme. EVM equivalence
# rests on the 100% execution fidelity; a separate EVM spot-check on another seed (seedchk_*) complements it.
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

with open('fashion_meta.txt') as f:
    mS,mS1,mS2,mH1,mH2=[int(x) for x in f.read().split()]
print(f"fashion_meta reference: S={mS} SHIFT1={mS1} SHIFT2={mS2} (seed-42 check; canonical 434/443/466)")

def train_and_quant(seed):
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
        if e%5==0 or e==24:
            z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
            print(f"    epoch {e}: test acc={(np.argmax(z3,axis=1)==yte).mean():.4f}", flush=True)
    S=4096
    W1q=np.round(W1*S).astype(np.int64);W2q=np.round(W2*S).astype(np.int64);W3q=np.round(W3*S).astype(np.int64)
    b1i=np.round(b1*255*S).astype(np.int64)
    s0=Xte_raw[0].astype(np.int64);z1s=s0@W1q+b1i;a1s=np.maximum(0,z1s)
    SHIFT1=0
    while a1s.max()>>SHIFT1>32767: SHIFT1+=1
    a1_=np.clip(a1s>>SHIFT1,0,32767)
    # layer-2 bias scale: a1_small*W2q carries 255*S^2/2^SHIFT1, so the bias carries the same scale
    b2i=np.round(b2*(255*S*S/(2**SHIFT1))).astype(np.int64)
    z2s=a1_@W2q+b2i;a2s=np.maximum(0,z2s)
    SHIFT2=0
    while a2s.max()>>SHIFT2>32767: SHIFT2+=1
    return (W1,b1,W2,b2,W3,b3),(W1q,b1i,W2q,b2i,W3q),(S,SHIFT1,SHIFT2)

def eval_seed(fp,qp,shifts):
    W1,b1,W2,b2,W3,b3=fp; W1q,b1i,W2q,b2i,W3q=qp; S,S1,S2=shifts
    z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
    pf=np.argmax(z3,axis=1)
    imgs=Xte_raw.astype(np.int64)
    q1=np.clip(np.maximum(0,imgs@W1q+b1i)>>S1,0,32767)
    q2=np.clip(np.maximum(0,q1@W2q+b2i)>>S2,0,32767)
    pq=np.argmax(q2@W3q,axis=1)
    r={}
    for tag,n in (("500",500),("10k",len(yte))):
        r["facc_"+tag]=int((pf[:n]==yte[:n]).sum())
        r["qacc_"+tag]=int((pq[:n]==yte[:n]).sum())
        r["agree_"+tag]=int((pq[:n]==pf[:n]).sum())
    return r

seeds=[42,7,123,2024,31337]
rows=[]
for sd in seeds:
    print(f"[seed {sd}] training...", flush=True)
    t0=time.time()
    fp,qp,sh=train_and_quant(sd)
    r=eval_seed(fp,qp,sh); r["seed"]=sd; r["S1"],r["S2"]=sh[1],sh[2]
    rows.append(r)
    print(f"[seed {sd}] done ({time.time()-t0:.0f}s): SHIFT={sh[1]}/{sh[2]} | "
          f"500: float {r['facc_500']} q {r['qacc_500']} agree {r['agree_500']} | "
          f"10k: float {r['facc_10k']} q {r['qacc_10k']} agree {r['agree_10k']}", flush=True)
    if sd==42:
        g1 = (sh[0]==mS and sh[1]==mS1 and sh[2]==mS2)
        g2 = (r['qacc_500']==434 and r['facc_500']==443 and r['agree_500']==466)
        print(f"  [CHECK seed42] S/SHIFT == fashion_meta: {'PASSED' if g1 else 'MISMATCH (report!)'} | "
              f"canonical 434/443/466: {'PASSED' if g2 else 'MISMATCH (report!)'}", flush=True)

print("\n================ SEED SWEEP SUMMARY (Fashion-MNIST, 5 seeds) ================")
print(" seed  | SH1/SH2 | f500 | q500 | agr500 | f10k  | q10k  | agr10k")
for r in rows:
    print(f" {r['seed']:5d} |  {r['S1']}/{r['S2']}   | {r['facc_500']:3d}  | {r['qacc_500']:3d}  |  {r['agree_500']:3d}   | {r['facc_10k']:5d} | {r['qacc_10k']:5d} | {r['agree_10k']:5d}")
qa=np.array([r['qacc_10k'] for r in rows])/100.0
fa=np.array([r['facc_10k'] for r in rows])/100.0
ag=np.array([r['agree_10k'] for r in rows])/100.0
dl=fa-qa
print(f"\n10k: float {fa.mean():.2f}%±{fa.std():.2f} | int16 {qa.mean():.2f}%±{qa.std():.2f} | "
      f"loss {dl.mean():.2f}±{dl.std():.2f} points | agreement {ag.mean():.2f}%±{ag.std():.2f}")
