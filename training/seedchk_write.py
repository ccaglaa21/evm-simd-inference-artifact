# seedchk_write.py — SPOT-CHECK 1/2: retrains Fashion seed-31337 (deterministic),
# writes the weights with the seedchk_* prefix (fashion_* UNTOUCHED). Also writes the first 100 test
# images + int16-simulation predictions; 2/2 runs them in the C++ EVM and compares.
# Reason for the choice: seed-31337 chose BOTH SHIFTs differently from the canonical (7/12) -> the most demanding case.
import numpy as np, struct
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
Xte_raw=read_images(f'{DATA}/t10k-images-idx3-ubyte')
yte=read_labels(f'{DATA}/t10k-labels-idx1-ubyte')
def oh(y,k=10):
    o=np.zeros((len(y),k));o[np.arange(len(y)),y]=1;return o
Ytr=oh(ytr)
def relu(x):return np.maximum(0,x)

np.random.seed(31337)
H1,H2=128,64
W1=np.random.randn(784,H1)*0.05;b1=np.zeros(H1)
W2=np.random.randn(H1,H2)*0.05;b2=np.zeros(H2)
W3=np.random.randn(H2,10)*0.05;b3=np.zeros(10)
lr=0.1;batch=128
print("seed-31337 training (deterministic repeat)...", flush=True)
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
    if e%5==0 or e==24: print(f"  epoch {e} ok", flush=True)

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
assert (SHIFT1,SHIFT2)==(7,12), f"the sweep gave 7/12, got {SHIFT1}/{SHIFT2} — STOP, report"
print(f"SHIFT={SHIFT1}/{SHIFT2} (consistent with the sweep)")

np.savetxt('seedchk_W1.txt',W1q.flatten(),fmt='%d')
np.savetxt('seedchk_W2.txt',W2q.flatten(),fmt='%d')
np.savetxt('seedchk_W3.txt',W3q.flatten(),fmt='%d')
np.savetxt('seedchk_b1.txt',b1i,fmt='%d')
np.savetxt('seedchk_b2.txt',b2i,fmt='%d')
with open('seedchk_meta.txt','w') as f: f.write(f"{S} {SHIFT1} {SHIFT2} {H1} {H2}\n")
N=100
with open('seedchk_imgs.txt','w') as f:
    f.write(f"{N}\n")
    for n in range(N):
        img=Xte_raw[n].astype(np.int64)
        a1=np.clip(np.maximum(0,img@W1q+b1i)>>SHIFT1,0,32767)
        a2=np.clip(np.maximum(0,a1@W2q+b2i)>>SHIFT2,0,32767)
        pq=int(np.argmax(a2@W3q))
        f.write(f"{int(yte[n])} {pq}\n")
        f.write(" ".join(map(str,img.tolist()))+"\n")
print(f"Written: seedchk_W1/W2/W3, seedchk_b1/b2, seedchk_meta, seedchk_imgs ({N} images + int16 predictions)")
