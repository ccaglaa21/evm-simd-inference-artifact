import numpy as np, struct
DATA='fashion_data'
def read_images(path):
    with open(path,'rb') as f:
        m,n,r,c=struct.unpack('>IIII',f.read(16))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r*c)
def read_labels(path):
    with open(path,'rb') as f:
        m,n=struct.unpack('>II',f.read(8)); return np.frombuffer(f.read(),dtype=np.uint8)

print("Loading Fashion-MNIST...")
Xtr_raw=read_images(f'{DATA}/train-images-idx3-ubyte'); Xtr=Xtr_raw.astype(np.float64)/255.0
ytr=read_labels(f'{DATA}/train-labels-idx1-ubyte')
Xte_raw=read_images(f'{DATA}/t10k-images-idx3-ubyte'); Xte=Xte_raw.astype(np.float64)/255.0
yte=read_labels(f'{DATA}/t10k-labels-idx1-ubyte')
def oh(y,k=10):
    o=np.zeros((len(y),k));o[np.arange(len(y)),y]=1;return o
Ytr=oh(ytr)

np.random.seed(42)
H1,H2=128,64
W1=np.random.randn(784,H1)*0.05;b1=np.zeros(H1)
W2=np.random.randn(H1,H2)*0.05;b2=np.zeros(H2)
W3=np.random.randn(H2,10)*0.05;b3=np.zeros(10)
lr=0.1;batch=128
def relu(x):return np.maximum(0,x)
print("Training (deep network, Fashion-MNIST)...")
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
    z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
    acc=(np.argmax(z3,axis=1)==yte).mean()
    if e%5==0 or e==24: print(f"  epoch {e}: test acc={acc:.4f}")
z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
float_acc=(np.argmax(z3,axis=1)==yte).mean()
print(f"Final Fashion deep network accuracy: {float_acc:.4f}")

# Quantization
S=4096
W1q=np.round(W1*S).astype(np.int64);W2q=np.round(W2*S).astype(np.int64);W3q=np.round(W3*S).astype(np.int64)
b1i=np.round(b1*255*S).astype(np.int64)
print(f"W1q:{W1q.min()}/{W1q.max()} W2q:{W2q.min()}/{W2q.max()} W3q:{W3q.min()}/{W3q.max()}")
s=Xte_raw[0].astype(np.int64);z1s=s@W1q+b1i;a1s=np.maximum(0,z1s)
SHIFT1=0
while a1s.max()>>SHIFT1>32767: SHIFT1+=1
a1_=np.clip(a1s>>SHIFT1,0,32767)
# layer-2 bias scale: a1_small*W2q carries 255*S^2/2^SHIFT1, so the bias carries the same scale
b2i=np.round(b2*(255*S*S/(2**SHIFT1))).astype(np.int64)
z2s=a1_@W2q+b2i;a2s=np.maximum(0,z2s)
SHIFT2=0
while a2s.max()>>SHIFT2>32767: SHIFT2+=1
print(f"SHIFT1={SHIFT1}, SHIFT2={SHIFT2}")

def qinfer(img):
    z1=img@W1q+b1i;a1=np.clip(np.maximum(0,z1)>>SHIFT1,0,32767).astype(np.int64)
    z2=a1@W2q+b2i; a2=np.clip(np.maximum(0,z2)>>SHIFT2,0,32767).astype(np.int64)
    z3=a2@W3q
    return int(np.argmax(z3))
N_test=1000;cq=0;mf=0
for n in range(N_test):
    img=Xte_raw[n].astype(np.int64)
    pq=qinfer(img)
    zz1=(img/255.0)@W1+b1;aa1=relu(zz1);zz2=aa1@W2+b2;aa2=relu(zz2);zz3=aa2@W3+b3
    pf=int(np.argmax(zz3))
    if pq==int(yte[n]):cq+=1
    if pq==pf:mf+=1
print(f"\nFashion Quantized accuracy ({N_test}): {cq}/{N_test}={cq/N_test:.4f}")
print(f"Fashion Quantized==Float ({N_test}): {mf}/{N_test}")

# Files (fashion_ prefix)
np.savetxt('fashion_W1.txt',W1q.flatten(),fmt='%d')
np.savetxt('fashion_W2.txt',W2q.flatten(),fmt='%d')
np.savetxt('fashion_W3.txt',W3q.flatten(),fmt='%d')
np.savetxt('fashion_b1.txt',b1i,fmt='%d')
np.savetxt('fashion_b2.txt',b2i,fmt='%d')
# output-layer bias at the scale of a2*W3q, 255*S^3/2^(SHIFT1+SHIFT2); not used by the contracts, written for output_bias.py
b3i=np.round(b3*(255*S*S*S/(2**(SHIFT1+SHIFT2)))).astype(np.int64)
np.savetxt('fashion_b3.txt',b3i,fmt='%d')
with open('fashion_meta.txt','w') as f: f.write(f"{S} {SHIFT1} {SHIFT2} {H1} {H2}\n")
# batch (images) + ref (float prediction), 500 images
N=500
with open('fashion_batch.txt','w') as f:
    f.write(f"{N} {S}\n")
    for n in range(N):
        img=Xte_raw[n].astype(np.int64)
        z1=img@W1q+b1i;a1=np.clip(np.maximum(0,z1)>>SHIFT1,0,32767).astype(np.int64)
        z2=a1@W2q+b2i;a2=np.clip(np.maximum(0,z2)>>SHIFT2,0,32767).astype(np.int64)
        z3=a2@W3q
        f.write(f"{int(yte[n])} {int(np.argmax(z3))} "+" ".join(map(str,z3.tolist()))+"\n")
        f.write(" ".join(map(str,Xte_raw[n].astype(int).tolist()))+"\n")
with open('fashion_ref.txt','w') as f:
    f.write(f"{N}\n")
    for n in range(N):
        img=Xte_raw[n].astype(np.float64)/255.0
        z1=img@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
        f.write(f"{int(yte[n])} {int(np.argmax(z3))}\n")
print("Files written: fashion_W1/W2/W3, fashion_b1/b2/b3, fashion_meta, fashion_batch, fashion_ref")
