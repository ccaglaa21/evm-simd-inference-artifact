import numpy as np, struct

def read_images(path):
    with open(path,'rb') as f:
        m,n,r,c=struct.unpack('>IIII',f.read(16))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r*c)
def read_labels(path):
    with open(path,'rb') as f:
        m,n=struct.unpack('>II',f.read(8)); return np.frombuffer(f.read(),dtype=np.uint8)

print("Loading MNIST...")
Xtr_raw=read_images('mnist_data/train-images-idx3-ubyte')
Xtr=Xtr_raw.astype(np.float64)/255.0
ytr=read_labels('mnist_data/train-labels-idx1-ubyte')
Xte_raw=read_images('mnist_data/t10k-images-idx3-ubyte')
Xte=Xte_raw.astype(np.float64)/255.0
yte=read_labels('mnist_data/t10k-labels-idx1-ubyte')
def oh(y,k=10):
    o=np.zeros((len(y),k));o[np.arange(len(y)),y]=1;return o
Ytr=oh(ytr)

# --- 784->128->64->10, two ReLU hidden layers (DEEP) ---
np.random.seed(42)
H1,H2=128,64
W1=np.random.randn(784,H1)*0.05; b1=np.zeros(H1)
W2=np.random.randn(H1,H2)*0.05;  b2=np.zeros(H2)
W3=np.random.randn(H2,10)*0.05;  b3=np.zeros(10)
lr=0.1; batch=128
def relu(x): return np.maximum(0,x)

print("Training (deep network, 2 hidden layers, ReLU)...")
for epoch in range(25):
    idx=np.random.permutation(len(Xtr))
    for i in range(0,len(Xtr),batch):
        bi=idx[i:i+batch]; xb,yb=Xtr[bi],Ytr[bi]
        z1=xb@W1+b1; a1=relu(z1)
        z2=a1@W2+b2; a2=relu(z2)
        z3=a2@W3+b3
        z3-=z3.max(1,keepdims=True); p=np.exp(z3); p/=p.sum(1,keepdims=True)
        dz3=(p-yb)/len(bi)
        dW3=a2.T@dz3; db3=dz3.sum(0)
        da2=dz3@W3.T; dz2=da2*(z2>0); dW2=a1.T@dz2; db2=dz2.sum(0)
        da1=dz2@W2.T; dz1=da1*(z1>0); dW1=xb.T@dz1; db1=dz1.sum(0)
        W1-=lr*dW1;b1-=lr*db1;W2-=lr*dW2;b2-=lr*db2;W3-=lr*dW3;b3-=lr*db3
    z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
    acc=(np.argmax(z3,axis=1)==yte).mean()
    if epoch%5==0 or epoch==24: print(f"  epoch {epoch}: test acc={acc:.4f}")

z1=Xte@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
print(f"Final DEEP network test accuracy: {(np.argmax(z3,axis=1)==yte).mean():.4f}")

# --- QUANTIZATION ---
S=4096
W1q=np.round(W1*S).astype(np.int32); W2q=np.round(W2*S).astype(np.int32); W3q=np.round(W3*S).astype(np.int32)
b1i=np.round(b1*255*S).astype(np.int64)
print(f"W1q:{W1q.min()}/{W1q.max()} W2q:{W2q.min()}/{W2q.max()} W3q:{W3q.min()}/{W3q.max()}")

# SHIFT calibration: bring every layer output down to 16 bits
s=Xte_raw[0].astype(np.int32)
z1s=s@W1q+b1i; a1s=np.maximum(0,z1s)
SHIFT1=0
while a1s.max()>>SHIFT1 > 32767: SHIFT1+=1
a1_small=np.clip(a1s>>SHIFT1,0,32767)
# bring bias2 to the a1_small scale: a1_small ~ a1_float * (255*S) / 2^SHIFT1
#   in practice b2 is also converted to int with the same scale
# layer-2 bias scale: a1_small*W2q carries 255*S^2/2^SHIFT1, so the bias carries the same scale
b2i=np.round(b2*(255*S*S/(2**SHIFT1))).astype(np.int64)
z2s=a1_small@W2q+b2i; a2s=np.maximum(0,z2s)
SHIFT2=0
while a2s.max()>>SHIFT2 > 32767: SHIFT2+=1
print(f"SHIFT1={SHIFT1}, SHIFT2={SHIFT2}")

# Full quantized inference (Python reference)
def qinfer(img):
    z1=img@W1q+b1i; a1=np.clip(np.maximum(0,z1)>>SHIFT1,0,32767).astype(np.int64)
    z2=a1@W2q+b2i;  a2=np.clip(np.maximum(0,z2)>>SHIFT2,0,32767).astype(np.int64)
    z3=a2@W3q
    return int(np.argmax(z3))

N_test=1000; cq=0; mf=0
for n in range(N_test):
    img=Xte_raw[n].astype(np.int32)
    pq=qinfer(img)
    zz1=(img/255.0)@W1+b1;aa1=relu(zz1);zz2=aa1@W2+b2;aa2=relu(zz2);zz3=aa2@W3+b3
    pf=int(np.argmax(zz3))
    if pq==int(yte[n]): cq+=1
    if pq==pf: mf+=1
print(f"\nQuantized accuracy ({N_test}): {cq}/{N_test}={cq/N_test:.4f}")
print(f"Quantized==Float ({N_test}): {mf}/{N_test}")

np.savetxt('deep_W1.txt',W1q.flatten(),fmt='%d')
np.savetxt('deep_W2.txt',W2q.flatten(),fmt='%d')
np.savetxt('deep_W3.txt',W3q.flatten(),fmt='%d')
np.savetxt('deep_b1.txt',b1i,fmt='%d')
np.savetxt('deep_b2.txt',b2i,fmt='%d')
# output-layer bias at the scale of a2*W3q, 255*S^3/2^(SHIFT1+SHIFT2); not used by the contracts, written for output_bias.py
b3i=np.round(b3*(255*S*S*S/(2**(SHIFT1+SHIFT2)))).astype(np.int64)
np.savetxt('deep_b3.txt',b3i,fmt='%d')
with open('deep_meta.txt','w') as f:
    f.write(f"{S} {SHIFT1} {SHIFT2} {H1} {H2}\n")
print("Files written: deep_W1/W2/W3, deep_b1/b2/b3, deep_meta")
