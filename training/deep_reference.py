import numpy as np, struct
def read_images(path):
    with open(path,'rb') as f:
        m,n,r,c=struct.unpack('>IIII',f.read(16))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r*c)
def read_labels(path):
    with open(path,'rb') as f:
        m,n=struct.unpack('>II',f.read(8)); return np.frombuffer(f.read(),dtype=np.uint8)

Xtr_raw=read_images('mnist_data/train-images-idx3-ubyte'); Xtr=Xtr_raw.astype(np.float64)/255.0
ytr=read_labels('mnist_data/train-labels-idx1-ubyte')
Xte_raw=read_images('mnist_data/t10k-images-idx3-ubyte'); Xte=Xte_raw.astype(np.float64)/255.0
yte=read_labels('mnist_data/t10k-labels-idx1-ubyte')
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

# FLOAT predictions of the deep network (reference) - 500 images
N=500
with open('deep_ref.txt','w') as f:
    f.write(f"{N}\n")
    for n in range(N):
        img=Xte_raw[n].astype(np.float64)/255.0
        z1=img@W1+b1;a1=relu(z1);z2=a1@W2+b2;a2=relu(z2);z3=a2@W3+b3
        pred_f=int(np.argmax(z3))
        f.write(f"{int(yte[n])} {pred_f}\n")
print(f"deep_ref.txt written ({N} images, float predictions of the deep network)")
