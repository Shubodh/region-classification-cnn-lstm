import numpy as np

a = np.load("./feature_vect_regionclass2.npy", allow_pickle=True)
b = []
n = a.shape
for i in range(n[0] - 1):
    b.append(a[i])

c = np.array(b)
print c.shape
c = c.reshape(-1,512)
print c.shape
#np.save("feature_vectttt",d)
