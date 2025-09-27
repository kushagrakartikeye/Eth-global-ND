import numpy as np
X = np.load("client/data/global_test/X.npy", allow_pickle=True)
print(type(X))
print(X.dtype)
print(X.shape)
y = np.load("client/data/global_test/y.npy", allow_pickle=True)
print(type(y))
print(y.dtype)
print(y.shape)