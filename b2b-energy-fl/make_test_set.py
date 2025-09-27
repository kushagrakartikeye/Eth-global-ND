import numpy as np
from sklearn.model_selection import train_test_split
import os

dirs = ["client/data/client1", "client/data/client2", "client/data/client3"]
Xs, ys = [], []

for d in dirs:
    Xs.append(np.load(f"{d}/X.npy"))
    ys.append(np.load(f"{d}/y.npy"))

X_all = np.concatenate(Xs, axis=0)
y_all = np.concatenate(ys, axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

os.makedirs("client/data/global_test/", exist_ok=True)
np.save("client/data/global_test/X.npy", X_test)
np.save("client/data/global_test/y.npy", y_test)

print(f"Saved test set X shape: {X_test.shape}, y shape: {y_test.shape}")
