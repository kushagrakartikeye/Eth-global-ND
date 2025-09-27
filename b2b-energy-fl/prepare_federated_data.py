import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Download and place 'wdbc.data' in this directory. Alternatively, adapt to another dataset.
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    header=None
)
X = df.iloc[:, 2:].values  # features: columns 2 onward
y = df.iloc[:, 1].values   # labels: M/B

le = LabelEncoder()
y = le.fit_transform(y)  # M=1, B=0

# Standardize features
X = StandardScaler().fit_transform(X)

# Split for 3 clients, ~60/20/20 for testability
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_c1, X_c2, y_c1, y_c2 = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp)
X_c3, y_c3 = X_test, y_test

clients = {"client1": (X_c1, y_c1), "client2": (X_c2, y_c2), "client3": (X_c3, y_c3)}

for client, (cx, cy) in clients.items():
    out_dir = f"client/data/{client}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), cx)
    np.save(os.path.join(out_dir, "y.npy"), cy)
print("Data prepared for 3 clients.")
