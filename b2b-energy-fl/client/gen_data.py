<<<<<<< HEAD
=======
<<<<<<< HEAD
import numpy as np
import os

def make_ds(seed, n=1000):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, size=n)
    demand = rng.normal(50, 10, size=n)
    gen = rng.normal(20, 5, size=n)
    spot = rng.normal(80, 15, size=n)
    y = spot + 0.2 * (demand - gen) + 0.1 * hour + rng.normal(0, 2, size=n)
    X = np.stack([hour, demand, gen, spot], axis=1).astype(np.float32)
    y = y.astype(np.float32)
    return X, y

if __name__ == "__main__":
    clients = [("client1", 1), ("client2", 2), ("client3", 3)]
    for name, seed in clients:
        X, y = make_ds(seed)
        os.makedirs(f"client/data/{name}", exist_ok=True)
        np.save(f"client/data/{name}/X.npy", X)
        np.save(f"client/data/{name}/y.npy", y)
=======
>>>>>>> 8d64af2 (replaced troublesome uagents with simulation)
import numpy as np
import os

def make_ds(seed, n=1000):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, size=n)
    demand = rng.normal(50, 10, size=n)
    gen = rng.normal(20, 5, size=n)
    spot = rng.normal(80, 15, size=n)
    y = spot + 0.2 * (demand - gen) + 0.1 * hour + rng.normal(0, 2, size=n)
    X = np.stack([hour, demand, gen, spot], axis=1).astype(np.float32)
    y = y.astype(np.float32)
    return X, y

if __name__ == "__main__":
    clients = [("client1", 1), ("client2", 2), ("client3", 3)]
    for name, seed in clients:
        X, y = make_ds(seed)
        os.makedirs(f"client/data/{name}", exist_ok=True)
        np.save(f"client/data/{name}/X.npy", X)
        np.save(f"client/data/{name}/y.npy", y)
<<<<<<< HEAD
=======
>>>>>>> 5b9db2d (agent networking sample)
>>>>>>> 8d64af2 (replaced troublesome uagents with simulation)
    print("Synthetic client datasets created.")