import subprocess
import sys
import os
import json

python_exec = sys.executable

# --- Load deployed info ---
with open("deployed.json", "r") as f:
    deployed = json.load(f)

coord_addr = deployed.get("coordinator")
owner_addr = deployed.get("owner")  # used to identify aggregator
# You need to set owner_pk manually if submitting transactions
owner_pk = deployed.get("owner_pk", "")  # optional, fill in if needed

# --- Define clients ---
clients = ["client1", "client2", "client3"]

def run_client(client_name, round_id, round_dir):
    cmd = [
        python_exec,
        "client/client.py",
        "--client", client_name,
        "--round_dir", round_dir
    ]
    print(f"[SIM] Running simulated client {client_name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Client {client_name} failed: {e}")
        raise

def run_aggregator(round_id, round_dir):
    cmd = [
        python_exec,
        "aggregator/aggregate.py",
        "--coord", coord_addr,
        "--pk", owner_pk,
        "--round_dir", round_dir,
        "--round_id", str(round_id)
    ]
    print(f"[SIM] Running aggregator for round {round_id}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Aggregator failed: {e}")
        raise

def run_rounds(num_rounds=3):
    for r in range(1, num_rounds + 1):
        round_dir = f"aggregator/round_{r}"
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n=== Starting Round {r} ===")
        # --- Launch all clients ---
        for client in clients:
            run_client(client, r, round_dir)
        # --- Run aggregator after all clients ---
        run_aggregator(r, round_dir)

if __name__ == "__main__":
    run_rounds(num_rounds=3)
