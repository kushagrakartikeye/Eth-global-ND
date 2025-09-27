# simulate_agents.py
"""
Simulate agents in-process to avoid uagents port issues.
- Reads deployed.json for coordinator address + keys.
- For each simulated client, either calls a python function (preferred)
  or runs client/client.py as a subprocess with args (fallback).
- Submits transactions to on-chain coordinator exactly as real agents would.
- Then runs aggregator (either via function or subprocess).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOYED_PATH = ROOT / "deployed.json"

# clients: mapping name -> pk (same as before)
CLIENTS = [
    ("client1", "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"),
    ("client2", "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"),
    ("client3", "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"),
]

PYTHON = sys.executable

def load_deployed():
    if not DEPLOYED_PATH.exists():
        raise RuntimeError("deployed.json not found. Run deploy script first.")
    return json.loads(DEPLOYED_PATH.read_text())

def run_client_subprocess(name, pk, coord_addr, round_dir, rnd):
    """
    Fallback: run client/client.py as a subprocess with the same CLI you used before.
    """
    cmd = [
        PYTHON, str(ROOT / "client" / "client.py"),
        "--client", name,
        "--coord", coord_addr,
        "--pk", pk,
        "--round_dir", round_dir,
        "--round", str(rnd)
    ]
    print("Running subprocess:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_aggregator_subprocess(coord_addr, owner_pk, round_dir, rnd):
    cmd = [
        PYTHON, str(ROOT / "aggregator" / "aggregate.py"),
        "--coord", coord_addr,
        "--pk", owner_pk,
        "--round_dir", round_dir,
        "--round_id", str(rnd)
    ]
    print("Running aggregator subprocess:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def simulate_round(deployed, rnd):
    coord_addr = deployed.get("coordinator")
    owner_pk = deployed.get("owner", None)
    if not coord_addr:
        raise RuntimeError("coordinator address missing in deployed.json")

    round_dir = ROOT / "aggregator" / f"round_{rnd}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # For each client, attempt to run client training (subprocess fallback)
    for name, pk in CLIENTS:
        print(f"\n[SIM] Running simulated client {name} (round {rnd})")
        try:
            # Prefer subprocess approach to keep your existing client code untouched:
            run_client_subprocess(name, pk, coord_addr, str(round_dir), rnd)
        except subprocess.CalledProcessError as e:
            print(f"[SIM] client {name} subprocess failed: {e}. Continuing with next client.")

    # Wait a short moment for txs to be mined (ganache/hardhat local)
    time.sleep(1.0)

    # Run aggregator to produce new global model
    print(f"\n[SIM] Running aggregator for round {rnd}")
    if owner_pk:
        try:
            run_aggregator_subprocess(coord_addr, owner_pk, str(round_dir), rnd)
        except subprocess.CalledProcessError as e:
            print(f"[SIM] Aggregator subprocess failed: {e}")
    else:
        print("[SIM] Owner pk missing in deployed.json; aggregator will be run without signing on-chain finalize step.")

def main(rounds=3):
    deployed = load_deployed()
    for rnd in range(1, rounds + 1):
        print(f"\n=== SIMULATED ROUND {rnd} ===")
        simulate_round(deployed, rnd)
        print(f"=== SIMULATED ROUND {rnd} COMPLETE ===")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    main(rounds=args.rounds)
