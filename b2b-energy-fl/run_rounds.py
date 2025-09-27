#!/usr/bin/env python3
import subprocess
import os
import json
import sys
from web3 import Web3
from eth_account import Account

# Hardhat default local deployer private key (safe only for local dev)
DEFAULT_HARDHAT_OWNER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

def load_deployed():
    with open("deployed.json", "r") as f:
        return json.load(f)

def run_client(python_exe, name, pk, coord_addr, round_dir, round_no):
    try:
        subprocess.run([
            python_exe, "client/client.py",
            "--client", name,
            "--coord", coord_addr,
            "--pk", pk,
            "--round_dir", round_dir,
            "--round", str(round_no)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}: {e}")

def run_aggregator(python_exe, coord_addr, owner_pk, round_dir, round_no):
    try:
        subprocess.run([
            python_exe, "aggregator/aggregate.py",
            "--coord", coord_addr,
            "--pk", owner_pk,
            "--round_dir", round_dir,
            "--round_id", str(round_no)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running aggregator: {e}")

def start_round(w3, coord_addr, owner_key, round_no):
    # Load Coordinator ABI
    with open("artifacts/contracts/Coordinator.sol/Coordinator.json", "r") as f:
        abi = json.load(f)["abi"]

    coord = w3.eth.contract(address=Web3.to_checksum_address(coord_addr), abi=abi)
    acct = w3.eth.account.from_key(owner_key)

    tx = coord.functions.startRound().build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        # EIP-1559-friendly fields are optional; keep simple for local
        "gas": 300_000,
        "chainId": w3.eth.chain_id,
    })
    signed = w3.eth.account.sign_transaction(tx, owner_key)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Round {round_no} started, tx: {tx_hash.hex()}")

def main():
    deployed = load_deployed()
    coord_addr = deployed["coordinator"]
    rpc_url = deployed.get("rpc", "http://127.0.0.1:8546")

    # Determine owner private key
    owner_pk = deployed.get("ownerPrivateKey") or os.environ.get("OWNER_PK") or DEFAULT_HARDHAT_OWNER_PK
    if not owner_pk:
        raise SystemExit("[Fatal] No owner private key found. Provide OWNER_PK env or add to deployed.json")

    # three client keys (Hardhat dev accounts)
    clients = [
        ("client1", "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"),
        ("client2", "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"),
        ("client3", "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"),
    ]

    # Use the venv python to run subprocesses
    python_exe = sys.executable

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"[Warning] Could not connect to RPC {rpc_url} — make sure a node is running.")
    else:
        print(f"Connected to RPC {rpc_url} (chainId: {w3.eth.chain_id})")

    for rnd in range(1, 4):
        print(f"\n=== Starting Round {rnd} ===")

        # Start the round on-chain (owner)
        try:
            start_round(w3, coord_addr, owner_pk, rnd)
        except Exception as e:
            print(f"[Warning] Could not start round on-chain: {e}")
            # continue anyway: some flows may not require startRound if contract logic differs

        round_dir = os.path.join("aggregator", f"round_{rnd}")
        os.makedirs(round_dir, exist_ok=True)

        for name, pk in clients:
            print(f"Running client: {name} (round {rnd})")
            run_client(python_exe, name, pk, coord_addr, round_dir, rnd)

        print(f"Running aggregator (round {rnd}).")
        run_aggregator(python_exe, coord_addr, owner_pk, round_dir, rnd)

        print(f"=== Round {rnd} complete ===")

if __name__ == "__main__":
    main()
