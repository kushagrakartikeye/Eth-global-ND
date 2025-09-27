import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from web3 import Web3
from eth_hash.auto import keccak
import json
import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import FedPersonalizedMLP  # personalized model
from uagents import Agent, Context, Protocol

with open("artifacts/contracts/Coordinator.sol/Coordinator.json", "r") as f:
    ABI_COORD = json.load(f)["abi"]

def keccak256(data: bytes) -> str:
    return "0x" + keccak(data).hex()

def safe_compute_delta(new_w, global_w):
    delta = {}
    for k in new_w:
        if k.startswith("shared"):
            if global_w is None or k not in global_w:
                delta[k] = new_w[k].clone()
            else:
                delta[k] = new_w[k] - global_w[k]
    return delta

def train_local(X, y, global_w, epochs=10, lr=1e-3):
    model = FedPersonalizedMLP(in_dim=X.shape[1])
    if global_w is not None:
        sd = model.state_dict()
        for k in global_w:
            if k in sd:
                sd[k] = global_w[k]
        model.load_state_dict(sd)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch + 1}/{epochs} loss: {loss.item():.6f}")
    new_w = model.state_dict()
    delta_shared = safe_compute_delta(new_w, global_w)
    delta_norm = sum(torch.norm(v).item() for v in delta_shared.values())
    print(f"Delta norm: {delta_norm:.6f}")
    return delta_shared

def load_global_model(path_npz):
    if not os.path.exists(path_npz):
        return None
    data = np.load(path_npz, allow_pickle=True)
    state_dict = {}
    for k in data.files:
        if k.startswith("shared"):
            state_dict[k] = torch.tensor(data[k], dtype=torch.float32)
    return state_dict

def save_masked_delta(masked_delta, path_npz):
    np.savez(path_npz, **masked_delta)

# --- Parse arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--client", type=str, default="client1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--round_dir", type=str, default="aggregator/round")
args = parser.parse_args()

AGENT_NAME = args.client
ROUND_DIR = args.round_dir

with open("deployed.json", "r") as f:
    deployed = json.load(f)

client_keys = deployed.get("client_keys", {})
client_addrs = deployed.get("client_addrs", {})
CLIENT_PK = client_keys.get(AGENT_NAME)
CLIENT_ADDR = client_addrs.get(AGENT_NAME)

if not CLIENT_PK or len(CLIENT_PK) != 64 or not all(c in '0123456789abcdefABCDEF' for c in CLIENT_PK):
    raise ValueError(f"CLIENT_PK for {AGENT_NAME} not found or invalid in deployed.json!")

client_agent = Agent(name=AGENT_NAME)

class ChatProtocol(Protocol):
    async def run(self, ctx: Context, sender: str, message: str):
        ctx.logger.info(f"[Chat] {sender}: {message}")
        if "dispute" in message:
            ctx.logger.info(f"Dispute raised by {sender}")

client_agent.include(ChatProtocol())

@client_agent.on_interval(period=30.0)
async def participate_round(ctx: Context):
    try:
        RPC_URL = deployed.get("rpc", "http://127.0.0.1:8546")
        COORD_ADDRESS = deployed["coordinator"]
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        acc = w3.eth.account.from_key(CLIENT_PK)
        coord = w3.eth.contract(address=Web3.to_checksum_address(COORD_ADDRESS), abi=ABI_COORD)
        current_round = coord.functions.currentRound().call()
        already_submitted = coord.functions.submitted(current_round, acc.address).call()
        if already_submitted:
            print(f"{AGENT_NAME} has already submitted for round {current_round}, skipping.")
            return
        os.makedirs(ROUND_DIR, exist_ok=True)
        X = np.load(f"client/data/{AGENT_NAME}/X.npy")
        y = np.load(f"client/data/{AGENT_NAME}/y.npy")
        print(f"{AGENT_NAME} dataset size: {X.shape[0]}")
        print(f"{AGENT_NAME} positive label ratio: {np.mean(y):.6f}")
        global_model_path = os.path.join(ROUND_DIR, "global_model.npz")
        global_shared_weights = load_global_model(global_model_path)
        delta = train_local(X, y, global_shared_weights)
        shape_map = {k: tuple(delta[k].shape) for k in delta}
        rng = np.random.default_rng(abs(hash(AGENT_NAME)) % (2 ** 32))
        mask = {k: rng.normal(0, 0.01, size=shape_map[k]).astype(np.float32) for k in shape_map}
        masked_delta = {k: delta[k].cpu().numpy().astype(np.float32) + mask[k] for k in delta}
        update_path = os.path.join(ROUND_DIR, f"{acc.address}_update.npz")
        save_masked_delta(masked_delta, update_path)
        with open(update_path, "rb") as f:
            file_bytes = f.read()
        update_hash = keccak256(file_bytes)
        nonce = w3.eth.get_transaction_count(acc.address)
        tx = coord.functions.submitUpdate(update_hash).build_transaction({
            "from": acc.address,
            "nonce": nonce,
            "maxFeePerGas": w3.to_wei("2", "gwei"),
            "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
            "gas": 2_000_000,
            "chainId": w3.eth.chain_id,
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=CLIENT_PK)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        _ = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"{AGENT_NAME} submitted update hash: {update_hash}")
        print(f"Transaction hash: {tx_hash.hex()}")
    except Exception as e:
        print(f"Error submitting update from {AGENT_NAME}: {e}")
        raise

if __name__ == "__main__":
    client_agent.run(host="0.0.0.0", port=args.port)
