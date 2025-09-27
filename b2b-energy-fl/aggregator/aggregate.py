import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from web3 import Web3
from hexbytes import HexBytes
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import FedPersonalizedMLP, merge_global_state
from uagents import Agent, Context, Protocol

with open("artifacts/contracts/Coordinator.sol/Coordinator.json", "r") as f:
    ABI_COORD = json.load(f)["abi"]
with open("artifacts/contracts/RewardToken.sol/RewardToken.json", "r") as f:
    ABI_REWARD = json.load(f)["abi"]

# Load keys and addresses from deployed.json automatically
with open("deployed.json", "r") as f:
    deployed = json.load(f)

AGENT_NAME = "aggregator"
AGGREGATOR_PK = deployed.get("owner_pk")
if not AGGREGATOR_PK or len(AGGREGATOR_PK) != 64 or not all(c in '0123456789abcdefABCDEF' for c in AGGREGATOR_PK):
    raise ValueError("AGGREGATOR_PK not found or invalid in deployed.json!")
ROUND_DIR = os.environ.get("ROUND_DIR", "aggregator/round")
ROUND_ID = int(os.environ.get("ROUND_ID", "1"))

agg_agent = Agent(name=AGENT_NAME)

class ChatProtocol(Protocol):
    async def run(self, ctx: Context, sender: str, message: str):
        ctx.logger.info(f"[Chat] {sender}: {message}")
        if "dispute" in message:
            ctx.logger.info(f"Dispute raised by {sender}")

agg_agent.include(ChatProtocol())

def save_global_as_npz(model: nn.Module, path_npz: str):
    sd = model.state_dict()
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    np.savez(path_npz, **arrays)

def load_update_npz(path):
    with np.load(path) as data:
        return {k: torch.tensor(data[k]) for k in data.files}

def unmask(masked_updates, seeds):
    shapes = {k: masked_updates[0][k].shape for k in masked_updates[0]}
    rngs = [np.random.default_rng(seed) for seed in seeds]
    masks = []
    for rng in rngs:
        mask = {k: torch.from_numpy(rng.normal(0, 0.01, size=shapes[k]).astype(np.float32)) for k in shapes}
        masks.append(mask)
    unmasked = []
    for i, upd in enumerate(masked_updates):
        client_unmasked = {}
        for k in upd:
            client_unmasked[k] = upd[k] - masks[i][k]
        unmasked.append(client_unmasked)
    return unmasked

def federated_average(updates):
    keys = updates[0].keys()
    avg = {}
    for k in keys:
        avg[k] = torch.mean(torch.stack([upd[k] for upd in updates]), dim=0)
    return avg

def load_global_model(path_npz):
    if not os.path.exists(path_npz):
        return None
    data = np.load(path_npz, allow_pickle=True)
    state_dict = {}
    for k in data.files:
        state_dict[k] = torch.tensor(data[k], dtype=torch.float32)
    return state_dict

@agg_agent.on_interval(period=60.0)
async def aggregate_round(ctx: Context):
    try:
        COORD_ADDRESS = deployed["coordinator"]
        REWARD_ADDRESS = deployed["reward"]
        RPC_URL = deployed.get("rpc", "http://127.0.0.1:8546")
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        acct = w3.eth.account.from_key(AGGREGATOR_PK)
        coord = w3.eth.contract(address=Web3.to_checksum_address(COORD_ADDRESS), abi=ABI_COORD)
        reward = w3.eth.contract(address=Web3.to_checksum_address(REWARD_ADDRESS), abi=ABI_REWARD)
        coord_owner = coord.functions.owner().call()
        if coord_owner.lower() != acct.address.lower():
            raise SystemExit(f"[Abort] Coordinator.owner() is {coord_owner}, but tx sender will be {acct.address}. Use correct PK.")
        reward_owner = reward.functions.owner().call()
        if reward_owner.lower() != Web3.to_checksum_address(COORD_ADDRESS).lower():
            raise SystemExit("[Abort] RewardToken.owner() is not the Coordinator.")
        os.makedirs(ROUND_DIR, exist_ok=True)
        # Load participants from deployed.json for full automation
        participants = deployed.get("client_addrs", {}).values()
        participants = [Web3.to_checksum_address(addr) for addr in participants]
        seeds = [1, 2, 3]
        masked_updates = []
        for addr in participants:
            path = os.path.join(ROUND_DIR, f"{addr}_update.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Update file for participant {addr} missing: {path}")
            masked = load_update_npz(path)
            masked_updates.append(masked)
        unmasked_updates = unmask(masked_updates, seeds)
        avg_shared = federated_average(unmasked_updates)
        model = FedPersonalizedMLP(in_dim=30)
        global_model_path = os.path.join(ROUND_DIR, "global_model.npz")
        prev_sd = load_global_model(global_model_path)
        if prev_sd:
            model.load_state_dict(prev_sd, strict=False)
        model_sd = model.state_dict()
        merge_global_state(model_sd, avg_shared)
        model.load_state_dict(model_sd)
        save_global_as_npz(model, global_model_path)
        # ===== Model Evaluation Block (with Debug Prints) =====
        eval_X_path = "client/data/global_test/X.npy"
        eval_y_path = "client/data/global_test/y.npy"
        if os.path.exists(eval_X_path) and os.path.exists(eval_y_path):
            X_test = np.load(eval_X_path)
            y_test = np.load(eval_y_path)
            model.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X_test.astype(np.float32))
                outputs = model(X_t)
                probs = torch.sigmoid(outputs).numpy().ravel()
                preds = (probs > 0.5).astype(int)
                print("Sample prediction probabilities:", probs[:10])
                print("Sample predicted labels:", preds[:10])
                print("Sample true labels:", y_test[:10])
                acc = (preds == y_test).mean()
            print(f"Global evaluation accuracy this round: {acc:.4f}")
        else:
            print("Warning: global test set files not found, skipping evaluation.")
        with open(global_model_path, "rb") as f:
            model_bytes = f.read()
        model_hash_b32 = HexBytes(w3.keccak(model_bytes))
        tx = coord.functions.finalizeRound(
            model_hash_b32,
            participants,
            100
        ).build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "maxFeePerGas": w3.to_wei("2", "gwei"),
            "maxPriorityFeePerGas": w3.to_wei("1", "gwei"),
            "gas": 3_000_000,
            "chainId": w3.eth.chain_id,
        })
        signed = w3.eth.account.sign_transaction(tx, AGGREGATOR_PK)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"🎉 Round {ROUND_ID} finalized with global hash: {model_hash_b32.hex()}")
        print(f"🔗 Transaction hash: {tx_hash.hex()}")
    except Exception as e:
        print(f"Error in aggregation: {e}")
        raise

if __name__ == "__main__":
    agg_agent.run()
