import os
import argparse
import numpy as np
import torch
import json
from web3 import Web3
from uagents import Agent, Context

with open("artifacts/contracts/Coordinator.sol/Coordinator.json", "r") as f:
    ABI_COORD = json.load(f)["abi"]

parser = argparse.ArgumentParser()
parser.add_argument("--coord", type=str, required=True)
parser.add_argument("--pk", type=str, required=True)
parser.add_argument("--round_dir", type=str, required=True)
parser.add_argument("--round_id", type=int, required=True)
parser.add_argument("--port", type=int, default=8050)
args = parser.parse_args()

RPC_URL = "http://127.0.0.1:8546"

aggregator_agent = Agent(name="aggregator")

@aggregator_agent.on_interval(period=30.0)
async def aggregate_round(ctx: Context):
    try:
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        acc = w3.eth.account.from_key(args.pk)
        coord = w3.eth.contract(address=Web3.to_checksum_address(args.coord), abi=ABI_COORD)

        participants = coord.functions.getParticipants().call()
        updates = []
        for addr in participants:
            path = os.path.join(args.round_dir, f"{addr}_update.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Update file for participant {addr} missing: {path}")
            data = np.load(path, allow_pickle=True)
            update = {k: torch.tensor(data[k]) for k in data.files}
            updates.append(update)

        avg_update = {}
        for k in updates[0]:
            avg_update[k] = torch.mean(torch.stack([u[k] for u in updates]), dim=0)

        save_path = os.path.join(args.round_dir, "global_model.npz")
        np.savez(save_path, **{k: v.numpy() for k, v in avg_update.items()})
        print(f"✅ Aggregated round {args.round_id}, saved model to {save_path}")
    except Exception as e:
        print(f"Error in aggregation: {e}")
        raise

if __name__ == "__main__":
    os.environ["PORT"] = str(args.port)
    aggregator_agent.run()
