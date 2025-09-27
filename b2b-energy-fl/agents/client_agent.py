import os
import json
import socket
import argparse
from pydantic import BaseModel
from uagents import Agent, Context, Protocol

# --- Argparse for simulate_agents.py ---
parser = argparse.ArgumentParser()
parser.add_argument("--client", type=str, required=True, help="Client name (e.g. client1)")
parser.add_argument("--coord", type=str, required=True, help="Coordinator contract address")
parser.add_argument("--pk", type=str, required=True, help="Private key")
parser.add_argument("--round_dir", type=str, required=True, help="Round directory path")
parser.add_argument("--round", type=int, required=True, help="Round ID")
parser.add_argument("--port", type=int, default=8010, help="Port for this client agent")
args = parser.parse_args()

AGENT_NAME = args.client
AGENT_PORT = args.port
COORDINATOR_ADDRESS = args.coord

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP

ENDPOINT = f"http://{get_local_ip()}:{AGENT_PORT}"

# Explicit endpoint
client_agent = Agent(name=AGENT_NAME, endpoint=[ENDPOINT])

class RegisterMessage(BaseModel):
    type: str = "register"
    agent_name: str
    eth_address: str
    endpoint: str

async def register_with_coordinator(ctx: Context):
    eth_address = ""
    try:
        with open(os.path.join(os.path.dirname(__file__), "../deployed.json")) as f:
            deployed = json.load(f)
            eth_address = deployed["client_addrs"].get(AGENT_NAME, "")
    except Exception:
        pass
    msg = RegisterMessage(agent_name=AGENT_NAME, eth_address=eth_address, endpoint=ENDPOINT)
    await ctx.send(COORDINATOR_ADDRESS, msg)
    ctx.logger.info(f"Registered {AGENT_NAME} with coordinator at {COORDINATOR_ADDRESS}")

@client_agent.on_event("startup")
async def on_startup(ctx: Context):
    await register_with_coordinator(ctx)

class ClientChatProtocol(Protocol):
    async def run(self, ctx: Context, sender: str, message):
        ctx.logger.info(f"[Chat] {sender}: {message}")

client_agent.include(ClientChatProtocol())

if __name__ == "__main__":
    os.environ["PORT"] = str(AGENT_PORT)   # <- Correct place
    client_agent.run()
