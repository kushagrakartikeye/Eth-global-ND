import os
import json
import socket
from pydantic import BaseModel
from uagents import Agent, Context, Protocol

AGENT_NAME = os.environ.get("AGENT_NAME", "client1")
AGENT_PORT = os.environ.get("AGENT_PORT", "8001")
COORDINATOR_ADDRESS = os.environ.get("COORDINATOR_ADDRESS", "http://127.0.0.1:8005")

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

ENDPOINT = f"http://{get_local_ip()}:{AGENT_PORT}"

# 👇 Explicit endpoint
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
    client_agent.run()
