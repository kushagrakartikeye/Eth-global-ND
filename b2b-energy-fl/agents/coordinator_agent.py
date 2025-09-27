import os
import json
from pydantic import BaseModel
from uagents import Agent, Context, Protocol

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "agent_registry.json")

class RegisterMessage(BaseModel):
    type: str
    agent_name: str
    eth_address: str
    endpoint: str

class Registry:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            with open(path, "r") as f:
                self.agents = json.load(f)
        else:
            self.agents = {}

    def register(self, agent_name, address, endpoint):
        self.agents[agent_name] = {
            "address": address,
            "endpoint": endpoint
        }
        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.agents, f, indent=2)

    def list_agents(self):
        return self.agents

class CoordinatorProtocol(Protocol):
    async def run(self, ctx: Context, sender: str, message):
        if isinstance(message, RegisterMessage) and message.type == "register":
            ctx.logger.info(f"Registering agent: {message.agent_name} addr: {message.eth_address} endpoint: {message.endpoint}")
            ctx.registry.register(message.agent_name, message.eth_address, message.endpoint)
            await ctx.send(sender, {"response": f"registered:{message.agent_name}"})

class CoordinatorAgent(Agent):
    def __init__(self):
        super().__init__(name="coordinator")
        self.registry = Registry(REGISTRY_PATH)
        self.include(CoordinatorProtocol())

if __name__ == "__main__":
    coord_agent = CoordinatorAgent()
    coord_agent.run()
