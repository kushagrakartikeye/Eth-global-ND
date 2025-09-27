import os
from uagents import Agent, Context

AGGREGATOR_PORT = os.environ.get("AGGREGATOR_PORT", "8004")

agg_agent = Agent(
    name="aggregator",
    seed="aggregator_seed",
    endpoint=[f"http://127.0.0.1:{AGGREGATOR_PORT}"]  # Force to port 8004
)

@agg_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Aggregator agent started on port {AGGREGATOR_PORT}")

if __name__ == "__main__":
    agg_agent.run()
