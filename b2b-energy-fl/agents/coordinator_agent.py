from uagents import Agent, Context

coord_agent = Agent(
    name="coordinator",
    seed="coordinator_seed",
    endpoint=["http://127.0.0.1:8005"]  # Force to port 8005
)

@coord_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info("Coordinator agent started on port 8005")

if __name__ == "__main__":
    coord_agent.run()
