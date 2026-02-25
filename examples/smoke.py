"""Smoke test — run with real API keys to verify end-to-end."""

import asyncio

from pydantic import BaseModel

from llmkit import Agent, Anthropic, OpenAI


class City(BaseModel):
    name: str
    country: str
    population: int


async def main() -> None:
    # Test basic send
    agent = Agent(OpenAI.GPT_4O_MINI, system="Be concise.")
    reply = await agent.send("What is 2+2?")
    print(f"OpenAI: {reply.text}")
    print(f"Usage: {reply.usage}")

    # Test structured output
    agent2 = Agent(Anthropic.CLAUDE_SONNET)
    reply2 = await agent2.send("Tell me about Amsterdam", response_model=City)
    print(f"Anthropic structured: {reply2.parsed}")

    # Test single tool call
    agent3 = Agent(OpenAI.GPT_4O_MINI)

    @agent3.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    reply3 = await agent3.send("What is 7 times 8?")
    print(f"Tool result: {reply3.text}")

    # Test parallel tool calls
    agent4 = Agent(OpenAI.GPT_4O_MINI)

    @agent4.tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @agent4.tool
    def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    reply4 = await agent4.send(
        "Calculate both of these: 15 + 27, and 100 - 37. Use the tools for both."
    )
    print(f"Parallel tools: {reply4.text}")

    # Test multi-turn
    agent5 = Agent(Anthropic.CLAUDE_SONNET, system="Be concise.")
    await agent5.send("My name is Job")
    reply5 = await agent5.send("What is my name?")
    print(f"Multi-turn: {reply5.text}")


asyncio.run(main())
