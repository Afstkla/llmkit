"""Smoke test — run with real API keys to verify end-to-end."""

import asyncio

from pydantic import BaseModel

from llmkit import Anthropic, Chat, OpenAI


class City(BaseModel):
    name: str
    country: str
    population: int


async def main():
    # Test basic send
    chat = Chat(OpenAI.GPT_4O_MINI, system="Be concise.")
    reply = await chat.send("What is 2+2?")
    print(f"OpenAI: {reply.text}")
    print(f"Usage: {reply.usage}")

    # Test structured output
    chat2 = Chat(Anthropic.CLAUDE_SONNET)
    reply2 = await chat2.send("Tell me about Amsterdam", response_model=City)
    print(f"Anthropic structured: {reply2.parsed}")

    # Test tools
    chat3 = Chat(OpenAI.GPT_4O_MINI)

    @chat3.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    reply3 = await chat3.send("What is 7 times 8?")
    print(f"Tool result: {reply3.text}")

    # Test multi-turn
    chat4 = Chat(Anthropic.CLAUDE_SONNET, system="Be concise.")
    await chat4.send("My name is Job")
    reply4 = await chat4.send("What is my name?")
    print(f"Multi-turn: {reply4.text}")


asyncio.run(main())
