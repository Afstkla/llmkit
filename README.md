# llmkit

[![CI](https://github.com/Afstkla/llmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Afstkla/llmkit/actions/workflows/ci.yml)
[![Docs](https://github.com/Afstkla/llmkit/actions/workflows/docs.yml/badge.svg)](https://afstkla.github.io/llmkit/)

Minimal, typed Python LLM wrapper. One `Agent` object, multiple providers, no boilerplate.

Supports **OpenAI**, **Anthropic**, and **Gemini**.

## Install

```bash
uv add llmkit[all]         # all providers
uv add llmkit[anthropic]   # just anthropic
uv add llmkit[openai]      # just openai
uv add llmkit[gemini]      # just gemini
```

Requires Python 3.14+.

## Quick Start

```python
import asyncio
from llmkit import Agent, Anthropic

async def main():
    agent = Agent(Anthropic.CLAUDE_SONNET, system="Be concise.")
    reply = await agent.send("What is 2+2?")
    print(reply.text)      # "4"
    print(reply.usage)     # Usage(input_tokens=..., output_tokens=...)

asyncio.run(main())
```

## Multi-turn Conversations

State is managed automatically:

```python
agent = Agent("openai/gpt-4o", system="You are helpful.")
await agent.send("My name is Job")
reply = await agent.send("What is my name?")
# reply.text -> "Job"
```

## Structured Output

Pass a Pydantic model, get a validated instance back:

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str
    population: int

reply = await agent.send("Tell me about Amsterdam", response_model=City)
reply.parsed  # City(name="Amsterdam", country="Netherlands", population=...)
```

## Tools

Register tools with a decorator — schema is extracted from type hints and docstrings:

```python
agent = Agent("openai/gpt-4o")

@agent.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

reply = await agent.send("What is 7 times 8?")
# Tool is called automatically, result fed back to the model
```

Async tools and programmatic registration:

```python
@agent.tool
async def fetch(url: str) -> str:
    """Fetch a URL."""
    return await aiohttp_get(url)

# Programmatic
agent.tools.register(some_function)
agent.tools.register(fn, name="custom", description="Override docstring")
agent.tools.unregister("fetch")
```

## Hosted Tools

Use provider-hosted tools like web search:

```python
from llmkit import Agent, Anthropic, WebSearch

agent = Agent(Anthropic.CLAUDE_SONNET, hosted_tools=[WebSearch()])
reply = await agent.send("What happened in the news today?")
```

Works with OpenAI, Anthropic, and Gemini.

## Agent-as-Tool

Turn an agent into a tool another agent can call:

```python
researcher = Agent("anthropic/claude-sonnet-4-20250514", system="You research topics thoroughly.")
writer = Agent("openai/gpt-4o", system="You write clear summaries.")

writer.tools.register(researcher.as_tool(name="research", description="Research a topic"))
reply = await writer.send("Write a summary about quantum computing")
```

## Hooks

Tap into the agent lifecycle with `@agent.on()`:

```python
from llmkit.types import Message, Reply

@agent.on("turn_start")
def on_turn(messages: list[Message]) -> None:
    print(f"Sending {len(messages)} messages...")

@agent.on("tool_call_start")
def on_tool(name: str, args: dict[str, Any]) -> None:
    print(f"Calling {name}({args})")
```

Events: `turn_start`, `turn_end`, `tool_call_start`, `tool_call_end`. Both sync and async handlers work.

## Streaming

```python
async for chunk in agent.stream("Write me a story"):
    print(chunk.text, end="")
```

## Sync Convenience

```python
reply = agent.send_sync("Quick question")
```

## Model Enums

Autocomplete-friendly model selection:

```python
from llmkit import Agent, OpenAI, Anthropic, Gemini

Agent(OpenAI.GPT_4O)
Agent(OpenAI.GPT_4O_MINI)
Agent(Anthropic.CLAUDE_OPUS)
Agent(Anthropic.CLAUDE_SONNET)
Agent(Gemini.GEMINI_2_5_PRO)

# Raw strings still work
Agent("openai/gpt-4o")
```

## Auth

API keys are read from environment variables by default:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Override per instance:

```python
agent = Agent("openai/gpt-4o", api_key="sk-...")
```

## Custom Providers

```python
from llmkit import Agent, register_provider

register_provider("ollama", OllamaProvider)
agent = Agent("ollama/llama3")
```

## Contributing

```bash
git clone git@github.com:Afstkla/llmkit.git
cd llmkit
uv sync --all-extras --all-groups
```

Run checks:

```bash
uv run ruff check src/ tests/    # lint
uv run ty check src/              # type check
uv run pytest -v                  # tests
```

All three must pass before submitting a PR.

## License

MIT
