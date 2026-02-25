# llmkit

[![CI](https://github.com/Afstkla/llmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/Afstkla/llmkit/actions/workflows/ci.yml)
[![Docs](https://github.com/Afstkla/llmkit/actions/workflows/docs.yml/badge.svg)](https://afstkla.github.io/llmkit/)

Minimal, typed Python LLM wrapper. One `Chat` object, multiple providers, no boilerplate.

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
from llmkit import Chat, Anthropic

async def main():
    chat = Chat(Anthropic.CLAUDE_SONNET, system="Be concise.")
    reply = await chat.send("What is 2+2?")
    print(reply.text)      # "4"
    print(reply.usage)     # Usage(input_tokens=..., output_tokens=...)

asyncio.run(main())
```

## Multi-turn Conversations

State is managed automatically:

```python
chat = Chat("openai/gpt-4o", system="You are helpful.")
await chat.send("My name is Job")
reply = await chat.send("What is my name?")
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

reply = await chat.send("Tell me about Amsterdam", response_model=City)
reply.parsed  # City(name="Amsterdam", country="Netherlands", population=...)
```

## Tools

Register tools with a decorator — schema is extracted from type hints and docstrings:

```python
chat = Chat("openai/gpt-4o")

@chat.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

reply = await chat.send("What is 7 times 8?")
# Tool is called automatically, result fed back to the model
```

Async tools and programmatic registration:

```python
@chat.tool
async def fetch(url: str) -> str:
    """Fetch a URL."""
    return await aiohttp_get(url)

# Programmatic
chat.tools.register(some_function)
chat.tools.register(fn, name="custom", description="Override docstring")
chat.tools.unregister("fetch")
```

## Streaming

```python
async for chunk in chat.stream("Write me a story"):
    print(chunk.text, end="")
```

## Sync Convenience

```python
reply = chat.send_sync("Quick question")
```

## Model Enums

Autocomplete-friendly model selection:

```python
from llmkit import OpenAI, Anthropic, Gemini

Chat(OpenAI.GPT_4O)
Chat(OpenAI.GPT_4O_MINI)
Chat(Anthropic.CLAUDE_OPUS)
Chat(Anthropic.CLAUDE_SONNET)
Chat(Gemini.GEMINI_2_5_PRO)

# Raw strings still work
Chat("openai/gpt-4o")
```

## Auth

API keys are read from environment variables by default:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Override per instance:

```python
chat = Chat("openai/gpt-4o", api_key="sk-...")
```

## Custom Providers

```python
from llmkit import register_provider

register_provider("ollama", OllamaProvider)
chat = Chat("ollama/llama3")
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
