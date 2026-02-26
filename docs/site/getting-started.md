# Getting Started

## Install

```bash
uv add llmkit[anthropic]
```

Or pick your provider: `openai`, `anthropic`, `gemini`, `azure`, `bedrock`, `vertex`, or `all`.

## Basic usage

```python
import asyncio
from llmkit import Agent, OpenAI

async def main():
    agent = Agent(OpenAI.GPT_4O_MINI, system="Be concise.")
    reply = await agent.send("What is the capital of France?")
    print(reply.text)
    print(reply.usage)  # Usage(input_tokens=..., output_tokens=...)

asyncio.run(main())
```

## Multi-turn

State is managed automatically. Just keep calling `send`:

```python
agent = Agent("anthropic/claude-sonnet-4-20250514")
await agent.send("My name is Job")
reply = await agent.send("What is my name?")
# reply.text -> "Job"
```

Access the full history via `agent.messages`.

## Persisting conversations

Save and restore conversation state:

```python
from llmkit.types import Message

# Save
saved: list[Message] = agent.messages

# Restore into a new agent — conversation continues
new_agent = Agent("openai/gpt-4o", messages=saved)
reply = await new_agent.send("What did we talk about?")
```

Both `agent.messages` and the `messages` parameter are defensively copied — mutating either side won't affect the other.

## Sync convenience

If you don't want async:

```python
reply = agent.send_sync("Quick question")
```

## Model selection

Use enums for autocomplete, or raw strings:

```python
from llmkit import Agent, OpenAI, Anthropic, Gemini

Agent(OpenAI.GPT_4O)
Agent(Anthropic.CLAUDE_SONNET)
Agent(Gemini.GEMINI_2_5_PRO)

# Raw strings work too
Agent("openai/gpt-4o")
```

## Auth

API keys are read from environment variables by default:

| Provider | Env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| Azure | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |

Override per instance:

```python
agent = Agent("openai/gpt-4o", api_key="sk-...")
```
