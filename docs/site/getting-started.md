# Getting Started

## Install

```bash
uv add llmkit[anthropic]
```

Or pick your provider: `openai`, `anthropic`, `gemini`, `azure`, `bedrock`, `vertex`, or `all`.

## Basic usage

```python
import asyncio
from llmkit import Chat, OpenAI

async def main():
    chat = Chat(OpenAI.GPT_4O_MINI, system="Be concise.")
    reply = await chat.send("What is the capital of France?")
    print(reply.text)
    print(reply.usage)  # Usage(input_tokens=..., output_tokens=...)

asyncio.run(main())
```

## Multi-turn

State is managed automatically. Just keep calling `send`:

```python
chat = Chat("anthropic/claude-sonnet-4-20250514")
await chat.send("My name is Job")
reply = await chat.send("What is my name?")
# reply.text -> "Job"
```

Access the full history via `chat.messages`.

## Sync convenience

If you don't want async:

```python
reply = chat.send_sync("Quick question")
```

## Model selection

Use enums for autocomplete, or raw strings:

```python
from llmkit import OpenAI, Anthropic, Gemini

Chat(OpenAI.GPT_4O)
Chat(Anthropic.CLAUDE_SONNET)
Chat(Gemini.GEMINI_2_5_PRO)

# Raw strings work too
Chat("openai/gpt-4o")
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
chat = Chat("openai/gpt-4o", api_key="sk-...")
```
