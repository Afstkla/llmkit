# llmkit

Minimal, typed Python LLM wrapper. One `Chat` object, multiple providers, no boilerplate.

```python
from llmkit import Chat, Anthropic

chat = Chat(Anthropic.CLAUDE_SONNET, system="Be concise.")
reply = await chat.send("What is 2+2?")
print(reply.text)  # "4"
```

## Why llmkit?

- **One object** — `Chat` handles conversation state, tools, structured output, streaming
- **Typed** — Pydantic models in, validated instances out. Full type hints everywhere
- **Minimal** — No abstractions between you and the LLM. Read the source in 10 minutes
- **Multi-provider** — OpenAI, Anthropic, Gemini, Azure, Bedrock, Vertex

## Install

```bash
uv add llmkit[all]         # all providers
uv add llmkit[anthropic]   # just one
```

Requires Python 3.14+.
