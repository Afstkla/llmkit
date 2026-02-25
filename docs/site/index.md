# llmkit

Minimal, typed Python LLM wrapper. One `Agent` object, multiple providers, no boilerplate.

```python
from llmkit import Agent, Anthropic

agent = Agent(Anthropic.CLAUDE_SONNET, system="Be concise.")
reply = await agent.send("What is 2+2?")
print(reply.text)  # "4"
```

## Why llmkit?

- **One object** — `Agent` handles conversation state, tools, structured output, streaming
- **Typed** — Pydantic models in, validated instances out. Full type hints everywhere
- **Minimal** — No abstractions between you and the LLM. Read the source in 10 minutes
- **Multi-provider** — OpenAI, Anthropic, Gemini, Azure, Bedrock, Vertex

## Install

```bash
uv add llmkit[all]         # all providers
uv add llmkit[anthropic]   # just one
```

Requires Python 3.14+.
