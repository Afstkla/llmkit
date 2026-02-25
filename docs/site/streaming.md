# Streaming

Stream responses as they're generated.

## Usage

```python
from llmkit import Chat, Anthropic

chat = Chat(Anthropic.CLAUDE_SONNET)

async for chunk in chat.stream("Write me a short story"):
    print(chunk.text, end="")
```

Each `chunk` is a `Reply` with:

- `chunk.text` — the text fragment
- `chunk.usage` — token counts (populated on final chunk for some providers)
- `chunk.raw` — the raw provider chunk
