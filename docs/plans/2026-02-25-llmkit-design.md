# llmkit Design

Minimal, typed Python LLM wrapper supporting OpenAI, Anthropic, and Gemini with a clean API that replaces the verbose chat completions pattern.

## Core API

```python
from llmkit import Chat

chat = Chat("anthropic/claude-sonnet-4-20250514", system="You are a helpful assistant.")
reply = await chat.send("Summarize this article")
print(reply.text)

# Multi-turn — state kept automatically
reply2 = await chat.send("Make it shorter")

# Sync convenience
reply = chat.send_sync("Quick question")

# Streaming — async iterator
async for chunk in chat.stream("Write me a story"):
    print(chunk.text, end="")

# Message history
chat.messages  # list[Message]
```

## Key Types

- `Chat` — main object, holds conversation state + config
- `Message` — role, content, tool_calls
- `Reply` — .text, .parsed, .tool_calls, .usage, .raw
- `ToolCall` — name, args, result
- `ToolDef` — name, description, parameters schema
- `Usage` — input_tokens, output_tokens

## Structured Output

```python
class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    steps: list[str]

reply = await chat.send("Give me a pasta recipe", response_model=Recipe)
reply.parsed  # Recipe instance
```

Uses each provider's native structured output. Validates with Pydantic, auto-retry on parse failure (configurable, default 1 retry).

## Tools

```python
@chat.tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return do_search(query)

@chat.tool
async def fetch_url(url: str) -> str:
    """Fetch contents of a URL."""
    return await aiohttp_get(url)

# Programmatic registration
chat.tools.register(some_function)
chat.tools.register(some_callable, name="custom_name", description="Override docstring")
chat.tools.unregister("search_web")
chat.tools.list()  # list[ToolDef]
```

Schema extraction: function name → tool name, docstring → description, type hints → parameter schema via Pydantic TypeAdapter.

Tool execution loop: model requests call → library executes → feeds result back → repeats until final text response. Max iterations configurable (default 10). Exceptions sent as error text to model.

## Auth

Env vars auto-detected (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY), overridable per instance:

```python
chat = Chat("openai/gpt-4o", api_key="sk-...")
```

## Provider Architecture

```python
class Provider(Protocol):
    async def send(self, messages: list[Message], **kwargs) -> RawResponse: ...
    async def stream(self, messages: list[Message], **kwargs) -> AsyncIterator[RawChunk]: ...
```

Three built-in providers: OpenAIProvider, AnthropicProvider, GeminiProvider. Optional dependencies — only install what you use.

Custom providers:
```python
from llmkit import register_provider
register_provider("ollama", OllamaProvider)
```

## Model Strings

litellm-style: `provider/model-name`. E.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `gemini/gemini-2.0-flash`.

## Project Setup

- Python 3.14 minimum
- uv for project management
- ruff for linting + formatting
- ty for type checking
- pytest + pytest-asyncio for tests
- Optional dependencies per provider

## File Structure

```
src/llmkit/
├── __init__.py          # public API exports
├── chat.py              # Chat class
├── types.py             # Message, Reply, ToolCall, ToolDef, Usage
├── tools.py             # ToolRegistry, @tool decorator, schema extraction
├── providers/
│   ├── __init__.py      # Provider protocol, registry, parse_model()
│   ├── openai.py
│   ├── anthropic.py
│   └── gemini.py
└── exceptions.py        # LLMKitError, ProviderError, ParseError, ToolError
```
