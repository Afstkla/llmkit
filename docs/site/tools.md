# Tools

Register functions as tools. The model calls them automatically.

## Decorator

```python
from llmkit import Chat, OpenAI

chat = Chat(OpenAI.GPT_4O)

@chat.tool
def search(query: str) -> str:
    """Search the web for information."""
    return do_search(query)

reply = await chat.send("What's the weather in Amsterdam?")
# search() is called automatically, result fed back to the model
```

The function name becomes the tool name. The docstring becomes the description. Type hints become the parameter schema.

## Async tools

```python
@chat.tool
async def fetch(url: str) -> str:
    """Fetch contents of a URL."""
    return await aiohttp_get(url)
```

## Programmatic registration

For tools you create at runtime:

```python
chat.tools.register(some_function)
chat.tools.register(fn, name="custom_name", description="Override the docstring")
chat.tools.unregister("search")
chat.tools.list()  # list[ToolDef]
```

## How it works

When the model wants to call a tool:

1. Model returns a tool call request
2. llmkit executes the function with the provided arguments
3. The result is sent back to the model
4. Model either calls another tool or returns a final text response

This loops up to `max_tool_iterations` times (default 10). If a tool raises an exception, the error message is sent to the model as the tool result.

## Schema extraction

Parameters are extracted from type hints using Pydantic's `TypeAdapter`:

```python
@chat.tool
def search(query: str, max_results: int = 10, filter: str | None = None) -> str:
    """Search for things."""
    ...
```

- `query` — required string
- `max_results` — optional integer, default 10
- `filter` — optional string
