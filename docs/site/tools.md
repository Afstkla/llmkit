# Tools

Register functions as tools. The model calls them automatically.

## Decorator

```python
from llmkit import Agent, OpenAI

agent = Agent(OpenAI.GPT_4O)

@agent.tool
def search(query: str) -> str:
    """Search the web for information."""
    return do_search(query)

reply = await agent.send("What's the weather in Amsterdam?")
# search() is called automatically, result fed back to the model
```

The function name becomes the tool name. The docstring becomes the description. Type hints become the parameter schema.

## Async tools

```python
@agent.tool
async def fetch(url: str) -> str:
    """Fetch contents of a URL."""
    return await aiohttp_get(url)
```

## Programmatic registration

For tools you create at runtime:

```python
agent.tools.register(some_function)
agent.tools.register(fn, name="custom_name", description="Override the docstring")
agent.tools.unregister("search")
agent.tools.list()  # list[ToolDef]
```

## Hosted tools

Providers offer built-in tools like web search. Use them via `hosted_tools`:

```python
from llmkit import Agent, Anthropic, WebSearch

agent = Agent(Anthropic.CLAUDE_SONNET, hosted_tools=[WebSearch()])
reply = await agent.send("What happened in the news today?")
```

Works with OpenAI, Anthropic, and Gemini — llmkit translates to each provider's format.

## Agent-as-tool

Turn an agent into a tool another agent can call:

```python
researcher = Agent("anthropic/claude-sonnet-4-20250514", system="You research topics thoroughly.")
writer = Agent("openai/gpt-4o", system="You write clear summaries.")

writer.tools.register(researcher.as_tool(name="research", description="Research a topic"))
reply = await writer.send("Write a summary about quantum computing")
# writer calls researcher as a tool, gets back research, writes summary
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
@agent.tool
def search(query: str, max_results: int = 10, filter: str | None = None) -> str:
    """Search for things."""
    ...
```

- `query` — required string
- `max_results` — optional integer, default 10
- `filter` — optional string
