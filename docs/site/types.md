# Types

All data types are frozen dataclasses importable from `llmkit.types`.

```python
from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage
```

## Reply

Returned by `agent.send()` and `agent.stream()`.

```python
@dataclass(frozen=True, slots=True)
class Reply:
    text: str | None
    parsed: Any           # validated Pydantic instance when using response_model
    tool_calls: list[ToolCall]
    usage: Usage
    raw: Any              # raw provider response
```

## Message

A single message in the conversation history. Access via `agent.messages`.

```python
@dataclass(frozen=True, slots=True)
class Message:
    role: str                              # "user", "assistant", or "tool"
    content: str | None
    tool_calls: list[ToolCall] | None = None
```

## Usage

Token counts for a request.

```python
@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
```

## ToolCall

A tool invocation requested by the model (or completed with a result).

```python
@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None
```

## ToolDef

A tool definition registered on the agent. Returned by `agent.tools.list()`.

```python
@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema
```

## Hook types

Typed callback aliases for lifecycle hooks, importable from `llmkit`.

```python
from llmkit import Event, TurnStartHook, TurnEndHook, ToolCallStartHook, ToolCallEndHook

type Event = Literal["turn_start", "turn_end", "tool_call_start", "tool_call_end"]

type TurnStartHook = Callable[[list[Message]], None] | Callable[[list[Message]], Awaitable[None]]
type TurnEndHook = Callable[[Reply], None] | Callable[[Reply], Awaitable[None]]
type ToolCallStartHook = Callable[[str, dict[str, Any]], None] | Callable[[str, dict[str, Any]], Awaitable[None]]
type ToolCallEndHook = Callable[[str, dict[str, Any], str], None] | Callable[[str, dict[str, Any], str], Awaitable[None]]
```

## Exceptions

```python
from llmkit.exceptions import LLMKitError, ProviderError, ParseError, ToolError
```

| Exception | When |
|---|---|
| `LLMKitError` | Base for all llmkit errors |
| `ProviderError` | Error from an LLM provider |
| `ParseError` | Failed to parse structured output |
| `ToolError` | Tool loop exceeded max iterations |
