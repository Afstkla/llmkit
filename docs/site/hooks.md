# Hooks

Tap into the agent lifecycle to log, monitor, or react to events.

## Usage

Register hooks with the `@agent.on()` decorator:

```python
from llmkit import Agent, OpenAI
from llmkit.types import Message, Reply

agent = Agent(OpenAI.GPT_4O)

@agent.on("turn_start")
def on_turn(messages: list[Message]) -> None:
    print(f"Sending {len(messages)} messages to the model...")

@agent.on("turn_end")
def on_reply(reply: Reply) -> None:
    print(f"Got reply: {reply.usage.input_tokens} in, {reply.usage.output_tokens} out")
```

## Events

| Event | Signature |
|---|---|
| `turn_start` | `(messages: list[Message]) -> None` |
| `turn_end` | `(reply: Reply) -> None` |
| `tool_call_start` | `(name: str, args: dict[str, Any]) -> None` |
| `tool_call_end` | `(name: str, args: dict[str, Any], result: str) -> None` |

During a tool loop, `turn_start`/`turn_end` fire on every iteration (each round-trip to the provider).

## Hook types

All hook types are exported for annotation:

```python
from llmkit import TurnStartHook, TurnEndHook, ToolCallStartHook, ToolCallEndHook

# Event is a Literal type
from llmkit import Event  # Literal["turn_start", "turn_end", "tool_call_start", "tool_call_end"]
```

## Async handlers

All hooks accept both sync and async handlers:

```python
@agent.on("tool_call_end")
async def log_to_db(name: str, args: dict[str, Any], result: str) -> None:
    await db.log(tool=name, result=result)
```

## Multiple handlers

Register multiple handlers per event — they fire in registration order:

```python
@agent.on("turn_start")
def log_it(messages: list[Message]) -> None:
    logger.info("turn starting")

@agent.on("turn_start")
def track_it(messages: list[Message]) -> None:
    metrics.increment("turns")
```
