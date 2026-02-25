# Hooks

Tap into the agent lifecycle to log, monitor, or react to events.

## Usage

Register hooks with the `@agent.on()` decorator:

```python
from llmkit import Agent, OpenAI

agent = Agent(OpenAI.GPT_4O)

@agent.on("turn_start")
def on_turn(messages):
    print(f"Sending {len(messages)} messages to the model...")

@agent.on("turn_end")
def on_reply(reply):
    print(f"Got reply: {reply.usage.input_tokens} in, {reply.usage.output_tokens} out")
```

## Events

| Event | When | Arguments |
|---|---|---|
| `turn_start` | Before each provider call | `messages: list[Message]` |
| `turn_end` | After each provider call | `reply: Reply` |
| `tool_call_start` | Before each tool execution | `name: str, args: dict` |
| `tool_call_end` | After each tool execution | `name: str, args: dict, result: str` |

During a tool loop, `turn_start`/`turn_end` fire on every iteration (each round-trip to the provider).

## Async handlers

Async hooks work too:

```python
@agent.on("tool_call_end")
async def log_to_db(name, args, result):
    await db.log(tool=name, result=result)
```

## Multiple handlers

Register multiple handlers per event — they fire in registration order:

```python
@agent.on("turn_start")
def log_it(messages):
    logger.info("turn starting")

@agent.on("turn_start")
def track_it(messages):
    metrics.increment("turns")
```
