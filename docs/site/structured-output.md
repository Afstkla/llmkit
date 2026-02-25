# Structured Output

Pass a Pydantic model to `response_model` and get a validated instance back.

## Basic usage

```python
from pydantic import BaseModel
from llmkit import Chat, Anthropic

class City(BaseModel):
    name: str
    country: str
    population: int

chat = Chat(Anthropic.CLAUDE_SONNET)
reply = await chat.send("Tell me about Amsterdam", response_model=City)

reply.parsed  # City(name="Amsterdam", country="Netherlands", population=872680)
```

## How it works

- Schema is extracted from your Pydantic model automatically
- Each provider's native structured output is used (OpenAI `json_schema`, Anthropic tool-use, Gemini `response_schema`)
- Response is validated with Pydantic — if validation fails, an auto-retry is attempted (configurable via `structured_retries`)

## Nested models

Anything Pydantic supports works:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    address: Address
    hobbies: list[str]

reply = await chat.send("Make up a person", response_model=Person)
```

## Access

- `reply.parsed` — the validated Pydantic instance
- `reply.text` — raw text (may be `None` when structured output is used)
- `reply.raw` — the raw provider response
