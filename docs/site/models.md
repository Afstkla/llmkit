# Models

Use `StrEnum` classes for autocomplete-friendly model selection, or pass raw strings.

## OpenAI

```python
from llmkit import Agent, OpenAI

Agent(OpenAI.GPT_4O)          # openai/gpt-4o
Agent(OpenAI.GPT_4O_MINI)     # openai/gpt-4o-mini
Agent(OpenAI.GPT_4_1)         # openai/gpt-4.1
Agent(OpenAI.GPT_4_1_MINI)    # openai/gpt-4.1-mini
Agent(OpenAI.GPT_4_1_NANO)    # openai/gpt-4.1-nano
Agent(OpenAI.O3)              # openai/o3
Agent(OpenAI.O3_MINI)         # openai/o3-mini
Agent(OpenAI.O4_MINI)         # openai/o4-mini
```

## Anthropic

```python
from llmkit import Agent, Anthropic

Agent(Anthropic.CLAUDE_OPUS)    # anthropic/claude-opus-4-20250514
Agent(Anthropic.CLAUDE_SONNET)  # anthropic/claude-sonnet-4-20250514
Agent(Anthropic.CLAUDE_HAIKU)   # anthropic/claude-haiku-3-5-20241022
```

## Gemini

```python
from llmkit import Agent, Gemini

Agent(Gemini.GEMINI_2_5_PRO)    # gemini/gemini-2.5-pro
Agent(Gemini.GEMINI_2_5_FLASH)  # gemini/gemini-2.5-flash
Agent(Gemini.GEMINI_2_0_FLASH)  # gemini/gemini-2.0-flash
```

## Bedrock

```python
from llmkit import Agent, Bedrock

Agent(Bedrock.CLAUDE_OPUS)    # bedrock/anthropic.claude-opus-4-20250514-v1:0
Agent(Bedrock.CLAUDE_SONNET)  # bedrock/anthropic.claude-sonnet-4-20250514-v1:0
Agent(Bedrock.CLAUDE_HAIKU)   # bedrock/anthropic.claude-haiku-3-5-20241022-v1:0
```

## Vertex

```python
from llmkit import Agent, Vertex

Agent(Vertex.CLAUDE_OPUS)    # vertex/claude-opus-4@20250514
Agent(Vertex.CLAUDE_SONNET)  # vertex/claude-sonnet-4@20250514
Agent(Vertex.CLAUDE_HAIKU)   # vertex/claude-haiku-3-5@20241022
```

## Raw strings

All enums are `StrEnum`, so they're just strings. You can always pass a raw model string:

```python
Agent("openai/gpt-4o")
Agent("anthropic/claude-sonnet-4-20250514")
Agent("gemini/gemini-2.5-pro")
```
