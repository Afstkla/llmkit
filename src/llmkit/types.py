
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True, slots=True)
class Reply:
    text: str | None
    parsed: Any
    tool_calls: list[ToolCall]
    usage: Usage
    raw: Any
