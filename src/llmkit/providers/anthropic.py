
import os
from collections.abc import AsyncIterator
from typing import Any

from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage


class AnthropicProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        self._model = model
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply:
        ant_messages = _build_messages(messages)
        req_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": ant_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system:
            req_kwargs["system"] = system

        if tools:
            req_kwargs["tools"] = [_tool_to_anthropic(t) for t in tools]

        if response_model is not None:
            from pydantic import BaseModel

            if issubclass(response_model, BaseModel):
                req_kwargs["tools"] = req_kwargs.get("tools", []) + [
                    {
                        "name": f"structured_{response_model.__name__}",
                        "description": f"Return structured {response_model.__name__} output",
                        "input_schema": response_model.model_json_schema(),
                    }
                ]
                tool_name = f"structured_{response_model.__name__}"
                req_kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        response = await self._client.messages.create(**req_kwargs)
        return _parse_response(response, response_model)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]:
        ant_messages = _build_messages(messages)
        req_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": ant_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system:
            req_kwargs["system"] = system

        if tools:
            req_kwargs["tools"] = [_tool_to_anthropic(t) for t in tools]

        async with self._client.messages.stream(**req_kwargs) as stream:
            async for text in stream.text_stream:
                yield Reply(
                    text=text,
                    parsed=None,
                    tool_calls=[],
                    usage=Usage(input_tokens=0, output_tokens=0),
                    raw=None,
                )


def _build_messages(messages: list[Message]) -> list[dict[str, Any]]:
    ant: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool" and msg.tool_calls:
            tc = msg.tool_calls[0]
            ant.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": msg.content or "",
                    }
                ],
            })
        elif msg.role == "assistant" and msg.tool_calls:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                })
            ant.append({"role": "assistant", "content": content})
        else:
            ant.append({"role": msg.role, "content": msg.content or ""})
    return ant


def _tool_to_anthropic(tool: ToolDef) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _parse_response(response: Any, response_model: type | None) -> Reply:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    parsed = None

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            if response_model is not None and block.name.startswith("structured_"):
                from pydantic import BaseModel

                if issubclass(response_model, BaseModel):
                    parsed = response_model.model_validate(block.input)
            else:
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=block.input,
                    )
                )

    return Reply(
        text="\n".join(text_parts) if text_parts else None,
        parsed=parsed,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
        raw=response,
    )
