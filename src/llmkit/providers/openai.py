
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from llmkit.hosted_tools import HostedTool, hosted_tools_for_openai
from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage


class OpenAIProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        hosted_tools: list[HostedTool] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply:
        oai_messages = _build_messages(messages, system)
        req_kwargs: dict[str, Any] = {"model": self._model, "messages": oai_messages}

        oai_tools: list[dict[str, Any]] = []
        if tools:
            oai_tools.extend(_tool_to_oai(t) for t in tools)
        if hosted_tools:
            oai_tools.extend(hosted_tools_for_openai(hosted_tools))
        if oai_tools:
            req_kwargs["tools"] = oai_tools

        if response_model is not None:
            req_kwargs["response_format"] = _response_format(response_model)

        response = await self._client.chat.completions.create(**req_kwargs)
        return _parse_response(response, response_model)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        hosted_tools: list[HostedTool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]:
        oai_messages = _build_messages(messages, system)
        req_kwargs: dict[str, Any] = {
            "model": self._model, "messages": oai_messages, "stream": True,
        }

        oai_tools: list[dict[str, Any]] = []
        if tools:
            oai_tools.extend(_tool_to_oai(t) for t in tools)
        if hosted_tools:
            oai_tools.extend(hosted_tools_for_openai(hosted_tools))
        if oai_tools:
            req_kwargs["tools"] = oai_tools

        stream = await self._client.chat.completions.create(**req_kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield Reply(
                    text=delta.content,
                    parsed=None,
                    tool_calls=[],
                    usage=Usage(input_tokens=0, output_tokens=0),
                    raw=chunk,
                )


def _build_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    oai: list[dict[str, Any]] = []
    if system:
        oai.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "tool" and msg.tool_calls:
            tc = msg.tool_calls[0]
            oai.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": msg.content or "",
            })
        elif msg.role == "assistant" and msg.tool_calls:
            oai.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in msg.tool_calls
                ],
            })
        else:
            oai.append({"role": msg.role, "content": msg.content or ""})
    return oai


def _tool_to_oai(tool: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _response_format(model: type) -> dict[str, Any]:
    from pydantic import BaseModel

    if issubclass(model, BaseModel):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": model.model_json_schema(),
                "strict": True,
            },
        }
    msg = "response_model must be a Pydantic BaseModel subclass"
    raise TypeError(msg)


def _parse_response(response: Any, response_model: type | None) -> Reply:
    choice = response.choices[0]
    text = choice.message.content
    tool_calls: list[ToolCall] = []

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                )
            )

    parsed = None
    if response_model is not None and text:
        from pydantic import BaseModel

        if issubclass(response_model, BaseModel):
            parsed = response_model.model_validate_json(text)

    return Reply(
        text=text,
        parsed=parsed,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        ),
        raw=response,
    )
