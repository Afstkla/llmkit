from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


class GeminiProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        if genai is None:
            msg = "google-genai package is required: pip install llmkit[gemini]"
            raise ImportError(msg)

        self._model = model
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply:
        contents = _build_contents(messages)
        config: dict[str, Any] = {}

        if system:
            config["system_instruction"] = system

        if tools:
            config["tools"] = [_tools_to_gemini(tools)]

        if response_model is not None:
            from pydantic import BaseModel

            if issubclass(response_model, BaseModel):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_model.model_json_schema()

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config) if config else None,
        )
        return _parse_response(response, response_model)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]:
        contents = _build_contents(messages)
        config: dict[str, Any] = {}

        if system:
            config["system_instruction"] = system

        if tools:
            config["tools"] = [_tools_to_gemini(tools)]

        stream = self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config) if config else None,
        )
        async for chunk in stream:  # type: ignore[union-attr]
            if chunk.text:
                yield Reply(
                    text=chunk.text,
                    parsed=None,
                    tool_calls=[],
                    usage=Usage(input_tokens=0, output_tokens=0),
                    raw=chunk,
                )


def _build_contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        if msg.role == "tool" and msg.tool_calls:
            tc = msg.tool_calls[0]
            func_resp = {
                "name": tc.name,
                "response": {"result": msg.content or ""},
            }
            contents.append({
                "role": "user",
                "parts": [{"function_response": func_resp}],
            })
        elif msg.role == "assistant" and msg.tool_calls:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for tc in msg.tool_calls:
                parts.append({"function_call": {"name": tc.name, "args": tc.args}})
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": role, "parts": [{"text": msg.content or ""}]})
    return contents


def _tools_to_gemini(tools: list[ToolDef]) -> dict[str, Any]:
    declarations = []
    for t in tools:
        declarations.append({
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        })
    return {"function_declarations": declarations}


def _parse_response(response: Any, response_model: type | None) -> Reply:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in response.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(
                ToolCall(
                    id=fc.name,  # Gemini doesn't have separate IDs
                    name=fc.name,
                    args=dict(fc.args) if fc.args else {},
                )
            )
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    text = "\n".join(text_parts) if text_parts else None

    parsed = None
    if response_model is not None and text:
        import json

        from pydantic import BaseModel

        if issubclass(response_model, BaseModel):
            parsed = response_model.model_validate(json.loads(text))

    return Reply(
        text=text,
        parsed=parsed,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        ),
        raw=response,
    )
