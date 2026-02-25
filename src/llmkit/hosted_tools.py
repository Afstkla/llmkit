"""Hosted tools that providers offer natively (e.g. web search)."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HostedTool:
    """Base class for hosted tools."""


@dataclass(frozen=True, slots=True)
class WebSearch(HostedTool):
    """Web search tool available on OpenAI, Anthropic, and Gemini."""


def hosted_tools_for_openai(tools: list[HostedTool]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, WebSearch):
            result.append({"type": "web_search_preview"})
    return result


def hosted_tools_for_anthropic(tools: list[HostedTool]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, WebSearch):
            result.append({"type": "web_search_20250305", "name": "web_search"})
    return result


def hosted_tools_for_gemini(tools: list[HostedTool]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, WebSearch):
            result.append({"google_search": {}})
    return result
