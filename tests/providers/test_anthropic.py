
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit.providers.anthropic import AnthropicProvider
from llmkit.types import Message, ToolDef


@pytest.fixture
def provider():
    return AnthropicProvider(model="claude-sonnet-4-20250514", api_key="sk-test")


def test_init(provider: AnthropicProvider):
    assert provider._model == "claude-sonnet-4-20250514"


async def test_send_simple(provider: AnthropicProvider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello!")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    with patch.object(
        provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="Hi")])

    assert reply.text == "Hello!"
    assert reply.usage.input_tokens == 10
    assert reply.tool_calls == []


async def test_send_with_system(provider: AnthropicProvider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="I am helpful")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 3

    with patch.object(
        provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
    ) as mock_create:
        await provider.send(
            [Message(role="user", content="Who are you?")],
            system="You are helpful.",
        )
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."


async def test_send_with_tools(provider: AnthropicProvider):
    mock_tc = MagicMock()
    mock_tc.type = "tool_use"
    mock_tc.id = "toolu_123"
    mock_tc.name = "search"
    mock_tc.input = {"query": "test"}

    mock_response = MagicMock()
    mock_response.content = [mock_tc]
    mock_response.stop_reason = "tool_use"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    tools = [
        ToolDef(
            name="search",
            description="Search",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]

    with patch.object(
        provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="search")], tools=tools)

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "search"
    assert reply.tool_calls[0].args == {"query": "test"}
