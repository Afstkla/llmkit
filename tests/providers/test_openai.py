
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit.providers.openai import OpenAIProvider
from llmkit.types import Message, ToolDef


@pytest.fixture
def provider():
    return OpenAIProvider(model="gpt-4o", api_key="sk-test")


def test_init(provider: OpenAIProvider):
    assert provider._model == "gpt-4o"


async def test_send_simple(provider: OpenAIProvider):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello!"
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    with patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="Hi")])

    assert reply.text == "Hello!"
    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 5
    assert reply.tool_calls == []


async def test_send_with_tools(provider: OpenAIProvider):
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"query": "test"}'

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    tools = [ToolDef(
        name="search", description="Search",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )]

    with patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="search for test")], tools=tools)

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "search"
    assert reply.tool_calls[0].args == {"query": "test"}


async def test_send_with_system(provider: OpenAIProvider):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "I am helpful"
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 3

    with patch.object(
        provider._client.chat.completions, "create",
        new_callable=AsyncMock, return_value=mock_response
    ) as mock_create:
        await provider.send(
            [Message(role="user", content="Who are you?")],
            system="You are helpful.",
        )
        call_args = mock_create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
