from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit.providers.gemini import GeminiProvider
from llmkit.types import Message


@pytest.fixture
def provider():
    with patch("llmkit.providers.gemini.genai"):
        p = GeminiProvider(model="gemini-2.0-flash", api_key="test-key")
        yield p


def test_init(provider: GeminiProvider):
    assert provider._model == "gemini-2.0-flash"


async def test_send_simple(provider: GeminiProvider):
    mock_response = MagicMock()
    mock_response.text = "Hello!"
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [MagicMock(text="Hello!", function_call=None)]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5

    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        reply = await provider.send([Message(role="user", content="Hi")])

    assert reply.text == "Hello!"
    assert reply.usage.input_tokens == 10
    assert reply.tool_calls == []
