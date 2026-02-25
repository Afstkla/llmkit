def test_public_api():
    from llmkit import Agent, Anthropic, Chat, Gemini, OpenAI, WebSearch, register_provider

    assert Agent is not None
    assert Chat is Agent  # backwards compat alias
    assert register_provider is not None
    assert OpenAI is not None
    assert Anthropic is not None
    assert Gemini is not None
    assert WebSearch is not None


def test_types_importable():
    from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage

    assert Message is not None
    assert Reply is not None
    assert ToolCall is not None
    assert ToolDef is not None
    assert Usage is not None


def test_exceptions_importable():
    from llmkit.exceptions import LLMKitError, ParseError, ProviderError, ToolError

    assert LLMKitError is not None
    assert ParseError is not None
    assert ProviderError is not None
    assert ToolError is not None
