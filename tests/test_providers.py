import pytest

from llmkit.providers import get_provider_class, parse_model, register_provider


def test_parse_model():
    provider, model = parse_model("openai/gpt-4o")
    assert provider == "openai"
    assert model == "gpt-4o"


def test_parse_model_with_slashes():
    provider, model = parse_model("bedrock/anthropic.claude-3")
    assert provider == "bedrock"
    assert model == "anthropic.claude-3"


def test_parse_model_no_slash():
    with pytest.raises(ValueError, match="must be in 'provider/model' format"):
        parse_model("gpt-4o")


def test_register_and_get_provider():
    class FakeProvider:
        pass

    register_provider("fake", FakeProvider)
    assert get_provider_class("fake") is FakeProvider


def test_get_unknown_provider():
    with pytest.raises(KeyError):
        get_provider_class("nonexistent_provider_xyz")
