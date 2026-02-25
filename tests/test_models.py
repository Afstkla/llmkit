from llmkit.models import Anthropic, Gemini, OpenAI


def test_openai_enum_values():
    assert OpenAI.GPT_4O == "openai/gpt-4o"
    assert OpenAI.GPT_4O_MINI == "openai/gpt-4o-mini"
    assert OpenAI.O3 == "openai/o3"
    assert OpenAI.O3_MINI == "openai/o3-mini"
    assert OpenAI.O4_MINI == "openai/o4-mini"


def test_anthropic_enum_values():
    assert Anthropic.CLAUDE_OPUS == "anthropic/claude-opus-4-20250514"
    assert Anthropic.CLAUDE_SONNET == "anthropic/claude-sonnet-4-20250514"
    assert Anthropic.CLAUDE_HAIKU == "anthropic/claude-haiku-3-5-20241022"


def test_gemini_enum_values():
    assert Gemini.GEMINI_2_5_PRO == "gemini/gemini-2.5-pro"
    assert Gemini.GEMINI_2_5_FLASH == "gemini/gemini-2.5-flash"
    assert Gemini.GEMINI_2_0_FLASH == "gemini/gemini-2.0-flash"


def test_enum_is_string():
    """Enum values should be usable as strings directly."""
    model: str = OpenAI.GPT_4O
    assert isinstance(model, str)
    assert model.startswith("openai/")
