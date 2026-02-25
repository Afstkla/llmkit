from enum import StrEnum


class OpenAI(StrEnum):
    """OpenAI model identifiers."""

    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    GPT_4_1 = "openai/gpt-4.1"
    GPT_4_1_MINI = "openai/gpt-4.1-mini"
    GPT_4_1_NANO = "openai/gpt-4.1-nano"
    O3 = "openai/o3"
    O3_MINI = "openai/o3-mini"
    O4_MINI = "openai/o4-mini"


class Anthropic(StrEnum):
    """Anthropic model identifiers."""

    CLAUDE_OPUS = "anthropic/claude-opus-4-20250514"
    CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"
    CLAUDE_HAIKU = "anthropic/claude-haiku-3-5-20241022"


class Gemini(StrEnum):
    """Google Gemini model identifiers."""

    GEMINI_2_5_PRO = "gemini/gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini/gemini-2.5-flash"
    GEMINI_2_0_FLASH = "gemini/gemini-2.0-flash"


class Bedrock(StrEnum):
    """AWS Bedrock model identifiers."""

    CLAUDE_OPUS = "bedrock/anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_SONNET = "bedrock/anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_HAIKU = "bedrock/anthropic.claude-haiku-3-5-20241022-v1:0"


class Vertex(StrEnum):
    """GCP Vertex AI (Anthropic) model identifiers."""

    CLAUDE_OPUS = "vertex/claude-opus-4@20250514"
    CLAUDE_SONNET = "vertex/claude-sonnet-4@20250514"
    CLAUDE_HAIKU = "vertex/claude-haiku-3-5@20241022"
