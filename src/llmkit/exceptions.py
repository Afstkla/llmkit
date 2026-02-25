class LLMKitError(Exception):
    """Base exception for llmkit."""


class ProviderError(LLMKitError):
    """Error from an LLM provider."""


class ParseError(LLMKitError):
    """Failed to parse structured output."""


class ToolError(LLMKitError):
    """Error during tool execution."""
