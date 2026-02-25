
from collections.abc import AsyncIterator
from typing import Any, Protocol

from llmkit.hosted_tools import HostedTool
from llmkit.types import Message, Reply, ToolDef


class Provider(Protocol):
    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        hosted_tools: list[HostedTool] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        hosted_tools: list[HostedTool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]: ...


_registry: dict[str, type] = {}


def register_provider(name: str, cls: type) -> None:
    """Register a provider class under a name."""
    _registry[name] = cls


def get_provider_class(name: str) -> type:
    """Get a registered provider class by name."""
    return _registry[name]


def parse_model(model: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_name)."""
    slash_idx = model.find("/")
    if slash_idx == -1:
        msg = f"Model '{model}' must be in 'provider/model' format (e.g. 'openai/gpt-4o')"
        raise ValueError(msg)
    return model[:slash_idx], model[slash_idx + 1 :]


def _register_builtins() -> None:
    """Lazy-register built-in providers."""
    try:
        from llmkit.providers.openai import OpenAIProvider
        register_provider("openai", OpenAIProvider)
    except ImportError:
        pass
    try:
        from llmkit.providers.anthropic import AnthropicProvider
        register_provider("anthropic", AnthropicProvider)
    except ImportError:
        pass
    try:
        from llmkit.providers.gemini import GeminiProvider
        register_provider("gemini", GeminiProvider)
    except ImportError:
        pass
    try:
        from llmkit.providers.azure import AzureOpenAIProvider
        register_provider("azure", AzureOpenAIProvider)
    except ImportError:
        pass
    try:
        from llmkit.providers.bedrock import BedrockProvider
        register_provider("bedrock", BedrockProvider)
    except ImportError:
        pass
    try:
        from llmkit.providers.vertex import VertexProvider
        register_provider("vertex", VertexProvider)
    except ImportError:
        pass


_register_builtins()
