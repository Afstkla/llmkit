from __future__ import annotations

from llmkit.providers.anthropic import AnthropicProvider


class BedrockProvider(AnthropicProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        aws_region: str | None = None,
    ) -> None:
        from anthropic import AsyncAnthropicBedrock

        self._model = model
        self._client = AsyncAnthropicBedrock(
            aws_region=aws_region or "us-east-1",
        )
