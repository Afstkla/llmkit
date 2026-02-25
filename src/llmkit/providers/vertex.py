from __future__ import annotations

from llmkit.providers.anthropic import AnthropicProvider


class VertexProvider(AnthropicProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
    ) -> None:
        from anthropic import AsyncAnthropicVertex

        self._model = model
        self._client = AsyncAnthropicVertex(
            project_id=project_id or "",
            region=region or "us-east5",
        )
