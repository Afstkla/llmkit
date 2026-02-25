from __future__ import annotations

import os

from llmkit.providers.openai import OpenAIProvider


class AzureOpenAIProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str = "2024-10-21",
    ) -> None:
        from openai import AsyncAzureOpenAI

        self._model = model
        self._client = AsyncAzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version=api_version,
        )
