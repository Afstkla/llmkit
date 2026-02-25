# Providers

## Built-in providers

| Provider | Install | Model string |
|---|---|---|
| OpenAI | `llmkit[openai]` | `openai/gpt-4o` |
| Anthropic | `llmkit[anthropic]` | `anthropic/claude-sonnet-4-20250514` |
| Gemini | `llmkit[gemini]` | `gemini/gemini-2.5-pro` |
| Azure OpenAI | `llmkit[azure]` | `azure/gpt-4o` |
| AWS Bedrock | `llmkit[bedrock]` | `bedrock/anthropic.claude-sonnet-4-20250514-v1:0` |
| GCP Vertex AI | `llmkit[vertex]` | `vertex/claude-sonnet-4@20250514` |

## Cloud provider setup

### Azure OpenAI

```python
from llmkit import Chat

chat = Chat(
    "azure/gpt-4o",
    api_key="...",
    base_url="https://my-resource.openai.azure.com",
    api_version="2024-10-21",
)
```

Or set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` env vars.

### AWS Bedrock

```python
from llmkit import Chat, Bedrock

chat = Chat(Bedrock.CLAUDE_SONNET, aws_region="us-west-2")
```

Uses your default AWS credentials (env vars, `~/.aws/credentials`, or IAM role).

### GCP Vertex AI

```python
from llmkit import Chat, Vertex

chat = Chat(Vertex.CLAUDE_SONNET, project_id="my-project", region="us-east5")
```

Uses your default GCP credentials (`gcloud auth application-default login`).

## Custom providers

Implement the `Provider` protocol and register:

```python
from llmkit import Chat, register_provider

class OllamaProvider:
    def __init__(self, *, model, api_key=None, **kwargs):
        self._model = model

    async def send(self, messages, **kwargs):
        ...

    async def stream(self, messages, **kwargs):
        ...

register_provider("ollama", OllamaProvider)
chat = Chat("ollama/llama3")
```
