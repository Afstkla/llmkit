# llmkit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal, typed Python LLM wrapper supporting OpenAI, Anthropic, and Gemini with a clean `Chat` API.

**Architecture:** Async-first `Chat` class holds conversation state and delegates to thin provider adapters behind a `Provider` protocol. Tools are registered via decorators or programmatically, with automatic schema extraction from type hints. Structured output uses Pydantic models with native provider support.

**Tech Stack:** Python 3.14, uv, ruff, ty, pytest, pytest-asyncio, pydantic, openai/anthropic/google-genai SDKs as optional deps.

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/llmkit/__init__.py`
- Create: `src/llmkit/py.typed`

**Step 1: Initialize uv project**

```bash
cd /Users/jobnijenhuis/Developer/llmkit
uv init --lib --python 3.14
```

This creates `pyproject.toml` and `src/llmkit/`. Remove any boilerplate files uv creates that we don't need.

**Step 2: Configure pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "llmkit"
version = "0.1.0"
description = "Minimal, typed Python LLM wrapper"
requires-python = ">=3.14"
dependencies = [
    "pydantic>=2.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.40"]
gemini = ["google-genai>=1.0"]
all = ["openai>=1.0", "anthropic>=0.40", "google-genai>=1.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.9",
    "ty>=0.0.1a0",
]

[tool.ruff]
target-version = "py314"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ty]
python-version = "3.14"
```

**Step 3: Install dependencies**

```bash
cd /Users/jobnijenhuis/Developer/llmkit
uv sync --all-extras --all-groups
```

**Step 4: Create empty init and py.typed marker**

`src/llmkit/__init__.py`:
```python
"""llmkit — minimal, typed Python LLM wrapper."""
```

`src/llmkit/py.typed`: empty file (marker for PEP 561).

**Step 5: Verify tooling works**

```bash
cd /Users/jobnijenhuis/Developer/llmkit
uv run ruff check src/
uv run ty check src/
uv run pytest --co -q
```

All should succeed with no errors and no tests collected.

**Step 6: Commit**

```bash
git add pyproject.toml src/ uv.lock
git commit -m "feat: scaffold project with uv, ruff, ty, pytest"
```

---

### Task 2: Core Types

**Files:**
- Create: `src/llmkit/types.py`
- Create: `src/llmkit/exceptions.py`
- Create: `tests/test_types.py`

**Step 1: Write the failing tests**

`tests/test_types.py`:
```python
from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage


def test_message_creation():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls is None


def test_message_with_tool_calls():
    tc = ToolCall(id="1", name="search", args={"q": "test"}, result=None)
    msg = Message(role="assistant", content=None, tool_calls=[tc])
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].name == "search"


def test_reply_text():
    reply = Reply(
        text="hello",
        parsed=None,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
        raw={},
    )
    assert reply.text == "hello"
    assert reply.usage.input_tokens == 10


def test_tool_def():
    td = ToolDef(name="search", description="Search the web", parameters={"type": "object"})
    assert td.name == "search"


def test_usage():
    u = Usage(input_tokens=100, output_tokens=50)
    assert u.input_tokens + u.output_tokens == 150
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_types.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'llmkit.types'`

**Step 3: Implement types.py**

`src/llmkit/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True, slots=True)
class Reply:
    text: str | None
    parsed: Any
    tool_calls: list[ToolCall]
    usage: Usage
    raw: Any
```

**Step 4: Implement exceptions.py**

`src/llmkit/exceptions.py`:
```python
class LLMKitError(Exception):
    """Base exception for llmkit."""


class ProviderError(LLMKitError):
    """Error from an LLM provider."""


class ParseError(LLMKitError):
    """Failed to parse structured output."""


class ToolError(LLMKitError):
    """Error during tool execution."""
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_types.py -v
```

Expected: all 5 PASS.

**Step 6: Lint and type check**

```bash
uv run ruff check src/ tests/
uv run ty check src/
```

**Step 7: Commit**

```bash
git add src/llmkit/types.py src/llmkit/exceptions.py tests/test_types.py
git commit -m "feat: add core types and exceptions"
```

---

### Task 3: Provider Protocol and Registry

**Files:**
- Create: `src/llmkit/providers/__init__.py`
- Create: `tests/test_providers.py`

**Step 1: Write the failing tests**

`tests/test_providers.py`:
```python
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_providers.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement providers/__init__.py**

`src/llmkit/providers/__init__.py`:
```python
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from llmkit.types import Message


class Provider(Protocol):
    async def send(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Any: ...

    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[Any]: ...


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
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_providers.py -v
```

Expected: all 5 PASS.

**Step 5: Commit**

```bash
git add src/llmkit/providers/__init__.py tests/test_providers.py
git commit -m "feat: add provider protocol and registry"
```

---

### Task 4: Tool Registry and Schema Extraction

**Files:**
- Create: `src/llmkit/tools.py`
- Create: `tests/test_tools.py`

**Step 1: Write the failing tests**

`tests/test_tools.py`:
```python
import pytest

from llmkit.tools import ToolRegistry


def test_register_with_decorator():
    reg = ToolRegistry()

    @reg.register
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"Hello, {name}!"

    assert "greet" in reg
    td = reg.get("greet")
    assert td.name == "greet"
    assert td.description == "Say hello to someone."


def test_register_programmatic():
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    reg.register(add)
    assert "add" in reg


def test_register_with_overrides():
    reg = ToolRegistry()

    def f(x: str) -> str:
        return x

    reg.register(f, name="custom", description="Custom description")
    td = reg.get("custom")
    assert td.name == "custom"
    assert td.description == "Custom description"


def test_unregister():
    reg = ToolRegistry()

    @reg.register
    def temp(x: str) -> str:
        """Temp."""
        return x

    reg.unregister("temp")
    assert "temp" not in reg


def test_unregister_unknown():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.unregister("nonexistent")


def test_list_tools():
    reg = ToolRegistry()

    @reg.register
    def a(x: str) -> str:
        """A."""
        return x

    @reg.register
    def b(x: int) -> int:
        """B."""
        return x

    tools = reg.list()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"a", "b"}


def test_schema_extraction():
    reg = ToolRegistry()

    @reg.register
    def search(query: str, max_results: int = 10) -> str:
        """Search for things."""
        return "results"

    td = reg.get("search")
    params = td.parameters
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["properties"]["query"]["type"] == "string"
    assert "max_results" in params["properties"]
    assert params["required"] == ["query"]


def test_schema_optional_param():
    reg = ToolRegistry()

    @reg.register
    def fetch(url: str, timeout: int | None = None) -> str:
        """Fetch a URL."""
        return ""

    td = reg.get("fetch")
    assert td.parameters["required"] == ["url"]


def test_call_sync_tool():
    reg = ToolRegistry()

    @reg.register
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    result = reg.call("double", {"n": 5})
    assert result == 10


async def test_call_async_tool():
    reg = ToolRegistry()

    @reg.register
    async def async_double(n: int) -> int:
        """Double a number async."""
        return n * 2

    result = await reg.acall("async_double", {"n": 5})
    assert result == 10
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_tools.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement tools.py**

`src/llmkit/tools.py`:
```python
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter

from llmkit.types import ToolDef


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a tool. Works as decorator or direct call."""

        def _register(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            tool_desc = description or (inspect.getdoc(f) or "")
            params = _extract_schema(f)
            self._tools[tool_name] = ToolDef(
                name=tool_name,
                description=tool_desc,
                parameters=params,
            )
            self._callables[tool_name] = f
            return f

        if fn is not None:
            return _register(fn)
        return _register

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(name)
        del self._tools[name]
        del self._callables[name]

    def get(self, name: str) -> ToolDef:
        return self._tools[name]

    def list(self) -> list[ToolDef]:
        return list(self._tools.values())

    def call(self, name: str, args: dict[str, Any]) -> Any:
        fn = self._callables[name]
        return fn(**args)

    async def acall(self, name: str, args: dict[str, Any]) -> Any:
        fn = self._callables[name]
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def _extract_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON Schema from function signature using Pydantic."""
    sig = inspect.signature(fn)
    hints = {k: v for k, v in fn.__annotations__.items() if k != "return"}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, Any)
        ta = TypeAdapter(annotation)
        schema = ta.json_schema()
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_tools.py -v
```

Expected: all 10 PASS.

**Step 5: Commit**

```bash
git add src/llmkit/tools.py tests/test_tools.py
git commit -m "feat: add tool registry with schema extraction"
```

---

### Task 5: Chat Class (No Provider — Core Logic)

**Files:**
- Create: `src/llmkit/chat.py`
- Create: `tests/test_chat.py`

This task builds the `Chat` class with a mock/fake provider so we can test all the orchestration logic (message history, tool decorator, send_sync) without hitting real APIs.

**Step 1: Write the failing tests**

`tests/test_chat.py`:
```python
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from llmkit.chat import Chat
from llmkit.types import Message, Reply, ToolCall, Usage


class FakeProvider:
    """Test provider that returns canned responses."""

    def __init__(self, responses: list[Reply] | None = None) -> None:
        self.responses = list(responses or [])
        self.sent_messages: list[list[Message]] = []

    async def send(self, messages: list[Message], **kwargs: Any) -> Reply:
        self.sent_messages.append(messages)
        return self.responses.pop(0)

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[Any]:
        reply = self.responses.pop(0)
        yield reply


def _simple_reply(text: str) -> Reply:
    return Reply(
        text=text,
        parsed=None,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
        raw={},
    )


def _tool_call_reply(name: str, args: dict[str, Any], call_id: str = "call_1") -> Reply:
    return Reply(
        text=None,
        parsed=None,
        tool_calls=[ToolCall(id=call_id, name=name, args=args)],
        usage=Usage(input_tokens=10, output_tokens=5),
        raw={},
    )


async def test_send_basic():
    provider = FakeProvider([_simple_reply("Hello!")])
    chat = Chat.__new__(Chat)
    chat._provider = provider
    chat._system = "You are helpful."
    chat._messages = []
    chat._tools = __import__("llmkit.tools", fromlist=["ToolRegistry"]).ToolRegistry()
    chat._model_name = "fake/test"
    chat._max_tool_iterations = 10
    chat._structured_retries = 1

    reply = await chat.send("Hi")
    assert reply.text == "Hello!"
    assert len(chat.messages) == 2  # user + assistant


async def test_send_maintains_history():
    provider = FakeProvider([_simple_reply("First"), _simple_reply("Second")])
    chat = Chat.__new__(Chat)
    chat._provider = provider
    chat._system = None
    chat._messages = []
    chat._tools = __import__("llmkit.tools", fromlist=["ToolRegistry"]).ToolRegistry()
    chat._model_name = "fake/test"
    chat._max_tool_iterations = 10
    chat._structured_retries = 1

    await chat.send("One")
    await chat.send("Two")
    assert len(chat.messages) == 4  # 2 user + 2 assistant
    assert chat.messages[0].content == "One"
    assert chat.messages[2].content == "Two"


async def test_tool_decorator():
    provider = FakeProvider([
        _tool_call_reply("double", {"n": 5}),
        _simple_reply("The answer is 10"),
    ])
    chat = Chat.__new__(Chat)
    chat._provider = provider
    chat._system = None
    chat._messages = []
    chat._tools = __import__("llmkit.tools", fromlist=["ToolRegistry"]).ToolRegistry()
    chat._model_name = "fake/test"
    chat._max_tool_iterations = 10
    chat._structured_retries = 1

    @chat.tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    reply = await chat.send("What is 5 doubled?")
    assert reply.text == "The answer is 10"


def test_send_sync():
    provider = FakeProvider([_simple_reply("Sync reply")])
    chat = Chat.__new__(Chat)
    chat._provider = provider
    chat._system = None
    chat._messages = []
    chat._tools = __import__("llmkit.tools", fromlist=["ToolRegistry"]).ToolRegistry()
    chat._model_name = "fake/test"
    chat._max_tool_iterations = 10
    chat._structured_retries = 1

    reply = chat.send_sync("Hello")
    assert reply.text == "Sync reply"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_chat.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement chat.py**

`src/llmkit/chat.py`:
```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from llmkit.exceptions import ToolError
from llmkit.tools import ToolRegistry
from llmkit.types import Message, Reply, ToolCall, Usage


class Chat:
    def __init__(
        self,
        model: str,
        *,
        system: str | None = None,
        api_key: str | None = None,
        max_tool_iterations: int = 10,
        structured_retries: int = 1,
    ) -> None:
        from llmkit.providers import get_provider_class, parse_model

        provider_name, model_name = parse_model(model)
        provider_cls = get_provider_class(provider_name)
        self._provider = provider_cls(model=model_name, api_key=api_key)
        self._model_name = model_name
        self._system = system
        self._messages: list[Message] = []
        self._tools = ToolRegistry()
        self._max_tool_iterations = max_tool_iterations
        self._structured_retries = structured_retries

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    def tool(self, fn: Any) -> Any:
        """Decorator to register a tool on this chat."""
        return self._tools.register(fn)

    async def send(
        self,
        content: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> Reply:
        self._messages.append(Message(role="user", content=content))

        for _ in range(self._max_tool_iterations):
            reply: Reply = await self._provider.send(
                self._messages,
                system=self._system,
                tools=self._tools.list() if self._tools.list() else None,
                response_model=response_model,
            )

            if not reply.tool_calls:
                self._messages.append(
                    Message(role="assistant", content=reply.text)
                )
                return reply

            # Handle tool calls
            self._messages.append(
                Message(role="assistant", content=None, tool_calls=reply.tool_calls)
            )
            for tc in reply.tool_calls:
                try:
                    result = await self._tools.acall(tc.name, tc.args)
                    result_str = str(result)
                except Exception as e:
                    result_str = f"Error: {e}"

                self._messages.append(
                    Message(
                        role="tool",
                        content=result_str,
                        tool_calls=[ToolCall(id=tc.id, name=tc.name, args=tc.args, result=result_str)],
                    )
                )

        msg = f"Tool loop exceeded {self._max_tool_iterations} iterations"
        raise ToolError(msg)

    def send_sync(
        self,
        content: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> Reply:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.send(content, response_model=response_model)
                ).result()
        return asyncio.run(self.send(content, response_model=response_model))

    async def stream(
        self,
        content: str,
    ) -> AsyncIterator[Reply]:
        self._messages.append(Message(role="user", content=content))
        async for chunk in self._provider.stream(
            self._messages,
            system=self._system,
            tools=self._tools.list() if self._tools.list() else None,
        ):
            yield chunk
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_chat.py -v
```

Expected: all 4 PASS.

**Step 5: Commit**

```bash
git add src/llmkit/chat.py tests/test_chat.py
git commit -m "feat: add Chat class with tool loop and sync convenience"
```

---

### Task 6: OpenAI Provider

**Files:**
- Create: `src/llmkit/providers/openai.py`
- Create: `tests/providers/__init__.py`
- Create: `tests/providers/test_openai.py`

This provider translates between llmkit types and the OpenAI SDK. Tests use mocking to avoid real API calls.

**Step 1: Write the failing tests**

`tests/providers/__init__.py`: empty file.

`tests/providers/test_openai.py`:
```python
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit.providers.openai import OpenAIProvider
from llmkit.types import Message, ToolDef


@pytest.fixture
def provider():
    return OpenAIProvider(model="gpt-4o", api_key="sk-test")


def test_init(provider: OpenAIProvider):
    assert provider._model == "gpt-4o"


async def test_send_simple(provider: OpenAIProvider):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello!"
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="Hi")])

    assert reply.text == "Hello!"
    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 5
    assert reply.tool_calls == []


async def test_send_with_tools(provider: OpenAIProvider):
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"query": "test"}'

    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    tools = [ToolDef(name="search", description="Search", parameters={"type": "object", "properties": {"query": {"type": "string"}}})]

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="search for test")], tools=tools)

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "search"
    assert reply.tool_calls[0].args == {"query": "test"}


async def test_send_with_system(provider: OpenAIProvider):
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "I am helpful"
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 3

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response
    ) as mock_create:
        await provider.send(
            [Message(role="user", content="Who are you?")],
            system="You are helpful.",
        )
        call_args = mock_create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/providers/test_openai.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement openai provider**

`src/llmkit/providers/openai.py`:
```python
from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage


class OpenAIProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply:
        oai_messages = _build_messages(messages, system)
        req_kwargs: dict[str, Any] = {"model": self._model, "messages": oai_messages}

        if tools:
            req_kwargs["tools"] = [_tool_to_oai(t) for t in tools]

        if response_model is not None:
            req_kwargs["response_format"] = _response_format(response_model)

        response = await self._client.chat.completions.create(**req_kwargs)
        return _parse_response(response, response_model)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]:
        oai_messages = _build_messages(messages, system)
        req_kwargs: dict[str, Any] = {"model": self._model, "messages": oai_messages, "stream": True}

        if tools:
            req_kwargs["tools"] = [_tool_to_oai(t) for t in tools]

        stream = await self._client.chat.completions.create(**req_kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield Reply(
                    text=delta.content,
                    parsed=None,
                    tool_calls=[],
                    usage=Usage(input_tokens=0, output_tokens=0),
                    raw=chunk,
                )


def _build_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    oai: list[dict[str, Any]] = []
    if system:
        oai.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "tool" and msg.tool_calls:
            tc = msg.tool_calls[0]
            oai.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": msg.content or "",
            })
        elif msg.role == "assistant" and msg.tool_calls:
            oai.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in msg.tool_calls
                ],
            })
        else:
            oai.append({"role": msg.role, "content": msg.content or ""})
    return oai


def _tool_to_oai(tool: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _response_format(model: type) -> dict[str, Any]:
    from pydantic import BaseModel

    if issubclass(model, BaseModel):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": model.model_json_schema(),
                "strict": True,
            },
        }
    msg = "response_model must be a Pydantic BaseModel subclass"
    raise TypeError(msg)


def _parse_response(response: Any, response_model: type | None) -> Reply:
    choice = response.choices[0]
    text = choice.message.content
    tool_calls: list[ToolCall] = []

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                )
            )

    parsed = None
    if response_model is not None and text:
        from pydantic import BaseModel

        if issubclass(response_model, BaseModel):
            parsed = response_model.model_validate_json(text)

    return Reply(
        text=text,
        parsed=parsed,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        ),
        raw=response,
    )
```

**Step 4: Register the provider**

Add to `src/llmkit/providers/__init__.py` at the bottom:

```python
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


_register_builtins()
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/providers/test_openai.py -v
```

Expected: all 4 PASS.

**Step 6: Commit**

```bash
git add src/llmkit/providers/ tests/providers/
git commit -m "feat: add OpenAI provider"
```

---

### Task 7: Anthropic Provider

**Files:**
- Create: `src/llmkit/providers/anthropic.py`
- Create: `tests/providers/test_anthropic.py`

**Step 1: Write the failing tests**

`tests/providers/test_anthropic.py`:
```python
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit.providers.anthropic import AnthropicProvider
from llmkit.types import Message, ToolDef


@pytest.fixture
def provider():
    return AnthropicProvider(model="claude-sonnet-4-20250514", api_key="sk-test")


def test_init(provider: AnthropicProvider):
    assert provider._model == "claude-sonnet-4-20250514"


async def test_send_simple(provider: AnthropicProvider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello!")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    with patch.object(
        provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="Hi")])

    assert reply.text == "Hello!"
    assert reply.usage.input_tokens == 10
    assert reply.tool_calls == []


async def test_send_with_system(provider: AnthropicProvider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="I am helpful")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 15
    mock_response.usage.output_tokens = 3

    with patch.object(
        provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
    ) as mock_create:
        await provider.send(
            [Message(role="user", content="Who are you?")],
            system="You are helpful.",
        )
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."


async def test_send_with_tools(provider: AnthropicProvider):
    mock_tc = MagicMock()
    mock_tc.type = "tool_use"
    mock_tc.id = "toolu_123"
    mock_tc.name = "search"
    mock_tc.input = {"query": "test"}

    mock_response = MagicMock()
    mock_response.content = [mock_tc]
    mock_response.stop_reason = "tool_use"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5

    tools = [ToolDef(name="search", description="Search", parameters={"type": "object", "properties": {"query": {"type": "string"}}})]

    with patch.object(
        provider._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="search")], tools=tools)

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "search"
    assert reply.tool_calls[0].args == {"query": "test"}
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/providers/test_anthropic.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement anthropic provider**

`src/llmkit/providers/anthropic.py`:
```python
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage


class AnthropicProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        self._model = model
        self._client = AsyncAnthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply:
        ant_messages = _build_messages(messages)
        req_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": ant_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system:
            req_kwargs["system"] = system

        if tools:
            req_kwargs["tools"] = [_tool_to_anthropic(t) for t in tools]

        if response_model is not None:
            from pydantic import BaseModel

            if issubclass(response_model, BaseModel):
                req_kwargs["tools"] = req_kwargs.get("tools", []) + [
                    {
                        "name": f"structured_{response_model.__name__}",
                        "description": f"Return structured {response_model.__name__} output",
                        "input_schema": response_model.model_json_schema(),
                    }
                ]
                req_kwargs["tool_choice"] = {"type": "tool", "name": f"structured_{response_model.__name__}"}

        response = await self._client.messages.create(**req_kwargs)
        return _parse_response(response, response_model)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]:
        ant_messages = _build_messages(messages)
        req_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": ant_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
        }

        if system:
            req_kwargs["system"] = system

        if tools:
            req_kwargs["tools"] = [_tool_to_anthropic(t) for t in tools]

        async with self._client.messages.stream(**{k: v for k, v in req_kwargs.items() if k != "stream"}) as stream:
            async for text in stream.text_stream:
                yield Reply(
                    text=text,
                    parsed=None,
                    tool_calls=[],
                    usage=Usage(input_tokens=0, output_tokens=0),
                    raw=None,
                )


def _build_messages(messages: list[Message]) -> list[dict[str, Any]]:
    ant: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool" and msg.tool_calls:
            tc = msg.tool_calls[0]
            ant.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": msg.content or "",
                    }
                ],
            })
        elif msg.role == "assistant" and msg.tool_calls:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                })
            ant.append({"role": "assistant", "content": content})
        else:
            ant.append({"role": msg.role, "content": msg.content or ""})
    return ant


def _tool_to_anthropic(tool: ToolDef) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _parse_response(response: Any, response_model: type | None) -> Reply:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    parsed = None

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            if response_model is not None and block.name.startswith("structured_"):
                from pydantic import BaseModel

                if issubclass(response_model, BaseModel):
                    parsed = response_model.model_validate(block.input)
            else:
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        args=block.input,
                    )
                )

    return Reply(
        text="\n".join(text_parts) if text_parts else None,
        parsed=parsed,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
        raw=response,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/providers/test_anthropic.py -v
```

Expected: all 4 PASS.

**Step 5: Commit**

```bash
git add src/llmkit/providers/anthropic.py tests/providers/test_anthropic.py
git commit -m "feat: add Anthropic provider"
```

---

### Task 8: Gemini Provider

**Files:**
- Create: `src/llmkit/providers/gemini.py`
- Create: `tests/providers/test_gemini.py`

**Step 1: Write the failing tests**

`tests/providers/test_gemini.py`:
```python
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit.providers.gemini import GeminiProvider
from llmkit.types import Message, ToolDef


@pytest.fixture
def provider():
    with patch("llmkit.providers.gemini.genai") as mock_genai:
        p = GeminiProvider(model="gemini-2.0-flash", api_key="test-key")
        yield p


def test_init(provider: GeminiProvider):
    assert provider._model == "gemini-2.0-flash"


async def test_send_simple(provider: GeminiProvider):
    mock_response = MagicMock()
    mock_response.text = "Hello!"
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [MagicMock(text="Hello!", function_call=None)]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5

    with patch.object(
        provider._client.aio.models, "generate_content", new_callable=AsyncMock, return_value=mock_response
    ):
        reply = await provider.send([Message(role="user", content="Hi")])

    assert reply.text == "Hello!"
    assert reply.usage.input_tokens == 10
    assert reply.tool_calls == []
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/providers/test_gemini.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement gemini provider**

`src/llmkit/providers/gemini.py`:
```python
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


class GeminiProvider:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        if genai is None:
            msg = "google-genai package is required: pip install llmkit[gemini]"
            raise ImportError(msg)

        self._model = model
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    async def send(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        response_model: type | None = None,
        **kwargs: Any,
    ) -> Reply:
        contents = _build_contents(messages)
        config: dict[str, Any] = {}

        if system:
            config["system_instruction"] = system

        if tools:
            config["tools"] = [_tools_to_gemini(tools)]

        if response_model is not None:
            from pydantic import BaseModel

            if issubclass(response_model, BaseModel):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_model.model_json_schema()

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config) if config else None,
        )
        return _parse_response(response, response_model)

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Reply]:
        contents = _build_contents(messages)
        config: dict[str, Any] = {}

        if system:
            config["system_instruction"] = system

        if tools:
            config["tools"] = [_tools_to_gemini(tools)]

        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config) if config else None,
        ):
            if chunk.text:
                yield Reply(
                    text=chunk.text,
                    parsed=None,
                    tool_calls=[],
                    usage=Usage(input_tokens=0, output_tokens=0),
                    raw=chunk,
                )


def _build_contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        if msg.role == "tool" and msg.tool_calls:
            tc = msg.tool_calls[0]
            contents.append({
                "role": "user",
                "parts": [{"function_response": {"name": tc.name, "response": {"result": msg.content or ""}}}],
            })
        elif msg.role == "assistant" and msg.tool_calls:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for tc in msg.tool_calls:
                parts.append({"function_call": {"name": tc.name, "args": tc.args}})
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": role, "parts": [{"text": msg.content or ""}]})
    return contents


def _tools_to_gemini(tools: list[ToolDef]) -> dict[str, Any]:
    declarations = []
    for t in tools:
        declarations.append({
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        })
    return {"function_declarations": declarations}


def _parse_response(response: Any, response_model: type | None) -> Reply:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for part in response.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(
                ToolCall(
                    id=fc.name,  # Gemini doesn't have separate IDs
                    name=fc.name,
                    args=dict(fc.args) if fc.args else {},
                )
            )
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    text = "\n".join(text_parts) if text_parts else None

    parsed = None
    if response_model is not None and text:
        import json

        from pydantic import BaseModel

        if issubclass(response_model, BaseModel):
            parsed = response_model.model_validate(json.loads(text))

    return Reply(
        text=text,
        parsed=parsed,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        ),
        raw=response,
    )
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/providers/test_gemini.py -v
```

Expected: all 2 PASS.

**Step 5: Commit**

```bash
git add src/llmkit/providers/gemini.py tests/providers/test_gemini.py
git commit -m "feat: add Gemini provider"
```

---

### Task 9: Public API Exports and __init__.py

**Files:**
- Modify: `src/llmkit/__init__.py`

**Step 1: Write the failing test**

`tests/test_init.py`:
```python
def test_public_api():
    from llmkit import Chat, register_provider
    from llmkit.types import Message, Reply, ToolCall, ToolDef, Usage
    from llmkit.exceptions import LLMKitError, ParseError, ProviderError, ToolError

    assert Chat is not None
    assert register_provider is not None
    assert Message is not None
    assert Reply is not None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_init.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Update __init__.py**

`src/llmkit/__init__.py`:
```python
"""llmkit — minimal, typed Python LLM wrapper."""

from llmkit.chat import Chat
from llmkit.providers import register_provider

__all__ = ["Chat", "register_provider"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_init.py -v
```

Expected: PASS.

**Step 5: Run full test suite**

```bash
uv run pytest -v
uv run ruff check src/ tests/
uv run ty check src/
```

All should pass.

**Step 6: Commit**

```bash
git add src/llmkit/__init__.py tests/test_init.py
git commit -m "feat: wire up public API exports"
```

---

### Task 10: Integration Smoke Test (Manual)

This is not an automated test — it verifies the library works end-to-end with real APIs.

**Step 1: Create a smoke test script**

`examples/smoke.py`:
```python
"""Smoke test — run with real API keys to verify end-to-end."""

import asyncio

from pydantic import BaseModel

from llmkit import Chat


class City(BaseModel):
    name: str
    country: str
    population: int


async def main():
    # Test basic send
    chat = Chat("openai/gpt-4o-mini", system="Be concise.")
    reply = await chat.send("What is 2+2?")
    print(f"OpenAI: {reply.text}")
    print(f"Usage: {reply.usage}")

    # Test structured output
    chat2 = Chat("anthropic/claude-sonnet-4-20250514")
    reply2 = await chat2.send("Tell me about Amsterdam", response_model=City)
    print(f"Anthropic structured: {reply2.parsed}")

    # Test tools
    chat3 = Chat("openai/gpt-4o-mini")

    @chat3.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    reply3 = await chat3.send("What is 7 times 8?")
    print(f"Tool result: {reply3.text}")

    # Test multi-turn
    chat4 = Chat("anthropic/claude-sonnet-4-20250514", system="Be concise.")
    await chat4.send("My name is Job")
    reply4 = await chat4.send("What is my name?")
    print(f"Multi-turn: {reply4.text}")


asyncio.run(main())
```

**Step 2: Run smoke test (requires API keys)**

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... uv run python examples/smoke.py
```

**Step 3: Commit**

```bash
git add examples/smoke.py
git commit -m "feat: add smoke test example"
```
