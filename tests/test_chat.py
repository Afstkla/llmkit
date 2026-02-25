
from collections.abc import AsyncIterator
from typing import Any

from llmkit.chat import Chat
from llmkit.tools import ToolRegistry
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


def _make_chat(provider: FakeProvider, system: str | None = None) -> Chat:
    """Create a Chat instance with a fake provider, bypassing __init__."""
    chat = object.__new__(Chat)
    chat._provider = provider
    chat._system = system
    chat._messages = []
    chat._tools = ToolRegistry()
    chat._model_name = "fake/test"
    chat._max_tool_iterations = 10
    chat._structured_retries = 1
    return chat


async def test_send_basic():
    provider = FakeProvider([_simple_reply("Hello!")])
    chat = _make_chat(provider, system="You are helpful.")

    reply = await chat.send("Hi")
    assert reply.text == "Hello!"
    assert len(chat.messages) == 2  # user + assistant


async def test_send_maintains_history():
    provider = FakeProvider([_simple_reply("First"), _simple_reply("Second")])
    chat = _make_chat(provider)

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
    chat = _make_chat(provider)

    @chat.tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    reply = await chat.send("What is 5 doubled?")
    assert reply.text == "The answer is 10"


def test_send_sync():
    provider = FakeProvider([_simple_reply("Sync reply")])
    chat = _make_chat(provider)

    reply = chat.send_sync("Hello")
    assert reply.text == "Sync reply"
