
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from llmkit.agent import Agent
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


def _make_chat(provider: FakeProvider, system: str | None = None) -> Agent:
    """Create a Agent instance with a fake provider, bypassing __init__."""
    chat = object.__new__(Agent)
    chat._provider = provider
    chat._system = system
    chat._messages = []
    chat._tools = ToolRegistry()
    chat._model_name = "fake/test"
    chat._hosted_tools = []
    chat._hooks = defaultdict(list)
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


async def test_as_tool():
    provider = FakeProvider([_simple_reply("Tool response")])
    agent = _make_chat(provider)

    tool_fn = agent.as_tool(name="my_agent", description="A helpful agent")
    assert tool_fn.__name__ == "my_agent"
    assert tool_fn.__doc__ == "A helpful agent"
    assert tool_fn.__annotations__ == {"message": str, "return": str}

    result = await tool_fn(message="Hello")
    assert result == "Tool response"


async def test_as_tool_empty_response():
    provider = FakeProvider([_simple_reply(None)])
    agent = _make_chat(provider)

    tool_fn = agent.as_tool(name="agent", description="An agent")
    result = await tool_fn(message="Hello")
    assert result == ""


async def test_parallel_tool_calls():
    provider = FakeProvider([
        Reply(
            text=None,
            parsed=None,
            tool_calls=[
                ToolCall(id="call_1", name="add", args={"a": 1, "b": 2}),
                ToolCall(id="call_2", name="multiply", args={"a": 3, "b": 4}),
            ],
            usage=Usage(input_tokens=10, output_tokens=5),
            raw={},
        ),
        _simple_reply("1+2=3 and 3*4=12"),
    ])
    agent = _make_chat(provider)

    @agent.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @agent.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    reply = await agent.send("What is 1+2 and 3*4?")
    assert reply.text == "1+2=3 and 3*4=12"
    # Should have: user, assistant (tool calls), tool result 1, tool result 2, assistant (final)
    assert len(agent.messages) == 5
    assert agent.messages[2].role == "tool"
    assert agent.messages[2].content == "3"
    assert agent.messages[3].role == "tool"
    assert agent.messages[3].content == "12"


async def test_hook_turn_start_end():
    provider = FakeProvider([_simple_reply("Hi")])
    agent = _make_chat(provider)
    events: list[str] = []

    @agent.on("turn_start")
    def on_start(messages: list[Message]) -> None:
        events.append(f"start:{len(messages)}")

    @agent.on("turn_end")
    def on_end(reply: Reply) -> None:
        events.append(f"end:{reply.text}")

    await agent.send("Hello")
    assert events == ["start:1", "end:Hi"]


async def test_hook_tool_call_start_end():
    provider = FakeProvider([
        _tool_call_reply("add", {"a": 1, "b": 2}),
        _simple_reply("3"),
    ])
    agent = _make_chat(provider)
    events: list[str] = []

    @agent.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @agent.on("tool_call_start")
    def on_tool_start(name: str, args: dict[str, Any]) -> None:
        events.append(f"call:{name}")

    @agent.on("tool_call_end")
    def on_tool_end(name: str, args: dict[str, Any], result: str) -> None:
        events.append(f"result:{name}={result}")

    await agent.send("What is 1+2?")
    assert events == ["call:add", "result:add=3"]


async def test_hook_async():
    provider = FakeProvider([_simple_reply("Hi")])
    agent = _make_chat(provider)
    events: list[str] = []

    @agent.on("turn_end")
    async def on_end(reply: Reply) -> None:
        events.append(f"async:{reply.text}")

    await agent.send("Hello")
    assert events == ["async:Hi"]


async def test_hook_multiple_handlers():
    provider = FakeProvider([_simple_reply("Hi")])
    agent = _make_chat(provider)
    events: list[str] = []

    @agent.on("turn_start")
    def first(messages: list[Message]) -> None:
        events.append("first")

    @agent.on("turn_start")
    def second(messages: list[Message]) -> None:
        events.append("second")

    await agent.send("Hello")
    assert events == ["first", "second"]
