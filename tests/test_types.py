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
