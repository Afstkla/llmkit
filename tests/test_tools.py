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
