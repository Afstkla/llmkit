
import builtins
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
            tool_name = name or getattr(f, "__name__", str(f))
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

    def list(self) -> builtins.list[ToolDef]:
        return builtins.list(self._tools.values())

    def call(self, name: str, args: dict[str, Any]) -> Any:
        fn = self._callables[name]
        return fn(**args)

    async def acall(self, name: str, args: dict[str, Any]) -> Any:
        fn = self._callables[name]
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def _extract_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON Schema from function signature using Pydantic."""
    sig = inspect.signature(fn)
    annotations = getattr(fn, "__annotations__", {})
    hints = {k: v for k, v in annotations.items() if k != "return"}

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
