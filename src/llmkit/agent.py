
import asyncio
import inspect
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, cast

from pydantic import BaseModel

from llmkit.exceptions import ToolError
from llmkit.hosted_tools import HostedTool
from llmkit.tools import ToolRegistry
from llmkit.types import Message, Reply, ToolCall

type Event = Literal["turn_start", "turn_end", "tool_call_start", "tool_call_end"]

type TurnStartHook = Callable[[list[Message]], None] | Callable[[list[Message]], Awaitable[None]]
type TurnEndHook = Callable[[Reply], None] | Callable[[Reply], Awaitable[None]]
type ToolCallStartHook = (
    Callable[[str, dict[str, Any]], None] | Callable[[str, dict[str, Any]], Awaitable[None]]
)
type ToolCallEndHook = (
    Callable[[str, dict[str, Any], str], None]
    | Callable[[str, dict[str, Any], str], Awaitable[None]]
)
type Hook = TurnStartHook | TurnEndHook | ToolCallStartHook | ToolCallEndHook


class Agent:
    def __init__(
        self,
        model: str,
        *,
        system: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        aws_region: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        hosted_tools: list[HostedTool] | None = None,
        max_tool_iterations: int = 10,
        structured_retries: int = 1,
    ) -> None:
        from llmkit.providers import get_provider_class, parse_model

        provider_name, model_name = parse_model(model)
        provider_cls = get_provider_class(provider_name)

        # Build provider kwargs from non-None values
        provider_kwargs: dict[str, Any] = {"model": model_name, "api_key": api_key}
        for key, val in [
            ("base_url", base_url),
            ("api_version", api_version),
            ("aws_region", aws_region),
            ("project_id", project_id),
            ("region", region),
        ]:
            if val is not None:
                provider_kwargs[key] = val

        self._provider = provider_cls(**provider_kwargs)
        self._model_name = model_name
        self._system = system
        self._messages: list[Message] = []
        self._tools = ToolRegistry()
        self._hosted_tools = hosted_tools or []
        self._hooks: dict[Event, list[Hook]] = defaultdict(list)
        self._max_tool_iterations = max_tool_iterations
        self._structured_retries = structured_retries

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    def tool(self, fn: Any) -> Any:
        """Decorator to register a tool on this agent."""
        return self._tools.register(fn)

    def on[F: Hook](self, event: Event) -> Callable[[F], F]:
        """Decorator to register a lifecycle hook.

        Events: turn_start, turn_end, tool_call_start, tool_call_end
        """
        def decorator(fn: F) -> F:
            self._hooks[event].append(fn)
            return fn
        return decorator

    async def _emit(self, event: Event, *args: Any) -> None:
        for fn in self._hooks.get(event, []):
            result = fn(*args)
            if inspect.isawaitable(result):
                await result

    def as_tool(self, *, name: str, description: str) -> Callable[..., Any]:
        """Turn this agent into a tool another agent can call."""
        async def _agent_tool(message: str) -> str:
            reply = await self.send(message)
            return reply.text or ""

        _agent_tool.__name__ = name
        _agent_tool.__doc__ = description
        _agent_tool.__annotations__ = {"message": str, "return": str}
        return _agent_tool

    async def send(
        self,
        content: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> Reply:
        self._messages.append(Message(role="user", content=content))

        for _ in range(self._max_tool_iterations):
            await self._emit("turn_start", self._messages)

            tools_list = self._tools.list()
            reply: Reply = await self._provider.send(
                self._messages,
                system=self._system,
                tools=tools_list if tools_list else None,
                hosted_tools=self._hosted_tools if self._hosted_tools else None,
                response_model=response_model,
            )

            await self._emit("turn_end", reply)

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
                await self._emit("tool_call_start", tc.name, tc.args)

                try:
                    result = await self._tools.acall(tc.name, tc.args)
                    result_str = str(result)
                except Exception as e:
                    result_str = f"Error: {e}"

                await self._emit("tool_call_end", tc.name, tc.args, result_str)

                tool_result = ToolCall(
                    id=tc.id, name=tc.name, args=tc.args, result=result_str,
                )
                self._messages.append(
                    Message(
                        role="tool",
                        content=result_str,
                        tool_calls=[tool_result],
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
                return cast(
                    Reply,
                    pool.submit(
                        asyncio.run, self.send(content, response_model=response_model)
                    ).result(),
                )
        return asyncio.run(self.send(content, response_model=response_model))

    async def stream(
        self,
        content: str,
    ) -> AsyncIterator[Any]:
        self._messages.append(Message(role="user", content=content))
        tools_list = self._tools.list()
        async for chunk in self._provider.stream(
            self._messages,
            system=self._system,
            tools=tools_list if tools_list else None,
            hosted_tools=self._hosted_tools if self._hosted_tools else None,
        ):
            yield chunk


# Backwards compatibility
Chat = Agent
