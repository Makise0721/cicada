"""启动组装: 加载插件 -> 拓扑激活 -> 按显式名单解析模型与工具 -> 注入内核.

唯一点允许同时依赖三层; 插件启动失败或必需能力缺失即 BootError.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect

from cicada.core.agent import Agent
from cicada.core.ports import ModelPort
from cicada.runtime.plugin import PluginDefinition
from cicada.runtime.runtime import CapabilityError, PluginRuntime, StartupReport


class BootError(RuntimeError):
    """插件启动失败或必需能力缺失."""


ModelPolicy = Callable[[ModelPort, PluginRuntime], ModelPort]


@dataclass
class App:
    runtime: PluginRuntime
    agent: Agent
    report: StartupReport

    async def aclose(self) -> None:
        await self.runtime.stop()


async def bootstrap(
    definitions: list[PluginDefinition],
    *,
    tool_capabilities: tuple[str, ...],
    model_capability: str = "model",
    system_prompt: str = "",
    model_policy: ModelPolicy | None = None,
    max_turns: int = 50,
) -> App:
    if not isinstance(system_prompt, str):
        raise TypeError(f"system_prompt must be a str, got {type(system_prompt).__name__}")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise ValueError("max_turns must be a positive integer")
    if model_policy is not None and not callable(model_policy):
        raise TypeError("model_policy must be a synchronous callable or None")
    runtime = PluginRuntime()
    for definition in definitions:
        runtime.register(definition)
    report = await runtime.start()
    if report.failed:
        details = "; ".join(f"{name}: {runtime.instance(name).error}" for name in report.failed)
        raise BootError(f"plugin startup failed: {details}")
    try:
        model = runtime.capability(model_capability)
        tools = {tool.spec.name: tool for name in tool_capabilities for tool in _as_tools(runtime.capability(name), name)}
        if model_policy is not None:
            model = model_policy(model, runtime)
            if inspect.isawaitable(model):
                if inspect.iscoroutine(model):
                    model.close()
                raise BootError("model_policy must return a ModelPort synchronously")
            if not callable(getattr(model, "stream", None)):
                raise BootError("model_policy did not return a ModelPort")
        return App(
            runtime=runtime,
            agent=Agent(model=model, tools=tools, system_prompt=system_prompt, max_turns=max_turns),
            report=report,
        )
    except BaseException as exc:
        # 策略和工具组装都发生在 start 之后；失败必须回收已发布能力。
        try:
            await runtime.stop()
        except BaseException as cleanup_error:
            exc.add_note(f"runtime cleanup failed: {cleanup_error}")
        if not isinstance(exc, Exception) or isinstance(exc, BootError):
            raise
        prefix = "required capability missing" if isinstance(exc, CapabilityError) else "application assembly failed"
        raise BootError(f"{prefix}: {exc}") from exc


def _as_tools(value: object, capability: str) -> list:
    """能力值可以是单个 Tool 或 Tool 列表; 其余形态视为组装错误."""
    if hasattr(value, "spec"):
        return [value]
    try:
        tools = list(value)  # type: ignore[arg-type]
    except TypeError:
        tools = []
    if not tools or not all(hasattr(tool, "spec") for tool in tools):
        raise BootError(f"capability {capability!r} did not provide tool(s)")
    return tools
