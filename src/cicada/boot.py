"""启动组装: 加载插件 -> 拓扑激活 -> 按显式名单解析模型与工具 -> 注入内核.

唯一点允许同时依赖三层; 插件启动失败或必需能力缺失即 BootError.
"""

from __future__ import annotations

from dataclasses import dataclass

from cicada.core.agent import Agent
from cicada.runtime.plugin import PluginDefinition
from cicada.runtime.runtime import CapabilityError, PluginRuntime, StartupReport


class BootError(RuntimeError):
    """插件启动失败或必需能力缺失."""


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
) -> App:
    if not isinstance(system_prompt, str):
        raise TypeError(f"system_prompt must be a str, got {type(system_prompt).__name__}")
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
    except CapabilityError as exc:
        raise BootError(f"required capability missing: {exc}") from exc
    return App(
        runtime=runtime,
        agent=Agent(model=model, tools=tools, system_prompt=system_prompt),
        report=report,
    )


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
