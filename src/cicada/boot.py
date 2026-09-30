"""启动组装: 加载插件 -> 拓扑激活 -> 解析模型与工具 -> 注入内核.

唯一点允许同时依赖三层; 必需能力缺失或插件失败即 BootError.
"""

from __future__ import annotations

from dataclasses import dataclass

from cicada.core.agent import Agent
from cicada.runtime.plugin import PluginDefinition
from cicada.runtime.runtime import PluginRuntime, StartupReport


class BootError(RuntimeError):
    """必需能力缺失或插件启动失败."""


@dataclass
class App:
    runtime: PluginRuntime
    agent: Agent
    report: StartupReport

    async def aclose(self) -> None:
        await self.runtime.stop()


async def bootstrap(definitions: list[PluginDefinition]) -> App:
    runtime = PluginRuntime()
    for definition in definitions:
        runtime.register(definition)
    report = await runtime.start()
    if report.failed:
        details = "; ".join(f"{name}: {runtime.instance(name).error}" for name in report.failed)
        raise BootError(f"plugin startup failed: {details}")
    model = runtime.capability("model")
    tools = {tool.spec.name: tool for tool in runtime.capability("tools")}
    return App(runtime=runtime, agent=Agent(model=model, tools=tools), report=report)
