"""插件运行时: 注册、拓扑激活、有序关闭. 不导入内核, 不知道会话消息."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from cicada.runtime.graph import resolve
from cicada.runtime.plugin import (
    DeclarationError,
    InstanceState,
    PluginContext,
    PluginDefinition,
    PluginInstance,
)
from cicada.runtime.scope import ResourceScope


class RegistrationError(RuntimeError):
    """重复注册或注册时机非法."""


class CapabilityError(RuntimeError):
    """能力不存在/已回收/运行时已停止."""


@dataclass(frozen=True)
class StartupReport:
    activated: tuple[str, ...]
    failed: tuple[str, ...]


class PluginRuntime:
    def __init__(self) -> None:
        self._definitions: list[PluginDefinition] = []
        self._instances: dict[str, PluginInstance] = {}
        self._capabilities: dict[str, tuple[str, Any]] = {}  # capability -> (provider 实例名, 值)
        self._activation_order: list[str] = []
        self._started = False
        self._stopped = False

    def register(self, definition: PluginDefinition) -> None:
        if self._started:
            raise RegistrationError("cannot register plugins after start")
        if any(d.name == definition.name for d in self._definitions):
            raise RegistrationError(f"duplicate plugin name {definition.name!r}")
        self._definitions.append(definition)

    def resolve(self) -> tuple[PluginDefinition, ...]:
        return resolve(tuple(self._definitions))

    async def start(self) -> StartupReport:
        if self._started:
            raise RuntimeError("runtime already started")
        self._started = True
        activated: list[str] = []
        failed: list[str] = []
        for definition in self.resolve():
            instance = PluginInstance(definition=definition, scope=ResourceScope(definition.name))
            self._instances[definition.name] = instance
            context = PluginContext(instance, self)
            try:
                outcome = definition.setup(context)
                if inspect.isawaitable(outcome):
                    await outcome
                missing = definition.provides - context._provided
                if missing:
                    raise DeclarationError(
                        f"plugin {definition.name!r} declared provides {sorted(missing)} but did not provide them"
                    )
            except BaseException as exc:
                instance.error = exc
                failed.append(definition.name)
                await self._teardown(instance)
                for name in reversed(activated):
                    await self._teardown(self._instances[name])
                activated.clear()
                break
            else:
                instance.state = InstanceState.ACTIVE
                self._activation_order.append(definition.name)
                activated.append(definition.name)
        return StartupReport(activated=tuple(activated), failed=tuple(failed))

    async def stop(self) -> None:
        """反向拓扑逐个关停: 消费者 scope 清理完成后才清理提供者 scope."""
        if self._stopped:
            return
        self._stopped = True
        errors: list[BaseException] = []
        for name in reversed(self._activation_order):
            instance = self._instances[name]
            if instance.state in (InstanceState.STOPPED, InstanceState.FAILED):
                continue
            outcome = await self._teardown(instance)
            if outcome is not None:
                errors.append(outcome)
        if errors:
            raise ExceptionGroup("runtime stop failed", errors)

    def capability(self, name: str) -> Any:
        if self._stopped:
            raise CapabilityError(f"runtime stopped; capability {name!r} unavailable")
        return self._lookup(name)

    def instance(self, name: str) -> PluginInstance:
        return self._instances[name]

    def _publish(self, capability: str, provider: str, value: Any) -> None:
        self._capabilities[capability] = (provider, value)

    def _retract(self, capability: str, provider: str) -> None:
        current = self._capabilities.get(capability)
        if current is not None and current[0] == provider:
            del self._capabilities[capability]

    def _lookup(self, capability: str) -> Any:
        current = self._capabilities.get(capability)
        if current is None:
            raise CapabilityError(f"capability {capability!r} is not available")
        return current[1]

    async def _teardown(self, instance: PluginInstance) -> ExceptionGroup | None:
        """清理实例 scope; 清理失败使实例 FAILED 并保留错误证据; 返回清理错误."""
        instance.state = InstanceState.STOPPING
        try:
            await instance.scope.aclose()
        except ExceptionGroup as exc:
            if instance.error is None:
                instance.error = exc
            instance.state = InstanceState.FAILED
            return exc
        instance.state = InstanceState.STOPPED if instance.error is None else InstanceState.FAILED
        return None
