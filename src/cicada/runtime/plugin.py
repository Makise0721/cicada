"""插件定义、实例与上下文."""

from __future__ import annotations

import enum
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cicada.runtime.scope import Cleanup, ResourceScope

if TYPE_CHECKING:
    from cicada.runtime.runtime import PluginRuntime

SetupFn = Callable[["PluginContext"], Awaitable[None] | None]


class DeclarationError(RuntimeError):
    """provide/require 与声明不符."""


@dataclass(frozen=True)
class PluginDefinition:
    """插件定义身份; 与运行实例身份分开."""

    name: str
    setup: SetupFn
    provides: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()


class InstanceState(enum.Enum):
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class PluginInstance:
    """插件实例身份; 与定义身份分开."""

    definition: PluginDefinition
    scope: ResourceScope
    state: InstanceState = InstanceState.STARTING
    error: BaseException | None = None


class PluginContext:
    """插件在 setup 中看到的运行时入口: 声明内 provide/require + 资源登记."""

    def __init__(self, instance: PluginInstance, runtime: "PluginRuntime") -> None:
        self._instance = instance
        self._runtime = runtime
        self._provided: set[str] = set()

    @property
    def plugin_name(self) -> str:
        return self._instance.definition.name

    def provide(self, capability: str, value: Any) -> None:
        if capability not in self._instance.definition.provides:
            raise DeclarationError(
                f"plugin {self.plugin_name!r} did not declare provides {capability!r}"
            )
        if capability in self._provided:
            raise DeclarationError(
                f"plugin {self.plugin_name!r} provided {capability!r} twice"
            )
        self._provided.add(capability)
        self._runtime._publish(capability, self.plugin_name, value)
        # 能力注册本身是资源: 随实例 scope 关闭而撤销
        self._instance.scope.defer(lambda: self._runtime._retract(capability, self.plugin_name))

    def require(self, capability: str) -> Any:
        if capability not in self._instance.definition.requires:
            raise DeclarationError(
                f"plugin {self.plugin_name!r} did not declare requires {capability!r}"
            )
        return self._runtime._lookup(capability)

    def defer(self, cleanup: Cleanup) -> None:
        self._instance.scope.defer(cleanup)

