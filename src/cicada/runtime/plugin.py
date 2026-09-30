"""插件定义、实例与上下文."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
