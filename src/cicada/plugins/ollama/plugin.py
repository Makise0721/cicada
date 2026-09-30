"""ollama 插件工厂: 提供 "model" 能力, AsyncClient 生命周期挂实例 scope."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from cicada.plugins.ollama.model import OllamaModel
from cicada.plugins.ollama.protocol import OllamaConfig
from cicada.runtime.plugin import PluginDefinition

if TYPE_CHECKING:
    from cicada.runtime.plugin import PluginContext


def ollama_plugin(config: OllamaConfig | None = None) -> PluginDefinition:
    """model 插件: setup 中创建 AsyncClient 并以 ctx.defer(client.aclose) 登记清理."""
    resolved = config or OllamaConfig()

    def setup(ctx: PluginContext) -> None:
        client = httpx.AsyncClient()
        ctx.defer(client.aclose)
        ctx.provide("model", OllamaModel(client, resolved))

    return PluginDefinition(
        name="ollama-model",
        setup=setup,
        provides=frozenset({"model"}),
        requires=frozenset(),
    )
