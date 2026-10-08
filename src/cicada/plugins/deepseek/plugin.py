"""DeepSeek 插件：环境凭据只在 setup 使用，client 归插件 scope 所有。"""

import os
import httpx

from cicada.runtime.plugin import PluginDefinition
from cicada.plugins.deepseek.protocol import DeepSeekConfig
from cicada.plugins.deepseek.model import DeepSeekModel


def deepseek_plugin(config: DeepSeekConfig | None = None, *, api_key: str | None = None,
                    transport: httpx.AsyncBaseTransport | None = None) -> PluginDefinition:
    resolved = config or DeepSeekConfig()

    def setup(ctx) -> None:
        key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        if not isinstance(key, str) or not key.strip() or any(c in key for c in "\r\n"):
            raise ValueError("DEEPSEEK_API_KEY is missing or invalid")
        client = httpx.AsyncClient(transport=transport, trust_env=False)
        ctx.defer(client.aclose)
        ctx.provide("model", DeepSeekModel(client, resolved, key))

    return PluginDefinition(name="deepseek-model", setup=setup,
                            provides=frozenset({"model"}), requires=frozenset())
