"""把已打开的会话存储交给插件 runtime 拥有 (会话保存恢复 v1).

store 由应用协调器在 bootstrap 前打开; 插件只负责 provide 与最后清理:
注册名单中排在首位且不 require 任何能力 -> 拓扑序最先激活 -> runtime stop
逆序清理时最后释放 (模型/进程资源先清理). 02 的活动包装器将通过显式
requires 建立同样的顺序约束.
"""

from __future__ import annotations

from cicada.plugins.session_store.store import SQLiteSessionStore
from cicada.runtime.plugin import PluginContext, PluginDefinition

SESSION_STORE_CAPABILITY = "session.store"


def session_store_plugin(store: SQLiteSessionStore) -> PluginDefinition:
    def setup(ctx: PluginContext) -> None:
        ctx.provide(SESSION_STORE_CAPABILITY, store)
        ctx.defer(store.aclose)

    return PluginDefinition(
        name="session-store",
        setup=setup,
        provides=frozenset({SESSION_STORE_CAPABILITY}),
    )
