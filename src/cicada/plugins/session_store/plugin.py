"""把已打开的会话存储交给插件 runtime 拥有 (会话保存恢复 v1).

store 由应用协调器在 bootstrap 前打开; 插件只负责 provide 与最后清理.
顺序权威在应用层的显式依赖: open_application 的持久装配 (_bind_store_last_cleanup)
给其余插件定义追加 require "session.store" -> store 拓扑序最先激活 -> runtime stop
逆序清理时最后释放 (模型/进程资源先清理). 本插件自身零 requires; 注册名单中
排在首位只是同一顺序的冗余保险, 不是机制本身.
"""

from __future__ import annotations

from cicada.plugins.session_store.store import SQLiteSessionStore
from cicada.persistence.contracts import SessionCatalogPort
from cicada.runtime.plugin import PluginContext, PluginDefinition

SESSION_STORE_CAPABILITY = "session.store"
SESSION_CATALOG_CAPABILITY = "session.catalog"


def session_store_plugin(store: SQLiteSessionStore, catalog: SessionCatalogPort | None = None) -> PluginDefinition:
    def setup(ctx: PluginContext) -> None:
        ctx.provide(SESSION_STORE_CAPABILITY, store)
        ctx.defer(store.aclose)
        if catalog is not None:
            ctx.provide(SESSION_CATALOG_CAPABILITY, catalog)
            ctx.defer(catalog.aclose)

    return PluginDefinition(
        name="session-store",
        setup=setup,
        provides=frozenset({SESSION_STORE_CAPABILITY, SESSION_CATALOG_CAPABILITY}
                           if catalog is not None else {SESSION_STORE_CAPABILITY}),
    )
