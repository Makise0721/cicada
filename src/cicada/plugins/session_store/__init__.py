"""会话保存恢复的 SQLite 存储能力 (会话保存恢复 v1).

打开/恢复、存储 port、原只读列表与有界目录观察 reader；插件把已创建的
存储和 reader 交给 runtime 拥有，具体装配仍由应用负责。
"""

from cicada.plugins.session_store.listing import list_sessions
from cicada.plugins.session_store.catalog import SessionCatalogReader
from cicada.plugins.session_store.plugin import (
    SESSION_STORE_CAPABILITY,
    SESSION_CATALOG_CAPABILITY,
    session_store_plugin,
)
from cicada.plugins.session_store.store import SQLiteSessionStore, open_session_store, read_session_identity

__all__ = [
    "SESSION_STORE_CAPABILITY",
    "SESSION_CATALOG_CAPABILITY",
    "SessionCatalogReader",
    "SQLiteSessionStore",
    "list_sessions",
    "open_session_store",
    "read_session_identity",
    "session_store_plugin",
]
