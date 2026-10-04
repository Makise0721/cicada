"""会话保存恢复的 SQLite 存储能力 (会话保存恢复 v1).

对外只暴露三样东西: open_session_store (打开/新建), session_store_plugin
(把已打开的 store 交给 runtime 拥有), SQLiteSessionStore (port 实现).
"""

from cicada.plugins.session_store.plugin import (
    SESSION_STORE_CAPABILITY,
    session_store_plugin,
)
from cicada.plugins.session_store.store import SQLiteSessionStore, open_session_store

__all__ = [
    "SESSION_STORE_CAPABILITY",
    "SQLiteSessionStore",
    "open_session_store",
    "session_store_plugin",
]
