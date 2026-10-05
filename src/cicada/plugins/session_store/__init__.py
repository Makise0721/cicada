"""会话保存恢复的 SQLite 存储能力 (会话保存恢复 v1).

对外只暴露四样东西: open_session_store (打开/新建), session_store_plugin
(把已打开的 store 交给 runtime 拥有), SQLiteSessionStore (port 实现),
list_sessions (工作区只读分页列表, 05).
"""

from cicada.plugins.session_store.listing import list_sessions
from cicada.plugins.session_store.plugin import (
    SESSION_STORE_CAPABILITY,
    session_store_plugin,
)
from cicada.plugins.session_store.store import SQLiteSessionStore, open_session_store

__all__ = [
    "SESSION_STORE_CAPABILITY",
    "SQLiteSessionStore",
    "list_sessions",
    "open_session_store",
    "session_store_plugin",
]
