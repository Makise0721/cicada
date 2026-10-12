"""@ 文件路径补全菜单 (project-context-v1 C6/C7): token、查询生命周期与渲染。

交互借鉴固定版本 Pi (pi-mono@1a584a7a packages/tui/src/autocomplete.ts 与
editor.ts, MIT): token 边界/引号、20ms 防抖、显式 Tab 绕过、结果按快照/
generation 核对。补全只改草稿 (Editor.replace_span 一次替换一次 undo);
候选不是附件, 提交保留原文, 正文由模型按现有 read 工具读取。

边界:
- 本模块是 UI 状态机: 查询只从输入路径启动, render 只读候选/短状态;
  候选端口经组合层注入 (应用只读出口), 本模块不导入 Coding 插件。
- UI 拥有它启动的查询任务: 最多一项运行/收尾工作加一份最新待执行输入;
  旧工作实际落定后才开新查询; 取消不显示错误, 过时成功/失败都丢弃。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from cicada.core.cancel import CancelToken
from cicada.file_completion import (
    SCOPE_UNAVAILABLE,
    FileCompletionCandidate,
    FileCompletionResult,
    is_separator,
)
from cicada.ui import theme
from cicada.ui.display_style import current_style
from cicada.ui.editor import Editor
from cicada.ui.main_screen import truncate_to_width

DEBOUNCE_S = 0.020

# 稳定短原因 -> 简短中文 (C3); 未知原因统一兜底, 不显示任意后端文本。
REASON_TEXT = {
    "capability_unavailable": "补全能力未装配",
    "session_closed": "会话已关闭",
    "unsupported_path": "不支持的路径",
    "not_git": "不是 Git 工作区",
    "root_mismatch": "工作区不是 Git 根目录",
    "git_unavailable": "Git 不可用",
    "git_launch_failed": "Git 启动失败",
    "git_failed": "Git 查询失败",
    "output_limit": "清单超过输出限额",
    "path_limit": "清单超过路径限额",
    "timeout": "查询超时",
    "path_unreadable": "路径不可读",
    "invalid_path": "路径不可用",
    "invalid_inventory": "Git 清单不可用",
    "alias_conflict": "路径别名冲突",
    "internal_error": "补全出错",
}
_UNAVAILABLE_STATUS = "候选不可用："
_TRUNCATED_STATUS = "仅显示前 20 项"


@dataclass(frozen=True)
class FileToken:
    """光标所在的独立 @ token: 从 `@` 到整段 token 结束 (含光标后尾部)。"""

    line: int  # 逻辑行号
    start: int  # '@' 起始列 (Python 字符索引)
    end: int  # token 结束列 (排他); 引号 token 含右引号
    query: str  # '@'/'@"' 之后、光标之前的路径文字
    quoted: bool  # token 是否使用 @"..." 双引号形式


def parse_at_token(line: str, cursor: int) -> FileToken | None:
    """解析当前逻辑行光标前的独立 @ token (line 固定为所在逻辑行号 0 视图)。

    独立 `@` 的前一位必须是行首、空白或分隔字符 (邮箱 `a@b` 不触发); 支持
    未加引号、`@"...` 与带右引号 token; 未闭合双引号是合法编辑中状态, 空格
    与标点在双引号内不分隔。其它引号排列或控制字符关闭菜单 (返回 None)。
    """
    if not 0 <= cursor <= len(line):
        return None
    before = line[:cursor]
    quotes = [index for index, ch in enumerate(before) if ch == '"']
    if len(quotes) % 2 == 1:
        start_quote = quotes[-1]  # 未闭合引号
        at_index = start_quote - 1
        if at_index < 0 or before[at_index] != "@" or not _is_token_start(before, at_index):
            return None  # 非 @" 引号形态: 关闭菜单并保留原草稿
        end = _quoted_token_end(line, cursor)
        return FileToken(0, at_index, end, before[start_quote + 1 :], True)
    last = max((index for index, ch in enumerate(before) if is_separator(ch)), default=-1)
    token_text = before[last + 1 :]
    if not token_text.startswith("@"):
        return None  # 普通文字/邮箱不触发
    end = _unquoted_token_end(line, cursor)
    token = FileToken(0, last + 1, end, token_text[1:], False)
    if any(ord(ch) < 0x20 or ch == "\x7f" for ch in token.query):
        return None  # 控制字符: 关闭菜单并保留原草稿
    return token


def _is_token_start(text: str, index: int) -> bool:
    return index == 0 or is_separator(text[index - 1])


def _unquoted_token_end(line: str, cursor: int) -> int:
    for index in range(cursor, len(line)):
        if is_separator(line[index]):
            return index
    return len(line)


def _quoted_token_end(line: str, cursor: int) -> int:
    closing = line.find('"', cursor)
    return len(line) if closing == -1 else closing + 1


class FileCompletionMenu:
    """候选菜单与查询生命周期; 由前端按输入路径驱动, 组合层注入候选端口。"""

    def __init__(
        self,
        editor: Editor,
        complete: Callable[[str, CancelToken], Awaitable[FileCompletionResult]],
        *,
        enter_barrier: Callable[[], bool] | None = None,
        on_update: Callable[[], None] | None = None,
    ) -> None:
        self._editor = editor
        self._complete = complete
        self._enter_barrier = enter_barrier if enter_barrier is not None else (lambda: False)
        self.on_update = on_update  # 异步结果落定后的重绘请求 (组合层注入)
        self._candidates: tuple[FileCompletionCandidate, ...] = ()
        self._index = 0
        self._status: str | None = None  # 已映射的短状态 (原因/截断标注)
        self._token: FileToken | None = None
        self._snapshot: tuple[str, int, int] | None = None  # (全文, 逻辑行, 逻辑列)
        self._query_snapshot: tuple[str, int, int] | None = None
        self._generation = 0
        self._task: asyncio.Task | None = None
        self._query_token: CancelToken | None = None
        self._pending: tuple[str, int] | None = None  # (query, generation) 最新待执行
        self._debounce: asyncio.TimerHandle | None = None
        self._lock: tuple[str, int, int] | None = None  # Esc 锁: 相同草稿/光标不重开
        self._closed = False

    # --- 只读视图 -------------------------------------------------------------------

    @property
    def candidates(self) -> tuple[FileCompletionCandidate, ...]:
        return self._candidates

    @property
    def selected(self) -> FileCompletionCandidate | None:
        return self._candidates[self._index] if self._candidates else None

    @property
    def visible(self) -> bool:
        return bool(self._candidates) or self._status is not None

    # --- 输入路径 (前端每次按键派发前调用) --------------------------------------------

    def sync(self, editor: Editor, *, eligible: bool) -> None:
        """按当前草稿全文、逻辑光标与资格建立快照; 变化即失效并重排查询。"""
        if self._closed:
            return
        if not eligible:
            self.invalidate()
            return
        text = editor.get_text()
        line = editor.state.cursor_line
        col = editor.state.cursor_col
        if self._lock is not None:
            if self._lock == (text, line, col):
                self.invalidate()  # Esc 锁住: 相同草稿/光标不重开
                return
            self._lock = None  # 文本/光标变化解锁
        snapshot = (text, line, col)
        if snapshot == self._snapshot:
            return  # 同一草稿/光标: 保留现有候选与在途查询
        self._snapshot = snapshot
        self._generation += 1
        self._cancel_work()
        self._candidates = ()
        self._index = 0
        self._status = None
        token = parse_at_token(editor.state.lines[line] or "", col)
        if token is None:
            self._token = None
            return
        self._token = FileToken(line, token.start, token.end, token.query, token.quoted)
        self._schedule_query(token.query)

    def move(self, delta: int) -> bool:
        """上下移动选择; 无候选时不消费按键 (归光标移动)。"""
        if not self._candidates:
            return False
        self._index = (self._index + delta) % len(self._candidates)
        return True

    def handle_tab(self) -> bool:
        """显式 Tab: 有候选接受; 有 token 无快照时跳过防抖立即查询。返回是否消费。"""
        if self._closed:
            return False
        self._lock = None  # 显式 Tab 解除 Esc 锁
        self.sync(self._editor, eligible=True)
        if self._candidates:
            return self.accept()
        if self._token is not None:
            self._schedule_query(self._token.query, immediate=True)
            return True
        return False

    def handle_enter(self) -> bool:
        """菜单展示候选时完成当前项并消费; 等待查询无候选则使查询失效走原提交。

        busy/finalizing 屏障优先: 即使存在候选也不能确认 (返回 False 落入
        原静默屏障), 屏障事实由组合层注入。
        """
        if self._closed:
            return False
        if self._candidates:
            if self._enter_barrier():
                return False
            return self.accept()
        self.invalidate()  # 迟到结果不可重开菜单
        return False

    def dismiss(self) -> bool:
        """Esc: 关闭、取消并锁住相同草稿/光标的触发。返回是否有可见变化。"""
        if self._closed:
            return False
        had = self.visible or self._task is not None or self._pending is not None
        self._lock = (
            self._editor.get_text(),
            self._editor.state.cursor_line,
            self._editor.state.cursor_col,
        )
        self.invalidate()
        return had

    def invalidate(self) -> None:
        """使候选与在途工作失效 (资格/焦点变化); 不设置 Esc 锁。"""
        self._generation += 1
        self._cancel_work()
        self._candidates = ()
        self._index = 0
        self._status = None
        self._token = None
        self._snapshot = None

    async def close(self) -> None:
        """封闭入口、取消防抖与当前查询并等待实际落定 (前端 finally 先于终端恢复)。"""
        self._closed = True
        self.invalidate()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    # --- 接受 -----------------------------------------------------------------------

    def accept(self) -> bool:
        """插入当前候选: 只替换当前 token, 一次 replace_span 一次 undo。

        借鉴 Pi: 已用双引号的 token 保持引号; 含空白/分隔字符的路径包双引号;
        文件后补一个分隔空格 (右侧已有空格则复用); 目录以 `/` 结尾不补空格,
        带引号时光标停在右引号前。完成文件后关闭; 完成目录生成新 query。
        """
        candidate = self.selected
        token = self._token
        if candidate is None or token is None:
            return False
        path = candidate.relative_path
        quoted = token.quoted or any(is_separator(ch) for ch in path)
        value = '@"' + path + '"' if quoted else "@" + path
        cursor_offset: int | None = None
        if candidate.is_directory:
            if quoted:
                cursor_offset = len(value) - 1  # 右引号前, 继续输入路径
        else:
            line_text = self._editor.state.lines[token.line] or ""
            if line_text[token.end : token.end + 1] != " ":
                value += " "  # 右侧已有空格时复用而不重复
        self._editor.replace_span(token.line, token.start, token.end, value,
                                  cursor_offset=cursor_offset)
        self._lock = None
        self.invalidate()
        self.sync(self._editor, eligible=True)  # 文件即关闭; 目录继续生成新 query
        return True

    # --- 查询生命周期 -----------------------------------------------------------------

    def _cancel_work(self) -> None:
        if self._debounce is not None:
            self._debounce.cancel()
            self._debounce = None
        self._pending = None
        if self._query_token is not None:
            self._query_token.cancel()  # 协作取消; 任务自身在 finally 清引用

    def _schedule_query(self, query: str, *, immediate: bool = False) -> None:
        self._pending = (query, self._generation)
        if self._debounce is not None:
            self._debounce.cancel()
            self._debounce = None
        if immediate:
            self._start_pending()
            return
        self._debounce = asyncio.get_running_loop().call_later(DEBOUNCE_S, self._start_pending)

    def _start_pending(self) -> None:
        self._debounce = None
        if self._closed or self._pending is None:
            return
        if self._task is not None and not self._task.done():
            return  # 旧工作未实际落定: 保留最新待执行, 由其 finally 触发
        query, generation = self._pending
        self._pending = None
        token = CancelToken()
        self._query_token = token
        self._query_snapshot = self._snapshot
        self._task = asyncio.ensure_future(self._run(query, generation, token))

    async def _run(self, query: str, generation: int, token: CancelToken) -> None:
        try:
            try:
                result = await self._complete(query, token)
            except asyncio.CancelledError:
                return  # 取消不显示错误
            except Exception:
                if generation == self._generation and self._still_current():
                    self._status = REASON_TEXT["internal_error"]
                    self._notify()
                return
            self._apply(generation, result)
        finally:
            self._task = None
            self._query_token = None
            if self._pending is not None and self._debounce is None:
                # 防抖已到期且最新输入一直在等旧工作落定; 未到期的防抖仍归定时器
                self._start_pending()

    def _still_current(self) -> bool:
        """结果核对草稿/光标: 查询发起快照必须仍等于当前快照。"""
        return self._query_snapshot is not None and self._query_snapshot == self._snapshot

    def _notify(self) -> None:
        if self.on_update is not None:
            self.on_update()

    def _apply(self, generation: int, result: FileCompletionResult) -> None:
        if generation != self._generation or self._closed:
            return  # 过时成功丢弃
        if not self._still_current():
            return  # 草稿/光标已变化
        self._index = 0
        if result.scope == SCOPE_UNAVAILABLE:
            self._candidates = ()
            reason = REASON_TEXT.get(result.reason or "", result.reason or "不可用")
            self._status = _UNAVAILABLE_STATUS + _sanitize(reason)
            self._notify()
            return
        self._status = _TRUNCATED_STATUS if result.truncated else None
        self._candidates = result.candidates
        self._notify()

    # --- 渲染 -----------------------------------------------------------------------

    def render(self, width: int, max_rows: int) -> list[str]:
        """只读候选/短状态; 不扫描目录、不创建任务。"""
        if max_rows <= 0:
            return []
        if self._candidates:
            overflow = len(self._candidates) > max_rows or self._status is not None
            count = min(len(self._candidates), max_rows - int(overflow and max_rows > 1))
            start = max(0, min(self._index - count // 2, len(self._candidates) - count))
            lines: list[str] = []
            for index in range(start, start + count):
                path = self._candidates[index].relative_path
                if index == self._index:
                    pad = " " * max(0, width - 4 - len(path))
                    lines.append(
                        theme.bg_open(theme.MENU_SEL_BG)
                        + theme.fg_open(theme.EDITOR_BORDER) + "→ "
                        + theme.fg_open(theme.MD_LINK) + path
                        + pad + current_style().sgr("0")
                    )
                else:
                    lines.append(
                        "  " + theme.fg(theme.MD_LINK, truncate_to_width(path, max(0, width - 2)))
                    )
            if overflow:
                label = _TRUNCATED_STATUS if self._status is None else self._status
                if count < len(self._candidates):
                    label = f"{self._index + 1}/{len(self._candidates)} {label}"
                lines.append(theme.fg(theme.FOOTER_DIM, truncate_to_width("  " + label, width)))
            return lines
        if self._status is not None:
            return [theme.fg(theme.MUTED_GRAY, truncate_to_width("  " + self._status, width))]
        return []


def _sanitize(text: str) -> str:
    """终端控制字符转义 (C3): 原因文本来自本地词表, 此处防御性剥离。"""
    return "".join(ch for ch in text if ord(ch) >= 32 and ch != "\x7f")
