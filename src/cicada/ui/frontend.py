"""前端组合 (票据 01): 调度、键分派、生命周期与故障恢复的唯一拥有者.

来源职责: pi-mono@1a584a7a packages/tui/src/tui.ts TuiBase (MIT,
Copyright (c) 2025 Mario Zechner) 的 requestRender 合并 (16ms 最小间隔,
输入即时帧) 与主屏生命周期; 键位语义按设计 §准入与键位表的 01 子集:
- Escape 前缀定时器归本前端: Escape 后 ESCAPE_PREFIX_TIMEOUT_S 内到达的
  Enter 是完整 Alt+Enter (仅换行, 不提交); 超时后才按孤立 Escape 派发
  on_escape。解析器层的孤立 Escape 由终端适配 flush_keys 落出。
- 粘贴是单事件 (Keys.BracketedPaste), 原文插入草稿, 换行不触发提交。
- Ctrl+C/Ctrl+D/Enter 的语义回调由宿主注入 (01 演示: Enter 提交到
  on_submit, Ctrl+D 空草稿退出); 本模块不接真实模型。

故障矩阵 (P10): 渲染写失败先记录 ui_failures 并尝试恢复终端一次, 再失败
则置 terminal_failed 并请求退出; 尺寸读取失败沿用上次良好值; 输入读取
失败记录并继续; 停止时收集恢复错误, 不伪称已恢复。
"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.keys import Keys

from cicada.ui.editor import Editor
from cicada.ui.main_screen import MainScreen
from cicada.ui.terminal_io import Terminal

MIN_RENDER_INTERVAL_S = 1.0 / 60.0  # Pi MIN_RENDER_INTERVAL_MS = 16
ESCAPE_PREFIX_TIMEOUT_S = 0.010  # Pi DEFAULT_ESCAPE_TIMEOUT_MS = 10
WRITE_RETRY_DELAY_S = 0.2  # 写失败后的恢复重试延迟 (Windows 时钟粒度 ~15.6ms, 取整数量级)


class MainScreenFrontend:
    """主屏前端: 组件行 + 编辑器 -> 差分 renderer -> 终端; 输入 -> 编辑器/回调."""

    def __init__(
        self,
        terminal: Terminal,
        *,
        document_lines: Callable[[int], list[str]],
        editor: Editor,
        on_submit: Callable[[str], object] | None = None,
        on_escape: Callable[[], None] | None = None,
        on_ctrl_c: Callable[[], None] | None = None,
        on_ctrl_o: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
        escape_prefix_timeout: float = ESCAPE_PREFIX_TIMEOUT_S,
        footer_lines: Callable[[int, int], list[str]] | None = None,
    ) -> None:
        self._terminal = terminal
        self._editor = editor
        self._document_lines = document_lines
        self._on_submit = on_submit
        self._on_escape = on_escape
        self._on_ctrl_c = on_ctrl_c
        self._on_ctrl_o = on_ctrl_o
        self._on_exit = on_exit
        self._footer_lines = footer_lines
        self.ui_failures: list[str] = []
        self.terminal_failed = False
        self._screen = MainScreen(self._render_frame_lines, terminal.write)
        self._render_requested = False
        self._render_timer: asyncio.TimerHandle | None = None
        self._last_render_at = 0.0
        self._immediate_scheduled = False
        self._escape_timer: asyncio.TimerHandle | None = None
        self._escape_pending = False
        self._exit_event = asyncio.Event()
        self._stopped = False
        self._escape_prefix_timeout = escape_prefix_timeout
        self._recovering = False
        self._frame_rows = 24  # 每帧 render_once 前快照 (帧合成内不重读尺寸)

    # --- 帧合成 -----------------------------------------------------------------

    def _render_frame_lines(self, width: int) -> list[str]:
        lines = list(self._document_lines(width))
        # 帧内不重读终端尺寸: rows 用本帧 render_once 前的快照, 避免一帧内
        # viewport 用旧尺寸而编辑器/底栏行数用新尺寸 (缩放恰落排版期间)
        rows = self._frame_rows
        # 设计 §4: 编辑器正文 1–5 行; 低高度 (8 行) 时 1–3 行, 剩余给对话
        max_rows = min(5, max(3, int(rows * 0.3)))
        lines.extend(self._editor.render(width, max_rows))
        if self._footer_lines is not None:
            lines.extend(self._footer_lines(width, rows))
        return lines

    # --- 调度 (TuiBase.requestRender 移植) -----------------------------------------

    def request_render(self, force: bool = False) -> None:
        if self._stopped:
            return
        if force:
            self._screen.reset_render_state()
            self._render_immediate()
            return
        if self._render_requested:
            return
        self._render_requested = True
        asyncio.get_running_loop().call_soon(self._schedule_render)

    def _render_immediate(self) -> None:
        if self._immediate_scheduled:
            return
        self._immediate_scheduled = True
        loop = asyncio.get_running_loop()
        loop.call_soon(self._run_immediate)

    def _run_immediate(self) -> None:
        self._immediate_scheduled = False
        if self._stopped or not self._render_requested:
            return
        self._cancel_render_timer()
        self._render_requested = False
        self._last_render_at = asyncio.get_running_loop().time()
        self._do_render()

    def _schedule_render(self) -> None:
        if self._stopped or self._render_timer is not None or not self._render_requested:
            return
        loop = asyncio.get_running_loop()
        elapsed = loop.time() - self._last_render_at
        delay = max(0.0, MIN_RENDER_INTERVAL_S - elapsed)
        self._render_timer = loop.call_later(delay, self._run_throttled)

    def _run_throttled(self) -> None:
        self._render_timer = None
        if self._stopped or not self._render_requested:
            return
        self._render_requested = False
        self._last_render_at = asyncio.get_running_loop().time()
        self._do_render()
        if self._render_requested:
            self._schedule_render()

    def _cancel_render_timer(self) -> None:
        if self._render_timer is not None:
            self._render_timer.cancel()
            self._render_timer = None

    def _do_render(self) -> None:
        try:
            self._frame_rows = self._terminal.rows
            self._screen.render_once(self._terminal.columns, self._frame_rows)
        except Exception as exc:
            self.ui_failures.append(f"render failed: {exc!r}")
            self._attempt_recovery()

    def _attempt_recovery(self) -> None:
        """写失败恢复: 重置渲染状态并延迟重试一次 (瞬时故障可恢复);
        重试仍失败则置 terminal_failed 并请求退出, 不伪称屏幕已恢复。"""
        if self.terminal_failed or self._recovering:
            return
        self._recovering = True
        asyncio.get_running_loop().call_later(WRITE_RETRY_DELAY_S, self._retry_render)

    def _retry_render(self) -> None:
        self._recovering = False
        if self._stopped:
            return
        try:
            self._frame_rows = self._terminal.rows
            self._screen.reset_render_state()
            self._screen.render_once(self._terminal.columns, self._frame_rows)
        except Exception as exc:
            self.terminal_failed = True
            self.ui_failures.append(f"terminal write failed permanently: {exc!r}")
            self._exit_event.set()

    # --- 输入 --------------------------------------------------------------------

    def _on_input_batch(self, keys: list) -> None:
        for key in keys:
            self._dispatch_key(key)
        self.request_render_now()

    def request_render_now(self) -> None:
        """输入路径: 跳过节流定时器立即渲染 (输入延迟敏感)。"""
        if self._stopped:
            return
        self._render_requested = True
        self._render_immediate()

    def _dispatch_key(self, key) -> None:
        data = key.data
        # Ctrl+C / Ctrl+D: 与 Escape 前缀无关, 直接派发
        if key.key in (Keys.ControlC,) or data == "\x03":
            if self._on_ctrl_c is not None:
                self._on_ctrl_c()
            return
        if key.key in (Keys.ControlD,) or data == "\x04":
            if self._editor.is_empty() and self._on_exit is not None:
                self._on_exit()
            return
        # Ctrl+_: 撤销
        if data == "\x1f":
            self._editor.undo()
            return
        # Ctrl+O: 工具输出展开切换 (03 工具卡; 01 阶段无宿主时为无动作)
        if key.key is Keys.ControlO or data == "\x0f":
            if self._on_ctrl_o is not None:
                self._on_ctrl_o()
            return
        # Escape 前缀状态机: Escape 后窗口内 Enter = Alt+Enter 换行
        if key.key is Keys.Escape or (data == "\x1b" and len(data) == 1):
            self._arm_escape_prefix()
            return
        if self._escape_pending:
            if key.key in (Keys.ControlM,) or data == "\r":
                self._disarm_escape_prefix()
                self._editor.insert_newline()  # 完整 Alt+Enter: 仅换行
                return
            self._disarm_escape_prefix()
            self._dispatch_escape()  # 前缀窗口内的其他键: Escape 先行派发
        if key.key is Keys.BracketedPaste:
            self._editor.insert_text(data)
            return
        if key.key in (Keys.ControlM,) or data == "\r":
            self._handle_enter()
            return
        if data == "\n" or key.key is Keys.ControlJ:
            self._editor.insert_newline()  # Ctrl+J 换行 (Pi newLine)
            return
        if key.key is Keys.Backspace or data in ("\x7f", "\x08"):
            self._editor.backspace()
            return
        if key.key is Keys.Delete or data == "\x1b[3~":
            self._editor.delete_forward()
            return
        if key.key is Keys.Left:
            self._editor.move_cursor(0, -1)
            return
        if key.key is Keys.Right:
            self._editor.move_cursor(0, 1)
            return
        if key.key is Keys.Up:
            self._editor.move_cursor(-1, 0)
            return
        if key.key is Keys.Down:
            self._editor.move_cursor(1, 0)
            return
        if key.key is Keys.Home:
            self._editor.move_to_line_start()
            return
        if key.key is Keys.End:
            self._editor.move_to_line_end()
            return
        # 可打印文本 (含 IME 上屏多字符): 原样插入
        if data and not data.startswith("\x1b") and all(ord(ch) >= 32 or ch == "\t" for ch in data):
            self._editor.insert_text(data)

    def _handle_enter(self) -> None:
        """受理制提交: on_submit 返回 False = 拒绝 (草稿保留, 02 准入语义);
        True/None = 接受 (清空草稿)。空白草稿直接清空不派发。"""
        text = self._editor.get_text()
        if not text.strip():
            if text:
                self._editor.clear()
            return
        stripped = text.strip()
        if self._on_submit is None:
            self._editor.clear()
            return
        result = self._on_submit(stripped)
        if result is False:
            return  # 拒绝: 草稿保留 (实际接受前不伪造用户条目)
        self._editor.clear()
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    # --- Escape 前缀定时器 -----------------------------------------------------------

    def _arm_escape_prefix(self) -> None:
        if self._escape_pending:
            return  # 连续 Escape: 已有定时器在走
        self._escape_pending = True
        loop = asyncio.get_running_loop()
        self._escape_timer = loop.call_later(self._escape_prefix_timeout, self._on_escape_timeout)

    def _disarm_escape_prefix(self) -> None:
        if self._escape_timer is not None:
            self._escape_timer.cancel()
            self._escape_timer = None
        self._escape_pending = False

    def _on_escape_timeout(self) -> None:
        self._escape_timer = None
        self._escape_pending = False
        self._dispatch_escape()

    def _dispatch_escape(self) -> None:
        if self._on_escape is not None:
            self._on_escape()

    # --- 生命周期 ---------------------------------------------------------------------

    async def run_async(self) -> list[str]:
        """运行至退出; 返回停止时的恢复错误列表 (空 = 干净恢复)。

        terminal.start 与 exit_event 等待整体在 try 内: 正常退出、写失败永久
        退出与任务被取消三条路径都执行收尾 (置停止态 -> 停定时器 -> 停止序列
        -> terminal.stop 恢复模式), 取消也不例外 (US21: 取消退出的终端恢复)。
        """
        loop = asyncio.get_running_loop()
        try:
            self._terminal.start(self._on_input_batch, lambda: self.request_render())
            self._last_render_at = loop.time()
            self.request_render_now()
            await self._exit_event.wait()
        finally:
            # 终态冲刷: 退出前的待渲染请求 (如 gate 错误条目) 在收尾前落屏一次,
            # 不随 _stopped 丢弃 (规格: 终态事件不丢弃); 失败照常记录不阻断收尾
            if self._render_requested:
                self._render_requested = False
                try:
                    self._frame_rows = self._terminal.rows
                    self._screen.render_once(self._terminal.columns, self._frame_rows)
                except Exception as exc:
                    self.ui_failures.append(f"final render failed: {exc!r}")
            self._stopped = True
            self._cancel_render_timer()
            self._disarm_escape_prefix()
            self._write_stop_sequence()
            stop_errors = self._terminal.stop()
        return stop_errors

    def request_exit(self) -> None:
        self._exit_event.set()

    def _write_stop_sequence(self) -> None:
        """renderer 收尾: 光标下移到内容底部 (Pi beforeTerminalStop), 显示光标。"""
        try:
            previous_lines = self._screen._previous_lines
            if previous_lines:
                self._terminal.write(" ")
                line_diff = len(previous_lines) - self._screen._hardware_cursor_row
                if line_diff > 0:
                    self._terminal.write(f"\x1b[{line_diff}B")
                elif line_diff < 0:
                    self._terminal.write(f"\x1b[{-line_diff}A")
                self._terminal.write("\r\n")
            self._terminal.write("\x1b[?25h")
        except Exception as exc:
            self.ui_failures.append(f"stop sequence failed: {exc!r}")


__all__ = ["ESCAPE_PREFIX_TIMEOUT_S", "MIN_RENDER_INTERVAL_S", "MainScreenFrontend"]
