"""前端组合 (票据 01): 调度、键分派、生命周期与故障恢复的唯一拥有者.

来源职责: pi-mono@1a584a7a packages/tui/src/tui.ts TuiBase (MIT,
Copyright (c) 2025 Mario Zechner) 的 requestRender 合并 (16ms 最小间隔,
输入即时帧) 与主屏生命周期; 键位语义按设计 §准入与键位表的 01 子集:
- Escape 前缀定时器归本前端: Escape 后 ESCAPE_PREFIX_TIMEOUT_S 内到达的
  Enter 是完整 Alt+Enter (仅换行, 不提交); 超时后才按孤立 Escape 派发
  on_escape。解析器层的孤立 Escape 由终端适配 flush_keys 落出。
- 粘贴是单事件 (Keys.BracketedPaste), 原文插入草稿, 换行不触发提交。
- Ctrl+T: 04 用户裁决移除思考展开切换功能; 控制键仍在此安全消费, 不写入文本。
- Ctrl+C/Ctrl+D/Enter 的语义回调由宿主注入 (01 演示: Enter 提交到
  on_submit, Ctrl+D 空草稿退出); 本模块不接真实模型。
- tui-compact-interaction 票据 01 (契约 C3): 注入 layout_document 时前端持有
  DocumentViewport 自有视口——呈现帧 ≤ terminal.rows, 文档窗口 = rows 扣除
  状态/编辑/菜单/底栏后的余量, 默认跟随底部; F6 在编辑器/文档焦点间切换
  (模态/模型菜单活跃时忽略), 文档焦点中 Up/Down 选动作、Enter 只执行可见
  选中动作、Esc 立即回草稿并消费, PageUp/PageDown/Home/End 回看; 文本/粘贴/
  换行键退出焦点后按原编辑行为处理。未注入 layout_document 保持旧整幅回放。
- tui-compact-interaction 票据 02 (契约 C4/D5): 输入批次扩展 PointerEvent;
  每次成功写入后保存该帧的命中图/几何/revision, 首帧由 renderer 锚定清屏
  (屏幕行 = 帧行)。模态与菜单活跃时 pointer 被消费不穿透; 只有同目标左键
  down/up、未拖动、无修饰符且目标仍命中同一区域才执行一次显示动作; 动作
  执行/滚轮/几何不符/写失败/模态菜单切换/停止都使命中图立即失效, 动作执行
  后本批后续 pointer 丢弃直至新帧成功。滚轮每 notch 移动 3 物理行, 小
  delta 累积, 回看脱离跟随、到底恢复; 鼠标启用失败经 on_mouse_unavailable
  给一次键盘回退提示, 双击第二按与未知输入安全消费。
- tui-compact-interaction 票据 03 (契约 C5/C6): Shift+Tab (Keys.BackTab) 交
  on_shift_tab 思考强度循环 (模态接管; 菜单/模型菜单活跃时归控件, 不透传,
  不改焦点或草稿); 底栏徽标的实际呈现列区间由 FooterComponent 经
  footer_badge_spans 提供并并入同一帧命中图, 点击徽标走 on_display_action
  的 effort_cycle, 与键盘共用同一同步准入入口。

故障矩阵 (P10): 渲染写失败先记录 ui_failures 并尝试恢复终端一次, 再失败
则置 terminal_failed 并请求退出; 尺寸读取失败沿用上次良好值; 输入读取
失败记录并继续; 停止时收集恢复错误, 不伪称已恢复。
"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.keys import Keys

from cicada.ui.commands import COMMAND_MENU_MAX_ROWS, SlashCommand, SlashCommandMenu
from cicada.ui import theme
from cicada.ui.document_layout import (
    ACTION_EFFORT_CYCLE,
    ACTION_JUMP_LATEST,
    TARGET_EFFORT,
    TARGET_LATEST,
    DisplayAction,
    DocumentLayout,
    DocumentViewport,
    HitSpan,
)
from cicada.ui.editor import Editor
from cicada.ui.file_menu import FileCompletionMenu
from cicada.ui.main_screen import MainScreen, strip_terminal_sequences
from cicada.ui.model_menu import ModelMenu
from cicada.ui.login_wizard import LoginModal
from cicada.ui.review_view import ReviewView
from cicada.ui.rewind_view import RewindView
from cicada.ui.session_resume import SessionResumeModal
from cicada.ui.terminal_io import (
    WHEEL_DELTA,
    InputBatchItem,
    PointerEvent,
    Terminal,
    TerminalGeometry,
)
from cicada.ui.display_style import DisplayStyle

MIN_RENDER_INTERVAL_S = 1.0 / 60.0  # Pi MIN_RENDER_INTERVAL_MS = 16
ESCAPE_PREFIX_TIMEOUT_S = 0.010  # Pi DEFAULT_ESCAPE_TIMEOUT_MS = 10
WRITE_RETRY_DELAY_S = 0.2  # 写失败后的恢复重试延迟 (Windows 时钟粒度 ~15.6ms, 取整数量级)

# pointer 派发结果 (02): none=消费无渲染, render=需立即渲染, poison=动作已
# 执行, 同批后续 pointer 丢弃直至新帧成功 (C4)
_POINTER_NONE = "none"
_POINTER_RENDER = "render"
_POINTER_POISON = "poison"


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
        command_menu: SlashCommandMenu | None = None,
        status_lines: Callable[[int], list[str]] | None = None,
        model_menu: ModelMenu | None = None,
        on_model_confirm: Callable[[str], None] | None = None,
        on_model_action: Callable[[str, str], None] | None = None,
        modal: LoginModal | None = None,
        display_style: DisplayStyle | None = None,
        layout_document: Callable[[int, DisplayAction | None], DocumentLayout] | None = None,
        on_display_action: Callable[[DisplayAction], object] | None = None,
        on_mouse_unavailable: Callable[[], None] | None = None,
        on_shift_tab: Callable[[], None] | None = None,
        footer_badge_spans: Callable[[], tuple[tuple[int, int, int], ...]] | None = None,
        review: ReviewView | None = None,
        rewind: RewindView | None = None,
        file_menu: FileCompletionMenu | None = None,
        session_modal: SessionResumeModal | None = None,
    ) -> None:
        self.display_style = display_style if display_style is not None else DisplayStyle.capture()
        self._terminal = terminal
        self._editor = editor
        self._document_lines = document_lines
        # 紧凑交互 01 (C3): 注入同源布局即启用自有视口 (帧 ≤ rows); 否则旧回放
        self._layout_document = layout_document
        self._on_display_action = on_display_action
        self._viewport = DocumentViewport() if layout_document is not None else None
        self._on_submit = on_submit
        self._on_escape = on_escape
        self._on_ctrl_c = on_ctrl_c
        self._on_ctrl_o = on_ctrl_o
        self._on_exit = on_exit
        self._on_mouse_unavailable = on_mouse_unavailable
        # 票据 03: Shift+Tab 思考强度循环 (准入/事务归组合层); 底栏徽标命中列
        self._on_shift_tab = on_shift_tab
        self._footer_badge_spans = footer_badge_spans
        self._footer_lines = footer_lines
        self._command_menu = command_menu
        self._status_lines = status_lines
        self._model_menu = model_menu
        self._on_model_confirm = on_model_confirm
        self._on_model_action = on_model_action
        self._modal = modal
        self._review = review
        self._rewind = rewind
        self._file_menu = file_menu
        self._session_modal = session_modal
        self.sensitive_input_used = False
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
        # 票据 02 (C4/D5): 最近一次成功呈现帧的命中图/几何; pending 命中图
        # 只在 render_once 成功返回后提升, 写失败即弃。_pending_left_down 是
        # 待配对左键按下 (x, y, action) —— 流式更新后的配对以「目标身份与命中
        # 区域仍相同」复核, 不绑 revision。滚轮小 delta 累积到标准 notch。
        self._frame_hits: tuple[HitSpan, ...] | None = None
        self._frame_geometry: TerminalGeometry | None = None
        self._pending_frame_hits: tuple[HitSpan, ...] | None = None
        self._pending_left_down: tuple[int, int, DisplayAction] | None = None
        self._wheel_accumulator = 0

    # --- 帧合成 -----------------------------------------------------------------

    def _render_frame_lines(self, width: int) -> list[str]:
        with self.display_style.use():
            return self._compose_frame_lines(width)

    def _compose_frame_lines(self, width: int) -> list[str]:
        if self._session_modal_active():
            self._pending_frame_hits = ()
            return self._session_modal.render(width, self._frame_rows)
        if self._review_active():
            layout = self._page.layout(width, self._frame_rows)
            self._pending_frame_hits = layout.hits
            return list(layout.lines)
        if self._modal_active():
            self.sensitive_input_used = True
            return self._modal.render(width, self._frame_rows)
        # 帧内不重读终端尺寸: rows 用本帧 render_once 前的快照, 避免一帧内
        # viewport 用旧尺寸而编辑器/底栏行数用新尺寸 (缩放恰落排版期间)
        rows = self._frame_rows
        status = self._status_lines(width) if self._status_lines is not None else []
        # 设计 §4: 编辑器正文 1–5 行; 低高度 (8 行) 时 1–3 行, 剩余给对话
        max_rows = min(5, max(3, int(rows * 0.3)))
        menu: list[str] = []
        if self._document_focused():
            slash_open = False  # 文档焦点期间斜线菜单不同步不渲染 (C3)
        else:
            self._sync_command_menu()
            # 斜线菜单打开时输入框下方只留候选 (2026-10-18 裁决): 其他信息隐去,
            # 候选窗放宽到 COMMAND_MENU_MAX_ROWS 行
            slash_open = (
                self._command_menu is not None
                and self._command_menu.is_open
                and not self._model_menu_active()
                and not (self._file_menu is not None and self._file_menu.visible)
            )
        footer = (
            [] if slash_open
            else (self._footer_lines(width, rows) if self._footer_lines is not None else [])
        )
        if not self._document_focused():
            menu_budget = min(
                COMMAND_MENU_MAX_ROWS if slash_open else 4,
                max(0, rows - len(footer) - 3),
            )
            if self._model_menu_active():
                # 模型菜单 (DeepSeek 04) 与斜线菜单共用编辑器下方槽位; 两者不会同时打开
                menu = self._model_menu.render(width, menu_budget)
            elif self._file_menu is not None and self._file_menu.visible:
                # 文件补全菜单 (C6): render 只读候选/短状态, 查询只在输入路径启动
                menu = self._file_menu.render(width, menu_budget)
            else:
                menu = (
                    self._command_menu.render(width, menu_budget)
                    if self._command_menu is not None else []
                )
        if menu or status:
            # 菜单/状态行与底栏先留空间，低窗口仍保留编辑光标所在的正文行。
            max_rows = max(1, min(max_rows, rows - len(menu) - len(footer) - len(status) - 2))
        editor_lines = self._editor.render(width, max_rows)
        fixed = len(status) + len(editor_lines) + len(menu) + len(footer)
        if self._layout_document is not None and self._viewport is not None:
            # 自有视口 (紧凑交互 01, C3): 呈现帧 ≤ rows; 文档窗口是固定部之外
            # 的余量, 跟随/回看/锚定与「有更新 · 最新」指示归 DocumentViewport
            layout = self._layout_document(width, self._viewport.selection)
            reserve = 1 if layout.lines else 0  # 文档与编辑器上边框之间留一行
            doc_height = max(0, rows - fixed - reserve)
            if doc_height == 0:
                reserve = 0  # 无文档窗口时不留间隔行: 帧 ≤ rows (C3, 低高度+菜单)
            lines = list(self._viewport.frame(
                layout, doc_height, theme.fg(theme.MUTED_GRAY, "有更新 · 最新")
            ))
            if reserve:
                lines.append("")
            # 票据 02 (D5 同源): 命中图来自同一次布局与呈现窗口; 是否有效
            # 由 render 成败决定 (_do_render 提升或丢弃)
            self._pending_frame_hits = self._viewport.frame_hits(width)
        else:
            lines = list(self._document_lines(width))
            if lines and strip_terminal_sequences(lines[-1]).strip():
                lines.append("")  # 输出与编辑器上边框之间留一行。
        lines.extend(status)  # 状态行: 文档与编辑器之间 (规格 §F), 空闲时 0 行
        lines.extend(editor_lines)
        lines.extend(menu)
        footer_base = len(lines)
        lines.extend(footer)
        if footer and self._viewport is not None and self._footer_badge_spans is not None:
            # 票据 03 (C6): 本次成功呈现的底栏徽标列区间并入同一帧命中图;
            # FooterComponent 已按"右段完整保留"过滤, 被裁掉的徽标无 span。
            # 无自有视口的旧回放路径不注册: 帧可超 rows 且无 viewport_top
            # 补偿, 行号会错位 (02 回执: 旧回放路径 pointer 安全消费)
            badge_action = DisplayAction(ACTION_EFFORT_CYCLE, (TARGET_EFFORT,))
            hits = list(self._pending_frame_hits or ())
            hits.extend(
                HitSpan(footer_base + row, start, end, badge_action)
                for row, start, end in self._footer_badge_spans()
            )
            self._pending_frame_hits = tuple(hits)
        return lines

    def _document_focused(self) -> bool:
        return self._viewport is not None and self._viewport.focus

    def _sync_viewport_layout(self) -> None:
        """导航/执行键派生前同步渲染一帧, 让视口读到当前布局。

        正常渲染经节流推迟 (call_soon/call_later), 同一批按键里上一个动作
        (如开详情) 的布局要等下一 tick 才进视口; 此时导航键会按过期布局
        移动选择/锚定 (实测竞态: 展开详情后紧跟 Down 选错目标)。同步一帧
        对齐状态; 无变化时 render_once 无增量写出, 成本可忽略。
        """
        if self._stopped or self._viewport is None:
            return
        self._cancel_render_timer()
        self._render_requested = False
        self._do_render()

    def _sync_command_menu(self) -> None:
        if self._command_menu is not None:
            self._command_menu.sync(self._editor.get_text(), at_end=self._editor.cursor_at_end())

    def _sync_file_menu(self) -> None:
        """输入路径同步 (C7): 查询只从这里启动, render 不启动查询。

        模态/模型菜单/审阅/文档焦点优先: 资格不成立即候选失效并取消。
        """
        if self._file_menu is None:
            return
        eligible = not (
            self._document_focused()
            or self._model_menu_active()
            or self._modal_active()
            or self._review_active()
            or self._session_modal_active()
        )
        self._file_menu.sync(self._editor, eligible=eligible)

    def _file_menu_invalidate(self) -> None:
        if self._file_menu is not None:
            self._file_menu.invalidate()

    def prepare_session_change(self) -> None:
        """同步封住候选入口；应用在管理预约内等待已启动查询真实落定。"""
        self._file_menu_invalidate()
        if self._command_menu is not None:
            self._command_menu.dismiss()
        if self._model_menu is not None:
            self._model_menu.dismiss()
        if self._viewport is not None:
            self._viewport.exit_focus()
        self._invalidate_pointer_frame()
        self._disarm_escape_prefix()

    def reset_session_display(self) -> None:
        self.prepare_session_change()
        if self._viewport is not None:
            self._viewport = DocumentViewport()
        self._invalidate_pointer_frame()
        self._wheel_accumulator = 0
        self._disarm_escape_prefix()
        self._screen.reset_render_state()
        self.request_render()

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
        exc = self._render_frame_captured()
        if exc is not None:
            if self._review_active():
                self._page.cancel()
            self.ui_failures.append('login render failed' if self.sensitive_input_used else f"render failed: {exc!r}")
            self._attempt_recovery()

    def _snapshot_geometry(self) -> TerminalGeometry:
        """一次几何快照: columns/rows/origin 同源, 帧合成与命中图共用 (C4)。"""
        geometry = getattr(self._terminal, "geometry", None)
        if isinstance(geometry, TerminalGeometry):
            return geometry
        return TerminalGeometry(self._terminal.columns, self._terminal.rows)

    def _render_frame_captured(self, *, reset: bool = False) -> Exception | None:
        """渲染一帧并管理命中图生命周期 (02, C4): 几何一次快照, 写入成功才
        提升本帧命中图与几何; 失败立即全部失效 (写出的屏幕不可信, 陈旧坐标
        不得受理)。返回捕获的异常或 None。"""
        geometry = self._snapshot_geometry()
        try:
            if reset:
                self._screen.reset_render_state()
            self._frame_rows = geometry.rows
            self._pending_frame_hits = None
            self._screen.render_once(geometry.columns, geometry.rows)
        except Exception as exc:
            self._invalidate_pointer_frame()
            return exc
        self._frame_hits = self._pending_frame_hits
        self._pending_frame_hits = None
        self._frame_geometry = geometry
        return None

    def _invalidate_pointer_frame(self) -> None:
        """命中图与待配对按下立即失效 (C4): 写失败/模态菜单切换/滚动/停止。"""
        self._frame_hits = None
        self._frame_geometry = None
        self._pending_frame_hits = None
        self._pending_left_down = None

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
        exc = self._render_frame_captured(reset=True)
        if exc is not None:
            self.terminal_failed = True
            self.ui_failures.append('login terminal write failed permanently' if self.sensitive_input_used else f"terminal write failed permanently: {exc!r}")
            self._exit_event.set()

    # --- 输入 --------------------------------------------------------------------

    def _on_input_batch(self, items: list[InputBatchItem]) -> None:
        if self._session_modal_active():
            self._session_modal.handle_batch([item for item in items if not isinstance(item, PointerEvent)])
            self._invalidate_pointer_frame()
            self.request_render_now()
            return
        capture_before = self._pointer_capture_active()
        poisoned = False
        needs_render = False
        for item in items:
            review_before = self._review_active()
            generation_before = self._page.generation if self._page is not None else 0
            if isinstance(item, PointerEvent):
                if poisoned:
                    continue  # 动作执行后的同批旧 pointer 丢弃直至新帧成功 (C4)
                outcome = self._dispatch_pointer(item)
                if outcome == _POINTER_POISON:
                    poisoned = True
                    needs_render = True
                elif outcome == _POINTER_RENDER:
                    needs_render = True
            else:
                session_was_active = self._session_modal_active()
                modal_was_open = self._modal_active()
                self._dispatch_key(item)
                # C7: 文件菜单按当前草稿全文/逻辑光标/资格建立快照; 同步在按键
                # 应用之后, 查询只从这里启动 (render 不启动)
                self._sync_file_menu()
                needs_render = True
                if session_was_active != self._session_modal_active():
                    self._invalidate_pointer_frame()
                    break
                if modal_was_open and not self._modal_active():
                    break  # 取消后的同批输入不能落入聊天编辑器或提交新请求。
            if self._page is not None and self._page.generation != generation_before:
                self._invalidate_pointer_frame()
                poisoned = True
            if review_before != self._review_active():
                self._invalidate_pointer_frame()
                break  # 开启/退出页面消费同批余键和旧坐标。
        if capture_before != self._pointer_capture_active():
            self._invalidate_pointer_frame()  # 模态/菜单切换: 命中图立即失效 (C4)
        if needs_render:
            self.request_render_now()

    def request_render_now(self) -> None:
        """输入路径: 跳过节流定时器立即渲染 (输入延迟敏感)。"""
        if self._stopped:
            return
        self._render_requested = True
        self._render_immediate()

    def _dispatch_key(self, key) -> None:
        if self._session_modal_active():
            self._disarm_escape_prefix()
            self._session_modal.handle_batch([key])
            return
        if self._review_active():
            self._file_menu_invalidate()  # C7: 审阅优先, 候选失效并取消
            self._sync_viewport_layout()
            self._disarm_escape_prefix()
            self._page.handle(key)
            return
        if self._modal_active():
            self._file_menu_invalidate()  # C7: 模态优先, 候选失效并取消
            self.sensitive_input_used = True
            self._disarm_escape_prefix()
            self._modal.handle(key)
            return
        self._sync_command_menu()
        data = key.data
        # Escape 前缀窗口内到达任何非 Enter/Escape 键 (输入处理 v1 D2): Escape
        # 先行派发再按原逻辑处理该键; 否则 Alt+功能键会先执行功能键、10 ms 后才
        # 派发 Escape, 顺序颠倒。Enter 保留 Alt+Enter 换行、Escape 自身保留连按
        # 合并, 二者由下方前缀状态机处理。
        if self._escape_pending and not (
            key.key is Keys.Escape
            or (data == "\x1b" and len(data) == 1)
            or key.key is Keys.ControlM
            or data == "\r"
        ):
            self._disarm_escape_prefix()
            self._dispatch_escape()
        # 原生鼠标 KeyPress (上游文本解析产物): 安全消费, 不写入编辑器 (C4)。
        # 置于冲刷之后, 使 Alt+鼠标键同样先派发 Escape (D2 无例外)。
        if key.key is Keys.WindowsMouseEvent:
            return
        # 目录内的动作先于主屏退出/工具展开，不改变菜单外原键位。
        if self._model_menu_active() and self._model_menu.search_enabled:
            action = {Keys.ControlD: 'default', Keys.ControlO: 'nonthinking', Keys.ControlL: 'configure'}.get(key.key)
            if action is not None:
                option = self._model_menu.selected
                if option is not None and self._on_model_action is not None:
                    self._on_model_action(action, option.id)
                return
        # Ctrl+C / Ctrl+D: 直接派发 (Escape 前缀已在上方先行派发)
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
        # Ctrl+T: 04 用户裁决移除切换功能; 安全消费控制键, 不写入文本。
        if key.key is Keys.ControlT or data == "\x14":
            return
        # Shift+Tab (BackTab, 票据 03/C6): 主会话思考强度循环, 不改焦点或草稿;
        # 模态已在上接管; 菜单/模型菜单活跃时归当前控件所有, 不向主会话透传。
        # 置于文档焦点与文本插入之前: 文档模式不当作反向焦点导航, "\t" 不进草稿
        if key.key is Keys.BackTab or data == "\x1b[Z":
            if (
                self._on_shift_tab is not None
                and not self._model_menu_active()
                and (self._command_menu is None or not self._command_menu.is_open)
            ):
                self._on_shift_tab()
            return
        # F6: 编辑器/文档焦点切换 (紧凑交互 01, C2/C3); 模态已在上方接管,
        # 模型菜单活跃时忽略; 未注入 layout_document (无自有视口) 安全消费
        if key.key is Keys.F6:
            if self._viewport is not None and not self._model_menu_active():
                if self._viewport.focus:
                    self._viewport.exit_focus()
                else:
                    if self._command_menu is not None:
                        self._command_menu.dismiss()  # 焦点期间菜单不同步不渲染
                    self._file_menu_invalidate()  # C7: 文档焦点优先, 候选失效并取消
                    self._sync_viewport_layout()
                    self._viewport.enter_focus()
            return
        # 文档焦点键位 (C2): Up/Down 选动作, Enter 只执行可见选中动作 (绝不
        # 提交草稿), PageUp/PageDown/Home/End 翻看, Esc 立即回草稿并消费;
        # 文本/粘贴/换行等其余键退出焦点后按原编辑行为继续分发
        if self._document_focused():
            if key.key is Keys.Escape or (data == "\x1b" and len(data) == 1):
                self._viewport.exit_focus()
                return
            if key.key in (Keys.Up, Keys.Down, Keys.PageUp, Keys.PageDown, Keys.Home, Keys.End):
                self._sync_viewport_layout()
            if key.key is Keys.Up:
                self._viewport.move_selection(-1)
                return
            if key.key is Keys.Down:
                self._viewport.move_selection(1)
                return
            if key.key is Keys.PageUp:
                self._viewport.page(-1)
                return
            if key.key is Keys.PageDown:
                self._viewport.page(1)
                return
            if key.key is Keys.Home:
                self._viewport.go_home()
                return
            if key.key is Keys.End:
                self._viewport.go_end()
                return
            if key.key in (Keys.ControlM,) or data == "\r":
                self._sync_viewport_layout()
                action = self._viewport.execute_selected()
                if action is not None and self._on_display_action is not None:
                    self._on_display_action(action)
                return
            self._viewport.exit_focus()
        # Escape 前缀状态机: Escape 后窗口内 Enter = Alt+Enter 换行
        if key.key is Keys.Escape or (data == "\x1b" and len(data) == 1):
            self._arm_escape_prefix()
            return
        if self._escape_pending:
            if key.key in (Keys.ControlM,) or data == "\r":
                self._disarm_escape_prefix()
                if self._model_menu_active():
                    self._model_menu.dismiss()  # Alt+Enter 是编辑: 收起菜单后仅换行
                self._editor.insert_newline()  # 完整 Alt+Enter: 仅换行
                return
            self._disarm_escape_prefix()
            self._dispatch_escape()  # 前缀窗口内的其他键: Escape 先行派发
        # 模型菜单优先键位 (契约 C7): 箭头/Enter 先于普通 editor 命令解释;
        # 跨供应商目录的普通文本进入独立搜索；旧目录其余键先收起再交编辑器。
        if self._model_menu_active():
            if key.key is Keys.Up:
                self._model_menu.move(-1)
                return
            if key.key is Keys.Down:
                self._model_menu.move(1)
                return
            if key.key in (Keys.ControlM,) or data == "\r":
                self._confirm_model_menu()
                return
            if self._model_menu.search_enabled:
                if key.key in (Keys.ControlH, Keys.Backspace):
                    self._model_menu.filter(self._model_menu.query[:-1])
                    return
                if key.key is Keys.ControlU:
                    self._model_menu.filter('')
                    return
                if key.key is Keys.BracketedPaste or not isinstance(key.key, Keys):
                    if not any(ord(char) < 32 for char in data):
                        self._model_menu.filter(self._model_menu.query + data)
                    return
            self._model_menu.dismiss()
        if key.key is Keys.BracketedPaste:
            self._editor.insert_text(data)
            return
        if key.key in (Keys.ControlM,) or data == "\r":
            if self._file_menu is not None and self._file_menu.handle_enter():
                return  # C6: 菜单展示候选时 Enter 完成当前项; 第二次 Enter 才走原准入
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
            if self._file_menu is not None and self._file_menu.move(-1):
                return
            if self._command_menu is not None and self._command_menu.move(-1):
                return
            self._editor.move_cursor(-1, 0)
            return
        if key.key is Keys.Down:
            if self._file_menu is not None and self._file_menu.move(1):
                return
            if self._command_menu is not None and self._command_menu.move(1):
                return
            self._editor.move_cursor(1, 0)
            return
        if key.key is Keys.Home:
            self._editor.move_to_line_start()
            return
        if key.key is Keys.End:
            self._editor.move_to_line_end()
            return
        # Tab (C6): 文件补全菜单有候选时接受, 有 @ token 时跳过防抖启动查询并
        # 消费该键; 无 @ 上下文保持原编辑行为 (插入制表符展开 4 空格)
        if key.key is Keys.Tab or data == "\t":
            if self._file_menu is not None and self._file_menu.handle_tab():
                return
            self._editor.insert_text(data if data else "\t")
            return
        # 可打印文本 (含 IME 上屏多字符): 原样插入
        if data and not data.startswith("\x1b") and all(ord(ch) >= 32 or ch == "\t" for ch in data):
            self._editor.insert_text(data)

    # --- 指针派发 (票据 02, C4/D5) --------------------------------------------------

    def _pointer_capture_active(self) -> bool:
        """模态与菜单先取得输入: 活跃时 pointer 被消费, 不穿透文档 (C4)。"""
        if self._review_active():
            return False  # 审阅提供自己的同源命中图，菜单不能抢走本页输入。
        if self._session_modal_active() or self._modal_active() or self._model_menu_active():
            return True
        if self._file_menu is not None and self._file_menu.candidates:
            return True  # C6: 候选打开时沿用菜单的指针捕获, v1 无点击选择
        return self._command_menu is not None and self._command_menu.is_open

    def _dispatch_pointer(self, event: PointerEvent) -> str:
        """指针事件派发。返回 _POINTER_NONE/_POINTER_RENDER/_POINTER_POISON。

        几何与呈现帧不符的事件忽略等待新帧; 只有同一目标上的左键 down/up、
        未拖动且修饰符为空才执行一次显示动作; 执行后命中图失效并毒化本批。
        点击不改编辑器/文档焦点, 不触碰草稿。
        """
        if self._pointer_capture_active():
            self._pending_left_down = None
            return _POINTER_NONE
        if self._viewport is None and not self._review_active():
            return _POINTER_NONE  # 旧整幅回放无自有视口: pointer 安全消费
        if event.kind == "cancel":
            self._pending_left_down = None  # 失焦/取消: 清除按下 (后端可观察时)
            return _POINTER_NONE
        if (
            self._frame_hits is None
            or self._frame_geometry is None
            or event.geometry != self._frame_geometry
        ):
            # 无有效帧或几何不一致: 等新帧; 陈旧按下不可信, 一并丢弃
            self._pending_left_down = None
            return _POINTER_NONE
        if event.kind == "move":
            pending = self._pending_left_down
            if pending is not None and (
                (event.x, event.y) != (pending[0], pending[1])
                or event.button not in ("none", "left")
            ):
                self._pending_left_down = None  # 拖出/其他按钮: 清除按下 (C4)
            return _POINTER_NONE
        if event.kind == "wheel":
            return self._dispatch_wheel(event)
        if event.kind == "down":
            if event.double_click:
                self._pending_left_down = None  # 原生双击第二按: 消费不配对
            elif event.modifiers or event.button != "left":
                self._pending_left_down = None  # 修饰拖动归宿主选择; 非左键不配对
            else:
                action = self._hit_action_at(event.x, event.y)
                self._pending_left_down = (event.x, event.y, action) if action is not None else None
            return _POINTER_NONE
        if event.kind == "up":
            pending = self._pending_left_down
            self._pending_left_down = None
            if pending is None or event.modifiers:
                return _POINTER_NONE
            x, y, action = pending
            if (event.x, event.y) != (x, y):
                return _POINTER_NONE  # 拖出又拖回/位移释放: 无动作 (C4)
            # 流式帧更新: 仅当目标身份与命中区域仍相同 (几何已同) 才执行
            if self._hit_action_at(x, y) != action:
                return _POINTER_NONE
            return self._execute_pointer_action(action)
        return _POINTER_NONE  # 未知种类: 安全消费

    def _dispatch_wheel(self, event: PointerEvent) -> str:
        """滚轮: 清除未完成配对, 小 delta 累积到 notch 后移动 3 行 (C3/C4)。"""
        self._pending_left_down = None
        self._wheel_accumulator += event.wheel_delta
        notches = int(self._wheel_accumulator / WHEEL_DELTA)  # 截断到零: 累积小 delta
        self._wheel_accumulator -= notches * WHEEL_DELTA
        if not notches:
            return _POINTER_NONE
        if self._review_active():
            self._page.wheel(notches)
        else:
            self._viewport.wheel_scroll(notches)
        self._frame_hits = None  # 视口位移后命中行失效, 等新帧 (C4)
        return _POINTER_RENDER

    def _hit_action_at(self, x: int, y: int) -> DisplayAction | None:
        """帧行命中查询 (帧内 0 基行/列): 命中最窄区域; 正文无动作。"""
        if self._frame_hits is None:
            return None
        best: HitSpan | None = None
        for span in self._frame_hits:
            if span.row == y and span.start_column <= x < span.end_column:
                if best is None or (
                    span.end_column - span.start_column
                ) < (best.end_column - best.start_column):
                    best = span
        return best.action if best is not None else None

    def _execute_pointer_action(self, action: DisplayAction) -> str:
        """执行命中的显示动作; 命中图失效, 本批后续 pointer 丢弃 (C4)。"""
        if action.kind == ACTION_JUMP_LATEST and action.target_id == (TARGET_LATEST,):
            self._viewport.jump_to_latest()  # 视口内部消费, 同 End 语义
        elif self._on_display_action is not None:
            self._on_display_action(action)  # 只读显示动作; 未知/失效目标静默
        self._invalidate_pointer_frame()
        return _POINTER_POISON

    def _handle_enter(self) -> None:
        """受理制提交: on_submit 返回 False = 拒绝 (草稿保留, 02 准入语义);
        True/None = 接受 (清空草稿)。空白草稿直接清空不派发。"""
        text = self._editor.get_text()
        if not text.strip():
            if text:
                self._editor.clear()
            return
        self._sync_command_menu()
        menu = self._command_menu
        option = menu.selected_option if menu is not None else None
        if option is not None and menu is not None:
            # 固定参数选项 (/mode 三档): 选中即把 `命令 值` 写回草稿并走原提交
            # 路径, 选完直接执行, 不需要用户再手打参数。
            command, value = option
            self._editor.set_text(f"{command.command} {value}")
            text = self._editor.get_text()
            menu.complete(text)
            selected = None
        else:
            selected = menu.selected if menu is not None else None
        if selected is not None and selected.takes_argument:
            # 需要参数的命令 (/export、/name、/skill 等): 菜单补齐只把命令写回
            # 草稿, 不提交 —— 参数由用户继续输入, 再次 Enter 才走准入。
            self._complete_argument_command(selected)
            return
        # /skill 的任务后缀是用户原文，保留尾随空白和换行直到请求封装。
        command_text = text.lstrip()
        skill_command = command_text.startswith("/skill") and (
            len(command_text) == 6 or command_text[6].isspace()
        )
        submitted = selected.command if selected is not None else (
            command_text if skill_command else text.strip()
        )
        if self._on_submit is None:
            self._editor.clear()
            return
        result = self._on_submit(submitted)
        if result is False:
            return  # 拒绝: 草稿保留 (实际接受前不伪造用户条目)
        if self._review_active():
            return  # 应用准入成功后才消费 /diff；失败仍保留原编辑器状态。
        if self._modal_active():
            return  # 模态输入期间聊天草稿/光标/撤销栈保持原样。
        self._editor.clear()
        if asyncio.iscoroutine(result):
            asyncio.ensure_future(result)

    def _complete_argument_command(self, selected: SlashCommand) -> None:
        """菜单补齐带参数命令: 草稿变成 `命令 + 空格`, 光标在末尾, 不提交。

        菜单同步会因草稿含空白自动收起; 下一次 Enter 只提交用户补齐的最终文案。
        """
        if self._command_menu is not None:
            self._command_menu.dismiss()
        self._editor.set_text(f"{selected.command} ")

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
        self._sync_command_menu()
        self._sync_file_menu()
        if self._model_menu_active() and self._model_menu.dismiss():
            # 模型菜单 Esc 只收起菜单 (契约 C7), 不触发会话取消
            self.request_render_now()
            return
        if self._file_menu is not None and self._file_menu.dismiss():
            # C6: Esc 只收起候选并锁住相同草稿/光标, 不触发会话取消
            self.request_render_now()
            return
        if self._command_menu is not None and self._command_menu.dismiss():
            self.request_render_now()
            return
        if self._on_escape is not None:
            self._on_escape()

    def _session_modal_active(self) -> bool:
        return self._session_modal is not None and self._session_modal.active

    def _model_menu_active(self) -> bool:
        return self._model_menu is not None and self._model_menu.is_open

    def _modal_active(self) -> bool:
        return self._modal is not None and self._modal.is_open

    @property
    def _page(self) -> ReviewView | RewindView | None:
        if self._rewind is not None and self._rewind.active:
            return self._rewind
        return self._review

    def _review_active(self) -> bool:
        return self._page is not None and self._page.active

    def _confirm_model_menu(self) -> None:
        """模型菜单 Enter: 收起并把 option ID 交回组合层 (屏障/任务归组合层与应用)。"""
        if self._model_menu is None:
            return
        option = self._model_menu.selected
        if option is not None and not option.available:
            return
        self._model_menu.dismiss()
        if option is not None and self._on_model_confirm is not None:
            self._on_model_confirm(option.id)

    # --- 生命周期 ---------------------------------------------------------------------

    def _on_terminal_resize(self) -> None:
        """尺寸/origin 变化: 命中图立即失效 (C4), 渲染按新几何重排。"""
        self._invalidate_pointer_frame()
        self.request_render()

    async def run_async(self) -> list[str]:
        """运行至退出; 返回停止时的恢复错误列表 (空 = 干净恢复)。

        terminal.start 与 exit_event 等待整体在 try 内: 正常退出、写失败永久
        退出与任务被取消三条路径都执行收尾 (置停止态 -> 停定时器 -> 停止序列
        -> terminal.stop 恢复模式), 取消也不例外 (US21: 取消退出的终端恢复)。
        """
        loop = asyncio.get_running_loop()
        shutdown_cancelled = False
        try:
            self._terminal.start(self._on_input_batch, self._on_terminal_resize)
            mouse_error = getattr(self._terminal, "mouse_setup_error", None)
            if mouse_error and self._on_mouse_unavailable is not None:
                # 鼠标能力启用失败: 一次键盘回退短提示 (C4), 不阻断会话
                self._on_mouse_unavailable()
            self._last_render_at = loop.time()
            self.request_render_now()
            while not self._exit_event.is_set():
                try:
                    await asyncio.wait_for(self._exit_event.wait(), timeout=0.05)
                except TimeoutError:
                    pass
                if self._modal_active() and getattr(self._terminal, 'input_failed', False):
                    self.terminal_failed = True
                    self.ui_failures.append('login terminal input failed')
                    self._exit_event.set()
        finally:
            if self._session_modal is not None:
                try:
                    await self._session_modal.close()
                except asyncio.CancelledError:
                    shutdown_cancelled = True
            if self._review_active():
                self._page.cancel()
            if self._file_menu is not None:
                # C7: 前端退出先封闭入口、取消防抖与当前查询并等待实际落定,
                # 再恢复终端; 不把 asyncio 取消等同于子进程/iterator 释放
                try:
                    await self._file_menu.close()
                except asyncio.CancelledError:
                    pass
            # 终态冲刷: 退出前的待渲染请求 (如 gate 错误条目) 在收尾前落屏一次,
            # 不随 _stopped 丢弃 (规格: 终态事件不丢弃); 失败照常记录不阻断收尾
            if self._render_requested:
                self._render_requested = False
                try:
                    self._frame_rows = self._terminal.rows
                    self._screen.render_once(self._terminal.columns, self._frame_rows)
                except Exception as exc:
                    self.ui_failures.append('login final render failed' if self.sensitive_input_used else f"final render failed: {exc!r}")
            self._stopped = True
            self._cancel_render_timer()
            self._disarm_escape_prefix()
            self._invalidate_pointer_frame()  # 停止: 命中图与待配对按下立即失效 (C4)
            self._write_stop_sequence()
            stop_errors = self._terminal.stop()
            if self.sensitive_input_used and stop_errors:
                stop_errors = ['login terminal restore errors']
        if shutdown_cancelled:
            raise asyncio.CancelledError
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
