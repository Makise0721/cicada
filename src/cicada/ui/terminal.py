"""终端对话入口 (票据 03): prompt_toolkit inline 主屏 + 基础文本 presenter.

职责边界:
- Application(full_screen=False) 主屏内联是唯一终端控制者; 所有输出经它协调;
  不使用 Rich.Live, 不存在第二个光标 Owner. Rich 非必需 (04 正式 presenter 再用).
- BasicTextPresenter 是 TuiPresenter Protocol 的过渡实现 (05 由正式 presenter
  替换): 只消费 ApplicationUpdate, 产出有界的展示记录/活动预览/状态快照;
  显示文本先经 sanitize_external_text 去除 CSI/OSC 与控制字符 (换行/制表除外),
  不把外部纯文本当 markup; 规范消息与会话历史保持原始内容.
- 有界 mailbox (结构通知容量 1024): 内核同步 subscriber 只合并文本增量并入队
  结构通知, 不阻塞、不直接写终端; 消费任务合并文本更新 (约 30 FPS) 后渲染;
  结构通知溢出记录 UI failure 并调用 session.cancel(), RequestOutcome 仍是收尾权威.
- 输入按 01 探针裁决: Enter 提交 (in_paste_mode 时当换行), Alt+Enter 换行,
  bracketed paste 是单事件原样插入, 孤立 Escape 经 KeyProcessor timeoutlen 消歧;
  忙碌可继续编辑草稿, 提交/finalizing/closing 阶段提交被拒绝且草稿保留,
  忙碌清空拒绝且保留草稿.
- 已接受的提问按原始多行内容保留为一次 [user] 记录; 清空输入框不移除对话流正文;
  忙碌拒绝不产生假用户消息; 最终回答只进入终端记录一次 (零增量也显示).
- 关闭顺序 (T11): 先退出 Application 恢复终端输入/光标模式, 再退订, 停 mailbox
  消费, 最后 await session.aclose(); 关闭不可逆. 忙碌 Ctrl+C 只协作取消,
  不伪造 idle.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from collections.abc import Callable
from typing import Protocol

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import in_paste_mode
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import print_formatted_text

from cicada.application import (
    AgentEventReceived,
    ApplicationClosed,
    ApplicationSession,
    ApplicationUpdate,
    HistoryCleared,
    RequestCompleted,
    RequestPhaseChanged,
)
from cicada.core.events import (
    AssistantCompleted,
    AssistantTextDelta,
    Event,
    ModelCallCompleted,
    RunStarted,
    TurnStarted,
)
from cicada.ui.contracts import (
    ActivePreview,
    PresentationRecord,
    PresentationUpdate,
    StatusSnapshot,
    TuiPresenter,
)

PREVIEW_MAX_LINES = 12
PREVIEW_MAX_BYTES = 4096
MAILBOX_CAPACITY = 1024
RENDER_MIN_INTERVAL_S = 1.0 / 30.0

_BUSY_STATES = frozenset({"executing", "cancelling", "finalizing"})

_CSI_RE = re.compile(r"\x1b\[[0-9:;<=>?]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# 保留 \n (0x0a) 与 \t (0x09); 其余 C0 控制字符与 DEL 全部移除
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_external_text(text: str) -> str:
    """T13: 外部显示文本先移除 CSI/OSC 及除换行/制表外的控制字符.

    只作用于展示路径; 规范消息、会话历史与证据保持原始内容.
    """
    text = _OSC_RE.sub("", _CSI_RE.sub("", text))
    return _CONTROL_RE.sub("", text)


def assistant_key(run_id: str, turn_index: int) -> str:
    """assistant 最终记录与同 run/turn 活动预览共享的稳定 key."""
    return f"assistant:{run_id}:{turn_index}"


def _clip_preview_text(text: str) -> tuple[str, bool]:
    """活动预览有界化: 从尾部保留最多 12 行且不超过 4096 UTF-8 bytes."""
    truncated = False
    lines = text.split("\n")
    if len(lines) > PREVIEW_MAX_LINES:
        lines = lines[-PREVIEW_MAX_LINES:]
        truncated = True
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        cost = len(line.encode("utf-8")) + (1 if kept else 0)
        if total + cost > PREVIEW_MAX_BYTES:
            truncated = True
            if not kept:
                # 单行超限: 保留尾部可容纳的字节 (UTF-8 边界安全截断)
                keep = max(0, PREVIEW_MAX_BYTES - 3)
                raw = line.encode("utf-8")[-keep:] if keep else b""
                kept.append("…" + raw.decode("utf-8", errors="ignore"))
            break
        kept.append(line)
        total += cost
    kept.reverse()
    return "\n".join(kept), truncated


class BasicTextPresenter:
    """TuiPresenter Protocol 的基础文本过渡实现 (05 由正式 presenter 替换).

    只消费 ApplicationUpdate, 不执行工具、不写会话、不计算自制 PASS;
    最终回答只追加一次; 工具/检查投影留给 04 的正式 presenter.
    """

    def __init__(self) -> None:
        self._final_keys: set[str] = set()
        self._run_id: str | None = None
        self._turn_index = 0
        # 文本按 (run_id, turn_index) 键控累积: 批次渲染下轮次不串扰
        self._turn_texts: dict[tuple[str, int], list[str]] = {}
        self._phase: str | None = None
        self._turn_for_status: int | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self.closed = False

    def accept(self, update: ApplicationUpdate) -> PresentationUpdate:
        if isinstance(update, AgentEventReceived):
            return self._accept_event(update.event)
        if isinstance(update, RequestPhaseChanged):
            self._phase = update.phase
            if update.run_id is not None:
                self._run_id = update.run_id
            if update.phase == "executing" and update.run_id is None:
                self._turn_for_status = None  # 新请求开始: 轮数待 RunStarted/TurnStarted 确认
            return PresentationUpdate(status=self._snapshot())
        if isinstance(update, RequestCompleted):
            outcome = update.outcome
            if outcome.result.stop_reason == "aborted":
                status = "cancelled"
            elif outcome.exit_code == 0:
                status = "done"
            else:
                status = "failed"
            record = PresentationRecord(
                key=f"request:{outcome.result.run_id}",
                role="system",
                title="",
                status=status,
                body=(f"请求完成: {outcome.result.stop_reason} (exit {outcome.exit_code})",),
            )
            return PresentationUpdate(appended=(record,), status=self._snapshot())
        if isinstance(update, HistoryCleared):
            record = PresentationRecord(
                key=f"history-cleared:{next(_RECORD_SEQ)}",
                role="system",
                title="",
                status="done",
                body=("已清空模型历史 (验证基线/账本保留)",),
            )
            return PresentationUpdate(appended=(record,))
        if isinstance(update, ApplicationClosed):
            self.closed = True
            record = PresentationRecord(
                key="application-closed",
                role="error" if update.error else "system",
                title="",
                status="failed" if update.error else "done",
                body=("应用已关闭",) if update.error is None else (f"应用已关闭: {update.error}",),
            )
            return PresentationUpdate(appended=(record,))
        return PresentationUpdate()

    def _accept_event(self, event: Event) -> PresentationUpdate:
        if isinstance(event, RunStarted):
            self._phase = "executing"
            return PresentationUpdate(status=self._snapshot())
        if isinstance(event, TurnStarted):
            self._run_id = event.run_id
            self._turn_index = event.turn_index
            self._turn_for_status = event.turn_index
            return PresentationUpdate(status=self._snapshot())
        if isinstance(event, AssistantTextDelta):
            parts = self._turn_texts.setdefault((event.run_id, event.turn_index), [])
            parts.append(event.text)
            merged = sanitize_external_text("".join(parts))
            clipped, truncated = _clip_preview_text(merged)
            preview = ActivePreview(
                run_id=event.run_id,
                turn_index=event.turn_index,
                text=clipped,
                truncated=truncated,
            )
            return PresentationUpdate(preview=preview, status=self._snapshot())
        if isinstance(event, AssistantCompleted):
            final = sanitize_external_text(event.message.text)
            key = assistant_key(event.run_id, self._turn_index)
            self._turn_texts.pop((event.run_id, self._turn_index), None)  # 该轮累积已终结
            if not final or key in self._final_keys:
                return PresentationUpdate()  # 纯工具调用轮 / 已追加过: 不重复
            self._final_keys.add(key)
            if event.message.stop_reason == "aborted":
                status = "cancelled"
            elif event.message.stop_reason == "stop":
                status = "done"
            else:
                status = "failed"
            record = PresentationRecord(
                key=key,
                role="assistant",
                title="",
                status=status,
                body=tuple(final.split("\n")),
            )
            return PresentationUpdate(appended=(record,), status=self._snapshot())
        if isinstance(event, ModelCallCompleted):
            if event.metrics is not None:
                self._input_tokens = event.metrics.input_tokens
                self._output_tokens = event.metrics.output_tokens
            return PresentationUpdate(status=self._snapshot())
        # ToolStarted / ToolCompleted: 基础 presenter 不做工具投影 (04 负责)
        return PresentationUpdate()

    def _snapshot(self) -> StatusSnapshot:
        return StatusSnapshot(
            phase=self._phase,
            turn_index=self._turn_for_status,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


_RECORD_SEQ = itertools.count(1)


class TerminalSink(Protocol):
    """终端输出 seam; 默认实现经 prompt_toolkit, 测试可注入记录器."""

    def print_line(self, text: str) -> None: ...


class PromptToolkitSink:
    """真实终端输出: 经 prompt_toolkit 的格式化输出排到 inline 主屏上方."""

    def print_line(self, text: str = "") -> None:
        print_formatted_text(text)


def _format_record(record: PresentationRecord) -> str:
    header = f"[{record.role}]"
    if record.title:
        header += f" {record.title}"
    return "\n".join([header, *record.body, ""])


_TEXT_DIRTY = object()  # 合并文本信号: 队列里最多一个, 不逐字排队


class TerminalExecutor:
    """prompt_toolkit inline 主屏: 唯一终端控制者, 协调展示/输入/mailbox 消费.

    消费 session.subscribe 的 ApplicationUpdate; subscribe 的 handler 是同步
    subscriber, 只做文本合并与入队, 不写终端. 展示渲染全部在消费任务内完成.
    """

    def __init__(
        self,
        session: ApplicationSession,
        *,
        presenter: TuiPresenter | None = None,
        sink: TerminalSink | None = None,
        input=None,
        output=None,
        use_patch_stdout: bool = False,
    ) -> None:
        self._session = session
        self._presenter = presenter if presenter is not None else BasicTextPresenter()
        self._sink = sink if sink is not None else PromptToolkitSink()
        self._use_patch_stdout = use_patch_stdout
        self._mailbox: asyncio.Queue = asyncio.Queue(maxsize=MAILBOX_CAPACITY)
        self._consumer_task: asyncio.Task | None = None
        self._unsubscribe: Callable[[], None] | None = None
        # 文本增量按 (run_id, turn_index) 合并; 插入序保持, 不丢轮次
        self._pending_text: dict[tuple[str, int], list[str]] = {}
        self._text_dirty = False
        self._printed_keys: set[str] = set()
        self.ui_failures: list[str] = []
        self._display_preview: ActivePreview | None = None
        self._display_status: StatusSnapshot | None = None
        self._user_seq = itertools.count(1)
        self._buffer = Buffer(multiline=True)
        self._app = self._build_app(input=input, output=output)

    # --- 只读视图 ------------------------------------------------------------------

    @property
    def app(self) -> Application:
        return self._app

    @property
    def buffer(self) -> Buffer:
        return self._buffer

    # --- subscriber (同步, 不写终端) ------------------------------------------------

    def _on_update(self, update: ApplicationUpdate) -> None:
        if isinstance(update, AgentEventReceived) and isinstance(update.event, AssistantTextDelta):
            delta = update.event
            self._pending_text.setdefault((delta.run_id, delta.turn_index), []).append(delta.text)
            self._mark_text_dirty()
            return
        self._enqueue_structural(update)

    def _mark_text_dirty(self) -> None:
        if self._text_dirty:
            return
        self._text_dirty = True
        try:
            self._mailbox.put_nowait(_TEXT_DIRTY)
        except asyncio.QueueFull:  # pragma: no cover - 满队即结构溢出, 下一分支处理
            self._record_ui_failure("text-merge signal dropped: mailbox overflow")

    def _enqueue_structural(self, update: ApplicationUpdate) -> None:
        try:
            self._mailbox.put_nowait(update)
        except asyncio.QueueFull:
            self._record_ui_failure(
                f"mailbox overflow; structural update dropped: {type(update).__name__}"
            )
            # T15: 结构通知溢出显式失败并协作取消; RequestOutcome 仍是收尾权威
            self._session.cancel()

    def _record_ui_failure(self, message: str) -> None:
        self.ui_failures.append(message)

    # --- 消费任务 ------------------------------------------------------------------

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        last_render = 0.0
        while True:
            item = await self._mailbox.get()
            items = [item]
            while True:
                try:
                    items.append(self._mailbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            # 约 30 FPS 的渲染节流: 等待期间的文本增量继续合并, 不逐帧排队
            wait = RENDER_MIN_INTERVAL_S - (loop.time() - last_render)
            if wait > 0:
                await asyncio.sleep(wait)
            last_render = loop.time()
            self._render_batch(items)

    def _render_batch(self, items: list[object]) -> None:
        updates: list[PresentationUpdate] = []
        if self._text_dirty:
            for (run_id, turn_index), parts in self._pending_text.items():
                merged = sanitize_external_text("".join(parts))
                updates.append(
                    self._presenter.accept(
                        AgentEventReceived(AssistantTextDelta(run_id, turn_index, merged))
                    )
                )
            self._pending_text.clear()
            self._text_dirty = False
        for item in items:
            if item is _TEXT_DIRTY:
                continue  # 已在上面统一 flush
            assert isinstance(item, (AgentEventReceived, RequestPhaseChanged, RequestCompleted,
                                     HistoryCleared, ApplicationClosed))
            updates.append(self._presenter.accept(item))  # type: ignore[arg-type]
        for update in updates:
            self._apply(update)
        if any(isinstance(item, ApplicationClosed) for item in items):
            self._app.exit()  # 关闭通知已送达: 退出主屏, run_async 完成收尾
        self._app.invalidate()

    def _apply(self, update: PresentationUpdate) -> None:
        for record in update.appended:
            if record.key in self._printed_keys:
                continue  # 已完成记录不重复打印
            self._printed_keys.add(record.key)
            self._sink.print_line(_format_record(record))
            # 最终完成替换预览: 同 run/turn 的最终回答已追加, 收起活动预览
            if (
                record.role == "assistant"
                and self._display_preview is not None
                and record.key
                == assistant_key(self._display_preview.run_id, self._display_preview.turn_index)
            ):
                self._display_preview = None
        if update.preview is not None:
            self._display_preview = update.preview
        if update.status is not None:
            self._display_status = update.status

    # --- 展示布局 ------------------------------------------------------------------

    def _get_preview_lines(self) -> list[tuple[str, str]]:
        preview = self._display_preview
        if preview is None:
            return [("", "(尚无输出 — 输入问题并回车; /help 查看命令)")]
        marker = " [截断]" if preview.truncated else ""
        lines = [("", f"┌ 预览 run={preview.run_id} turn={preview.turn_index}{marker}")]
        lines.extend(("", f"│ {line}") for line in preview.text.split("\n"))
        return lines

    def _status_summary(self) -> str:
        snap = self._display_status or StatusSnapshot()
        config = self._session.config
        model = snap.model if snap.model is not None else (config.model_label or None)
        workspace = snap.workspace if snap.workspace is not None else str(config.workspace)
        phase = snap.phase if snap.phase is not None else self._session.state
        turn = "-" if snap.turn_index is None else str(snap.turn_index)
        input_tokens = "-" if snap.input_tokens is None else str(snap.input_tokens)
        output_tokens = "-" if snap.output_tokens is None else str(snap.output_tokens)
        parts = [
            f"model={model or '?'}",
            f"workspace={workspace}",
            f"phase={phase}",
            f"turn={turn}",
            f"in={input_tokens}",
            f"out={output_tokens}",
        ]
        if self._session.verification_run_id is not None:
            parts.append(f"verification={self._session.verification_run_id}")
        if self.ui_failures:
            parts.append(f"ui_failures={len(self.ui_failures)}")
        return " ".join(parts)

    def _get_status_line(self) -> list[tuple[str, str]]:
        return [("reverse", f" {self._status_summary()} ")]

    # --- 输入 ----------------------------------------------------------------------

    def _build_app(self, *, input=None, output=None) -> Application:
        kb = KeyBindings()
        executor = self

        @kb.add("enter", filter=~in_paste_mode)
        def _submit(event) -> None:  # noqa: ANN001
            raw = executor._buffer.text
            stripped = raw.strip()
            if stripped.startswith("/"):
                if stripped.startswith("//"):
                    executor._accept_prompt(raw[1:], event)  # // 转义: 提交 "/..."
                    return
                if executor._handle_command(stripped, event):
                    executor._buffer.reset()
                return
            executor._accept_prompt(raw, event)

        @kb.add("escape", "enter")
        def _newline(event) -> None:  # noqa: ANN001
            executor._buffer.insert_text("\n")

        @kb.add("escape")
        def _escape(event) -> None:  # noqa: ANN001
            # 孤立 Escape 经 KeyProcessor timeoutlen 消歧后到达 (01 探针裁决)
            if executor._session.state in _BUSY_STATES:
                executor._session.cancel()

        @kb.add("c-c")
        def _ctrl_c(event) -> None:  # noqa: ANN001
            if executor._session.state in _BUSY_STATES:
                executor._session.cancel()  # 协作取消, 不伪造 idle
            else:
                event.app.exit()  # 空闲 Ctrl+C: 进入关闭序列

        @kb.add("c-d")
        def _ctrl_d(event) -> None:  # noqa: ANN001
            if executor._session.state not in _BUSY_STATES and not executor._buffer.text:
                event.app.exit()

        preview_win = Window(
            FormattedTextControl(executor._get_preview_lines),
            height=D(min=1, max=PREVIEW_MAX_LINES),
        )
        separator_win = Window(height=1, char="─")
        input_win = Window(BufferControl(buffer=executor._buffer), wrap_lines=True)
        status_win = Window(FormattedTextControl(executor._get_status_line), height=1)
        return Application(
            layout=Layout(
                HSplit([preview_win, separator_win, input_win, status_win]),
                focused_element=input_win,
            ),
            key_bindings=kb,
            full_screen=False,  # 主屏内联: 不进 alternate screen, 不清 scrollback
            input=input,
            output=output,
            style=Style.from_dict({"reverse": "reverse"}),
        )

    def _accept_prompt(self, raw: str, event) -> None:  # noqa: ANN001
        state = self._session.state
        if state != "idle":
            # executing/cancelling/finalizing/closing/closed 一律拒绝; 不排队, 不伪造用户消息
            self._sink.print_line(f"[system] {state}: 提交被拒绝, 草稿保留 (第一版不排队)")
            self._app.invalidate()
            return
        if not raw.strip():
            return
        # 已接受的提问: 原始多行内容保留为一次 [user] 记录 (显示文本先净化);
        # 清空输入框只是为下一条草稿, 不移除对话流正文
        record = PresentationRecord(
            key=f"user:{next(self._user_seq)}",
            role="user",
            title="",
            status="done",
            body=tuple(sanitize_external_text(raw).split("\n")),
        )
        self._sink.print_line(_format_record(record))
        self._buffer.reset()
        asyncio.ensure_future(self._run_prompt(raw))

    async def _run_prompt(self, prompt: str) -> None:
        try:
            await self._session.execute(prompt)
        except RuntimeError as exc:
            self._sink.print_line(f"[error] 请求未执行: {exc}")
        except Exception as exc:  # 内核/UI 异常: 展示后经应用关闭路径退出
            self._sink.print_line(f"[error] 请求异常终止: {exc!r}")
        finally:
            if self._session.state in ("closing", "closed") and self._app.is_running:
                self._app.exit()
            self._app.invalidate()

    def _handle_command(self, command: str, event) -> bool:  # noqa: ANN001
        """处理斜杠命令; 返回输入是否已被消费 (False = 草稿保留)."""
        if command == "/help":
            self._sink.print_line("\n".join([
                "[system] 可用命令:",
                "  /help    显示本帮助",
                "  /status  显示当前状态快照",
                "  /clear   清空模型历史 (仅空闲; 验证基线/账本保留)",
                "  /exit    退出 (忙碌时取消活动请求后关闭)",
                "  //text   输入以 / 开头的普通文本 (转义)",
                "",
            ]))
            return True
        if command == "/status":
            self._sink.print_line(f"[system] {self._status_summary()}")
            return True
        if command == "/clear":
            try:
                self._session.clear_history()  # 忙碌时抛 RuntimeError: 拒绝且草稿保留
            except RuntimeError as exc:
                self._sink.print_line(f"[system] 清空被拒绝: {exc}")
                return False
            return True  # HistoryCleared 通知经 mailbox 展示
        if command == "/exit":
            event.app.exit()
            return True
        self._sink.print_line(f"[system] 未知命令 {command} (输入 /help 查看可用命令)")
        return False  # 未知命令只提示, 草稿保留可修正

    # --- 生命周期 ------------------------------------------------------------------

    async def run_async(self) -> None:
        """运行主屏直至退出; 退出后按 T11 顺序收尾.

        顺序: Application 退出 (prompt_toolkit 恢复终端输入/光标模式) -> 退订 ->
        停 mailbox 消费 -> await session.aclose().
        """
        self._unsubscribe = self._session.subscribe(self._on_update)
        self._consumer_task = asyncio.get_running_loop().create_task(self._consume())
        try:
            if self._use_patch_stdout:
                from prompt_toolkit.patch_stdout import patch_stdout

                with patch_stdout():
                    await self._app.run_async()
            else:
                await self._app.run_async()
        finally:
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            if unsubscribe is not None:
                unsubscribe()
            consumer, self._consumer_task = self._consumer_task, None
            if consumer is not None:
                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass
            await self._session.aclose()

    def run(self) -> None:
        """同步入口: 运行主屏直至退出并完成关闭收尾."""
        asyncio.run(self.run_async())


__all__ = [
    "BasicTextPresenter",
    "PromptToolkitSink",
    "TerminalExecutor",
    "TerminalSink",
    "assistant_key",
    "sanitize_external_text",
    "MAILBOX_CAPACITY",
    "PREVIEW_MAX_BYTES",
    "PREVIEW_MAX_LINES",
    "RENDER_MIN_INTERVAL_S",
]
