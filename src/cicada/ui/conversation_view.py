"""会话组合入口 (票据 02): ApplicationSession 事件 -> 显示文档投影 + 输入准入.

职责 (规格 §一个前端接口与一个终端代理 / 设计 §运行结构):
- 订阅 ApplicationUpdate 并投影到 ui/transcript.py 的显示文档:
  AssistantTextDelta 累积、AssistantCompleted 校准同一条目 (按内核
  TurnStarted 跟踪的当前轮定位, 完成事件本身不带 turn_index);
  ModelCallCompleted 计量累计; RequestCompleted 记录错误/取消条目;
  HistoryCleared 追加系统条目 (显示文档保留)。
- 提供可注入 MainScreenFrontend 的开发组合 build(): document_lines 回调是
  01 冻结的显示文档替换点; on_submit 为受理制 (返回 False = 拒绝且草稿保留)。
- 斜线命令只在 idle 且无 reservation 时执行: /help /status /clear /exit 与
  //转义文本; 未知命令提示且草稿保留。/clear 在持久会话经共享 async
  clear_context() 可靠提交 (05), 提交期间输入屏障; 非持久保持旧同步入口。
- 本模块不起 renderer、不写终端: 组合层只投递状态, 渲染请求经 on_update
  回调 (宿主传 frontend.request_render 合并节流)。
- 正常对话不出现内部 phase 标签; 诊断细节归 /status (只读 diagnostics 对象
  经 build(diagnostics=...) 注入, 本模块默认只给已发布事实)。
- 工具卡经 ToolDocument 投影并按到达序挂载显示文档 (add_entry); 检查事实经
  CheckProjection 消费 RequestCompleted.outcome; /status 汇集检查事实与注入
  诊断并施加有界限额。
- 票据 05: 订阅前先把恢复快照 (restored_requests/restored_interrupted) 投影
  成只读历史条目 (ui/restore_view), 查看状态原因如实成条; 保存失败显示真实
  未保存事实 (后继编码由应用 storage_failed 锁存拒绝)。
- 票据 04: /status 新增预算段 (占用/上限/预留/余量/来源/锚/校准偏差) 与摘要段
  (压缩次数/边界/摘要计量单列), 只读出口为 session.budget 与
  session.summary_records; 恢复显示在压缩边界插入程序条目。
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from cicada.application import (
    AgentEventReceived,
    ApplicationClosed,
    ApplicationSession,
    ApplicationUpdate,
    ContextCompacted,
    HistoryCleared,
    OperationBusy,
    RequestCompleted,
    RequestPhaseChanged,
)
from cicada.budget import BudgetReport, BudgetSnapshot
from cicada.core.events import (
    AssistantCompleted,
    AssistantTextDelta,
    Event,
    ModelCallCompleted,
    TurnStarted,
)
from cicada.ui.admission import AdmissionGate, cancel_if_active, is_enter_barrier
from cicada.ui.checks import (
    PROCESS_UNCERTAIN_LABEL,
    STATUS_MAX_BYTES,
    STATUS_MAX_LINES,
    CheckProjection,
    clip_lines,
    clip_status_field,
    truncation_note,
)
from cicada.ui.editor import Editor, EditorStyle
from cicada.ui.commands import SLASH_COMMANDS
from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width
from cicada.ui.restore_view import project_restored_history
from cicada.ui.theme import FooterComponent
from cicada.ui.tool_view import ToolCardEntry, ToolDocument
from cicada.ui.transcript import Transcript

# 状态行 (规格 §F): braille spinner 与节奏; 帧由可注入时钟决定 (测试可冻结)
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SPINNER_INTERVAL_S = 0.1


class ConversationView:
    """ApplicationSession 与显示文档/输入回调之间的唯一组合层。"""

    def __init__(
        self,
        session: ApplicationSession,
        *,
        editor: Editor | None = None,
        on_exit: Callable[[], None] | None = None,
        on_update: Callable[[], None] | None = None,
        diagnostics: Callable[[], list[str]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._session = session
        self._editor = editor if editor is not None else Editor(
            border_color=theme.editor_border(),
            padding_x=2,
            style=EditorStyle(
                line_bg=theme.EDITOR_BG,
                prompt_glyph="›",
                prompt_color=theme.EDITOR_BORDER,
                placeholder="输入消息… Enter 发送, Shift+Enter 换行",
                placeholder_color=theme.EDITOR_PLACEHOLDER,
            ),
        )
        self._on_exit = on_exit
        self._on_update = on_update
        self._diagnostics = diagnostics
        self.transcript = Transcript()
        self._gate = AdmissionGate(session)
        self._gate.on_error = self._on_gate_error
        # 02: 压缩等待任务的强引用 (PA05: 迟到收尾任务不被垃圾回收, 保留诊断面)
        self._compact_task: asyncio.Task | None = None
        # 状态行 (规格 §F): spinner 帧时钟与动画任务; 忙时驱动 on_update, 转闲即停
        self._clock = clock if clock is not None else time.monotonic
        self._spinner_task: asyncio.Task | None = None
        # 票据 05 组合钩子: 会话进入 closing/closed 且 execute 异常到达时由宿主
        # 决定退出前端 (避免 barrier 内界面失去退出路径); 默认无动作
        self.on_unrecoverable: Callable[[], None] | None = None
        # 票据 05: 工具卡文档片段与检查事实投影
        self.tools = ToolDocument()
        self._checks = CheckProjection(session)
        # AssistantCompleted 不带 turn_index: 按内核 TurnStarted 跟踪当前轮
        self._current_turn: dict[str, int] = {}
        # 04 (审查 04-F1): /clear 后到下一请求执行前, 预算快照仍是旧代次的最后
        # 估计 (clear 不触碰 budget policy); 置位时 /status 预算段如实标注时效
        self._budget_stale = False
        # 票据 05: 建立订阅前注入只读恢复显示; 查看状态原因如实成条。
        # 04: 预算启用时传当前代次摘要记录 (压缩边界程序条目); 未启用时模型
        # 看到完整原文, 不得声称"模型所见为摘要"
        if session.restored_requests or session.restored_interrupted:
            project_restored_history(
                self.transcript,
                self.tools,
                requests=session.restored_requests,
                interrupted=session.restored_interrupted,
                summaries=session.summary_records if session.budget is not None else (),
            )
            if session.view_only_reason is not None:
                self.transcript.append_system(
                    f"恢复为查看状态（新请求被拒绝）：{session.view_only_reason}"
                )
        self._unsubscribe = session.subscribe(self._on_application_update)
        # 04 底栏: 工作区/模型来自配置事实, 计量来自显示文档累计
        self.footer = FooterComponent(
            str(session.config.workspace),
            session.config.model_label,
            self.transcript.metrics_fact,
        )

    # --- 生命周期 -----------------------------------------------------------------

    def close(self) -> None:
        """退订应用事件并停掉状态行动画 (组合层不拥有 session 关闭, aclose 归宿主路径)。"""
        self._unsubscribe()
        if self._spinner_task is not None:  # 防御: 宿主"关 view 不关 session"时不泄漏
            self._spinner_task.cancel()
            self._spinner_task = None

    @property
    def editor(self) -> Editor:
        """组合层编辑器: Ctrl+C 清草稿与宿主注入共用同一实例。"""
        return self._editor

    # --- 应用事件投影 ----------------------------------------------------------------

    def _on_application_update(self, update: ApplicationUpdate) -> None:
        if isinstance(update, AgentEventReceived):
            self._project_agent_event(update.event)
        elif isinstance(update, RequestPhaseChanged):
            self._checks.note_phase(update.run_id, update.phase)
            self._gate.note_phase(update.phase)
            if update.phase == "executing":
                # 04 (审查 04-F1): 新请求已开始执行, 预算段随后按当前请求重估
                self._budget_stale = False
        elif isinstance(update, RequestCompleted):
            outcome = update.outcome
            self.transcript.record_request_result(outcome.result.stop_reason, outcome.result.error)
            # 检查结论条目只在 decision 非空 (检查模式) 时追加; 正常完成安静
            summary = self._checks.on_request_completed(outcome)
            if summary is not None:
                self.transcript.append_system("\n".join(summary))
            if outcome.persistence is not None and not outcome.persistence.saved:
                # 05: 保存失败如实显示; 后继编码由应用 storage_failed 锁存拒绝
                self.transcript.append_error(
                    f"[session save failed: {outcome.persistence.error}]"
                )
            self._gate.note_request_completed()
        elif isinstance(update, OperationBusy):
            # 05/02: 持久清空/压缩提交中 (屏障在命令处置时已同步置位, 此处兜底)
            if update.operation == "compact":
                self._gate.note_compact_started()
                self.transcript.append_system("正在提交压缩（摘要较早请求，原始记录保留）…")
            else:
                self._gate.note_clear_started()
                self.transcript.append_system("正在提交清空（显示记录与验证状态保留）…")
        elif isinstance(update, ContextCompacted):
            # 02: 压缩切换成功; 确认条目 + 兜底释放屏障 (主释放由等待任务 finally)
            self._gate.note_compact_settled()
            self.transcript.append_system(
                "已完成压缩：较早请求已摘要（覆盖 "
                f"seq {update.first_request_seq}..{update.last_request_seq}，"
                f"{update.message_count} 条消息），原始记录保留可回看。"
            )
        elif isinstance(update, HistoryCleared):
            # /clear 只清模型历史: 显示文档保留, 追加系统条目说明
            self.transcript.append_system("已清空模型历史（已显示记录与验证状态保留）。")
            # 04 (审查 04-F1): 当前窗口已释放, 既有预算快照自此时效待重估
            self._budget_stale = True
            if self._session.has_process_uncertain():
                self.transcript.append_system(f"{PROCESS_UNCERTAIN_LABEL}：清空历史不解锁。")
        elif isinstance(update, ApplicationClosed):
            self._gate.note_closed()
        self._sync_spinner()  # 状态行跟随最近一次应用状态 (规格 §F)
        if self._on_update is not None:
            self._on_update()

    def _on_gate_error(self, message: str) -> None:
        """execute 异常如实成条; 会话已进入 closing/closed 时通知宿主退出前端。"""
        self.transcript.append_error(message)
        if (
            self._session.state in ("closing", "closed")
            and self.on_unrecoverable is not None
        ):
            self.on_unrecoverable()
        if self._on_update is not None:
            self._on_update()  # 错误条目不经应用事件, 必须显式触发渲染

    def _project_agent_event(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            self._current_turn[event.run_id] = event.turn_index
        elif isinstance(event, AssistantTextDelta):
            self.transcript.stream_assistant_delta(event.run_id, event.turn_index, event.text)
        elif isinstance(event, AssistantCompleted):
            turn_index = self._current_turn.get(event.run_id, 0)
            self.transcript.complete_assistant(event.run_id, turn_index, event.message)
        elif isinstance(event, ModelCallCompleted):
            self.transcript.add_metrics(event.metrics)
        # 工具卡与检查回执收集: 全部内核事件过一遍 (各自过滤)
        self._checks.observe_event(event)
        before = len(self.tools.entries)
        self.tools.apply(event)
        new_entries = self.tools.entries
        for index in range(before, len(new_entries)):
            self.transcript.add_entry(ToolCardEntry(self.tools, index))

    # --- MainScreenFrontend 注入面 ----------------------------------------------------

    def document_lines(self, width: int) -> list[str]:
        """01 冻结的显示文档替换点: 整份文档按当前宽度排版。"""
        return self.transcript.render(width)

    # --- 状态行 (tui-aesthetics 规格 §F) ---------------------------------------------

    def status_lines(self, width: int) -> list[str]:
        """输入框上方状态行: 忙时 1 行 (spinner + 真实状态文案), 空闲 0 行。

        宽度 < 4 放不下 spinner + 空格 + 1 列文案, 不占行 (防御截断无噪音)。
        """
        text = self._status_text()
        if text is None or width < 4:
            return []
        frame = SPINNER_FRAMES[int(self._clock() / SPINNER_INTERVAL_S) % len(SPINNER_FRAMES)]
        text = truncate_to_width(text, max(1, width - 2))
        return [
            theme.fg_open(theme.EDITOR_BORDER) + frame
            + theme.fg_open(theme.MUTED_GRAY) + " " + text
            + "\x1b[0m"
        ]

    def _status_text(self) -> str | None:
        """状态文案只取真实状态: 屏障/取消/执行/reservation 窗口, 空闲 None。"""
        if self._gate.clearing:
            return "正在清空…"
        if self._gate.compacting:
            return "正在压缩…"
        if self._session.state == "cancelling":
            return "正在取消…"
        if self._session.state == "finalizing":
            return "正在收尾…"  # 模型已停笔, 持久化/检查收尾中 (审查 06-3)
        if self._session.state == "executing" or self._gate.busy():
            return "正在输出…"
        return None

    def _sync_spinner(self) -> None:
        """忙时确保动画任务在跑, 转闲即停; 无 on_update 宿主或无事件循环不起任务。"""
        active = self._status_text() is not None
        if active and self._spinner_task is None and self._on_update is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._spinner_task = loop.create_task(self._spin())
            self._spinner_task.add_done_callback(self._spin_done)
        elif not active and self._spinner_task is not None:
            self._spinner_task.cancel()
            self._spinner_task = None

    def _spin_done(self, task: asyncio.Task) -> None:
        if self._spinner_task is task:
            self._spinner_task = None

    async def _spin(self) -> None:
        try:
            while self._status_text() is not None:
                await asyncio.sleep(SPINNER_INTERVAL_S)
                if self._on_update is not None:
                    self._on_update()
        except asyncio.CancelledError:
            pass

    def on_submit(self, text: str) -> bool:
        """受理制提交: 返回 False = 拒绝 (草稿保留); True = 接受 (前端清草稿)。"""
        if text.startswith("//"):
            return self._admit_coding(text[1:])  # // 转义: 去一个斜杠按普通文本发送
        if text.startswith("/"):
            return self._handle_command(text)
        return self._admit_coding(text)

    def _admit_coding(self, prompt: str) -> bool:
        """编码请求准入: 拒绝时无伪用户条目, 接受时同步占位后创建任务。"""
        task = self._gate.try_admit(prompt)
        if task is None:
            return False
        # 应用准入事实已取得 (reservation 已占, 任务已建): 此时才创建用户条目
        self.transcript.append_user(prompt)
        self._sync_spinner()  # reservation 窗口即忙: 状态行与动画立即上线
        return True

    def handle_ctrl_c(self) -> None:
        """Ctrl+C: executing/cancelling 公共 cancel (幂等); reservation 窗口记录
        取消意图; 持久清空/压缩提交中/finalizing/closing/closed 静默; 空闲清草稿。"""
        if self._gate.clearing or self._gate.compacting:
            return
        if cancel_if_active(self._session):
            return
        if self._session.state in ("finalizing", "closing", "closed"):
            return
        if self._gate.busy():
            self._gate.arm_pending_cancel()
            return
        self._editor.clear()

    def handle_escape(self) -> None:
        """Escape: executing/cancelling 公共 cancel; reservation 窗口记录取消意图;
        持久清空/压缩提交中与空闲均无动作; finalizing/closing/closed 静默。"""
        if self._gate.clearing or self._gate.compacting:
            return
        if cancel_if_active(self._session):
            return
        if self._gate.busy() and self._session.state == "idle":
            self._gate.arm_pending_cancel()

    def handle_ctrl_o(self) -> None:
        """Ctrl+O 接缝 (票据 05): 切换工具输出展开; 只改展示, 不触请求或工具。"""
        self.tools.toggle_expand()

    def handle_exit(self) -> None:
        """退出回调 (Ctrl+D 与 /exit 共用): 仅 idle 且无 reservation 时执行。"""
        if is_enter_barrier(self._session, self._gate):
            return
        if self._on_exit is not None:
            self._on_exit()

    # --- 斜线命令 ----------------------------------------------------------------------

    def _handle_command(self, text: str) -> bool:
        # 斜线命令统一只在 idle 且无 reservation 时执行; 屏障内静默拒绝
        if is_enter_barrier(self._session, self._gate):
            return False
        command = text.strip()
        if command == "/help":
            command_width = max(len(item.command) for item in SLASH_COMMANDS)
            self.transcript.append_information("命令帮助", [
                ("可用命令", [
                    f"{item.command:<{command_width}}  {item.description}" for item in SLASH_COMMANDS
                ]),
                ("普通文本", ["//内容  发送以斜杠开头的普通文本（转义）"]),
            ])
            return True
        if command == "/status":
            self.transcript.append_information("会话信息", self._status_sections())
            return True
        if command == "/clear":
            if self._session.persistence is not None:
                # 05: 持久清空经共享 async clear_context() 可靠提交; 命令处置时
                # 同步置屏障 (先于任务起跑窗口), 落定 (成功/失败) 后释放
                self._gate.note_clear_started()
                self._sync_spinner()
                asyncio.get_running_loop().create_task(self._run_persistent_clear())
                return True
            try:
                self._session.clear_history()  # 非空闲已被屏障拦截; 防御捕获
            except RuntimeError as exc:
                self.transcript.append_error(str(exc))
                return True
            # 确认条目由 HistoryCleared 事件投影追加
            return True
        if command == "/compact":
            if self._session.persistence is None:
                self.transcript.append_system(
                    "压缩需要持久会话（启动时带 --save-session/--tui/--resume）；"
                    "非持久会话只有预算准入，无压缩。"
                )
                return True
            if self._session.budget is None:
                self.transcript.append_system(
                    "压缩需要启用预算（真实模型按 --num-ctx 自动启用；fake 模式无预算）。"
                )
                return True
            # 02: 持久压缩经共享 async compact_context() 可靠提交; 屏障与清空同型
            self._gate.note_compact_started()
            self._sync_spinner()
            self._compact_task = asyncio.get_running_loop().create_task(
                self._run_persistent_compact()
            )
            return True
        if command == "/exit":
            self.handle_exit()
            return True
        self.transcript.append_system(f"未知命令: {command}（/help 查看可用命令）")
        return False  # 未知命令提示且草稿保留

    async def _run_persistent_clear(self) -> None:
        """等待持久清空提交落定; 成功的确认条目由 HistoryCleared 投影追加。

        取消路径 (外层取消且事务最终成功, 应用同步内存后传播 CancelledError)
        也必须释放屏障 (审查 05-F8): 用 BaseException 分支保证, 取消如实再传播。
        """
        try:
            await self._session.clear_context()
        except Exception as exc:
            # 失败: 内存历史不变 (应用保证), 屏障释放并如实成条
            self._gate.note_clear_settled()
            self.transcript.append_error(f"清空提交失败: {exc}")
            if self._on_update is not None:
                self._on_update()
            return
        except BaseException:
            # CancelledError 等: 屏障释放后如实传播, 不吞异常
            self._gate.note_clear_settled()
            raise
        self._gate.note_clear_settled()

    async def _run_persistent_compact(self) -> None:
        """等待持久压缩提交落定 (02); 成功确认条目由 ContextCompacted 投影追加。

        落定 (成功/失败/取消/放弃) 都释放屏障; 取消如实传播; 失败如实成条。
        close 封口放弃 (返回 None) 不是失败: 静默结束, 投影由应用保证不变。
        """
        try:
            await self._session.compact_context()
        except Exception as exc:
            self._gate.note_compact_settled()
            self.transcript.append_error(f"压缩失败: {exc}")
            if self._on_update is not None:
                self._on_update()
            return
        except BaseException:
            self._gate.note_compact_settled()
            raise
        self._gate.note_compact_settled()

    def _status_sections(self) -> list[tuple[str, list[str]]]:
        """/status: 已发布状态 + 持久化事实 + 检查事实 + 计量口径 + 注入诊断; 有界 (P12)。"""
        session_lines = [
            f"状态: {self._session.state}",
            f"模型: {self._session.config.model_label or '不可得'}",
        ]
        persistence = self._session.persistence
        if persistence is not None:
            # 05: 会话/保存/恢复来源事实进入状态页
            session_lines.extend([
                f"会话 ID: {persistence.session_id}",
                f"模式: {persistence.mode} · epoch {persistence.epoch_id[:8]}",
                (
                    f"上下文代次: {persistence.context_generation} · "
                    f"已保存 {persistence.saved_request_count} 条 · "
                    f"恢复 {persistence.restored_request_count} 条"
                ),
                f"历史模型调用: {persistence.restored_model_call_count} 次",
            ])
            if persistence.storage_failed is not None:
                session_lines.append(f"保存失败已锁存: {persistence.storage_failed}")
            if self._session.view_only_reason is not None:
                session_lines.append(f"查看状态: {self._session.view_only_reason}")
            if self._session.has_process_uncertain():
                session_lines.append(PROCESS_UNCERTAIN_LABEL)
        diagnostics = self._diagnostics() if self._diagnostics is not None else []
        # CLI 诊断重复提供相同模型事实时只展示一次，保留任何不同的诊断值。
        diagnostics = [line for line in diagnostics if line != session_lines[1]]
        sections = [
            ("会话", session_lines),
            *self._budget_sections(),
            ("计量", [self.transcript.metrics_summary()]),
            ("检查", self._checks.status_lines()),
            ("配置", diagnostics),
        ]
        # 保留原有字段顺序及整体限额; 装饰标题/空行不挤占事实行预算。
        clipped_sections = [
            (heading, [clip_status_field(row) for row in rows]) for heading, rows in sections
        ]
        fields = [row for _, rows in clipped_sections for row in rows]
        total = "\n".join(fields)
        if len(fields) <= STATUS_MAX_LINES and len(total.encode("utf-8")) <= STATUS_MAX_BYTES:
            return clipped_sections
        kept, original_lines, original_bytes, truncated = clip_lines(
            total, STATUS_MAX_LINES, STATUS_MAX_BYTES,
        )
        result: list[tuple[str, list[str]]] = []
        offset = 0
        for heading, rows in clipped_sections:
            # clip_lines 展开字段内换行; 加入字段间换行以保留尾部空行归属。
            count = sum(len((row + "\n").splitlines()) for row in rows)
            group = list(kept[offset:offset + count])
            if group:
                result.append((heading, group))
            offset += count
        notes = truncation_note(kept, original_lines, original_bytes, truncated)
        if notes:
            result.append(("", list(notes)))
        return result

    # --- 预算与摘要段 (04, A12) ------------------------------------------------------

    def _budget_sections(self) -> list[tuple[str, list[str]]]:
        """/status 预算段 + 摘要段: 只读出口 session.budget / session.summary_records.

        占用 (当前窗口估计) 与 计量 (跨调用累计) 分行分区, 文字不混用; 来源
        等级与估计偏差如实显示, 缺实测/缺估计时如实不可得。预算未启用时单行
        说明, 摘要段省略 (无压缩能力)。校准事实的机器出口是 session.budget
        (BudgetSnapshot.anchor) 与 session.summary_records, 05 直接读取。
        """
        snapshot = self._session.budget
        if snapshot is None:
            return [("预算", ["未启用（真实模型按 --num-ctx 自动启用；fake 模式无预算）"])]
        report = snapshot.report
        lines: list[str] = []
        if report is None:
            lines.append(f"预算: 上限 {snapshot.limit} · 占用估计不可得（本进程尚未发送请求）")
        else:
            if self._budget_stale:
                # 04 (审查 04-F1): clear 不重置预算策略, 此处数字描述清空前旧代次
                lines.append("提示: /clear 后尚无新请求，以下为清空前旧代次的最后估计")
            lines.append(
                f"预算: 上限 {report.limit} · 占用估计 {report.total_estimate}"
                f" · 余量 {report.margin_remaining}"
            )
            lines.append(f"来源: {report.source} · 估计器 {report.estimator} v{report.estimator_version}")
            lines.append(f"占用分解: {self._budget_breakdown(report)}")
            lines.extend(self._budget_anchor_lines(snapshot, report))
        return [("预算", lines), ("摘要", self._summary_section_lines())]

    @staticmethod
    def _budget_breakdown(report: BudgetReport) -> str:
        """占用逐项分解: 非消息组件逐项, 消息组件按类别合并 (条数+合计)。

        各项之和恒等于总估计; 显示聚合只省行数不省事实 (状态页有界裁剪口径)。
        """
        parts: list[str] = []
        message_sums: dict[str, list[int]] = {}  # kind -> [条数, tokens 合计]
        for component in report.components:
            if component.label.startswith("message:"):
                kind = component.label.rsplit(":", 1)[-1]
                count, total = message_sums.get(kind, [0, 0])
                message_sums[kind] = [count + 1, total + (component.tokens or 0)]
            else:
                tokens = "不可得" if component.tokens is None else str(component.tokens)
                parts.append(f"{component.label}~{tokens}")
        for kind, (count, total) in message_sums.items():
            parts.append(f"消息({kind})x{count}~{total}")
        return " · ".join(parts)

    @staticmethod
    def _budget_anchor_lines(snapshot: BudgetSnapshot, report: BudgetReport) -> list[str]:
        """锚状态与校准偏差 (04/A12): 如实文字, 不虚构失效原因。"""
        anchor = snapshot.anchor
        if anchor is None:
            return ["锚: 未立锚（尚无带实测计量的成功请求；/clear 或恢复后重置）"]
        lines = [
            f"锚: 已立锚 · 实测输入 {anchor.measured_input_tokens}"
            f" · 锚定请求含 {anchor.message_count} 条消息",
            (
                "最近估计沿用锚"
                if report.source in ("measured", "anchored")
                else "最近估计未使用锚（估计先于本锚确立，或请求前缀已变化）"
            ),
        ]
        if anchor.estimated_input_tokens is None:
            lines.append("校准偏差: 不可得（锚缺少立锚时的估计值）")
        else:
            deviation = anchor.estimated_input_tokens - anchor.measured_input_tokens
            lines.append(
                f"校准偏差: 估计 {anchor.estimated_input_tokens} − "
                f"实测 {anchor.measured_input_tokens} = {deviation:+d}（正=启发式偏高）"
            )
        return lines

    def _summary_section_lines(self) -> list[str]:
        """摘要段: 压缩次数/最近原因/当前代次边界 + 摘要调用计量单列。

        计量从摘要记录派生 (恢复后随记录一次可得), 不进累计计量、不随显示
        重复累计; None 字段保持不可得。
        """
        if self._session.persistence is None:
            return ["压缩: 不可用（需持久会话）"]
        records = self._session.summary_records
        lines: list[str] = []
        if records:
            last = records[-1]
            lines.append(
                f"压缩: {len(records)} 次 · 最近原因 {last.reason}"
                f" · 已摘要至消息 {last.end_message_index}（保留起点，其后原文可见）"
            )
        else:
            lines.append("压缩: 0 次 · 摘要边界: 无（未压缩）")
        if not records:
            lines.append("摘要计量: 不可得（本代次尚无已提交摘要）")
            return lines
        inputs = [r.metrics.input_tokens for r in records if r.metrics is not None and r.metrics.input_tokens is not None]
        outputs = [r.metrics.output_tokens for r in records if r.metrics is not None and r.metrics.output_tokens is not None]
        durations = [r.metrics.provider_duration_s for r in records if r.metrics is not None and r.metrics.provider_duration_s is not None]

        def fmt(values: list, partial: bool, unit: str = "") -> str:
            if not values:
                return "不可得"
            text = f"{sum(values):.1f}{unit}" if unit else str(sum(values))
            return f"{text}（部分调用不可得）" if partial else text

        partial_in = any(r.metrics is None or r.metrics.input_tokens is None for r in records)
        partial_out = any(r.metrics is None or r.metrics.output_tokens is None for r in records)
        partial_dur = any(r.metrics is None or r.metrics.provider_duration_s is None for r in records)
        lines.append(
            f"摘要计量: {len(records)} 次 · 累计输入 {fmt(inputs, partial_in)}"
            f" · 累计输出 {fmt(outputs, partial_out)} · 累计时长 {fmt(durations, partial_dur, 's')}"
        )
        return lines


def build(
    session: ApplicationSession,
    *,
    editor: Editor | None = None,
    on_exit: Callable[[], None] | None = None,
    on_update: Callable[[], None] | None = None,
    diagnostics: Callable[[], list[str]] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict:
    """开发组合入口: 返回可直接注入 MainScreenFrontend 的回调集合。

    用法: bundle = build(session, on_exit=frontend.request_exit);
    MainScreenFrontend(terminal, editor=bundle["editor"],
    document_lines=bundle["document_lines"], on_submit=bundle["on_submit"], ...)。
    本函数不创建 renderer/终端, 单元测试直接调 document_lines 与 on_submit。
    """
    view = ConversationView(
        session,
        editor=editor,
        on_exit=on_exit,
        on_update=on_update,
        diagnostics=diagnostics,
        clock=clock,
    )
    return {
        "document_lines": view.document_lines,
        "on_submit": view.on_submit,
        "on_escape": view.handle_escape,
        "on_ctrl_c": view.handle_ctrl_c,
        "on_exit": view.handle_exit,
        "editor": view.editor,
        "view": view,
        "footer_lines": view.footer.render,  # (width, rows) -> 底栏行
        "status_lines": view.status_lines,  # (width) -> 状态行 (规格 §F), 空闲 []
    }


__all__ = ["ConversationView", "build"]
