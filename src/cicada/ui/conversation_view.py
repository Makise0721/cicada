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
  //转义文本; 未知命令提示且草稿保留。/clear 只清模型历史
  (session.clear_history), 显示文档保留并追加系统条目; 命令输出也是显示条目。
- 本模块不起 renderer、不写终端: 组合层只投递状态, 渲染请求经 on_update
  回调 (宿主传 frontend.request_render 合并节流)。
- 正常对话不出现内部 phase 标签; 诊断细节归 /status (票据 05 的只读
  diagnostics 对象可经 build(diagnostics=...) 注入, 本模块默认只给已发布事实)。
- 票据 05 组合接线: 工具卡经 ToolDocument 投影并按到达序挂载显示文档
  (add_entry); 检查事实经 CheckProjection 消费 RequestCompleted.outcome
  (结论条目只在 decision 非空时追加, 正常完成安静); /status 汇集检查事实
  与注入诊断并施加有界限额。
"""

from __future__ import annotations

from typing import Callable

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
    TurnStarted,
)
from cicada.ui.admission import AdmissionGate, cancel_if_active, is_enter_barrier
from cicada.ui.checks import (
    PROCESS_UNCERTAIN_LABEL,
    CheckProjection,
    bound_status_lines,
)
from cicada.ui.editor import Editor
from cicada.ui import theme
from cicada.ui.theme import FooterComponent
from cicada.ui.tool_view import ToolCardEntry, ToolDocument
from cicada.ui.transcript import Transcript

HELP_TEXT = "\n".join(
    (
        "可用命令:",
        "/help 显示本帮助",
        "/status 显示状态与计量",
        "/clear 清空模型历史（已显示记录保留）",
        "/exit 退出程序",
        "//内容 发送以斜杠开头的普通文本（转义）",
    )
)

KNOWN_COMMANDS = ("/help", "/status", "/clear", "/exit")


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
    ) -> None:
        self._session = session
        self._editor = editor if editor is not None else Editor(border_color=theme.editor_border())
        self._on_exit = on_exit
        self._on_update = on_update
        self._diagnostics = diagnostics
        self.transcript = Transcript()
        self._gate = AdmissionGate(session)
        self._gate.on_error = self._on_gate_error
        # 票据 05 组合钩子: 会话进入 closing/closed 且 execute 异常到达时由宿主
        # 决定退出前端 (避免 barrier 内界面失去退出路径); 默认无动作
        self.on_unrecoverable: Callable[[], None] | None = None
        # 票据 05: 工具卡文档片段与检查事实投影
        self.tools = ToolDocument()
        self._checks = CheckProjection(session)
        # AssistantCompleted 不带 turn_index: 按内核 TurnStarted 跟踪当前轮
        self._current_turn: dict[str, int] = {}
        self._unsubscribe = session.subscribe(self._on_application_update)
        # 04 底栏: 工作区/模型来自配置事实, 计量来自显示文档累计
        self.footer = FooterComponent(
            str(session.config.workspace),
            session.config.model_label,
            self.transcript.metrics_fact,
        )

    # --- 生命周期 -----------------------------------------------------------------

    def close(self) -> None:
        """退订应用事件 (组合层不拥有 session 关闭, aclose 归宿主路径)。"""
        self._unsubscribe()

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
        elif isinstance(update, RequestCompleted):
            outcome = update.outcome
            self.transcript.record_request_result(outcome.result.stop_reason, outcome.result.error)
            # 检查结论条目只在 decision 非空 (检查模式) 时追加; 正常完成安静
            summary = self._checks.on_request_completed(outcome)
            if summary is not None:
                self.transcript.append_system("\n".join(summary))
            self._gate.note_request_completed()
        elif isinstance(update, HistoryCleared):
            # /clear 只清模型历史: 显示文档保留, 追加系统条目说明
            self.transcript.append_system("已清空模型历史（已显示记录与验证状态保留）。")
            if self._session.has_process_uncertain():
                self.transcript.append_system(f"{PROCESS_UNCERTAIN_LABEL}：清空历史不解锁。")
        elif isinstance(update, ApplicationClosed):
            self._gate.note_closed()
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
        return True

    def handle_ctrl_c(self) -> None:
        """Ctrl+C: executing/cancelling 公共 cancel (幂等); reservation 窗口记录
        取消意图; finalizing/closing/closed 静默; 空闲清草稿。"""
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
        空闲与 finalizing/closing/closed 均无动作。"""
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
            self.transcript.append_system(HELP_TEXT)
            return True
        if command == "/status":
            self.transcript.append_system(self._status_text())
            return True
        if command == "/clear":
            try:
                self._session.clear_history()  # 非空闲已被屏障拦截; 防御捕获
            except RuntimeError as exc:
                self.transcript.append_error(str(exc))
                return True
            # 确认条目由 HistoryCleared 事件投影追加
            return True
        if command == "/exit":
            self.handle_exit()
            return True
        self.transcript.append_system(f"未知命令: {command}（/help 查看可用命令）")
        return False  # 未知命令提示且草稿保留

    def _status_text(self) -> str:
        """/status: 已发布状态 + 检查事实 + 计量口径 + 注入诊断; 整体有界 (P12)。"""
        lines = [
            f"状态: {self._session.state}",
            f"模型: {self._session.config.model_label or '不可得'}",
            self.transcript.metrics_summary(),
        ]
        lines.extend(self._checks.status_lines())
        if self._diagnostics is not None:
            lines.extend(self._diagnostics())
        return "\n".join(bound_status_lines(lines))


def build(
    session: ApplicationSession,
    *,
    editor: Editor | None = None,
    on_exit: Callable[[], None] | None = None,
    on_update: Callable[[], None] | None = None,
    diagnostics: Callable[[], list[str]] | None = None,
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
    }


__all__ = ["ConversationView", "build"]
