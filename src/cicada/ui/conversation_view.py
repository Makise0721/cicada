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
- 工具组经 ToolDocument 投影 (tui-compact-interaction 票据 01): 同一模型轮次
  聚合为一个紧凑组, 新建组经 add_entry 挂载显示文档; 检查事实经
  CheckProjection 消费 RequestCompleted.outcome; /status 汇集检查事实与注入
  诊断并施加有界限额。
- 票据 05: 订阅前先把恢复快照 (restored_requests/restored_interrupted) 投影
  成只读历史条目 (ui/restore_view), 查看状态原因如实成条; 保存失败显示真实
  未保存事实 (后继编码由应用 storage_failed 锁存拒绝)。
- tui-startup-welcome 01: 订阅前在显示文档开头追加本视图唯一的启动欢迎条目
  (ui/welcome.py, 启动事实一次捕获后冻结); 查看状态原因不依附"有恢复记录"
  条件, 零记录恢复的查看状态同样如实成条。
- 票据 04: /status 新增预算段 (占用/上限/预留/余量/来源/锚/校准偏差) 与摘要段
  (压缩次数/边界/摘要计量单列), 只读出口为 session.budget 与
  session.summary_records; 恢复显示在压缩边界插入程序条目。
- DeepSeek 阶段票据 04 (契约 C7): /model 打开模型菜单 (本地操作, 非 UserMessage);
  菜单确认同步置 model 屏障 (先于任务创建), select_model 只提交 option ID;
  ModelChanged 刷新底栏有效模型并追加确认条目, 失败有界成条, closing 不重开。
  思考正文不进入普通答案 (transcript 只渲染 message.text 为答案)。
- thinking-budget-visibility 票据 01: 完成／恢复 message.reasoning 经
  ui/reasoning_view 进入答案前的独立思考块 (缺失省略, 默认收起); 04 用户
  裁决移除 Ctrl+T 切换, 点击/文档焦点为唯一展开入口。
- thinking-budget-visibility 票据 02: AssistantReasoningDelta 进有界思考
  预览 (transcript.stream_reasoning_delta); 状态行按真实事件区分等待模型/
  思考/输出, 屏障、取消、收尾优先; RunFinished/RequestCompleted/闸门错误/
  ApplicationClosed 幂等封存活动运行未完成预览并清阶段, 迟到事件不复活。
- thinking-budget-visibility 票据 03: 底栏经 FooterComponent budget 回调
  常驻"最近预算"（总需求/上限/近似标记/输入来源；待评估/未启用/超限如实
  降级）；/help 增加一句口径说明。只读 session.budget, 不新增刷新任务。
- tui-compact-interaction 票据 01 (契约 C1-C3): layout_document 是同源布局
  出口 (lines/hits/anchors 同一次排版), document_lines 只是其兼容包装;
  execute_display_action 执行组/工具/思考/Skill 的 toggle 与工具分页动作
  (F6 键盘与 02 鼠标共用入口), Ctrl+O 统一工具组与 Skill 开合, 逐项动作
  只覆盖当前条目。
- tui-compact-interaction 票据 03 (契约 C5/C6): Shift+Tab 与点击底栏徽标
  共用 handle_shift_tab 同步准入入口——屏障内静默, 阻断态短路, 目标只由
  UI 计算 (关闭→第一档/循环/同值 no-op), 受理同步置 model 屏障后建任务;
  ReasoningEffortChanged 只清 pending (04 用户裁决: 切换不发确认条目、
  期间无状态行, 底栏徽标即生效值), 失败有界成条; 底栏徽标事实经
  FooterComponent effort 回调读取 session.reasoning_effort, 事务目标只作
  pending 展示, 不伪造已生效。
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Callable

from cicada.application import (
    AgentEventReceived,
    ApplicationClosed,
    ApplicationSession,
    ApplicationUpdate,
    ConfigurationChanged,
    ContextCompacted,
    HistoryCleared,
    ModelChanged,
    LoginCompleted,
    OperationBusy,
    ReasoningEffortChanged,
    RequestCompleted,
    RequestPhaseChanged,
    SessionManagementError,
    SessionManagementSubmission,
    SessionNameChanged,
    SessionChanged,
)
from cicada.budget import BudgetReport, BudgetSnapshot
from cicada.persistence.contracts import validate_session_name
from cicada.skills import MAX_SKILL_LIST_BYTES, MAX_SKILL_LIST_LINES, SkillsError
from cicada.core.events import (
    AssistantCompleted,
    AssistantReasoningDelta,
    AssistantTextDelta,
    Event,
    ModelCallCompleted,
    RunFinished,
    RunStarted,
    ToolStarted,
    TurnStarted,
)
from cicada.ui.review_view import ACTION_REVIEW, ReviewView
from cicada.ui.rewind_view import ACTION_REWIND, RewindView
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
from cicada.ui.model_menu import ModelMenu
from cicada.ui.login_wizard import LoginModal
from cicada.user_config import LoginSubmission, UserConfigError
from cicada.ui.restore_view import project_restored_history
from cicada.ui.session_resume import SessionResumeModal
from cicada.ui.theme import FooterComponent
from cicada.ui.document_layout import (
    ACTION_EFFORT_CYCLE,
    ACTION_MORE,
    ACTION_RESET,
    ACTION_TOGGLE,
    TARGET_EFFORT,
    TARGET_GROUP,
    TARGET_REASONING,
    TARGET_SKILL,
    TARGET_TOOL,
    DisplayAction,
    DocumentLayout,
)
from cicada.ui.tool_view import ToolDocument, ToolGroupEntry
from cicada.ui.transcript import CacheMetricsTotal, Transcript
from cicada.ui.welcome import build_welcome_entry
from cicada.ui.display_style import DisplayStyle

# 状态行 (规格 §F): braille spinner 与节奏; 帧由可注入时钟决定 (测试可冻结)
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SPINNER_INTERVAL_S = 0.1

# 03 (契约 C7): 分词观察行的固定前缀; 状态页超限时按此后置优先级省略整行。
TOKEN_OBSERVATION_PREFIXES = ("本地计数:", "同请求偏差:")


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
        self.display_style = DisplayStyle.capture()
        self._editor = editor if editor is not None else Editor(
            border_color=theme.editor_border(),
            padding_x=2,
            style=EditorStyle(
                prompt_glyph="›",
                prompt_color=theme.EDITOR_BORDER,
                placeholder="输入消息… Enter 发送, Shift+Enter 换行",
                placeholder_color=theme.EDITOR_PLACEHOLDER,
            ),
        )
        self._on_exit = on_exit
        self._on_update = on_update
        self._diagnostics = diagnostics
        # 票据 05: 工具卡文档片段与检查事实投影 (先于显示文档: 展开标志注入)
        self.tools = ToolDocument()
        # local-skills-v1 票据 03: Skill 正文展开与工具输出共用全局展开标志
        self.transcript = Transcript(expand_getter=lambda: self.tools.expanded)
        self._gate = AdmissionGate(session)
        self._gate.on_error = self._on_gate_error
        self.review_view = ReviewView(session, self._editor, self._review_updated, self.transcript.append_error)
        self.rewind_view = RewindView(session, self._editor, self._review_updated, self.transcript.append_error)
        # 02: 压缩等待任务的强引用 (PA05: 迟到收尾任务不被垃圾回收, 保留诊断面)
        self._compact_task: asyncio.Task | None = None
        self._name_submission: SessionManagementSubmission | None = None
        self._name_task: asyncio.Task | None = None
        self._new_submission: SessionManagementSubmission | None = None
        self._new_task: asyncio.Task | None = None
        self.on_session_prepare: Callable[[], None] | None = None
        self.on_session_changed: Callable[[], None] | None = None
        self._view_closed = False
        self.session_modal = SessionResumeModal(session, on_update=self._session_modal_updated,
                                                on_error=self.transcript.append_error)
        # DeepSeek 04: 模型选择等待任务的强引用 (同 PA05 理由); 菜单组件归本层拥有
        self._model_task: asyncio.Task | None = None
        # 票据 03: 思考强度等待任务的强引用与事务目标 (底栏 pending 展示事实)
        self._effort_task: asyncio.Task | None = None
        self._effort_pending: str | None = None
        self.model_menu = ModelMenu()
        self.login_modal = LoginModal(self._cancel_login, self._submit_login,
                                      display_style=self.display_style)
        self._login_task: asyncio.Task | None = None
        # 状态行 (规格 §F): spinner 帧时钟与动画任务; 忙时驱动 on_update, 转闲即停
        self._clock = clock if clock is not None else time.monotonic
        self._spinner_task: asyncio.Task | None = None
        # 票据 05 组合钩子: 会话进入 closing/closed 且 execute 异常到达时由宿主
        # 决定退出前端 (避免 barrier 内界面失去退出路径); 默认无动作
        self.on_unrecoverable: Callable[[], None] | None = None
        self._checks = CheckProjection(session)
        # AssistantCompleted 不带 turn_index: 按内核 TurnStarted 跟踪当前轮
        self._current_turn: dict[str, int] = {}
        # 票据 02: 真实活动阶段 (显示事实, 不改生命周期): waiting/thinking/answering;
        # 旧运行迟到事件不覆盖新阶段, 终结/异常关闭幂等清理并封存活动运行预览
        self._active_run_id: str | None = None
        self._model_stage: str | None = None
        # welcome 01: 建立订阅前追加本视图唯一欢迎条目 (启动快照一次捕获后冻结,
        # 渲染/resize/模型切换/登录配置/clear/compact 都不重建; 与首条请求间无 await)
        self.transcript.add_entry(build_welcome_entry(session, display_style=self.display_style))
        # 票据 05: 建立订阅前注入只读恢复显示。
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
        # welcome 01 (规格 D4): 查看状态原因移出"有恢复记录"条件, 零记录恢复同样可见
        if session.view_only_reason is not None:
            self.transcript.append_system(
                f"恢复为查看状态（新请求被拒绝）：{session.view_only_reason}"
            )
        self._unsubscribe = session.subscribe(self._on_application_update)
        # 04 底栏: 工作区/模型来自配置事实, 计量来自显示文档累计;
        # 03 (契约 C5): 预算回调每帧至多读取一次 session.budget, 底栏显示最近预算
        self.footer = FooterComponent(
            str(session.config.workspace),
            session.config.model_label,
            self.transcript.metrics_fact,
            budget=lambda: self._session.budget,
            effort=self._effort_badge_facts,
            display_style=self.display_style,
        )

    # --- 生命周期 -----------------------------------------------------------------

    def _session_modal_updated(self) -> None:
        if self.session_modal.active:
            self._gate.note_session_started()
        else:
            self._gate.note_session_settled()
        if not self._view_closed and self._on_update:
            self._on_update()

    def _review_updated(self) -> None:
        if (self.review_view.active or self.rewind_view.active):
            self._gate.note_review_started()
        else:
            self._gate.note_review_settled()
        if self._on_update:
            self._on_update()

    def close(self) -> None:
        """退订应用事件并停掉状态行动画 (组合层不拥有 session 关闭, aclose 归宿主路径)。"""
        self._view_closed = True
        self.session_modal.cancel()
        self.review_view.cancel()
        self.rewind_view.cancel()
        if self._new_submission is not None:
            self._new_submission.cancel()
        if self._name_submission is not None:
            self._name_submission.cancel()
        self._unsubscribe()
        self.login_modal.dismiss()
        self._session.cancel_login()
        if self._spinner_task is not None:  # 防御: 宿主"关 view 不关 session"时不泄漏
            self._spinner_task.cancel()
            self._spinner_task = None

    @property
    def editor(self) -> Editor:
        """组合层编辑器: Ctrl+C 清草稿与宿主注入共用同一实例。"""
        return self._editor

    # --- 应用事件投影 ----------------------------------------------------------------

    def _on_application_update(self, update: ApplicationUpdate) -> None:
        if isinstance(update, SessionChanged):
            if update.session_generation != self._session.session_generation:
                return
            self.tools = ToolDocument()
            self.transcript.reset_session()
            self._checks = CheckProjection(self._session)
            self._current_turn.clear()
            self._active_run_id = self._model_stage = None
            self.model_menu.dismiss()
            self._editor.clear()
            self.transcript.append_system("已切换会话，验证状态保留。" if update.persistence.mode == "resume"
                                          else "已新建会话，验证状态保留。")
            project_restored_history(
                self.transcript, self.tools, requests=update.snapshot.requests,
                interrupted=update.snapshot.interrupted,
                summaries=update.snapshot.summaries if self._session.budget is not None else (),
                namespace=f"restored:{update.session_generation}")
            if self._session.view_only_reason:
                self.transcript.append_system("恢复为查看状态（新请求被拒绝）：" + self._session.view_only_reason)
            if self.on_session_changed is not None:
                self.on_session_changed()
        elif isinstance(update, AgentEventReceived):
            self._project_agent_event(update.event)
        elif isinstance(update, RequestPhaseChanged):
            self._checks.note_phase(update.run_id, update.phase)
            self._gate.note_phase(update.phase)
        elif isinstance(update, RequestCompleted):
            outcome = update.outcome
            # 票据 02: 请求级终结封存该运行预览 (RunFinished 已封存则幂等);
            # 防御 RunFinished 缺席时仍清活动状态
            self.transcript.finish_reasoning_previews(outcome.result.run_id)
            # 紧凑交互 01 (C1): 请求终结封存无结果工具子项为未确认 (幂等)
            self.tools.seal_run(outcome.result.run_id)
            if outcome.result.run_id == self._active_run_id:
                self._model_stage = None
                self._active_run_id = None
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
            elif update.operation == "clear":
                self._gate.note_clear_started()
                self.transcript.append_system("正在提交清空（显示记录与验证状态保留）…")
            elif update.operation == 'login':
                self._gate.note_login_started()
            elif update.operation == 'session':
                self._gate.note_session_started()
            else:
                # model: 屏障在菜单确认时已同步置位, 此处兜底; 进度由状态行表达
                self._gate.note_model_started()
        elif isinstance(update, SessionNameChanged):
            if update.session_generation == self._session.session_generation:
                self.transcript.append_system(
                    "已保存会话名称：" + update.user_name if update.user_name is not None
                    else "已清除会话名称。"
                )
        elif isinstance(update, ModelChanged):
            # DeepSeek 04 (C7): 底栏从有效配置读取 (启动诊断仍是启动事实);
            # 确认条目如实记录边界, 历史/摘要/验证状态由应用保留
            self.footer.set_model_label(update.model_label)
            self.transcript.append_system(
                f"已切换模型：{update.previous_label} → {update.model_label}"
                "（仅影响后继请求；对话、摘要与验证状态保留）。"
            )
        elif isinstance(update, LoginCompleted):
            if update.result is None or self._login_task is None or self._login_task.done():
                self._settle_login()
            result = update.result
            if result is not None:
                if result.defaults_saved:
                    outcome = result.application_outcome
                    if outcome == 'applied':
                        self.transcript.append_system('用户默认值已保存；当前会话已应用新的登录配置。')
                    elif outcome == 'unchanged':
                        self.transcript.append_system('用户默认值已保存；当前会话配置与登录设置一致，未变更。')
                    elif outcome == 'not-requested':
                        self.transcript.append_system('用户默认值已保存（下次新会话生效）；当前会话配置保持。')
                    else:
                        self.transcript.append_error(
                            '用户默认值已保存（下次新会话生效）；当前会话未应用：'
                            + (result.error_code or outcome))
                elif result.error_code is not None:
                    self.transcript.append_error('登录配置未完成：' + result.error_code)
        elif isinstance(update, ConfigurationChanged):
            # 登录 04 (C7): 底栏从有效配置读取; 事件只含安全身份与 epoch
            self.footer.set_model_label(update.identity.model)
            effort = (dict(update.identity.options).get('reasoning_effort', '-')
                      if update.identity.think else 'off')
            self.transcript.append_system(
                f"已应用登录配置：{update.identity.model} · 思考={effort}"
                "（仅影响后继请求；对话、摘要与验证状态保留）。"
            )
        elif isinstance(update, ReasoningEffortChanged):
            # 票据 03 (C5/D9): 只表达强度变化, 不解释为登录或模型名称变化;
            # 04 用户裁决: 切换成功不发确认条目, 底栏徽标即生效值, 仅清 pending
            self._effort_pending = None
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
            if self._session.has_process_uncertain():
                self.transcript.append_system(f"{PROCESS_UNCERTAIN_LABEL}：清空历史不解锁。")
        elif isinstance(update, ApplicationClosed):
            self._settle_login()
            self._gate.note_closed()
            self._seal_active_run()  # 票据 02: 关闭兜底封存, 阻止迟到复活
        self._sync_spinner()  # 状态行跟随最近一次应用状态 (规格 §F)
        if self._on_update is not None:
            self._on_update()

    def _seal_active_run(self) -> None:
        """票据 02/A07 异常关闭兜底: 封存当前运行未完成预览并清活动阶段。

        幂等; 无活动运行时只清状态, 不新建条目、不制造规范记录。紧凑交互 01
        (C1): 同运行无结果工具子项一并封存为未确认, 不伪造完成。
        """
        if self._active_run_id is not None:
            self.transcript.finish_reasoning_previews(self._active_run_id)
            self.tools.seal_run(self._active_run_id)
        self._active_run_id = None
        self._model_stage = None

    def _on_gate_error(self, message: str) -> None:
        """execute 异常如实成条; 会话已进入 closing/closed 时通知宿主退出前端。"""
        self.transcript.append_error(message)
        self._seal_active_run()  # 票据 02: 终结事件缺席路径的幂等封存
        if (
            self._session.state in ("closing", "closed")
            and self.on_unrecoverable is not None
        ):
            self.on_unrecoverable()
        if self._on_update is not None:
            self._on_update()  # 错误条目不经应用事件, 必须显式触发渲染

    def _project_agent_event(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            # 票据 02: 新运行建立活动身份; 阶段清空 (TurnStarted 前为"正在执行")
            self._active_run_id = event.run_id
            self._model_stage = None
        elif isinstance(event, TurnStarted):
            if event.run_id == self._active_run_id:
                self._model_stage = "waiting"
            self._current_turn[event.run_id] = event.turn_index
        elif isinstance(event, AssistantReasoningDelta):
            if (
                event.run_id == self._active_run_id
                and event.turn_index == self._current_turn.get(event.run_id)
                and self._model_stage is not None
            ):
                self._model_stage = "thinking"  # 空增量同样表达"正在思考"
            self.transcript.stream_reasoning_delta(event.run_id, event.turn_index, event.text)
        elif isinstance(event, AssistantTextDelta):
            if (
                event.run_id == self._active_run_id
                and event.turn_index == self._current_turn.get(event.run_id)
                and self._model_stage is not None
            ):
                self._model_stage = "answering"  # 内核只发布非空答案增量
            self.transcript.stream_assistant_delta(event.run_id, event.turn_index, event.text)
        elif isinstance(event, AssistantCompleted):
            if event.run_id == self._active_run_id:
                self._model_stage = None  # 完成模型记录清除模型阶段
            turn_index = self._current_turn.get(event.run_id, 0)
            self.transcript.complete_assistant(event.run_id, turn_index, event.message)
        elif isinstance(event, ToolStarted):
            if event.run_id == self._active_run_id:
                self._model_stage = None  # 工具执行期间回到"正在执行"
        elif isinstance(event, RunFinished):
            # 内核运行结束: 清阶段并封存该运行未完成预览 (幂等; 防御缺完成记录)
            self.transcript.finish_reasoning_previews(event.run_id)
            if event.run_id == self._active_run_id:
                self._model_stage = None
                self._active_run_id = None
        elif isinstance(event, ModelCallCompleted):
            self.transcript.add_metrics(event.metrics, call_key=(event.run_id, event.turn_index))
        # 工具组与检查回执收集: 全部内核事件过一遍 (各自过滤)
        self._checks.observe_event(event)
        # 紧凑交互 01 (C1): 组只在创建时挂载一个显示条目, 子项更新原位重排
        before = len(self.tools.groups)
        self.tools.apply(event)
        groups = self.tools.groups
        for index in range(before, len(groups)):
            self.transcript.add_entry(ToolGroupEntry(self.tools, index))

    # --- MainScreenFrontend 注入面 ----------------------------------------------------

    def document_lines(self, width: int) -> list[str]:
        """01 冻结的显示文档替换点: 同源布局 lines 的兼容包装 (C3)。"""
        with self.display_style.use():
            return self.transcript.render(width)

    def layout_document(
        self, width: int, selected: DisplayAction | None = None
    ) -> DocumentLayout:
        """同源文档布局 (紧凑交互 01, C3): 整份文档一次排版的 lines/hits/anchors。

        selected 是文档焦点当前选中的显示动作 (标题/分页标记着色与命中同帧)。
        """
        with self.display_style.use():
            return self.transcript.layout(width, selected)

    def execute_display_action(self, action: DisplayAction) -> bool:
        """执行显示动作 (紧凑交互 01: F6 Enter 与 02 鼠标点击共用入口)。

        只改展示状态 (组/工具/思考/Skill toggle、工具分页), 不触请求、工具或
        草稿; 未知/失效目标静默返回 False。jump_latest 由视口内部消费, 不到这里。
        """
        if action.kind == ACTION_REWIND:
            return self.rewind_view.execute(action)
        if action.kind == ACTION_REVIEW:
            return self.review_view.execute(action)
        if (self.review_view.active or self.rewind_view.active):
            return False
        kind = action.kind
        target = action.target_id
        handled = False
        if kind == ACTION_TOGGLE and target[:1] == (TARGET_GROUP,) and len(target) == 4:
            self.tools.toggle_group(target[1:4])
            handled = True
        elif kind == ACTION_TOGGLE and target[:1] == (TARGET_TOOL,) and len(target) == 5:
            self.tools.toggle_tool(target[1:5])
            handled = True
        elif kind == ACTION_MORE and target[:1] == (TARGET_TOOL,) and len(target) == 5:
            self.tools.tool_output_more(target[1:5])
            handled = True
        elif kind == ACTION_RESET and target[:1] == (TARGET_TOOL,) and len(target) == 5:
            self.tools.tool_output_reset(target[1:5])
            handled = True
        elif kind == ACTION_TOGGLE and target[:1] == (TARGET_REASONING,) and len(target) == 3:
            handled = self.transcript.toggle_reasoning_item(target[1], target[2])
        elif kind == ACTION_TOGGLE and target[:1] == (TARGET_SKILL,) and len(target) == 2:
            handled = self.transcript.toggle_skill(target[1])
        elif kind == ACTION_EFFORT_CYCLE and target == (TARGET_EFFORT,):
            self.handle_shift_tab()  # 徽标点击与 Shift+Tab 共用同一同步准入入口 (C6)
            handled = True
        if handled and self._on_update is not None:
            self._on_update()
        return handled

    # --- 状态行 (tui-aesthetics 规格 §F) ---------------------------------------------

    def status_lines(self, width: int) -> list[str]:
        with self.display_style.use():
            return self._render_status_lines(width)

    def _render_status_lines(self, width: int) -> list[str]:
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
            + self.display_style.sgr("0")
        ]

    def _status_text(self) -> str | None:
        """状态文案只取真实状态: 屏障/取消/执行/reservation 窗口, 空闲 None。"""
        if self._gate.login_active:
            wizard = self.login_modal.wizard
            if wizard and wizard.submitting:
                return ('正在保存并应用登录配置…' if wizard.target == 'save-and-apply'
                        else '正在保存登录配置…')
            return None
        if self._gate.clearing:
            return "正在清空…"
        if self._gate.session_managing:
            return "正在新建会话…" if self._new_submission is not None else "正在保存名称…"
        if self._gate.compacting:
            return "正在压缩…"
        if self._effort_pending is not None and self._gate.model_selecting:
            return None  # 04 用户裁决: 切换期间不显示状态行 (出现/消失引起版面跳动), 反馈由底栏徽标承担
        if self._gate.model_selecting:
            return "正在切换模型…"
        if self._session.state == "cancelling":
            return "正在取消…"
        if self._session.state == "finalizing":
            return "正在收尾…"  # 模型已停笔, 持久化/检查收尾中 (审查 06-3)
        if self._session.state == "executing" or self._gate.busy():
            # 票据 02 (D4): 模型阶段以真实事件为准, 无增量时只显示执行事实
            if self._model_stage == "thinking":
                return "正在思考…"
            if self._model_stage == "answering":
                return "正在输出…"
            if self._model_stage == "waiting":
                return "等待模型…"
            return "正在执行…"
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
        # 新请求预留清旧活动阶段 (票据 02 C4); 应用准入事实已取得 (reservation
        # 已占, 任务已建): 此时才创建用户条目
        self._active_run_id = None
        self._model_stage = None
        self.transcript.append_user(prompt)
        self._sync_spinner()  # reservation 窗口即忙: 状态行与动画立即上线
        return True

    def handle_ctrl_c(self) -> None:
        """Ctrl+C: executing/cancelling 公共 cancel (幂等); reservation 窗口记录
        取消意图; 持久清空/压缩/模型选择提交中/finalizing/closing/closed 静默; 空闲清草稿。"""
        if self.session_modal.active:
            self.session_modal.cancel()
            return
        if self._gate.session_managing:
            if self._new_submission is not None:
                self._new_submission.cancel()
            if self._name_submission is not None:
                self._name_submission.cancel()
            return
        if (self.review_view.active or self.rewind_view.active):
            self.review_view.cancel()
            self.rewind_view.cancel()
            return
        if self._gate.clearing or self._gate.compacting or self._gate.model_selecting or self._gate.login_active:
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
        持久清空/压缩/模型选择提交中与空闲均无动作; finalizing/closing/closed 静默。"""
        if self.session_modal.active:
            self.session_modal.cancel()
            return
        if self._gate.session_managing:
            if self._new_submission is not None:
                self._new_submission.cancel()
            if self._name_submission is not None:
                self._name_submission.cancel()
            return
        if (self.review_view.active or self.rewind_view.active):
            self.review_view.cancel()
            self.rewind_view.cancel()
            return
        if self._gate.clearing or self._gate.compacting or self._gate.model_selecting or self._gate.login_active:
            return
        if cancel_if_active(self._session):
            return
        if self._gate.busy() and self._session.state == "idle":
            self._gate.arm_pending_cancel()

    def handle_ctrl_o(self) -> None:
        """Ctrl+O (紧凑交互 01, C2): 统一所有工具组/单工具/Skill 为同一开合状态
        并作为后续默认值; 只改展示, 不触请求或工具。"""
        if (self.review_view.active or self.rewind_view.active):
            return
        expanded = self.tools.toggle_expand()
        self.transcript.set_skills_expanded(expanded)

    def handle_shift_tab(self) -> None:
        """Shift+Tab / 点击底栏徽标 (票据 03, C6): 同步准入后循环思考强度.

        忙碌/收尾/模态/菜单/屏障内静默 (is_enter_barrier, 不排队); view-only/
        storage_failed/process_uncertain 同步短路并短提示, 不建任务; 不支持的
        模型一次短提示且原值不动。目标只由本层计算 (关闭→第一档, 已启用循环,
        同值 no-op); 受理前同步置 model 屏障 (先于任务创建), 连续输入不并发;
        准备/提交/再次校验归应用事务, 失败保留旧值。
        """
        if is_enter_barrier(self._session, self._gate):
            return
        persistence = self._session.persistence
        if (
            self._session.view_only_reason is not None
            or self._session.has_process_uncertain()
            or (persistence is not None and persistence.storage_failed is not None)
        ):
            self.transcript.append_system("会话状态不允许切换思考强度。")
        else:
            state = self._session.reasoning_effort
            target = self._next_effort_target(state) if state.enabled else None
            if state.enabled and target is not None and target != state.current:
                self._gate.note_model_started()
                self._effort_pending = target
                self._sync_spinner()
                self._effort_task = asyncio.get_running_loop().create_task(
                    self._run_effort_cycle(target)
                )
            elif not state.enabled or target is None:
                # 关闭或 enabled 但 levels 为空 (第三方 provider 边界): 一次短提示, 原值不动
                self.transcript.append_system("当前模型不支持思考强度切换。")
        if self._on_update is not None:
            self._on_update()

    @staticmethod
    def _next_effort_target(state) -> str | None:
        """UI 仅计算目标 (C5): 关闭→第一档; 已启用按 levels 顺序循环。"""
        if not state.levels:
            return None
        if state.current is None or state.current not in state.levels:
            return state.levels[0]
        return state.levels[(state.levels.index(state.current) + 1) % len(state.levels)]

    def _effort_badge_facts(self) -> tuple[str | None, str | None] | None:
        """底栏思考强度徽标事实 (C6): (生效档位|None=关闭, 事务目标); 不支持 None.

        pending 只是展示事实, 底栏文案仍以生效档位开头, 不伪造已生效。
        """
        state = self._session.reasoning_effort
        if not state.enabled:
            return None
        return (state.current, self._effort_pending)

    async def _run_effort_cycle(self, target: str) -> None:
        """等待思考强度事务落定 (03); 成功经 ReasoningEffortChanged 清 pending.

        落定 (成功/失败/取消) 都释放屏障并清 pending; 失败显示有界原因
        (ModelSelectionError 不含凭据/请求历史), 取消如实传播。
        """
        try:
            await self._session.set_reasoning_effort(target)
        except Exception as exc:
            self._gate.note_model_settled()
            self._effort_pending = None
            self.transcript.append_error(f"思考强度切换失败: {exc}")
            if self._on_update is not None:
                self._on_update()
            return
        except BaseException:
            self._gate.note_model_settled()
            self._effort_pending = None
            raise
        self._gate.note_model_settled()
        self._effort_pending = None

    def handle_exit(self) -> None:
        """退出回调 (Ctrl+D 与 /exit 共用): 仅 idle 且无 reservation 时执行。"""
        if is_enter_barrier(self._session, self._gate):
            return
        if self._on_exit is not None:
            self._on_exit()

    def enter_barrier(self) -> bool:
        """busy/finalizing Enter 屏障事实 (C7): 文件菜单确认让位于原静默屏障。"""
        return is_enter_barrier(self._session, self._gate)

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
                ("快捷键", [
                    "点击工具/思考/Skill 标题逐层展开；滚轮回看，点「最新」返回",
                    "Ctrl+O 工具/Skill 文档展开/收起",
                    "Shift+Tab 或点击底栏思考标记：循环 low/high/max（关闭时先启用 low）",
                    "F6 文档回看：Up/Down 选择 · Enter 展开/分页 · Esc 返回"
                    "（PageUp/PageDown/Home/End 翻看）",
                ]),
                ("预算", [
                    "底栏「最近预算」为最近一次评估的总需求（输入＋输出预留＋安全余量，"
                    "~ 为近似）；估/锚/测/未 表示输入来源；完整分解与余量见 /status。",
                ]),
            ])
            return True
        if command == "/status":
            self.transcript.append_information("会话信息", self._status_sections())
            return True
        if text.startswith("/name") and (len(text) == 5 or text[5].isspace()):
            return self._handle_name_command(text)
        if command == "/resume":
            if self.model_menu.is_open or self.login_modal.is_open or (self.review_view.active or self.rewind_view.active):
                return False
            if self.on_session_prepare:
                self.on_session_prepare()
            self.session_modal.open()
            return False  # 模态使用独立搜索输入，取消完整保留草稿/光标/undo。
        if command == "/new":
            self._gate.note_session_started()
            try:
                if self.on_session_prepare is not None:
                    self.on_session_prepare()
                submission = self._session.submit_new_session()
            except SessionManagementError as exc:
                self._gate.note_session_rejected()
                self.transcript.append_error(str(exc))
                return False
            self._new_submission = submission
            self._new_task = asyncio.create_task(self._run_new_session(submission))
            return False  # 发布成功事件才清草稿，准备失败/取消保留编辑器全部状态。
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
        if command.startswith("/rewind") and (len(command) == 7 or command[7].isspace()):
            if command != "/rewind":
                self.transcript.append_system("用法：/rewind（不接受参数）")
                return True
            if self.model_menu.is_open or self.login_modal.is_open:
                return False
            return self.rewind_view.open()
        if command.startswith("/diff") and (len(command) == 5 or command[5].isspace()):
            if command != "/diff":
                self.transcript.append_system("用法：/diff（不接受参数）")
                return True
            if self.model_menu.is_open or self.login_modal.is_open:
                return False
            return self.review_view.open()
        if command == "/model":
            return self._open_model_menu()
        if command.startswith('/login') and (len(command) == 6 or command[6].isspace()):
            if command != '/login':
                self.transcript.append_system('用法：/login（不接受参数）')
                return True
            return self._open_login()
        if command.startswith("/skill") and (len(command) == 6 or command[6].isspace()):
            return self._handle_skill_command(text.lstrip())
        self.transcript.append_system(f"未知命令: {command}（/help 查看可用命令）")
        return False  # 未知命令提示且草稿保留

    def _handle_name_command(self, text: str) -> bool:
        if text == "/name":
            state = self._session.persistence
            title = ((state.user_name or self._session.session_short_title or "未命名")
                     if state is not None else "当前会话未启用保存")
            self.transcript.append_system(title + "\n/name <名称> · /name --clear")
            return True
        # 保留完整参数，校验控制字符后再 trim；只精确保留 --clear。
        try:
            name = validate_session_name(text[5:], trim=True)
            self._gate.note_session_started()
            submission = self._session.submit_session_name(None if name == "--clear" else name)
        except (ValueError, SessionManagementError) as exc:
            self._gate.note_session_rejected()
            self.transcript.append_error(str(exc))
            return True
        self._name_submission = submission
        self._name_task = asyncio.create_task(self._run_session_name(submission))
        self._sync_spinner()
        return True

    async def _run_session_name(self, submission: SessionManagementSubmission) -> None:
        waiter = asyncio.create_task(submission.wait())
        try:
            while True:
                try:
                    await asyncio.shield(waiter)
                    break
                except asyncio.CancelledError:
                    # 重复取消 UI 等待者也不能提前释放；事务自身取消则已经落定。
                    if waiter.done():
                        break
                except Exception as exc:
                    self.transcript.append_error(str(exc))
                    break
        finally:
            self._name_submission = None
            self._gate.note_session_settled()
            self._sync_spinner()
            if self._on_update is not None:
                self._on_update()

    async def _run_new_session(self, submission: SessionManagementSubmission) -> None:
        waiter = asyncio.create_task(submission.wait())
        try:
            while True:
                try:
                    await asyncio.shield(waiter)
                    break
                except asyncio.CancelledError:
                    submission.cancel()
                    if waiter.done():
                        break
                except Exception as exc:
                    if not self._view_closed:
                        self.transcript.append_error(str(exc))
                    break
        finally:
            self._new_submission = None
            self._gate.note_session_settled()
            if not self._view_closed and self._on_update is not None:
                self._on_update()

    def _open_login(self) -> bool:
        self._gate.note_login_started()  # 同步屏障先于读取/任务创建。
        self.model_menu.dismiss()
        try:
            view = self._session.login_view()
            self.login_modal.open(view)
        except UserConfigError as exc:
            self._gate.note_login_settled()
            self.transcript.append_error('登录配置不可用：' + exc.error_code)
        except Exception:
            self._gate.note_login_settled()
            self._session.cancel_login()
            self.transcript.append_error('登录配置不可用：read-failed')
        if self._on_update:
            self._on_update()
        return True

    def _settle_login(self) -> None:
        self.login_modal.dismiss()
        self._gate.note_login_settled()

    def _cancel_login(self) -> None:
        self._session.cancel_login()
        self._settle_login()
        self._sync_spinner()
        if self._on_update:
            self._on_update()

    def _submit_login(self, submission: LoginSubmission) -> None:
        self._login_task = asyncio.create_task(self._run_login(submission))
        self._sync_spinner()

    async def _run_login(self, submission: LoginSubmission) -> None:
        try:
            await self._session.configure_login(submission)
        except UserConfigError as exc:
            self.transcript.append_error('登录配置未完成：' + exc.error_code)
        except Exception:
            self.transcript.append_error('登录配置未完成：write-failed')
        finally:
            self._session.cancel_login()
            self._settle_login()
            self._sync_spinner()
            if self._on_update:
                self._on_update()

    def _open_model_menu(self) -> bool:
        """/model (DeepSeek 04, 契约 C7/D8): 打开模型目录的本地操作.

        不是 UserMessage/模型请求/工具调用; 目录只读来自 session.model_options。
        当前会话无选择能力 (fake) 时给有界说明, 不触碰历史/存储。
        view-only/storage_failed/process_uncertain 下仍可查看目录; 选择能否
        提交由应用 select_model 双向核对 (失败有界成条, 不解除编码阻断)。
        """
        options = self._session.model_options
        if not options:
            self.transcript.append_system("当前会话不提供模型选择（无可用模型目录）。")
            return True
        self.model_menu.open(options, current_id=self._session.config.model_label)
        if self._on_update is not None:
            self._on_update()
        return True

    def confirm_model_selection(self, option_id: str) -> None:
        """模型菜单 Enter 确认: UI 同步置 model 屏障先于 create_task (C7).

        只提交 option ID 给应用 select_model; 不触碰 raw 模型/历史/存储, 不产生
        UserMessage/intent/模型轮。忙碌/收尾/关闭屏障内静默 (菜单已被前端收起);
        同选 no-op (应用层同选亦不写); closing 永不重开由闸门状态检查保证。
        """
        self.model_menu.dismiss()  # Enter 确认即收起 (幂等; 前端已收起时无动作)
        if is_enter_barrier(self._session, self._gate):
            return  # 忙碌/finalizing/closing/closed: Enter 静默无动作
        if option_id == self._session.config.model_label:
            return  # 同选 no-op: 不置屏障、不建任务、不发事件
        self._gate.note_model_started()
        self._sync_spinner()
        self._model_task = asyncio.get_running_loop().create_task(
            self._run_model_select(option_id)
        )

    async def _run_model_select(self, option_id: str) -> None:
        """等待模型选择落定; 成功确认由 ModelChanged 投影追加 (底栏同时刷新).

        落定 (成功/失败/取消) 都释放屏障 (finally 语义同清空/压缩路径);
        失败显示有界原因 (ModelSelectionError 不含凭据/请求历史), 取消如实传播。
        """
        try:
            await self._session.select_model(option_id)
        except Exception as exc:
            self._gate.note_model_settled()
            self.transcript.append_error(f"模型选择失败: {exc}")
            if self._on_update is not None:
                self._on_update()
            return
        except BaseException:
            self._gate.note_model_settled()
            raise
        self._gate.note_model_settled()

    def _handle_skill_command(self, command: str) -> bool:
        """/skill (local-skills-v1 票据 03, spec D5/D8).

        裸命令显示有界目录与用法; 带名称经 session 纯准备封装为一次普通请求,
        再走原 _admit_coding 准入 (屏障由 _handle_command 与闸门双重检查)。
        名称后第一个空白串是分隔符, 任务余文原样保留 (不解释 shell 引号);
        未启用/未知名称/超限显示原因并保留草稿 (返回 False, 无伪用户条目)。
        """
        payload = command[len("/skill"):].lstrip()
        if not payload:
            if self._session.skills is None:
                self.transcript.append_system(
                    "未启用 Skills：启动时用 --skills-dir 指定目录后可用。"
                )
                return False  # 无目录是拒绝: 显示原因并保留草稿 (A07)
            self.transcript.append_information("Skills 目录", self._skill_catalog_sections())
            return True
        match = re.match(r"(\S+)(?:\s+(.*))?$", payload, re.DOTALL)
        name = match.group(1)
        task = match.group(2) or ""
        try:
            prompt = self._session.prepare_skill_request(name, task)
        except SkillsError as exc:
            self.transcript.append_system(str(exc))
            return False  # 未知名称/超限: 显示原因并保留草稿 (D8)
        return self._admit_coding(prompt)

    def _skill_catalog_sections(self) -> list[tuple[str, list[str]]]:
        """裸 /skill 的有界目录 (D8): 全部条目 + 用法; 裁剪带省略计数。

        manual-only 条目照常列出 (用户可显式选择), 标注模型不可自行加载。
        """
        skills = self._session.skills
        if skills is None:  # 防御: 调用方已拦截未启用
            return [("Skills", ["未启用 Skills：启动时用 --skills-dir 指定目录后可用。"])]
        rows = [
            f"root: {skills.root}",
            f"manifest sha256:{skills.manifest_sha256[:12]} · 共 {len(skills.infos)} 个 Skill",
        ]
        for info in skills.infos:
            marker = "（手动专用，模型不可自行加载）" if info.disable_model_invocation else ""
            rows.append(f"/skill {info.name}{marker}  {info.description}")
        rows.append("用法: /skill <名称> [任务] — 将该 Skill 正文作为下一条请求发送。")
        fields = [clip_status_field(row) for row in rows]
        total = "\n".join(fields)
        kept, original_lines, original_bytes, truncated = clip_lines(
            total, MAX_SKILL_LIST_LINES, MAX_SKILL_LIST_BYTES
        )
        lines = [*kept, *truncation_note(kept, original_lines, original_bytes, truncated)]
        return [("Skills", lines)]

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
                f"名称: {persistence.user_name or self._session.session_short_title or '未命名'}",
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
            if self._session.session_cleanup_failed is not None:
                session_lines.append(f"会话清理失败已锁存: {self._session.session_cleanup_failed}")
            if self._session.has_process_uncertain():
                session_lines.append(PROCESS_UNCERTAIN_LABEL)
        diagnostics = self._diagnostics() if self._diagnostics is not None else []
        # CLI 诊断重复提供相同模型事实时只展示一次，保留任何不同的诊断值。
        diagnostics = [line for line in diagnostics if line != session_lines[1]]
        sections = [
            ("会话", session_lines),
            *self._budget_sections(),
            ("计量", [
                self.transcript.metrics_summary(),
                self.transcript.cache_summary(),
                self._observation_line(),
                *self._token_observation_lines(),
            ]),
            ("检查", self._checks.status_lines()),
            ("配置", [*diagnostics, *self._session.login_status]),
        ]
        # 保留原有字段顺序及整体限额; 装饰标题/空行不挤占事实行预算。
        clipped_sections = [
            (heading, [clip_status_field(row) for row in rows]) for heading, rows in sections
        ]
        # 新缓存/发送诊断/分词观察行不挤掉原有检查/阻断事实。超限先省略缓存与发送诊断行,
        # 仍超限再省略分词观察行, 最后沿用原裁剪。
        cache_omitted = False
        observation_omitted = False
        token_omitted = False

        def over_limit(sections: list[tuple[str, list[str]]]) -> bool:
            fields = [row for _, rows in sections for row in rows]
            return (
                len(fields) > STATUS_MAX_LINES
                or len("\n".join(fields).encode("utf-8")) > STATUS_MAX_BYTES
            )

        def without(sections: list[tuple[str, list[str]]], predicate) -> list[tuple[str, list[str]]]:
            return [
                (heading, [row for row in rows if not predicate(heading, row)])
                for heading, rows in sections
            ]

        def present(sections: list[tuple[str, list[str]]], predicate) -> bool:
            return any(predicate(heading, row) for heading, rows in sections for row in rows)

        if over_limit(clipped_sections):
            cache_omitted = present(clipped_sections, self._is_cache_row)
            observation_omitted = present(clipped_sections, self._is_observation_row)
            clipped_sections = without(
                clipped_sections,
                lambda heading, row: self._is_cache_row(heading, row)
                or self._is_observation_row(heading, row),
            )
            if over_limit(clipped_sections):
                token_omitted = present(clipped_sections, self._is_token_row)
                clipped_sections = without(clipped_sections, self._is_token_row)
        fields = [row for _, rows in clipped_sections for row in rows]
        total = "\n".join(fields)
        omitted = "/".join(
            name for flag, name in (
                (cache_omitted, "缓存计量"), (observation_omitted, "发送诊断"),
                (token_omitted, "分词观察"),
            ) if flag
        )
        if len(fields) <= STATUS_MAX_LINES and len(total.encode("utf-8")) <= STATUS_MAX_BYTES:
            if omitted:
                clipped_sections.append(("", [f"…[{omitted}因状态页限额省略]"]))
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
            suffix = f"（{omitted}因状态页限额省略）" if omitted else ""
            result.append(("", [note + suffix for note in notes]))
        return result

    def _observation_line(self) -> str:
        """缓存 03 (契约 C7): 最近实际发送相对前次的本地形状变化行。

        无能力/未观察显示不可得; 只显示发送次数、变化与保留前导消息数,
        不显示正文、hash、凭据或回执, 不作为服务端 token 前缀或命中证明。
        """
        observation = self._session.cache_observation
        if observation is None:
            return "发送诊断: 不可得"
        labels = {
            "first_request": "首次可观察",
            "system_changed": "system变化",
            "tools_changed": "工具变化",
            "messages_extended": "历史延长",
            "messages_replaced": "历史替换",
            "request_context_changed": "尾部变化",
            "unchanged": "与上次相同",
        }
        parts = [f"第{observation.sequence}次发送"]
        parts.extend(labels.get(code, code) for code in observation.changes)
        if observation.preserved_messages is not None:
            parts.append(f"保留前导消息 {observation.preserved_messages} 条")
        return f"发送诊断: {' · '.join(parts)}（本地对比，非服务端命中）"

    def _token_observation_lines(self) -> list[str]:
        """03 (契约 C7): 最近正式普通调用的本地计数与同请求偏差观察 (至多两条)。

        只读 session.budget 已发布快照, 不加载资源、不重复计数: 无预算/无报告/未
        配置本地计数时不出行; 本地不可用时只报安全原因; profile 为固定公开参考,
        线上一致性未验证, 短显示身份可回溯完整 manifest。
        """
        snapshot = self._session.budget
        if snapshot is None or snapshot.report is None:
            return []
        local = snapshot.report.local_count
        if local is None:
            return []
        if local.tokens is None:
            return [f"本地计数: 不可用（{local.reason}）"]
        lines = [
            f"本地计数: {local.tokens} tokens · {self._short_profile(local.profile)}"
            "（固定公开参考，线上一致性未验证）"
        ]
        comparison = snapshot.comparison
        if comparison is None:
            return lines
        actual = comparison.input_tokens
        if actual is None:
            lines.append("同请求偏差: 实际输入不可得（本地计数偏高与否未知）")
        else:
            deviation = comparison.local_count.tokens - actual
            lines.append(
                f"同请求偏差: 实际输入 {actual} · 本地 − 实际 = {deviation:+d}（正=本地偏高）"
            )
        return lines

    @staticmethod
    def _short_profile(profile: str) -> str:
        """短显示身份 (03): profile 名 + manifest hash 前 8 位, 可回溯完整 manifest。"""
        name, separator, digest = profile.partition(":manifest-sha256:")
        return f"{name} · manifest {digest[:8]}…" if separator else profile

    @staticmethod
    def _is_cache_row(heading: str, row: str) -> bool:
        """可先省略的缓存观察行: 计量段缓存行与摘要段已提交摘要缓存行。"""
        return (heading == "计量" and row.startswith("缓存（")) or (
            heading == "摘要" and row.startswith("已提交摘要缓存:")
        )

    @staticmethod
    def _is_observation_row(heading: str, row: str) -> bool:
        return heading == "计量" and row.startswith("发送诊断:")

    @staticmethod
    def _is_token_row(heading: str, row: str) -> bool:
        return heading == "计量" and row.startswith(TOKEN_OBSERVATION_PREFIXES)

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
            # 首次发送、成功清空或配置切换后, 均以应用公开快照的失效事实为准。
            lines.append(f"预算: 上限 {snapshot.limit} · 占用估计不可得（当前上下文尚无预算报告）")
        else:
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
                f"实测 {anchor.measured_input_tokens} = {deviation:+d}（正=估计偏高）"
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
            lines.append("已提交摘要缓存: 尚无调用")
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
        cache = CacheMetricsTotal()
        for record in records:
            cache.add(record.metrics)
        lines.append(cache.summary("已提交摘要缓存"))
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
        "layout_document": view.layout_document,  # 紧凑交互 01: 同源布局 (C3)
        "on_display_action": view.execute_display_action,  # 紧凑交互 01: 显示动作执行
        "on_submit": view.on_submit,
        "on_escape": view.handle_escape,
        "on_ctrl_c": view.handle_ctrl_c,
        "on_exit": view.handle_exit,
        "editor": view.editor,
        "view": view,
        "display_style": view.display_style,
        "footer_lines": view.footer.render,  # (width, rows) -> 底栏行
        "status_lines": view.status_lines,  # (width) -> 状态行 (规格 §F), 空闲 []
        "model_menu": view.model_menu,  # DeepSeek 04: /model 目录 (契约 C7)
        "on_model_confirm": view.confirm_model_selection,
        "on_shift_tab": view.handle_shift_tab,  # 票据 03: 思考强度循环 (C6)
        "footer_badge_spans": lambda: view.footer.badge_spans,  # 03: 徽标命中列
        "modal": view.login_modal,
        "review": view.review_view,
        "rewind": view.rewind_view,
        "session_modal": view.session_modal,
    }


__all__ = ["ConversationView", "build"]
