"""共享应用编排: 单次 CLI 与 TUI 的唯一执行 seam (TUI 第一版 02 发布).

职责边界:
- 持有一个启动 App (runtime + Agent) 与检查模式下的一套 Verifier/RunPolicy;
  只在 open 时 initialize 一次, 不为每条用户消息重新 bootstrap.
- 请求生命周期 executing -> cancelling -> finalizing -> idle; 内核 RunFinished
  只把应用置为 finalizing, P4 收尾完成才回 idle; 完成前不接受新请求.
- 历史 H 由应用层维护: 内核只返回本次后缀, 应用把后缀隔离复制后追加一次.
- closing 是不可逆接收门; 迟到完成不能把状态转回 idle; 只有一个 close owner.
- 观察者通知先更新 state/history, 每个观察者收到隔离副本; 通知异常只记录.

不修改插件 runtime、工具算法、检查回执结构或 provider JSON.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from cicada.boot import App, BootError, bootstrap
from cicada.budget import (
    BudgetConfig,
    BudgetSnapshot,
    ComponentEstimate,
    assemble_report,
    budget_exceeded_text,
    estimate_input_components,
)
from cicada.context_policy import (
    SUMMARY_PROMPT_SHA256,
    SUMMARY_PROMPT_VERSION,
    BudgetPolicy,
    ContextProjection,
    ProjectionRelationError,
    SummaryBlock,
    canonical_prefix_sha256,
    generate_summary,
    summary_block_message,
    summary_input_messages,
    summary_prompt_text,
)
from cicada.core.agent import RunResult
from cicada.core.cancel import CancelToken
from cicada.core.events import Event, RunFinished, RunStarted, ToolCompleted
from cicada.core.messages import (
    Message,
    ToolResultMessage,
    UserMessage,
    isolate_message,
    isolate_tool_result,
)
from cicada.core.ports import ModelRequest, ToolSpec
from cicada.models import ModelBinding, ModelOption, ModelRoute, ModelSelectionPort
from cicada.persistence.codec import dumps_strict, encode_messages
from cicada.persistence.contracts import (
    ArchiveLimitError,
    InterruptedRequest,
    LaunchIdentity,
    ModelChangeSubmission,
    PersistenceState,
    PersistenceError,
    RecoverySnapshot,
    RequestOutcomeFacts,
    RequestPersistenceFacts,
    RestoredRequest,
    SessionStorePort,
    StorageUnavailableError,
    StorageWriteError,
    StoredIdentity,
    StoredReceipt,
    SummaryRecord,
    conflicting_launch_fields,
    receipt_terminal_trustworthy,
)
from cicada.process_activity import ActivityCoordinator, ProcessActivityWrapper
from cicada.run_policy import (
    CONTROL_SECTION_MAX_BYTES,
    FRESHNESS_MARKER_MAX_BYTES,
    FRESHNESS_TOOLS,
    HistoricalPrefixProof,
    HistoricalProjection,
    PrefixEntry,
    RunPolicy,
    verification_policy,
)
from cicada.plugins.session_store import SQLiteSessionStore, open_session_store, session_store_plugin
from cicada.plugins.ollama import OllamaModel, classify_context_overflow
from cicada.plugins.coding.inventory import inventory_plugin
from cicada.plugins.coding.process import process_plugin
from cicada.plugins.coding.tool_check import check_plugin
from cicada.plugins.coding.tool_edit import edit_plugin
from cicada.plugins.coding.tool_glob import glob_plugin
from cicada.plugins.coding.tool_grep import grep_plugin
from cicada.plugins.coding.tool_powershell import powershell_plugin
from cicada.plugins.coding.tool_read import read_plugin
from cicada.plugins.coding.tool_write import write_plugin
from cicada.plugins.coding.verification import verification_plugin
from cicada.plugins.coding.verification_contracts import (
    VERIFICATION_CAPABILITY,
    ChangeEvidence,
    VerificationPlan,
    VerificationView,
)
from cicada.plugins.coding.workspace import workspace_plugin
from cicada.plugins.skills import CATALOG_CAPABILITY, skills_plugin
from cicada.prompting import PromptInfo
from cicada.reporting import DeliveryDecision, decide_delivery
from cicada.runtime.plugin import PluginDefinition
from cicada.skills import (
    LOAD_SKILL_TOOL_NAME,
    SKILLS_PROMPT_VERSION,
    SkillCatalogSnapshot,
    SkillCatalogView,
    SkillSelectionError,
    build_skill_request,
    catalog_prompt_section,
    find_skill,
)

ApplicationPhase = Literal["executing", "cancelling", "finalizing", "idle"]
SessionStateName = Literal[
    "idle", "executing", "cancelling", "finalizing", "closing", "closed"
]

TOOL_CAPABILITIES = (
    "tool.read", "tool.edit", "tool.write", "tool.powershell", "tool.glob", "tool.grep",
)
CHECK_TOOL_CAPABILITIES = TOOL_CAPABILITIES + ("tool.check",)

# TUI 第一版拟定的监督窗口默认值 (contracts §4); 冻结值以共享回执为准
REQUEST_FINALIZE_SUPERVISION_S = 15.0
CLOSE_SUPERVISION_S = 15.0
_UNDEAD_GRACE_S = 2.0


def default_definitions(
    workspace: Path, model_plugin: PluginDefinition, plan: VerificationPlan | None = None,
    *, skills: SkillCatalogSnapshot | None = None
) -> list[PluginDefinition]:
    """内置 Coding 能力 + 模型插件 (+ 检查模式验证服务 + Skills 能力) 的启动名单.

    skills 为启用 Skills 的冻结快照: 追加目录能力与 load_skill 工具插件;
    None 时名单与未启用逐位一致。
    """
    definitions: list[PluginDefinition] = [
        workspace_plugin(workspace),
        inventory_plugin(),
        process_plugin(),
        read_plugin(),
        edit_plugin(),
        write_plugin(),
        powershell_plugin(),
        glob_plugin(),
        grep_plugin(),
    ]
    if plan is not None:
        definitions.append(verification_plugin(plan))
        definitions.append(check_plugin())
    if skills is not None:
        if not isinstance(skills, SkillCatalogSnapshot):
            raise ValueError("skills must be a SkillCatalogSnapshot or None")
        definitions.append(skills_plugin(skills))
    definitions.append(model_plugin)
    return definitions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(256 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt_artifacts(view: VerificationView) -> tuple[str, ...]:
    """核对最近回执引用的输出工件与写入时记录的 hash; 缺失/篡改都是阻断原因.

    只核对交付判定实际依赖的最近回执; 被取代的历史回执工件是历史证据, 不阻断。
    """
    problems: list[str] = []
    for state in view.checks:
        receipt = state.receipt
        if receipt is None or receipt.output_artifact_path is None:
            continue
        path = receipt.output_artifact_path
        try:
            observed = _sha256_file(path)
        except OSError as exc:
            problems.append(
                f"check {state.check_id} output artifact is missing or unreadable ({path}): {exc}")
            continue
        if receipt.output_artifact_sha256 is not None and observed != receipt.output_artifact_sha256:
            problems.append(
                f"check {state.check_id} output artifact no longer matches the hash recorded "
                f"when it was written ({path})")
    return tuple(problems)


@dataclass(frozen=True)
class ApplicationConfig:
    """不可变启动配置; CLI 保留参数解析与 fake JSON 转换, open 接收已校验结构."""

    workspace: Path
    system_prompt: str
    prompt_info: PromptInfo
    model_plugin: PluginDefinition
    plan: VerificationPlan | None = None
    max_turns: int = 50
    model_label: str = ""
    persistence: PersistenceConfig | None = None
    # 01 尾随扩展: 显式预算配置才启用预算层 (D9); fake 默认 None, CLI 由 num_ctx 实填
    budget: BudgetConfig | None = None
    # 02 尾随扩展: 启用 Skills 的冻结快照; None = 旧行为逐位保持
    skills: SkillCatalogSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise ValueError("workspace must be a Path")
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a str")
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if self.plan is not None and not isinstance(self.plan, VerificationPlan):
            raise ValueError("plan must be a VerificationPlan or None")
        if self.persistence is not None:
            if not isinstance(self.persistence, PersistenceConfig):
                raise ValueError("persistence must be a PersistenceConfig or None")
            # 03 起检查模式 + 持久会话组合放开 (票据 03); 恢复要求 checks_signature 一致
        if self.budget is not None and not isinstance(self.budget, BudgetConfig):
            raise ValueError("budget must be a BudgetConfig or None")
        if self.skills is not None and not isinstance(self.skills, SkillCatalogSnapshot):
            raise ValueError("skills must be a SkillCatalogSnapshot or None")


@dataclass(frozen=True)
class PersistenceConfig:
    """持久会话入口配置; session_id=None 表示创建新会话, 否则按精确 ID 恢复.

    identity 是启动层给出的有效配置身份 (默认值已解析)。默认恢复要求 identity
    与存档一致; adopt_current_config=True (--resume-with-current-config) 显式
    采用当前配置: 身份/工具指纹差异不拒绝, 差异记入新 epoch 的身份来源, 不解除
    view-only/process_uncertain 阻断 (04)。
    """

    session_id: str | None
    identity: LaunchIdentity
    adopt_current_config: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LaunchIdentity):
            raise ValueError("PersistenceConfig.identity must be a LaunchIdentity")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise ValueError("PersistenceConfig.session_id must be a str or None")
        if not isinstance(self.adopt_current_config, bool):
            raise ValueError("PersistenceConfig.adopt_current_config must be a bool")


@dataclass(frozen=True)
class RequestOutcome:
    """单个用户请求的权威终结; 复用现有记录, 不另造第二份检查回执."""

    result: RunResult
    view: VerificationView | None
    evidence: ChangeEvidence | None
    decision: DeliveryDecision | None
    exit_code: int
    persistence: RequestPersistenceFacts | None = None


@dataclass(frozen=True)
class AgentEventReceived:
    """包装原内核事件; 不改写."""

    event: Event


@dataclass(frozen=True)
class RequestPhaseChanged:
    """应用阶段变化; 首个 run ID 由 RunStarted 确认, 此前允许 run_id=None."""

    run_id: str | None
    phase: ApplicationPhase


@dataclass(frozen=True)
class RequestCompleted:
    """验证收尾后的唯一请求终结; 正常状态下收到时 session 已回 idle."""

    outcome: RequestOutcome


@dataclass(frozen=True)
class HistoryCleared:
    """已清空模型历史; 验证基线/账本/锁存继续保留."""


@dataclass(frozen=True)
class OperationBusy:
    """非请求操作进行中 (当前: 持久清空/压缩提交).

    只读补充原请求 phase: 请求生命周期已有 RequestPhaseChanged, 清空/压缩等
    操作经本事件让 UI 输入屏障可消费 (05/02 接线); 完成分别由 HistoryCleared
    与 ContextCompacted 标记.
    """

    operation: Literal["clear", "compact", "model"]


@dataclass(frozen=True)
class ModelChanged:
    previous_label: str
    model_label: str
    epoch_id: str | None


class ModelSelectionError(RuntimeError):
    """本地模型选择被拒绝; 有界原因, 无请求历史或凭据."""


@dataclass(frozen=True)
class ContextCompacted:
    """压缩切换成功 (02, D6): 摘要已持久并成为模型可见投影的一部分.

    payload 携带覆盖范围与保留边界 (规范历史下标); 原始记录不变, UI 据此
    追加确认条目并释放屏障。失败/取消不产生本事件。
    """

    summary_id: str
    first_request_seq: int
    last_request_seq: int
    message_count: int
    end_message_index: int
    reason: str  # manual / auto


@dataclass(frozen=True)
class ApplicationClosed:
    """关闭完成或明确清理失败; error 非空不代表所有资源均已成功释放."""

    error: str | None = None


ApplicationUpdate = (
    AgentEventReceived | RequestPhaseChanged | RequestCompleted | HistoryCleared
    | OperationBusy | ContextCompacted | ModelChanged | ApplicationClosed
)


class ApplicationOpenError(RuntimeError):
    """启动组装或检查基线建立失败; 显示文本即异常文本."""


class CompactionSealedError(RuntimeError):
    """close 已封住未接受的摘要提交 (PA05); 归关闭取消, 非存储故障."""


@dataclass(frozen=True)
class _CompactionPlan:
    """一次手动压缩的切点计划 (D4): 覆盖区间与保留边界."""

    messages: tuple[Message, ...]     # 摘要输入: 原始规范历史区间 (不经投影)
    first_request_seq: int
    last_request_seq: int
    message_count: int                # 本块覆盖消息数 (end - start)
    start_index: int                  # 覆盖起点 (= 投影保留起点)
    end_index: int                    # 覆盖前缀终点 = 保留边界


def _isolate_event(event: Event) -> Event:
    if isinstance(event, ToolCompleted):
        return ToolCompleted(run_id=event.run_id, result=isolate_tool_result(event.result))
    return event


def _isolate_outcome(outcome: RequestOutcome) -> RequestOutcome:
    result = outcome.result
    messages = tuple(isolate_message(m) for m in result.messages)
    return RequestOutcome(
        result=RunResult(
            run_id=result.run_id,
            stop_reason=result.stop_reason,
            error=result.error,
            messages=messages,
            model_calls=result.model_calls,
        ),
        view=outcome.view,
        evidence=outcome.evidence,
        decision=outcome.decision,
        exit_code=outcome.exit_code,
        persistence=outcome.persistence,
    )


def _isolate_update(update: ApplicationUpdate) -> ApplicationUpdate:
    if isinstance(update, AgentEventReceived):
        return AgentEventReceived(event=_isolate_event(update.event))
    if isinstance(update, RequestCompleted):
        return RequestCompleted(outcome=_isolate_outcome(update.outcome))
    return update


def _outcome_exit_code(
    result: RunResult, decision: DeliveryDecision | None
) -> int:
    """最近完成请求的既有 0/1/3 含义: 普通模式 stop=0 其余 1; 检查模式按交付判定."""
    if decision is None:
        return 0 if result.stop_reason == "stop" else 1
    if not decision.model_stopped:
        return 1
    return 0 if decision.can_deliver else 3


class ApplicationSession:
    """单一执行 seam: 持有一个 App、一个 Agent, 检查模式下还有一套验证服务."""

    def __init__(
        self,
        *,
        app: App,
        config: ApplicationConfig,
        service: Verifier | None,
        policy: RunPolicy | None,
        store: SessionStorePort | None = None,
        epoch_id: str | None = None,
        persistence_mode: Literal["save", "resume"] | None = None,
        restored_history: tuple[Message, ...] = (),
        restored_requests: tuple[RestoredRequest, ...] = (),
        restored_context_generation: int = 0,
        restored_interrupted: tuple[InterruptedRequest, ...] = (),
        restored_receipts: tuple[StoredReceipt, ...] = (),
        restored_summaries: tuple[SummaryRecord, ...] = (),
        view_only_reason: str | None = None,
        projection: HistoricalProjection | None = None,
        budget_policy: BudgetPolicy | None = None,
        context_projection: ContextProjection | None = None,
        model_selection: ModelSelectionPort | None = None,
        model_route: ModelRoute | None = None,
        summary_route: ModelRoute | None = None,
    ) -> None:
        self._app = app
        self._config = config
        self._service = service
        self._policy = policy
        self._budget_policy = budget_policy
        self._model_selection = model_selection
        self._model_route = model_route
        self._summary_route = summary_route
        self._model_in_flight = False
        self._model_done: asyncio.Future | None = None
        self._model_task: asyncio.Task | None = None
        # 模型可见投影容器 (02, D4): 压缩成功后追加摘要块, clear 重置, 恢复重建;
        # 与 BudgetPolicy 持同一实例
        self._context_projection = context_projection if context_projection is not None else ContextProjection()
        self._history: list[Message] = [isolate_message(m) for m in restored_history]
        self._store = store
        self._epoch_id = epoch_id
        self._persistence_mode = persistence_mode
        self._restored_requests = restored_requests
        self._restored_request_count = len(restored_requests)
        self._restored_model_call_count = sum(
            len(request.outcome.model_calls) for request in restored_requests
        )
        self._restored_interrupted = restored_interrupted
        self._view_only_reason = view_only_reason
        self._projection = projection                    # 前缀证明容器; 清空成功后重置 (03)
        self._restored_receipts = restored_receipts
        self._persisted_receipt_ids = {r.receipt_id for r in restored_receipts}
        # 04: 恢复的当前代次摘要记录 (store 恢复视图同口径) + 本进程已提交记录;
        # 只读出口 summary_records 供显示摘要计量/压缩事实与 05 校准取数
        self._restored_summaries = restored_summaries
        self._summary_records: list[SummaryRecord] = []
        self._context_generation = restored_context_generation
        self._request_id: str | None = None
        self._saved_request_count = 0
        self._last_committed_seq: int | None = None
        # 当前代次已提交完整请求的消息区间 (start,end) 与 seq, 按提交序; 压缩切点
        # 与覆盖范围取数 (D4)。恢复时按当前代次请求重建, 提交成功后追加, clear 清空
        self._committed_spans: list[tuple[int, int, int]] = []
        span_start = 0
        for request in restored_requests:
            if request.context_generation != restored_context_generation:
                continue
            self._committed_spans.append(
                (request.seq, span_start, span_start + len(request.messages))
            )
            span_start += len(request.messages)
        self._storage_failed: str | None = None
        self._state: SessionStateName = "idle"
        self._run_id: str | None = None
        self._observers: list[Callable[[ApplicationUpdate], None]] = []
        self.observer_errors: list[BaseException] = []
        self._cancel_token: CancelToken | None = None
        self._active_done: asyncio.Future | None = None
        self._clear_done: asyncio.Future | None = None  # 清空生产者完成标记 (close 监督对象)
        self._clearing = False  # 持久清空进行中: 与 execute 互斥 (契约 §3)
        # 02: 手动压缩生产者 (D6/PA05); close 封住未接受提交并协作取消生成
        self._compacting = False
        self._compact_done: asyncio.Future | None = None
        self._compact_token: CancelToken | None = None
        self._compact_sealed = False
        # 阶段收口审查 STG-F1: execute 的 begin 等待窗内 _state 仍为 idle,
        # 同步旗标把压缩/清空/新请求互斥补齐到该窗 (不依赖 UI 屏障)
        self._begin_in_flight = False
        self._process_uncertain = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task | None = None
        self._close_deadline: float | None = None
        self._close_latch = asyncio.Event()  # 锁存置位: 收尾等待据此收缩截止时间
        self._close_problems: list[str] = []
        self._undead_tasks: list[asyncio.Task] = []
        self._unsubscribe_core = app.agent.subscribe(self._on_core_event)

    # --- 只读快照 ------------------------------------------------------------------

    @property
    def state(self) -> SessionStateName:
        return self._state

    @property
    def history(self) -> tuple[Message, ...]:
        """隔离快照; 修改其中 details 不影响下一模型请求或验证判定."""
        return tuple(isolate_message(m) for m in self._history)

    @property
    def config(self) -> ApplicationConfig:
        return self._config

    @property
    def verification_run_id(self) -> str | None:
        return self._config.plan.verification_run_id if self._config.plan is not None else None

    def has_process_uncertain(self) -> bool:
        return self._process_uncertain

    @property
    def persistence(self) -> PersistenceState | None:
        """只读持久状态快照; 无持久化为 None."""
        if self._store is None:
            return None
        return PersistenceState(
            session_id=self._store.session_id,
            epoch_id=self._epoch_id or "",
            mode=self._persistence_mode or "save",
            context_generation=self._context_generation,
            restored_request_count=self._restored_request_count,
            restored_model_call_count=self._restored_model_call_count,
            saved_request_count=self._saved_request_count,
            last_committed_seq=self._last_committed_seq,
            storage_failed=self._storage_failed,
        )

    @property
    def restored_requests(self) -> tuple[RestoredRequest, ...]:
        """打开时恢复的已提交请求事实 (含逐轮计量); 无持久化为空元组."""
        return self._restored_requests

    @property
    def restored_interrupted(self) -> tuple[InterruptedRequest, ...]:
        """打开时存档中的未完成 intent (诊断与显示; 不进入可续写历史)."""
        return self._restored_interrupted

    @property
    def restored_receipts(self) -> tuple[StoredReceipt, ...]:
        """打开时恢复的历史检查回执事实 (05 显示消费的取数入口; 只读)."""
        return self._restored_receipts

    @property
    def skills(self) -> SkillCatalogView | None:
        """启用 Skills 时的只读目录视图 (元数据/root/manifest, 不含正文); 未启用 None."""
        snapshot = self._config.skills
        return None if snapshot is None else snapshot.view()

    def prepare_skill_request(self, name: str, task: str = "") -> str:
        """显式一次请求的普通 prompt 文本 (契约 §3/§4): 同步纯准备, 不创建请求.

        从启动冻结快照查找 (手动专用 Skill 允许显式选择), 封装为带来源/正文/
        任务的用户文本。未启用 Skills、未知名称抛 SkillSelectionError, 超过
        MAX_SKILL_REQUEST_BYTES 抛 SkillRequestError。本方法无历史追加、无
        存储、无模型调用; 准备成功不等于应用已接受或保存, 执行仍必须走原
        execute。
        """
        snapshot = self._config.skills
        if snapshot is None:
            raise SkillSelectionError(
                "skills are not enabled for this session; start with --skills-dir to enable them"
            )
        return build_skill_request(find_skill(snapshot, name, allow_manual=True), task)

    @property
    def view_only_reason(self) -> str | None:
        """非空 = 恢复准入发现未证终结活动, 仅可查看; 新编码请求被拒绝 (契约 D3).

        清空不解锁此状态; 普通/检查模式及配置覆盖都不能绕过 (A15).
        """
        return self._view_only_reason

    @property
    def active_request_id(self) -> str | None:
        """当前 (或最近一次) 请求的持久 ID; 活动包装器经协调器读取."""
        return self._request_id

    @property
    def budget(self) -> BudgetSnapshot | None:
        """最新预算报告与锚状态的只读快照 (04 显示消费); 未启用预算层为 None."""
        if self._budget_policy is None:
            return None
        return self._budget_policy.snapshot()

    @property
    def summary_records(self) -> tuple[SummaryRecord, ...]:
        """当前代次的已提交摘要记录 (04 显示/摘要计量出口; 05 校准机器取数).

        恢复的当前代次记录在前、本进程已提交记录按提交序在后; clear 推进代次后
        旧代次记录退出视图 (与 store 恢复视图同一口径, 存档留底不受影响)。
        摘要计量/压缩次数/最近原因/保留边界据此派生, 不另设累计器。
        """
        generation = self._context_generation
        restored = tuple(
            record for record in self._restored_summaries
            if record.context_generation == generation
        )
        committed = tuple(
            record for record in self._summary_records
            if record.context_generation == generation
        )
        return restored + committed

    def latch_storage_failure(self, detail: str) -> None:
        """活动包装器等生产者的公开锁存入口; 幂等, 首次失败固定原因."""
        self._latch_storage_failed(detail)

    # --- 订阅 ----------------------------------------------------------------------

    def subscribe(self, handler: Callable[[ApplicationUpdate], None]) -> Callable[[], None]:
        self._observers.append(handler)

        def unsubscribe() -> None:
            if handler in self._observers:
                self._observers.remove(handler)

        return unsubscribe

    def _notify(self, update: ApplicationUpdate) -> None:
        # 先更新 state/history 再调用本方法; 每个观察者拿到隔离副本
        for handler in list(self._observers):
            try:
                handler(_isolate_update(update))
            except Exception as exc:  # 通知异常不改变内核或验证权威
                self.observer_errors.append(exc)

    def _on_core_event(self, event: Event) -> None:
        self._notify(AgentEventReceived(event))
        if isinstance(event, RunStarted):
            self._run_id = event.run_id
            self._notify(RequestPhaseChanged(event.run_id, "executing"))
        elif isinstance(event, RunFinished):
            # 内核结束 != 应用空闲; 收尾由 execute 的 finalizing 段完成
            self._notify(RequestPhaseChanged(event.run_id, "finalizing"))

    # --- 执行 ----------------------------------------------------------------------

    @property
    def model_options(self) -> tuple[ModelOption, ...]:
        return self._model_selection.options if self._model_selection is not None else ()

    async def select_model(self, option_id: str) -> None:
        if self._closing or self._closed or self._state != "idle":
            raise ModelSelectionError("model selection requires an idle, open session")
        if self._model_in_flight or self._begin_in_flight or self._clearing or self._compacting:
            raise ModelSelectionError("another operation is in progress")
        if self._view_only_reason or self._storage_failed or self._process_uncertain:
            raise ModelSelectionError("model selection is blocked by the session state")
        if self._model_selection is None or self._model_route is None:
            raise ModelSelectionError("model selection is unavailable for this provider")
        if option_id not in {option.id for option in self.model_options}:
            raise ModelSelectionError("unknown model option")
        if option_id == self._model_route.binding.identity.model:
            return
        self._model_in_flight = True
        self._model_done = asyncio.get_running_loop().create_future()
        try:
            try:
                binding = self._model_selection.prepare_model(option_id)
                binding.validate_request(ModelRequest(
                    messages=self._context_projection.project(self._history),
                    tools=self._request_tool_specs(), system_prompt=self._config.system_prompt,
                ))
            except ValueError as exc:
                raise ModelSelectionError("model selection rejected by configuration or history") from exc
            persistence = self._config.persistence
            identity = None
            if persistence is not None:
                safe = binding.identity
                identity = replace(persistence.identity, provider=safe.provider, model=safe.model,
                    base_url=safe.base_url, think=safe.think, options=dict(safe.options),
                    max_request_bytes=safe.max_request_bytes,
                    sources={**persistence.identity.sources, "model": "selection"})
                persistence = replace(persistence, identity=identity)
            new_config = replace(self._config, model_label=binding.identity.model,
                                 budget=binding.budget, persistence=persistence)
            self._notify(OperationBusy(operation="model"))
            if self._store is None:
                if self._closing or self._closed:
                    raise ModelSelectionError("application is closing; model selection was not accepted")
                await self._finish_model_change(binding, new_config, None)
                return
            # 接受前唯一取消点; close 封口后不能把新事务排进队列。
            await asyncio.sleep(0)
            if self._closing or self._closed:
                raise ModelSelectionError("application is closing; model selection was not accepted")
            submission = None
            if self._store is not None:
                snapshot = self._store.recovery_snapshot()
                assert identity is not None and snapshot.identity is not None
                try:
                    submission = self._store.submit_model_change(
                        StoredIdentity(identity, snapshot.identity.tool_fingerprint),
                        expected_epoch_id=self._epoch_id or "",
                    )
                except PersistenceError as exc:
                    if isinstance(exc, (StorageUnavailableError, StorageWriteError, ArchiveLimitError)):
                        self._latch_storage_failed(str(exc))
                    raise ModelSelectionError("model configuration was not accepted by storage") from exc
            # 强引用覆盖事务和内存发布; close 等待同一完整生产者。
            self._model_task = asyncio.create_task(
                self._finish_model_change(binding, new_config, submission)
            )
            cancelled = False
            while True:
                try:
                    await asyncio.shield(self._model_task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError()
        finally:
            self._model_in_flight = False
            if self._model_done is not None and not self._model_done.done():
                self._model_done.set_result(None)

    async def _finish_model_change(
        self, binding: ModelBinding, config: ApplicationConfig,
        submission: ModelChangeSubmission | None,
    ) -> None:
        epoch_id = self._epoch_id
        if submission is not None:
            try:
                epoch_id = await submission.wait()
            except PersistenceError as exc:
                self._latch_storage_failed(str(exc))
                raise ModelSelectionError("model configuration failed to persist; current model is unchanged") from exc
        previous = self._config.model_label
        assert self._model_route is not None and self._summary_route is not None
        assert self._budget_policy is not None
        self._epoch_id = epoch_id
        self._model_route.binding = binding
        self._summary_route.binding = binding
        self._config = config
        policy = self._budget_policy
        policy.config = binding.budget
        policy.summary_budget = binding.summary_budget
        policy.summary_identity = binding.summary_identity
        policy.validate_request = binding.validate_request
        policy.invalidate()
        self._notify(ModelChanged(previous, config.model_label, epoch_id))

    async def execute(self, prompt: str) -> RequestOutcome:
        if self._model_in_flight:
            raise RuntimeError("model selection is in progress; execute is refused")
        if self._closing or self._closed:
            raise RuntimeError(f"application is {self._state}; execute rejected before start")
        if self._state != "idle":
            raise RuntimeError(f"application is {self._state}; a request must finish before the next one")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty request")
        if self._view_only_reason is not None:
            raise RuntimeError(
                f"session is view-only after restore ({self._view_only_reason}); "
                "new coding requests are refused"
            )
        if self._config.plan is not None and self._process_uncertain:
            raise RuntimeError(
                "process_uncertain: new coding requests are refused; only status, clear or exit remain"
            )
        if self._store is not None:
            # 可靠提交请求开始 intent; 失败零模型调用并锁存 storage_failed (D2)。
            # 等待受拥有: 外层取消不把已派发事务当作未发生, 取得明确结果后才收尾
            if self._storage_failed is not None:
                raise RuntimeError(
                    f"storage_failed: earlier session save failed ({self._storage_failed}); "
                    "new coding requests are refused"
                )
            if self._clearing:
                raise RuntimeError(
                    "a persistent clear is in progress; new requests are refused until it settles"
                )
            if self._compacting:
                # 02/PA05: execute/clear/compact 三向互斥 (应用层, 不依赖 UI 屏障);
                # 压缩在途时新请求必须等待, 否则其提交会与摘要切换交错
                raise RuntimeError(
                    "a compaction is in progress; new requests are refused until it settles"
                )
            if self._begin_in_flight:
                # 阶段收口审查 STG-F1: 前一请求仍在 begin 等待窗 (state 尚为 idle)
                raise RuntimeError(
                    "a previous request is still being started; "
                    "new requests are refused until it settles"
                )
        # 请求边界接缝 (D1/PA02): 准入通过后、首次预算预估之前调用一次,
        # 建立本次 execute 尝试的预算状态范围; begin 不再调用或清零
        if self._budget_policy is not None:
            self._budget_policy.note_request_start(len(self._history))
        # 03: 请求 token 提前到接缝处建立——begin 前自动压缩期间 session.cancel()
        # 即取消本次尝试 (零 begin/零模型调用); begin 后取消语义不变
        token = CancelToken()
        self._cancel_token = token
        # 阶段收口审查 STG-F1: 从首次预估到 begin 落定期间同步占住互斥
        # (预估内自动压缩另由 _compacting 覆盖); 失败/取消/完成均经 finally 释放
        self._begin_in_flight = True
        try:
            # 请求前预估 (D7 上道口): 超限先尝试本次 execute 至多一次的自动压缩,
            # 再对有效投影重估; 仍超限零 begin/零模型调用, 不锁存 storage_failed
            await self._refuse_overbudget_or_auto_compact(prompt)
            if self._budget_policy is not None:
                self._budget_policy.validate_request(ModelRequest(
                    messages=(*self._context_projection.project(self._history), UserMessage(prompt)),
                    tools=self._request_tool_specs(), system_prompt=self._config.system_prompt,
                ))
            if self._store is not None:
                begin_task = asyncio.ensure_future(
                    self._store.begin_request(self._epoch_id or "", prompt)
                )
                begin_error: PersistenceError | None = None
                begin_cancelled = False
                while True:
                    try:
                        self._request_id = await asyncio.shield(begin_task)
                        break
                    except asyncio.CancelledError:
                        begin_cancelled = True
                    except PersistenceError as exc:
                        begin_error = exc
                        break
                if begin_cancelled:
                    # 已取得事务明确结果后的取消属程序性终止: 与执行期异常一样进入 closing
                    if self._state != "closed":
                        self._state = "closing"
                        self._closing = True
                    raise asyncio.CancelledError()
                if begin_error is not None:
                    # 落盘故障 (不可用/写失败/越限) 才锁存 storage_failed; 关系冲突
                    # (pending 已存在/epoch 不符) 是状态问题, 存储健康, 不得误锁存
                    if isinstance(begin_error, (StorageUnavailableError, StorageWriteError, ArchiveLimitError)):
                        self._latch_storage_failed(str(begin_error))
                        raise RuntimeError(
                            f"session save failed before the request started; "
                            f"no model call or tool execution was made: {begin_error}"
                        ) from begin_error
                    raise RuntimeError(
                        f"cannot start a persisted request: {begin_error}"
                    ) from begin_error
        finally:
            self._begin_in_flight = False
        # 非持久会话无 begin: 接缝同为首次预估前 (准入通过后), 预算拒绝零副作用
        # (D9); 自动压缩需持久会话, 回调内判定后原样拒绝
        loop = asyncio.get_running_loop()
        self._state = "executing"
        self._run_id = None
        # token 已在接缝处建立 (03): 覆盖预估/自动压缩/begin/全部轮次
        self._active_done = loop.create_future()
        self._notify(RequestPhaseChanged(None, "executing"))
        try:
            result = await self._app.agent.run(prompt, token, history=tuple(self._history))
            # 内核结束 != 应用空闲: 显式进入收尾状态, 完成前不接受新请求 (T07);
            # closed 是终态, close 监督超时后的迟到完成不得改写它 (02-R-F1)
            if self._state != "closed":
                self._state = "finalizing"
            outcome = await self._finalize_request(result)
        except BaseException:
            # 异常路径也不得把应用留在"可执行"状态: 内核/UI 异常一律经关闭路径处理
            if self._state != "closed":
                self._state = "closing"
                self._closing = True
            raise
        finally:
            if self._active_done is not None and not self._active_done.done():
                self._active_done.set_result(None)
        # 先更新隔离的 state/history, 再通知订阅者; 应用把后缀追加一次
        if self._store is not None:
            # 完成提交成功先于 saved/idle/RequestCompleted; 失败锁存且退出码置 1。
            # 提交 completion 由协调器持有: 外层重复取消不取消事务、不当作未发生
            commit_task = asyncio.ensure_future(self._commit_request(result, outcome))
            while True:
                try:
                    persisted = await asyncio.shield(commit_task)
                    break
                except asyncio.CancelledError:
                    continue
            outcome = replace(
                outcome,
                persistence=persisted,
                exit_code=1 if not persisted.saved else outcome.exit_code,
            )
            if persisted.saved:
                # 提交成功: 当前代次新增一个完整请求区间 (压缩切点/覆盖范围取数, D4)
                self._committed_spans.append(
                    (
                        persisted.seq or 0,
                        len(self._history),
                        len(self._history) + len(result.messages),
                    )
                )
        self._history.extend(isolate_message(m) for m in result.messages)
        if outcome.view is not None and outcome.view.process_uncertain:
            self._process_uncertain = True
        if self._closing:
            # closing 是不可逆接收门; 关闭中仍发一次 RequestCompleted, 但状态不得
            # 回 idle, 也不得把已到达的 closed 终态改回 closing (02-R-F1)
            if self._state != "closed":
                self._state = "closing"
        else:
            self._state = "idle"
            self._notify(RequestPhaseChanged(result.run_id, "idle"))
        self._notify(RequestCompleted(outcome))
        return outcome

    def _request_tool_specs(self) -> tuple[ToolSpec, ...]:
        """当前组装解析后的实际工具声明 (与 _tool_fingerprint 同一取数面).

        直接访问 Agent._tools, 缺失即 AttributeError (fail-loud): 预估口静默
        漏计工具 schema 会系统性低估 (审查 01-F1), 不做 getattr 缺省回退。
        """
        return tuple(tool.spec for tool in self._app.agent._tools.values())

    def _refuse_overbudget_request(self, prompt: str) -> None:
        """请求前预估准入 (D7 两道口之一): 超限在 begin_request 之前拒绝.

        零 begin、零模型调用; 预算拒绝不是存储故障, 不锁存 storage_failed。
        持久模式预估拒绝不落库 (无 intent; 01 裁决, 完成回执钉住"无 intent"断言);
        每轮权威准入 (BudgetPolicy) 发生在 begin 之后, 按既有错误路径收尾并落
        finalized 错误记录。预估不含锚 (锚状态在 BudgetPolicy), 检查模式另加
        RunPolicy 控制段字节上限的允许量, 保持保守方向。
        """
        policy = self._budget_policy
        config = self._config.budget
        if policy is None or config is None:
            return
        # 01-F10/PA02: 预估计量面 = 当前 ContextProjection 的有效消息
        # ([摘要块] + 保留后缀) + 本次 prompt; 旧规范历史不再参与 (压缩后重估
        # 必须能回到限内)。控制段/freshness 允许量仍覆盖最终发送形态 (01 裁决)。
        messages = (
            *self._context_projection.project(self._history),
            UserMessage(text=prompt),
        )
        input_estimate, components = estimate_input_components(
            self._config.system_prompt, self._request_tool_specs(), messages, config
        )
        if self._config.plan is not None:
            allowance = math.ceil(CONTROL_SECTION_MAX_BYTES / 3)
            components = components + (
                ComponentEstimate(
                    "verification_control_allowance",
                    allowance,
                    "estimated",
                    note="upper bound of the RunPolicy control section appended before send",
                ),
            )
            input_estimate += allowance
            # RunPolicy 还对每条 check/powershell 工具结果在投影中追加 freshness
            # 标记 (单条上限 FRESHNESS_MARKER_MAX_BYTES 字节); 规范历史不含标记,
            # 逐条按上限补足, 使预估成为投影后请求的保守上界 (审查 01-F9/F13)
            fresh_count = sum(
                1 for m in messages
                if isinstance(m, ToolResultMessage) and m.result.name in FRESHNESS_TOOLS
            )
            if fresh_count:
                fresh_allowance = fresh_count * math.ceil(FRESHNESS_MARKER_MAX_BYTES / 3)
                components = components + (
                    ComponentEstimate(
                        "verification_freshness_allowance",
                        fresh_allowance,
                        "estimated",
                        note="upper bound of per-result freshness markers appended before send",
                    ),
                )
                input_estimate += fresh_allowance
        report = assemble_report(
            config, input_estimate=input_estimate, input_components=components, source="estimated"
        )
        if report.total_estimate > config.num_ctx:
            raise RuntimeError(f"request refused before start: {budget_exceeded_text(report)}")

    async def _refuse_overbudget_or_auto_compact(self, prompt: str) -> None:
        """请求前预估 + 本次 execute 至多一次的自动压缩 (D7, 03).

        预估超限先经 _auto_compact_for_request 尝试压缩 (条件与额度在回调内
        判定), 成功后对有效投影重估走同一预估接缝 (审查 01-F10); 仍超限按
        begin 前口径拒绝 (PA01: 无 begin/无 intent/无 finalized, 不锁存)。
        自动压缩期间取消 (session.cancel/close) 结束本次尝试: 零 begin/零
        模型调用, 抛 CancelledError 而不伪装成预算拒绝 (D7)。
        """
        try:
            self._refuse_overbudget_request(prompt)
        except RuntimeError:
            if self._budget_policy is None or not await self._auto_compact_for_request("admission"):
                if self._cancel_token is not None and self._cancel_token.cancelled:
                    raise asyncio.CancelledError() from None
                raise
            self._refuse_overbudget_request(prompt)

    async def _finalize_request(self, result: RunResult) -> RequestOutcome:
        """验证收尾: 扫描末次终结事实 -> refresh -> finalize -> 工件核对 -> 交付判定.

        建立拟定 15 秒监督窗口; 中途锁存 closing 时与 close 截止时间取较早值;
        超时取消收尾任务并返回明确不可交付结果.
        """
        if self._config.plan is None or self._service is None or self._policy is None:
            decision = None
            view = None
            evidence = None
        else:
            view, evidence, decision = await self._run_finalization(
                result, time.monotonic() + REQUEST_FINALIZE_SUPERVISION_S
            )
        return RequestOutcome(
            result=result,
            view=view,
            evidence=evidence,
            decision=decision,
            exit_code=_outcome_exit_code(result, decision),
        )

    def _effective_deadline(self, base_deadline: float) -> float:
        """closing 锁存后, 有效截止时间取请求窗口与 close 截止时间的较早值."""
        if self._closing and self._close_deadline is not None:
            return min(base_deadline, self._close_deadline)
        return base_deadline

    async def _run_finalization(
        self, result: RunResult, base_deadline: float
    ) -> tuple[VerificationView, ChangeEvidence, DeliveryDecision]:
        assert self._service is not None and self._policy is not None
        service, policy = self._service, self._policy
        token = CancelToken()

        async def work() -> tuple[VerificationView, ChangeEvidence, DeliveryDecision]:
            # 末次工具结果 (含最后一轮 check/powershell 错误) 先锁存, 再取最终视图与证据
            policy._scan_terminal_facts(result.messages)
            view = await service.refresh(token)
            evidence = await service.finalize(token)
            decision = decide_delivery(result, view, evidence, verify_receipt_artifacts(view))
            return view, evidence, decision

        async def _quarantine() -> None:
            """取消收尾工作; 到宽限期仍未终结的任务保留强引用与诊断."""
            token.cancel()
            task.cancel()
            _done, pending = await asyncio.wait([task], timeout=_UNDEAD_GRACE_S)
            if pending:
                self._undead_tasks.append(task)

        task = asyncio.create_task(work())
        while True:
            # close 可能在等待中途锁存并收缩截止时间: 用 FIRST_COMPLETED 重估
            shielded = asyncio.ensure_future(asyncio.shield(task))
            latch = asyncio.ensure_future(self._close_latch.wait())
            try:
                remaining = self._effective_deadline(base_deadline) - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                timed_out = False
                done, _ = await asyncio.wait(
                    {shielded, latch}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                latch.cancel()
            if shielded.done():
                try:
                    return shielded.result()
                except BaseException:
                    # 外层取消/异常: 收尾任务不得成为无人持有的后台任务
                    await _quarantine()
                    raise
            if latch.done():
                continue  # close 刚锁存: 用较早的 close 截止时间重新评估
            break  # remaining 到点: 超时路径
        await _quarantine()
        reason = (
            "verification finalization did not complete before the effective "
            "supervision deadline (request finalize window; bounded by the close "
            "deadline once closing is latched)"
        )
        return (
            _unavailable_view("finalization timed out"),
            _incomplete_evidence("finalization timed out"),
            DeliveryDecision(
                model_stopped=result.stop_reason == "stop",
                can_deliver=False,
                blocking_reasons=(reason,),
            ),
        )

    # --- 持久提交 -------------------------------------------------------------------

    def _latch_storage_failed(self, detail: str) -> None:
        if self._storage_failed is None:
            self._storage_failed = detail

    async def _commit_request(self, result: RunResult, outcome: RequestOutcome) -> RequestPersistenceFacts:
        """完整请求单事务提交; 返回持久事实, 失败不抛出而是锁存 storage_failed."""
        assert self._request_id is not None and self._store is not None
        decision = outcome.decision
        facts = RequestOutcomeFacts(
            run_id=result.run_id,
            stop_reason=result.stop_reason,
            error=result.error,
            exit_code=outcome.exit_code,
            finalized=True,
            finalization_notes=(
                tuple(decision.blocking_reasons)
                if decision is not None and not decision.can_deliver else ()
            ),
            model_calls=result.model_calls,
        )
        messages = tuple(isolate_message(m) for m in result.messages)
        receipts = self._new_receipts(outcome)
        try:
            seq = await self._store.commit_request(
                self._request_id, messages=messages, outcome=facts, receipts=receipts
            )
        except PersistenceError as exc:
            self._latch_storage_failed(str(exc))
            return RequestPersistenceFacts(
                request_id=self._request_id, saved=False, error=str(exc)
            )
        self._saved_request_count += 1
        self._last_committed_seq = seq
        self._persisted_receipt_ids.update(receipt.receipt_id for receipt in receipts)
        return RequestPersistenceFacts(request_id=self._request_id, saved=True, seq=seq)

    def _new_receipts(self, outcome: RequestOutcome) -> tuple[StoredReceipt, ...]:
        """本请求新产生的检查回执 (尚未持久化的); 与请求记录同事务落库 (03)."""
        view = outcome.view
        if self._config.plan is None or view is None or self._epoch_id is None:
            return ()
        return tuple(
            StoredReceipt(
                verification_run_id=receipt.verification_run_id,
                receipt_id=receipt.receipt_id,
                epoch_id=self._epoch_id,
                request_id=self._request_id or "",
                definition_fingerprint=_receipt_fingerprint(
                    receipt.check_id, receipt.command, receipt.timeout_s
                ),
                execution_status=receipt.execution_status,
                exit_code=receipt.exit_code,
                timed_out=receipt.timed_out,
                cancelled=receipt.cancelled,
                output_complete=receipt.output_complete,
                failure_kind=receipt.failure_kind,
            )
            for receipt in view.receipts
            if receipt.receipt_id not in self._persisted_receipt_ids
        )

    # --- 取消 / 清空 ---------------------------------------------------------------

    def cancel(self) -> None:
        """请求取消活动 token; 幂等; 不伪造完成, 不直接杀 UI."""
        if self._state == "executing":
            self._state = "cancelling"
            self._notify(RequestPhaseChanged(self._run_id, "cancelling"))
        if self._cancel_token is not None:
            self._cancel_token.cancel()

    def clear_history(self) -> None:
        """仅空闲允许; 清除模型历史, 保留 App/Agent、基线/账本/锁存和已有工件.

        持久会话拒绝同步入口: 清空边界必须经 clear_context() 可靠提交.
        """
        if self._model_in_flight:
            raise RuntimeError("model selection is in progress; clear is refused")
        if self._store is not None:
            raise RuntimeError(
                "persistent sessions must use clear_context(); the synchronous entry is refused "
                "to keep the clear boundary reliable"
            )
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        self._history.clear()
        if self._budget_policy is not None:
            self._budget_policy.invalidate()
        self._notify(HistoryCleared())

    async def clear_context(self) -> None:
        """持久清空: 先可靠提交清空边界, 成功后才清内存历史; 失败内存不变.

        无持久化模式委托旧同步入口; storage_failed 不阻止清空尝试, 失败如实报错.
        清空进行中 (_clearing) 与 execute 互斥, 双向拒绝 (契约 §3).
        """
        if self._model_in_flight:
            raise RuntimeError("model selection is in progress; clear is refused")
        if self._store is None:
            self.clear_history()
            return
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        if self._clearing:
            raise RuntimeError("a persistent clear is already in progress")
        if self._compacting:
            # 02: 压缩在途时清空被拒——clear 推进代次会让在途摘要提交撞关系校验,
            # 且其失败会被误归存储故障; 等压缩落定后再清 (三向互斥, 审查 02-F8)
            raise RuntimeError("a compaction is in progress; clear is refused until it settles")
        if self._begin_in_flight:
            # 阶段收口审查 STG-F1: 请求 begin 等待窗内 state 仍为 idle, 同步旗标
            # 补齐互斥 (该窗缺口的 clear 变体在基线已存在)
            raise RuntimeError(
                "a request start is being persisted; clear is refused until it settles"
            )
        self._clearing = True
        self._clear_done = asyncio.get_running_loop().create_future()
        self._notify(OperationBusy(operation="clear"))
        try:
            # 清空事务的 completion 受拥有: 外层取消先等明确结果; 成功则先完成内存
            # 同步再传播取消, 避免存档代次前进而内存未清的分叉 (契约 §1)
            clear_task = asyncio.ensure_future(self._store.commit_clear(self._epoch_id or ""))
            clear_error: PersistenceError | None = None
            clear_cancelled = False
            new_generation: int | None = None
            while True:
                try:
                    new_generation = await asyncio.shield(clear_task)
                    break
                except asyncio.CancelledError:
                    clear_cancelled = True
                except PersistenceError as exc:
                    clear_error = exc
                    break
            if clear_error is not None:
                self._latch_storage_failed(str(clear_error))
                raise RuntimeError(
                    f"session clear failed to persist; in-memory history is unchanged: {clear_error}"
                ) from clear_error
            assert new_generation is not None
            self._context_generation = new_generation
            self._history.clear()
            if self._budget_policy is not None:
                self._budget_policy.invalidate()
            self._notify(HistoryCleared())
            if self._projection is not None:
                # 前缀证明与内存历史同步重置; 失败路径不会到达这里 (契约 §5)
                self._projection.reset()
            if self._context_projection is not None:
                # 02: 新代次投影为空 (D8); 旧代次摘要记录保留在档为历史事实
                self._context_projection.reset()
            self._committed_spans.clear()
            if clear_cancelled:
                raise asyncio.CancelledError()
        finally:
            self._clearing = False
            if self._clear_done is not None and not self._clear_done.done():
                self._clear_done.set_result(None)

    # --- 手动压缩 (02, D4-D6) ---------------------------------------------------------

    async def compact_context(self) -> ContextCompacted | None:
        """手动压缩: 摘要当前代次较早已提交完整请求, 先持久后切换 (D4-D6).

        成功返回覆盖事实并发布 ContextCompacted; close 封口先于接受时返回
        None (归关闭取消, 零写入, 旧投影不变)。其余拒绝/失败抛 RuntimeError。
        协议 (PA03): 响应校验 → 封口核对且未取消 → store 接受 → 明确成功 →
        切换投影 + 事件。派发前取消零写入; store 接受后取消不取消事务, 成功
        先切换再传播外层 CancelledError, 失败保留旧投影并锁存存储故障。
        """
        if self._model_in_flight:
            raise RuntimeError("model selection is in progress; compact is refused")
        if self._store is None:
            raise RuntimeError(
                "compaction requires a persistent session (start with --save-session, "
                "--tui or --resume); non-persistent sessions have no compaction"
            )
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(
                f"history can only be compacted while idle and open (state={self._state})"
            )
        if self._view_only_reason is not None:
            raise RuntimeError(
                f"session is view-only after restore ({self._view_only_reason}); "
                "compaction is refused"
            )
        if self._storage_failed is not None:
            raise RuntimeError(
                f"storage_failed: earlier session save failed ({self._storage_failed}); "
                "compaction is refused because the projection switch relies on reliable commit"
            )
        policy = self._budget_policy
        channel = policy.summarize_channel if policy is not None else None
        if policy is None or self._config.budget is None or channel is None:
            raise RuntimeError(
                "compaction requires an enabled budget with a summary channel "
                "(a persistent session with a real model provider)"
            )
        if self._compacting:
            raise RuntimeError("a compaction is already in progress")
        if self._clearing:
            # 三向互斥 (审查 02-F8): 清空在途时压缩拒绝, 与 execute/clear 同一面
            raise RuntimeError(
                "a persistent clear is in progress; compaction is refused until it settles"
            )
        if self._begin_in_flight:
            # 阶段收口审查 STG-F1: 请求 begin 等待窗内 state 仍为 idle, 同步旗标
            # 补齐互斥 (不依赖 UI 屏障), 否则压缩可与在途请求交错
            raise RuntimeError(
                "a request start is being persisted; compaction is refused until it settles"
            )
        if self._compact_sealed:
            raise RuntimeError("application is closing; compaction is sealed")
        # 可压缩区间与摘要输入预算在置位屏障前判定: 拒绝不留下 busy/屏障 (D1)
        plan = self._compaction_plan()
        self._compacting = True
        self._compact_done = asyncio.get_running_loop().create_future()
        self._compact_token = CancelToken()
        self._notify(OperationBusy(operation="compact"))
        try:
            return await self._run_compaction(
                plan, reason="manual", cancel_token=self._compact_token
            )
        finally:
            self._compacting = False
            self._compact_token = None
            if self._compact_done is not None and not self._compact_done.done():
                self._compact_done.set_result(None)

    async def _run_compaction(
        self, plan: _CompactionPlan, *, reason: str, cancel_token: CancelToken
    ) -> ContextCompacted | None:
        """压缩核心协议 (D4-D6, PA03/PA05): 手动与自动共用同一实现 (03).

        摘要生成 → 响应校验 → 封口核对且未取消 → store 接受 → 明确成功 →
        切换投影 + ContextCompacted 事件。close 封口先于接受时返回 None
        (归关闭取消, 零写入, 旧投影不变); 摘要失败/提交失败抛 RuntimeError
        (投影不变; 锁存按 begin 同一白名单); store 接受后取消不取消事务,
        成功先切换再传播外层 CancelledError。
        """
        channel = self._budget_policy.summarize_channel
        outcome = await generate_summary(channel, plan.messages, cancel_token)
        if not outcome.ok:
            if outcome.failure == "aborted" and self._compact_sealed:
                # PA05: close 封口协作取消 → 归关闭取消, 静默放弃 (旧投影不变)
                return None
            # D5/PA03: 摘要失败不派发提交, 投影不变, 无摘要记录落库
            raise RuntimeError(
                f"summary generation failed ({outcome.failure}); the model projection "
                "is unchanged" + (f": {outcome.detail}" if outcome.detail else "")
            )
        # D6: 摘要模型身份取启动层有效身份 (PersistenceConfig.identity 本就是
        # LaunchIdentity; 审查修复: 此前误取 .launch 使真实持久组合的压缩必失败)
        identity_launch = self._config.persistence.identity if self._config.persistence is not None else None
        summary_identity = self._budget_policy.summary_identity
        record = SummaryRecord(
            summary_id=str(uuid.uuid4()),
            epoch_id=self._epoch_id or "",
            context_generation=self._context_generation,
            first_request_seq=plan.first_request_seq,
            last_request_seq=plan.last_request_seq,
            message_count=plan.message_count,
            end_message_index=plan.end_index,
            prefix_sha256=canonical_prefix_sha256(self._history[: plan.end_index]),
            text=outcome.text,
            provider=(
                summary_identity.provider if summary_identity is not None else (
                identity_launch.provider
                if identity_launch is not None
                else (self._config.model_label or "unknown")
                )
            ),
            model=(
                summary_identity.model if summary_identity is not None else (
                identity_launch.model
                if identity_launch is not None
                else (self._config.model_label or "unknown")
                )
            ),
            prompt_version=SUMMARY_PROMPT_VERSION,
            prompt_sha256=SUMMARY_PROMPT_SHA256,
            metrics=outcome.metrics,
            reason=reason,
            created_at=time.time(),
            call_identity=summary_identity,
            previous_projection_identity=canonical_prefix_sha256(
                self._context_projection.project(self._history)
            ),
        )
        commit_task = asyncio.ensure_future(self._commit_summary_dispatch(record))
        commit_cancelled = False
        while True:
            try:
                await asyncio.shield(commit_task)
                break
            except asyncio.CancelledError:
                # PA03: 派发后取消不取消事务; 重复取消也等明确结果
                commit_cancelled = True
            except CompactionSealedError:
                # PA05: 封口先于接受 → 零写入; 归关闭取消, 旧投影不变,
                # 不锁存 storage_failed, 迟到响应丢弃
                return None
            except PersistenceError as exc:
                # 提交失败: 旧投影不变。锁存面与 begin 路径同一白名单 (审查
                # 02-F12): 真实落盘故障才锁存 storage_failed; 关系冲突
                # (代次/seq/重复) 是状态问题, 存储健康, 误锁存会不可逆拒绝
                # 本进程后续请求
                if isinstance(exc, (StorageUnavailableError, StorageWriteError, ArchiveLimitError)):
                    self._latch_storage_failed(str(exc))
                raise RuntimeError(
                    "compaction failed to persist; the model projection is unchanged: "
                    f"{exc}"
                ) from exc
        # 明确成功 → 切换投影 + 事件; 已被取消时先落定再传播 (PA03)
        # 04: store 已接受即已提交摘要事实, 进入只读出口 (摘要计量/压缩次数取数)
        self._summary_records.append(record)
        block = SummaryBlock(
            message=summary_block_message(
                first_request_seq=plan.first_request_seq,
                last_request_seq=plan.last_request_seq,
                message_count=plan.message_count,
                text=record.text,
            ),
            end_message_index=plan.end_index,
            first_request_seq=plan.first_request_seq,
            last_request_seq=plan.last_request_seq,
            message_count=plan.message_count,
            prefix_sha256=record.prefix_sha256,
        )
        self._context_projection.apply_summary(
            block, canonical_length=len(self._history)
        )
        self._budget_policy.invalidate()
        compacted = ContextCompacted(
            summary_id=record.summary_id,
            first_request_seq=record.first_request_seq,
            last_request_seq=record.last_request_seq,
            message_count=record.message_count,
            end_message_index=record.end_message_index,
            reason=reason,
        )
        self._notify(compacted)
        if commit_cancelled:
            raise asyncio.CancelledError()
        return compacted

    async def _auto_compact_for_request(self, quota: str) -> bool:
        """自动压缩回调 (D7, 03): 准入自动尝试与 overflow 重试共用的应用侧实现.

        属于已有 execute, 不重复获取会自锁的互斥 (execute 已排除并发
        compact/clear; 本方法置位同一屏障供 close 监督与封口)。全部准入条件
        (锁存/互斥/可压缩区间) 通过后才消费对应额度, 失败不返还; 成功返回
        True (投影已切换, 调用方对有效投影重估), 其余返回 False, 压缩失败不
        中断请求 (按原准入/重试路径收尾)。不发 OperationBusy: 自动属于请求
        生命周期, UI 反馈随请求事件。
        """
        if self._store is None:
            return False
        policy = self._budget_policy
        if policy is None or self._config.budget is None or policy.summarize_channel is None:
            return False
        if self._closing or self._closed or self._compact_sealed:
            return False
        if self._compacting or self._clearing:
            return False
        if self._view_only_reason is not None:
            # 两种锁存状态下零摘要调用 (D9): view-only 不把历史发往模型
            return False
        if self._storage_failed is not None:
            # storage_failed 锁存 (含请求中途由活动包装器锁存): 零摘要调用
            return False
        try:
            plan = self._compaction_plan()
        except RuntimeError:
            return False
        if quota == "admission":
            if not policy.try_consume_admission_auto_attempt():
                return False
        elif quota == "overflow":
            if not policy.try_consume_overflow_retry():
                return False
        else:
            raise ValueError(f"unknown auto-compaction quota: {quota!r}")
        token = self._cancel_token
        assert token is not None, "auto compaction belongs to an execute attempt"
        self._compacting = True
        self._compact_done = asyncio.get_running_loop().create_future()
        self._compact_token = token
        try:
            compacted = await self._run_compaction(plan, reason="auto", cancel_token=token)
            return compacted is not None
        except RuntimeError:
            # 摘要失败/提交失败: 投影不变; 真实落盘故障已在 _run_compaction 锁存
            return False
        finally:
            self._compacting = False
            self._compact_token = None
            if self._compact_done is not None and not self._compact_done.done():
                self._compact_done.set_result(None)

    def _compaction_plan(self) -> _CompactionPlan:
        """D4: 当前代次、已提交完整请求、保留最近 1 个; 摘要输入自身须满足预算.

        覆盖起点 = 当前投影保留起点 (再次压缩只追加新区间, 不重写旧摘要);
        区间按请求边界收缩 (最长最旧前缀), 最旧单个请求也装不下时明确拒绝。
        摘要输入是原始规范历史 (D5: 不经 RunPolicy/BudgetPolicy 投影)。
        """
        config = self._budget_policy.summary_budget
        assert config is not None
        spans = self._committed_spans
        if len(spans) < 2:
            raise RuntimeError(
                "nothing to compact: at least two committed full requests are required "
                "(the most recent one is always retained)"
            )
        start = self._context_projection.retained_start
        eligible = [span for span in spans[:-1] if span[2] > start]
        if not eligible:
            raise RuntimeError(
                "nothing to compact: the committed requests before the retained "
                "boundary are already summarized"
            )
        for span in eligible:
            if span[1] < start:
                raise RuntimeError(
                    "compaction bookkeeping is inconsistent with the projection"
                )
        summary_system = summary_prompt_text()
        chosen_span: tuple[int, int, int] | None = None
        for span in reversed(eligible):
            range_messages = summary_input_messages(
                tuple(self._history[start : span[2]])
            )
            input_estimate, components = estimate_input_components(
                summary_system, (), range_messages, config
            )
            report = assemble_report(
                config, input_estimate=input_estimate, input_components=components,
                source="estimated",
            )
            if report.total_estimate <= config.num_ctx:
                chosen_span = span
                break
        if chosen_span is None:
            raise RuntimeError(
                "compaction unavailable: even the oldest single request does not fit "
                "the summary request budget"
            )
        return _CompactionPlan(
            messages=summary_input_messages(
                tuple(self._history[start : chosen_span[2]])
            ),
            first_request_seq=eligible[0][0],
            last_request_seq=chosen_span[0],
            message_count=chosen_span[2] - start,
            start_index=start,
            end_index=chosen_span[2],
        )

    async def _commit_summary_dispatch(self, record: SummaryRecord) -> None:
        """PA05 接受落定点: 封口核对与 store 派发之间无 await, 不可交错.

        store._dispatch 的 closing 检查与入队同为同步前缀, 因此"封口先 → 零
        写入 / 接受先 → 等明确结果"在本协程首个执行块内一次判定;
        create_task 不越过核对 (任务未起跑时封口先生效即零提交)。
        """
        if self._compact_sealed:
            raise CompactionSealedError(
                "compaction is sealed by close; the summary commit was not sent"
            )
        await self._store.commit_summary(record)

    # --- 关闭 ----------------------------------------------------------------------

    async def aclose(self) -> None:
        """幂等关闭; 重复调用共享同一次关闭任务和结果, 不重复 stop."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.get_running_loop().create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        # 关闭意图已锁存 (调用方 UI 应先行恢复其输入/光标模式); closing 不可逆
        self._closing = True
        if self._state == "idle":
            self._state = "closing"
        deadline = time.monotonic() + CLOSE_SUPERVISION_S
        self._close_deadline = deadline
        self._close_latch.set()
        # 02/PA05: 永久封住未接受的摘要提交并协作取消摘要生成;
        # 已被 store 接受的提交由其生产者等明确结果 (PA03), 不被取消掩盖
        self._compact_sealed = True
        if self._compact_token is not None:
            self._compact_token.cancel()
        # 取消活动请求并等待现有请求/清空/压缩生产者终结 (含持久提交与活动写入),
        # 共享同一截止时间; execute/clear/compact 互斥, 至多一个在途
        producers = [
            done for done in (self._active_done, self._clear_done, self._compact_done, self._model_done)
            if done is not None and not done.done()
        ]
        if producers and self._cancel_token is not None:
            self._cancel_token.cancel()
        for producer in producers:
            try:
                await asyncio.wait_for(
                    asyncio.shield(producer),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                self._close_problems.append(
                    "the active operation did not finalize before the close supervision deadline"
                )
                break  # 截止时间已耗尽; 未完成生产者保持强引用与关闭诊断
        # 解除核心到应用的订阅; 应用 observer 保留到 ApplicationClosed 送达后
        self._unsubscribe_core()
        stop_error: str | None = None
        try:
            await self._app.runtime.stop()
        except BaseException as exc:
            stop_error = f"runtime stop failed: {exc!r}"
        self._closed = True
        self._state = "closed"
        problems = list(self._close_problems)
        if stop_error is not None:
            problems.append(stop_error)
        self._notify(ApplicationClosed("; ".join(problems) if problems else None))
        self._observers.clear()


def _unavailable_view(reason: str) -> VerificationView:
    return VerificationView(
        verification_run_id="unavailable",
        snapshot_ref=None,
        scope_id=None,
        checks=(),
        receipts=(),
        process_uncertain=False,
        blocking_reasons=(reason,),
    )


def _incomplete_evidence(reason: str) -> ChangeEvidence:
    return ChangeEvidence(
        baseline_snapshot_ref=None,
        snapshot_ref=None,
        scope_id=None,
        changes=(),
        artifact_path=None,
        artifact_sha256=None,
        complete=False,
        failure_kind="supervision_timeout",
        error=reason,
    )


def _tool_fingerprint(app: App) -> str:
    """bootstrap 解析后的实际工具声明指纹: 稳定顺序 + 完整 schema 内容."""
    tools = getattr(app.agent, "_tools", {})
    specs = [
        {
            "name": tools[name].spec.name,
            "description": tools[name].spec.description,
            "parameters": tools[name].spec.parameters,
        }
        for name in sorted(tools)
    ]
    encoded = json.dumps(specs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_skills_prompt_identity(config: "ApplicationConfig") -> None:
    """Skills 启用时的纯配置一致性校验 (F1 处置; 不做任何 I/O).

    实际 system_prompt 的 hash 必须对应 prompt_info, 目录段必须由同一快照构造,
    版本必须是 Skills 版本; 有 PersistenceConfig 时持久身份的 prompt 字段必须与
    实际 PromptInfo 相等 (adopt_current_config 不豁免)。错配 = 启动错误。
    """
    info = config.prompt_info
    if info.version != SKILLS_PROMPT_VERSION:
        raise ValueError(
            f"skills enabled requires prompt version {SKILLS_PROMPT_VERSION}, got {info.version}"
        )
    actual = hashlib.sha256(config.system_prompt.encode("utf-8")).hexdigest()
    if actual != info.prompt_sha256:
        raise ValueError("prompt_info.prompt_sha256 does not match the actual system_prompt")
    if catalog_prompt_section(config.skills) not in config.system_prompt:
        raise ValueError(
            "system_prompt does not contain the catalog section of the configured skills snapshot"
        )
    persistence = config.persistence
    if persistence is None:
        return
    identity = persistence.identity
    mismatches = []
    if identity.prompt_version != info.version:
        mismatches.append("prompt_version")
    if identity.prompt_sha256 != info.prompt_sha256:
        mismatches.append("prompt_sha256")
    if identity.instructions_sha256 != info.instructions_sha256:
        mismatches.append("instructions_sha256")
    if mismatches:
        raise ValueError(
            "persistence identity prompt fields do not match the actual PromptInfo: "
            + ", ".join(mismatches)
        )


def _verify_skills_assembly(app: App, config: "ApplicationConfig") -> None:
    """bootstrap 后核对实际装配: 同一快照/工具注册/指纹含 load_skill (契约 §5)."""
    try:
        catalog = app.runtime.capability(CATALOG_CAPABILITY)
    except Exception as exc:
        raise ApplicationOpenError(f"skills catalog capability missing: {exc}") from exc
    if catalog != config.skills:
        raise ApplicationOpenError(
            "actual skills catalog capability does not match the configured snapshot "
            "(root/manifest/content identity)"
        )
    tools = getattr(app.agent, "_tools", {})
    if LOAD_SKILL_TOOL_NAME not in tools:
        raise ApplicationOpenError(
            f"the {LOAD_SKILL_TOOL_NAME} tool is not registered while skills are enabled"
        )


def verification_checks_signature(checks) -> str:
    """检查定义的稳定摘要; 进入 LaunchIdentity.checks_signature 参与恢复比对 (03)."""
    payload = [
        {"check_id": check.check_id, "command": check.command, "timeout_s": check.timeout_s}
        for check in checks
    ]
    return hashlib.sha256(dumps_strict(payload).encode("utf-8")).hexdigest()


def _receipt_fingerprint(check_id: str, command: str, timeout_s: float) -> str:
    return hashlib.sha256(dumps_strict(
        {"check_id": check_id, "command": command, "timeout_s": timeout_s}
    ).encode("utf-8")).hexdigest()


def _receipt_facts_mismatch(receipt: StoredReceipt, details: dict) -> str | None:
    """消息 details 的终结事实与回执记录不一致 -> 返回原因; 一致返回 None.

    契约 §5「事实不匹配……属于恢复格式/关系错误」: 缺键不比 (None 容忍),
    显式事实冲突 (exit_code/timed_out/cancelled/output_complete) 拒绝恢复.
    """
    exit_code = details.get("exit_code")
    if (
        isinstance(exit_code, int) and not isinstance(exit_code, bool)
        and receipt.exit_code != exit_code
    ):
        return f"exit_code {exit_code} but the receipt records {receipt.exit_code}"
    for key in ("timed_out", "cancelled", "output_complete"):
        value = details.get(key)
        if isinstance(value, bool) and getattr(receipt, key) != value:
            return f"{key} is {value} but the receipt records {getattr(receipt, key)}"
    return None


def _build_prefix_proof(snapshot: RecoverySnapshot) -> HistoricalProjection:
    """从恢复事实构建前缀证明容器; 回执引用缺失/错挂/事实不符属恢复关系错误, 模型零请求."""
    receipts_by_id = {receipt.receipt_id: receipt for receipt in snapshot.restored_receipts}
    entries: list[PrefixEntry] = []
    for request in snapshot.requests:
        if request.context_generation != snapshot.context_generation:
            continue
        for message in request.messages:
            receipt_id: str | None = None
            verification_run_id: str | None = None
            if isinstance(message, ToolResultMessage) and message.result.name == "check":
                details = (
                    message.result.details if isinstance(message.result.details, dict) else {}
                )
                candidate = details.get("receipt_id")
                if isinstance(candidate, str) and candidate:
                    receipt = receipts_by_id.get(candidate)
                    if receipt is None:
                        raise ApplicationOpenError(
                            f"restored history references check receipt {candidate} "
                            "that is not in the archive (relation error)"
                        )
                    if receipt.request_id != request.request_id:
                        raise ApplicationOpenError(
                            f"restored receipt {candidate} does not belong to its request"
                        )
                    mismatch = _receipt_facts_mismatch(receipt, details)
                    if mismatch is not None:
                        raise ApplicationOpenError(
                            f"restored check result contradicts receipt {candidate}: {mismatch}"
                        )
                    receipt_id = candidate
                    verification_run_id = receipt.verification_run_id
            entries.append(PrefixEntry(
                epoch_id=request.epoch_id,
                request_id=request.request_id,
                seq=request.seq,
                message_sha256=hashlib.sha256(
                    encode_messages((message,)).encode("utf-8")
                ).hexdigest(),
                receipt_id=receipt_id,
                verification_run_id=verification_run_id,
            ))
    # 显式关系校验: 证明逐位置背书的内容必须就是实际注入的规范历史
    # (隐式依赖 store 两条推导路径一致不够; 分叉按 fail-closed 拒绝, 见 A09)
    if len(entries) != len(snapshot.canonical_history):
        raise ApplicationOpenError(
            "restored requests do not reconstruct the canonical history (relation error)"
        )
    for entry, message in zip(entries, snapshot.canonical_history):
        observed = hashlib.sha256(
            encode_messages((message,)).encode("utf-8")
        ).hexdigest()
        if observed != entry.message_sha256:
            raise ApplicationOpenError(
                "restored requests do not reconstruct the canonical history (relation error)"
            )
    return HistoricalProjection(HistoricalPrefixProof(entries=tuple(entries)))


def _rebuild_context_projection(snapshot: RecoverySnapshot) -> tuple[SummaryBlock, ...]:
    """按摘要记录确定性重建投影块 (02, D6/A08); 关系/指纹不符属恢复错误, 模型零请求.

    逐块核对: generation 与当前代次一致、seq 引用当前代次已提交完整请求、
    覆盖前缀 [0:end) 的规范编码 sha256 与规范历史逐位一致。块消息由
    summary_block_message 程序生成, 与压缩时字节逐位一致。
    """
    generation_messages = snapshot.canonical_history
    seqs = {
        request.seq
        for request in snapshot.requests
        if request.context_generation == snapshot.context_generation
    }
    blocks: list[SummaryBlock] = []
    for record in snapshot.summaries:
        if record.context_generation != snapshot.context_generation:
            raise ApplicationOpenError(
                f"restored summary {record.summary_id} belongs to generation "
                f"{record.context_generation}, not the current {snapshot.context_generation}"
            )
        for seq in (record.first_request_seq, record.last_request_seq):
            if seq not in seqs:
                raise ApplicationOpenError(
                    f"restored summary {record.summary_id} references request {seq} "
                    "outside the current generation (relation error)"
                )
        if record.end_message_index > len(generation_messages):
            raise ApplicationOpenError(
                f"restored summary {record.summary_id} coverage end exceeds the "
                "canonical history (relation error)"
            )
        observed = canonical_prefix_sha256(generation_messages[: record.end_message_index])
        if observed != record.prefix_sha256:
            raise ApplicationOpenError(
                f"restored summary {record.summary_id} coverage does not match the "
                "canonical history (relation error)"
            )
        blocks.append(SummaryBlock(
            message=summary_block_message(
                first_request_seq=record.first_request_seq,
                last_request_seq=record.last_request_seq,
                message_count=record.message_count,
                text=record.text,
            ),
            end_message_index=record.end_message_index,
            first_request_seq=record.first_request_seq,
            last_request_seq=record.last_request_seq,
            message_count=record.message_count,
            prefix_sha256=record.prefix_sha256,
        ))
    return tuple(blocks)


def _bind_store_last_cleanup(
    definitions: list[PluginDefinition],
) -> list[PluginDefinition]:
    """持久拓扑: 每个资源插件显式 require session.store (契约 §1)。

    store 最先激活 -> runtime 逆序清理时模型/进程等资源先清理、store 最后释放;
    依赖方向显式可核对, 不靠注册顺序推断。
    """
    return [
        definition if "session.store" in definition.provides
        else replace(definition, requires=definition.requires | {"session.store"})
        for definition in definitions
    ]


def _open_persistent_store(
    config: ApplicationConfig,
) -> tuple[SQLiteSessionStore, RecoverySnapshot]:
    """打开或新建会话存储; 失败时已取得资源由本函数释放, 统一映射为启动错误."""
    persistence = config.persistence
    assert persistence is not None
    try:
        store = open_session_store(
            directory=Path(config.workspace) / ".cicada" / "sessions",
            session_id=persistence.session_id,
            workspace=config.workspace,
        )
    except PersistenceError as exc:
        raise ApplicationOpenError(f"session store open failed: {exc}") from exc
    return store, store.recovery_snapshot()


def _admit_persistent_resume(
    persistence: PersistenceConfig, snapshot: RecoverySnapshot
) -> str | None:
    """恢复准入: 配置冲突在 bootstrap 前拒绝 (无副作用); 返回 view-only 原因或 None.

    无未证终结活动的中断请求可以继续 (pending intent 只标中断并显示);
    存在未证终结/未闭合活动时恢复为查看状态, 不能经模式或配置覆盖解除 (A15).
    """
    if persistence.session_id is None:
        return None
    saved_identity = snapshot.identity
    if saved_identity is None:
        raise ApplicationOpenError(
            "the session archive has no saved launch identity to compare against"
        )
    if not persistence.adopt_current_config:
        conflicts = conflicting_launch_fields(saved_identity.launch, persistence.identity)
        if conflicts:
            raise ApplicationOpenError(
                "restored session config conflicts with the current launch: " + ", ".join(conflicts)
            )
    unproven = snapshot.unproven_process_activities()
    if unproven:
        first = unproven[0]
        return (
            f"restored session has {len(unproven)} process activity without proven "
            f"termination (first: {first.tool_name} call {first.call_id}, "
            + (
                f"reason: {first.terminal.reason}"
                if first.terminal is not None and first.terminal.reason
                else "no terminal record"
            ) + ")"
        )
    untrusted = [
        receipt for receipt in snapshot.restored_receipts
        if not receipt_terminal_trustworthy(receipt)
    ]
    if untrusted:
        # 存档级 process_uncertain 证据 (契约 D3): 检查回执终结不可信, 仅可查看
        return (
            f"restored archive records process_uncertain "
            f"({len(untrusted)} check receipt without trustworthy terminal facts; "
            f"first: receipt {untrusted[0].receipt_id})"
        )
    return None


async def _run_preflight(preflight: Callable[[], object] | None) -> None:
    """恢复路径的模型可达性探测: 必须在格式/配置预检之后、bootstrap 之前执行."""
    if preflight is None:
        return
    if not callable(preflight):
        raise TypeError("preflight must be a callable or None")
    outcome = preflight()
    if inspect.isawaitable(outcome):
        await outcome


async def open_application(
    config: ApplicationConfig, *, preflight: Callable[[], object] | None = None
) -> ApplicationSession:
    """输入/preflight/boot/initialize 成功才返回; 部分启动失败关闭已取得资源.

    持久模式次序: 校验配置及 workspace -> 独占 lease -> 打开/校验存档 ->
    恢复准入与已知配置冲突 -> preflight -> bootstrap (store 插件注册首位, 最后清理) ->
    核对实际工具指纹 -> 写入新的启动代次 -> 注入闭合历史并发布会话.
    与契约 §1 的记载差异: 契约要求注入先于启动代次, 实现中 epoch_id 是会话构造
    参数故 start_epoch 先行, 语义等价 (会话发布前完成全部写入); 偏差记录见
    shared-boundary-handoff.md §4.
    """
    plan = config.plan
    budget_config = config.budget
    policy_state: dict[str, object] = {}
    # 02/F1: Skills 启用时的纯配置一致性校验 (提示/身份绑定) 先于一切应用发起 I/O
    if config.skills is not None:
        try:
            _validate_skills_prompt_identity(config)
        except ValueError as exc:
            raise ApplicationOpenError(f"skills config invalid: {exc}") from exc
    # 恢复前缀证明容器 (03): 检查模式 + 持久会话时由 RunPolicy 每次发送前核对
    history_projection: HistoricalProjection | None = None
    # 模型可见投影容器 (02, D4): 预算启用即存在 (空 = 恒等投影); 压缩/clear/恢复
    # 由应用驱动, BudgetPolicy 与会话持同一实例
    context_projection: ContextProjection | None = None
    if budget_config is not None:
        context_projection = ContextProjection()

    def _make_budget_policy(model, binding=None):
        # D1 次序: BudgetPolicy 在 RunPolicy 下游、适配器上游;
        # 计量因此覆盖验证投影 (控制段/freshness 标记) 的最终形态
        budget = BudgetPolicy(
            model,
            budget_config,
            # D5: 摘要通道注入同一无状态适配器实例; 仅持久会话可压缩 (D9)
            summarize=(policy_state["summary_route"] if binding is not None else model) if store is not None else None,
            summary_budget=binding.summary_budget if binding is not None else None,
            summary_identity=binding.summary_identity if binding is not None else None,
            validate_request=binding.validate_request if binding is not None else None,
            # 03/D7: overflow 分类器只接 ollama 形态的适配器——分类按上游
            # 版本化错误文本形态匹配, 其他适配器接线属误配 (不匹配即不重试)
            classify_overflow=(
                classify_context_overflow if isinstance(model, OllamaModel) else None
            ),
            projection=context_projection,
        )
        policy_state["budget_policy"] = budget
        return budget

    def _model_policy(model, runtime):
        nonlocal config, budget_config, context_projection
        binding = model.initial_binding if isinstance(model, ModelSelectionPort) else None
        if binding is not None:
            policy_state["model_selection"] = model
            policy_state["model_route"] = ModelRoute(binding)
            policy_state["summary_route"] = ModelRoute(binding, summary=True)
            budget_config = binding.budget
            config = replace(config, budget=budget_config, model_label=binding.identity.model)
            if context_projection is None:
                context_projection = ContextProjection()
            model = policy_state["model_route"]
        if budget_config is not None:
            model = _make_budget_policy(model, binding)
        if plan is not None:
            service = runtime.capability(VERIFICATION_CAPABILITY)
            model = verification_policy(model, service, history=history_projection)
            policy_state["policy"] = model
            policy_state["service"] = service
        return model

    # D1 四态: 检查+预算 = verification_policy(budget_policy(inner));
    # 非检查+预算 = 仅 budget_policy(inner) (无 RunPolicy、不取 verification capability);
    # 仅检查 = 现状; 皆无 = 不装 model_policy
    model_policy = _model_policy

    store: SQLiteSessionStore | None = None
    snapshot: RecoverySnapshot | None = None
    if config.persistence is not None:
        store, snapshot = _open_persistent_store(config)
    if store is not None and plan is not None:
        history_projection = HistoricalProjection()

    # 活动包装器经 tool_policy 装入; 协调器在会话构造后绑定 (契约 §3)
    coordinator = ActivityCoordinator()
    tool_policy = None
    if store is not None:
        assert store is not None

        def tool_policy(tool, runtime, _store=store, _coordinator=coordinator):
            return ProcessActivityWrapper(tool, _store, _coordinator)

    # 02: Skills 启用时实际注册工具在既有名单上追加 load_skill
    tool_capabilities: tuple[str, ...] = (
        CHECK_TOOL_CAPABILITIES if plan is not None else TOOL_CAPABILITIES
    )
    if config.skills is not None:
        tool_capabilities = tool_capabilities + ("tool.load_skill",)

    definitions: list[PluginDefinition] = []
    if store is not None:
        definitions.append(session_store_plugin(store))
        definitions.extend(_bind_store_last_cleanup(
            default_definitions(config.workspace, config.model_plugin, plan, skills=config.skills)
        ))
    else:
        definitions.extend(
            default_definitions(config.workspace, config.model_plugin, plan, skills=config.skills)
        )

    if store is not None:
        assert snapshot is not None and config.persistence is not None
        try:
            view_only_reason = _admit_persistent_resume(config.persistence, snapshot)
            await _run_preflight(preflight)
            app: App = await bootstrap(
                definitions,
                tool_capabilities=tool_capabilities,
                system_prompt=config.system_prompt,
                model_policy=model_policy,
                tool_policy=tool_policy,
                max_turns=config.max_turns,
            )
        except BootError as exc:
            await store.aclose()
            raise ApplicationOpenError(f"boot failed: {exc}") from exc
        except BaseException:
            await store.aclose()
            raise
    else:
        try:
            app = await bootstrap(
                definitions,
                tool_capabilities=tool_capabilities,
                system_prompt=config.system_prompt,
                model_policy=model_policy,
                max_turns=config.max_turns,
            )
        except BootError as exc:
            raise ApplicationOpenError(f"boot failed: {exc}") from exc
    if config.skills is not None:
        # 02/契约 §5: 实际装配核对 — 同一快照/工具注册/工具指纹含 load_skill
        try:
            _verify_skills_assembly(app, config)
        except ApplicationOpenError:
            await app.aclose()
            raise

    if store is not None and snapshot is not None and config.persistence is not None:
        # 指纹核对/启动代次/会话构造的每条失败路径都必须释放 app (store 随 runtime 清理)
        try:
            fingerprint = _tool_fingerprint(app)
            saved_identity = snapshot.identity
            adopt = config.persistence.adopt_current_config
            if saved_identity is not None and not adopt:
                if saved_identity.tool_fingerprint != fingerprint:
                    raise ApplicationOpenError(
                        "restored session tool declarations differ from the current launch "
                        "(tool_fingerprint)"
                    )
            epoch_launch = config.persistence.identity
            if (
                saved_identity is not None
                and adopt
                and config.persistence.session_id is not None
            ):
                # 04: 显式采用当前配置——身份/指纹差异不拒绝, 差异记入新 epoch 的
                # 身份来源 (sources 只作记录, 不参与同配置比对)
                changed = conflicting_launch_fields(saved_identity.launch, epoch_launch)
                sources = {**epoch_launch.sources, "resume_config": "current"}
                if changed:
                    sources["changed_fields"] = ",".join(changed)
                epoch_launch = replace(epoch_launch, sources=sources)
            try:
                epoch_id = await store.start_epoch(
                    StoredIdentity(launch=epoch_launch, tool_fingerprint=fingerprint)
                )
            except PersistenceError as exc:
                raise ApplicationOpenError(f"session epoch start failed: {exc}") from exc
            # start_epoch 已把 pending intent 标中断: 重取点时快照, 会话交接事实与存档一致
            snapshot = store.recovery_snapshot()
            if history_projection is not None:
                try:
                    built = _build_prefix_proof(snapshot)
                except ApplicationOpenError:
                    raise
                except Exception as exc:
                    raise ApplicationOpenError(
                        f"restored prefix proof build failed: {exc}"
                    ) from exc
                history_projection.proof = built.proof
            if context_projection is not None:
                # 02/A08: 按摘要记录确定性重建投影; 覆盖/指纹不符属恢复关系错误
                try:
                    rebuilt_blocks = _rebuild_context_projection(snapshot)
                except ApplicationOpenError:
                    raise
                except Exception as exc:
                    raise ApplicationOpenError(
                        f"restored context projection rebuild failed: {exc}"
                    ) from exc
                context_projection.adopt(rebuilt_blocks)
            session = ApplicationSession(
                app=app,
                config=config,
                service=policy_state.get("service"),  # type: ignore[arg-type]
                policy=policy_state.get("policy"),  # type: ignore[arg-type]
                store=store,
                epoch_id=epoch_id,
                persistence_mode="resume" if config.persistence.session_id is not None else "save",
                restored_history=snapshot.canonical_history,
                restored_requests=snapshot.requests,
                restored_context_generation=snapshot.context_generation,
                restored_interrupted=snapshot.interrupted,
                restored_receipts=snapshot.restored_receipts,
                restored_summaries=snapshot.summaries,
                view_only_reason=view_only_reason,
                projection=history_projection,
                budget_policy=policy_state.get("budget_policy"),  # type: ignore[arg-type]
                model_selection=policy_state.get("model_selection"),
                model_route=policy_state.get("model_route"),
                summary_route=policy_state.get("summary_route"),
                context_projection=context_projection,
            )
            # 03/D7: 自动压缩回调在会话构造后绑定 (回调是会话的边界方法);
            # 仅持久会话接线 (D9), 非持久 auto_compact 保持 None=无自动行为
            _wired = policy_state.get("budget_policy")
            if _wired is not None:
                _wired.set_auto_compact(session._auto_compact_for_request)
            coordinator.bind(session)
        except ApplicationOpenError:
            await app.aclose()
            raise
        except BaseException as exc:
            await app.aclose()
            if isinstance(exc, Exception):
                raise ApplicationOpenError(f"session bootstrap failed: {exc}") from exc
            raise
    else:
        session = ApplicationSession(
            app=app,
            config=config,
            service=policy_state.get("service"),  # type: ignore[arg-type]
            policy=policy_state.get("policy"),  # type: ignore[arg-type]
            budget_policy=policy_state.get("budget_policy"),  # type: ignore[arg-type]
            model_selection=policy_state.get("model_selection"),
            model_route=policy_state.get("model_route"),
            summary_route=policy_state.get("summary_route"),
            context_projection=context_projection,
        )
    if plan is None:
        return session
    try:
        capture = await session._service.initialize(CancelToken())  # type: ignore[union-attr]
    except BaseException:
        await app.aclose()
        raise
    if not capture.available:
        await app.aclose()
        detail = f" ({capture.failure_kind}): {capture.error}" if capture.error else ""
        raise ApplicationOpenError(f"verification initialize failed{detail}")
    if (
        store is not None
        and snapshot is not None
        and snapshot.recorded_process_uncertain
        and session._service is not None
    ):
        # 跨恢复 process_uncertain: 应用准入与新验证服务控制段一致 (A15);
        # 清空不解除, 交付判断沿用 P4 既有锁存语义
        session._process_uncertain = True
        session._service.mark_process_uncertain(
            "restored archive records process_uncertain "
            "(check receipts without trustworthy terminal facts)"
        )
    return session
