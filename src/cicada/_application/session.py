"""运行期唯一会话协调者；持有规范历史、请求生命周期与交付判定。"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from cicada.boot import App
from cicada.approval import ApprovalRequest, Decision
from cicada.execution_mode import ExecutionModeController
from cicada.budget import (
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
    SummaryBlock,
    SummaryOutcome,
    canonical_prefix_sha256,
    generate_summary,
    build_summary_request,
    summary_block_message,
    summary_input_messages,
)
from cicada.core.agent import RunResult
from cicada.core.cancel import CancelToken
from cicada.core.events import Event, ModelCallCompleted, RunFinished, RunStarted, ToolCompleted
from cicada.core.messages import Message, UserMessage, isolate_message, isolate_tool_result
from cicada.core.ports import ModelRequest, ToolSpec
from cicada.costs import CostError, CostSnapshot
from cicada.diagnostic_export import ReportError, build_report, write_report
from cicada.models import (
    CacheDiagnosticsPort,
    CacheRequestObservation,
    ConfigurationRequest,
    ModelBinding,
    ModelConfigurationPort,
    ModelOption,
    ModelRoute,
    ModelSelectionPort,
    ModelSessionRoutingPort,
    ReasoningEffortPort,
    ReasoningEffortState,
)
from cicada.user_config import (
    LoginResult,
    LoginSubmission,
    LoginView,
    UserDefaults,
    UserConfigError,
    UserConfigStore,
)
from cicada.persistence.routing import SessionStoreRoute, settle
from cicada.persistence.contracts import (
    ArchiveLimitError,
    InterruptedRequest,
    ModelChangeSubmission,
    PersistenceState,
    PersistenceError,
    RequestOutcomeFacts,
    RequestPersistenceFacts,
    RestoredRequest,
    SessionStorePort,
    SessionCatalogPort,
    SessionCatalogSnapshot,
    StorageUnavailableError,
    StorageWriteError,
    StoredIdentity,
    StoredReceipt,
    SummaryRecord,
    conflicting_launch_fields,
    validate_session_name,
)
from cicada.run_policy import HistoricalProjection, RunPolicy
from cicada.file_completion import FileCompletionResult
from cicada.plugins.coding.file_completion import FILE_COMPLETION_CAPABILITY
from cicada.plugins.coding.change_review import REVIEW_TIMEOUT_S, ChangeReview
from cicada.review import (
    ReviewCheckDetails,
    ReviewCheckRow,
    ReviewError,
    ReviewPreview,
    ReviewSnapshot,
)
from cicada.restore_points import (
    REWIND_TIMEOUT_S,
    RewindError,
    RewindListView,
    RewindPointView,
    RewindPreview,
    RewindRestorePlan,
    RewindRestoreResult,
)
from cicada.plugins.coding.rewind_reader import RewindReader
from cicada.plugins.coding.rewind_restore import RewindRestorer
from cicada.plugins.coding.restore_store import RestoreStore, RestoreStoreError
from cicada.plugins.coding.verification_contracts import ChangeEvidence, VerificationView
from cicada.reporting import DeliveryDecision, decide_delivery
from cicada.runtime.runtime import CapabilityError
from cicada.skills import SkillCatalogView, SkillSelectionError, build_skill_request, find_skill

from .contracts import (
    AgentEventReceived,
    ApplicationClosed,
    ApplicationConfig,
    ApplicationUpdate,
    ApprovalRequested,
    CompactionSealedError,
    ConfigurationChanged,
    ContextCompacted,
    ExecutionModeChanged,
    HistoryCleared,
    LoginCompleted,
    ModelChanged,
    ModelSelectionError,
    OperationBusy,
    PersistenceConfig,
    ReasoningEffortChanged,
    RequestCompleted,
    RequestOutcome,
    RequestPhaseChanged,
    RewindRestoreCompleted,
    SessionChanged,
    SessionManagementError,
    SessionManagementResult,
    SessionManagementSubmission,
    SessionNameChanged,
    SessionStateName,
    _runtime_dir,
)
from . import recovery
from .ownership import FOREIGN_OWNERS, OperationLedger, busy_refusal


REQUEST_FINALIZE_SUPERVISION_S = 90.0


CLOSE_SUPERVISION_S = 15.0


_UNDEAD_GRACE_S = 2.0


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
class _CompactionPlan:
    """一次手动压缩的切点计划 (D4): 覆盖区间与保留边界."""

    request: ModelRequest             # 完整摘要输入: 原始历史区间及固定最后 user
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
    if isinstance(update, SessionChanged):
        from copy import deepcopy
        return replace(update, snapshot=deepcopy(update.snapshot))
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


_ARTIFACT_NOTICES = {
    "missing": "工件缺失或不可读",
    "mismatch": "工件与记录 hash 不符",
    "over_limit": "工件超过全量核验限额",
    "changed": "工件在核验期间变化",
    "unreadable": "工件不可读",
    "timeout": "工件核验超出时间预算",
    "binary": "二进制工件不显示正文",
}


@dataclass(frozen=True)
class _ReviewTarget:
    """审阅页登记的检查/工件目标; 只由应用从真实事实构造, 不接受 UI 输入."""

    kind: str  # "checks" | "artifact"
    path: Path | None = None
    expected_sha256: str | None = None
    evidence: ChangeEvidence | None = None


def _evidence_label(evidence: ChangeEvidence | None) -> str | None:
    """complete/failure 事实的有界显示; 不掩盖不完整工件, 无证据为 None."""
    if evidence is None:
        return None
    if evidence.complete:
        return "累计变更工件（生成时完整）"
    return f"累计变更工件（生成时不完整 · {evidence.failure_kind or 'unknown'}）"


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
        login_store_factory: Callable[[], UserConfigStore] | None = None,
        login_environment: Callable[[], str | None] | None = None,
        review_reader: ChangeReview | None = None,
        catalog_reader: SessionCatalogPort | None = None,
        new_store_factory: Callable[[], SessionStorePort] | None = None,
        resume_store_factory: Callable[[str], SessionStorePort] | None = None,
        restore_store: RestoreStore | None = None,
        restore_warnings: tuple[str, ...] = (),
        mode_controller: ExecutionModeController | None = None,
    ) -> None:
        self._app = app
        self._config = config
        self._service = service
        self._policy = policy
        self._mode_controller = mode_controller or ExecutionModeController()
        self._budget_policy = budget_policy
        self._model_selection = model_selection
        self._model_route = model_route
        self._summary_route = summary_route
        self._review_reader = review_reader
        self._catalog_reader = catalog_reader
        # P0-3: 单一所有权令牌; 页面类事务 (审阅/还原点/目录) 持租约到页面关闭
        self._ownership = OperationLedger()
        self._catalog_task: asyncio.Task | None = None
        self._catalog_token: CancelToken | None = None
        self._file_queries: dict[asyncio.Task, CancelToken] = {}
        self._rewind_reader: RewindReader | None = None
        self._rewind_restorer: RewindRestorer | None = None
        self._rewind_task: asyncio.Task | None = None
        self._rewind_token: CancelToken | None = None
        self._review_id: str | None = None
        self._review_snapshot: ReviewSnapshot | None = None
        self._review_token: CancelToken | None = None
        self._review_task: asyncio.Task | None = None
        # 02 (C8): 最近本进程 outcome 的隔离只读引用与登记目标; 恢复不建立,
        # clear 不清除。页面不持久化, 旧工件不因此获得当前交付权威。
        self._last_outcome: RequestOutcome | None = None
        self._last_request_model: str | None = None
        self._review_targets: dict[str, _ReviewTarget] = {}
        # 04 (C6): 最新接受的配置准备源; 发布时按结构能力更新, 不从启动缓存准备
        self._model_configuration: ModelConfigurationPort | None = None
        if model_route is not None and isinstance(model_route.binding.model, ModelConfigurationPort):
            self._model_configuration = model_route.binding.model
        self._model_done: asyncio.Future | None = None
        self._model_task: asyncio.Task | None = None
        self._session_done: asyncio.Future | None = None
        self._session_task: asyncio.Task | None = None
        self._session_token: CancelToken | None = None
        self._session_generation = 0
        self._login_store_factory = login_store_factory
        self._login_environment = login_environment
        self._login_store: UserConfigStore | None = None
        self._login_view: LoginView | None = None
        self._login_task: asyncio.Task | None = None
        self._login_save_task: asyncio.Task | None = None
        self._login_cancel_requested = False
        # 模型可见投影容器 (02, D4): 压缩成功后追加摘要块, clear 重置, 恢复重建;
        # 与 BudgetPolicy 持同一实例
        self._context_projection = context_projection if context_projection is not None else ContextProjection()
        self._history: list[Message] = [isolate_message(m) for m in restored_history]
        self._store = store
        if isinstance(self._model_selection, ModelSessionRoutingPort):
            transient_session_id = str(uuid.uuid4())
            self._model_selection.set_session_id_provider(
                lambda: self._store.session_id if self._store is not None else transient_session_id)
        self._new_store_factory = new_store_factory
        self._resume_store_factory = resume_store_factory
        # 请求级快照与撤回 01: 捕获汇 (仅持久会话 + restore_capture 开启时非 None);
        # None 时 begin/note_commit/end 三处接线全部短路, 行为与既有逐位一致
        self._restore_store = restore_store
        # C3: 启动机会式清理的告警事实 (只读呈现; 清理失败不阻断启动)
        self._restore_warnings: tuple[str, ...] = tuple(restore_warnings)
        self._session_cleanup_failed: str | None = None
        self._request_store: SessionStorePort | None = None
        self._request_epoch: str | None = None
        self._request_generation = 0
        self._receipt_owners: dict[str, tuple[str, str, str]] = {}
        self._request_receipts_before: frozenset[str] = frozenset()
        self._process_model_calls: list[ModelCallCompleted] = []
        self._process_summary_calls: list[SummaryOutcome] = []
        self._user_name = store.recovery_snapshot().user_name if store is not None else None
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
        # 02: 手动压缩生产者 (D6/PA05); close 封住未接受提交并协作取消生成
        self._compact_done: asyncio.Future | None = None
        self._compact_token: CancelToken | None = None
        self._compact_sealed = False
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

    def _in_flight_workers(self) -> tuple[asyncio.Future, ...]:
        """令牌已让出、但真实 IO/提交尚未落定的生产者.

        与所有权是两件事: 事务可以在 finally 里归还令牌, 而线程里的事务还没
        结束; 开新页面/导出要求两者都静默。
        """
        return tuple(
            worker for worker in (
                self._active_done, self._clear_done, self._compact_done, self._model_done,
                self._login_task, self._login_save_task, self._review_task,
                self._rewind_task, *self._undead_tasks,
            ) if worker is not None and not worker.done()
        )

    @property
    def rewind_active(self) -> bool:
        return self._ownership.holds("rewind")

    async def open_rewind(self) -> RewindListView:
        if (self._closing or self._closed or self._state != "idle"
                or self._ownership.owner not in (None, "rewind")
                or self._in_flight_workers()
                or (self.rewind_active and self._rewind_reader is None)):
            raise RewindError("busy", "查看还原点需要空闲会话，等当前操作实际结束后再打开。")
        if self._store is None:
            raise RewindError("unavailable", "还原点查看需要持久会话。")
        # 关闭捕获只影响后继写入，仍可只读查看本会话已有历史点。
        store = self._restore_store or RestoreStore(
            Path(self._config.workspace), data_dir=_runtime_dir(self._config))
        workspace = self._app.runtime.capability("coding.workspace")
        reader = RewindReader(store, workspace, self._store.session_id, self._epoch_id)
        if not self.rewind_active:
            self._ownership.claim("rewind")
        self._rewind_reader = reader
        self._rewind_restorer = RewindRestorer(reader)
        try:
            return await self._rewind_io(reader.capture)
        except BaseException:
            self.cancel_rewind()
            raise

    async def read_rewind_point(self, point_id: str) -> RewindPointView:
        reader = self._require_rewind()
        return await self._rewind_io(lambda token: reader.read_point(point_id, token))

    async def read_rewind_preview(self, point_id: str, entry_id: str) -> RewindPreview:
        reader = self._require_rewind()
        return await self._rewind_io(lambda token: reader.preview(point_id, entry_id, token))

    def _admit_rewind_write(self) -> RewindRestorer:
        self._require_rewind()
        if (self._process_uncertain or self._storage_failed or self._view_only_reason
                or self._session_cleanup_failed):
            raise RewindError("blocked", "会话处于写入阻断状态，只能查看还原点。")
        if self._rewind_restorer is None:
            raise RewindError("invalid_target", "请重新打开还原点页面。")
        return self._rewind_restorer

    async def plan_rewind_restore(self, point_id: str) -> RewindRestorePlan:
        restorer = self._admit_rewind_write()
        return await self._rewind_io(lambda token: restorer.prepare(point_id, token))

    async def execute_rewind_restore(self, plan_id: str) -> RewindRestoreResult:
        restorer = self._admit_rewind_write()
        return await self._rewind_io(lambda token: restorer.execute(plan_id, token),
                                     completed=self._rewind_restore_completed)

    def _rewind_restore_completed(self, result: RewindRestoreResult) -> None:
        workspace = self._app.runtime.capability("coding.workspace")
        workspace.files.invalidate_read_basis(r.relative_path for r in result.files)
        self._notify(RewindRestoreCompleted(result))

    def _require_rewind(self) -> RewindReader:
        if not self.rewind_active or self._rewind_reader is None or self._closing:
            raise RewindError("invalid_target", "还原点页面已关闭。")
        if self._rewind_task is not None and not self._rewind_task.done():
            raise RewindError("busy", "还原点读取尚未结束。")
        return self._rewind_reader

    async def _rewind_io(self, operation, *, completed=None):
        reader = self._rewind_reader
        token = CancelToken()
        task = asyncio.create_task(asyncio.to_thread(operation, token))
        self._rewind_task, self._rewind_token = task, token
        if completed is not None:
            def publish(done):
                if not done.cancelled() and done.exception() is None:
                    completed(done.result())
            task.add_done_callback(publish)
        task.add_done_callback(self._rewind_io_settled)
        try:
            result = await asyncio.wait_for(asyncio.shield(task), REWIND_TIMEOUT_S)
            if self._rewind_reader is not reader or self._closing:
                raise RewindError("invalid_target", "还原点页面已关闭。")
            return result
        except asyncio.TimeoutError:
            self.cancel_rewind()
            raise RewindError("unavailable", "还原点读取超时，正在等待实际读取结束。") from None
        except asyncio.CancelledError:
            self.cancel_rewind()
            raise
        except (OSError, RestoreStoreError):
            raise RewindError("unavailable", "还原点数据不可读或已损坏。") from None

    def _rewind_io_settled(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()
        if self._rewind_task is task:
            self._rewind_token = None
            if self._rewind_reader is None:
                self._ownership.release("rewind")

    def cancel_rewind(self) -> None:
        self._rewind_reader = None
        self._rewind_restorer = None
        if self._rewind_token is not None:
            self._rewind_token.cancel()
        if self._rewind_task is None or self._rewind_task.done():
            self._ownership.release("rewind")

    @property
    def review_active(self) -> bool:
        return self._ownership.holds("review")

    @property
    def catalog_active(self) -> bool:
        return self._ownership.holds("catalog")

    async def observe_session_catalog(self, cancel: CancelToken) -> SessionCatalogSnapshot:
        if self._closing or self._closed:
            raise SessionManagementError("closing", "应用正在关闭。")
        if (self._state != "idle" or self._ownership.owner is not None
                or self._in_flight_workers()):
            raise SessionManagementError("busy", "会话目录需要空闲会话。")
        if self._catalog_reader is None:
            raise SessionManagementError("unavailable", "当前组合没有会话目录能力。")
        cancel.throw_if_cancelled()
        self._ownership.claim("catalog")
        async def observe_after_files():
            for query_token in self._file_queries.values():
                query_token.cancel()
            if self._file_queries:
                await self._settle_session_work(
                    asyncio.gather(*tuple(self._file_queries), return_exceptions=True), cancel)
            cancel.throw_if_cancelled()
            return await self._catalog_reader.observe(cancel)

        task = asyncio.create_task(observe_after_files())
        self._catalog_task, self._catalog_token = task, cancel
        cancelled = False
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                    cancel.cancel()
            if cancelled or cancel.cancelled or self._closing:
                if not task.cancelled():
                    task.exception()
                raise asyncio.CancelledError
            return task.result()
        finally:
            if self._catalog_task is task:
                self._catalog_task = None
                self._catalog_token = None
            self._ownership.release("catalog")

    async def complete_files(self, query: str, cancel: CancelToken) -> FileCompletionResult:
        """只读文件候选出口 (C3): 不创建模型消息、ToolCall/ToolResult 或读取依据。

        closing/closed 先拒绝新调用; 进行中的编码请求不阻止候选查询。未装配
        coding.file_completion 的组合 (旧直接构造) 返回 capability_unavailable。
        """
        if self._closing or self._closed or self._ownership.holds("session"):
            return FileCompletionResult((), "unavailable", reason="session_closed")
        try:
            port = self._app.runtime.capability(FILE_COMPLETION_CAPABILITY)
        except CapabilityError:
            return FileCompletionResult((), "unavailable", reason="capability_unavailable")
        task = asyncio.create_task(port.complete(query, cancel))
        self._file_queries[task] = cancel
        generation = self.session_generation
        try:
            result = await self._settle_session_work(task, cancel)
            if generation != self.session_generation:
                return FileCompletionResult((), "unavailable", reason="session_closed")
            return result
        finally:
            self._file_queries.pop(task, None)

    def _check_review_admission(self) -> None:
        if (self._closing or self._closed or self._state != "idle"
                or self._ownership.owner not in (None, "review")
                or self._in_flight_workers()):
            raise ReviewError("busy", "审阅需要空闲会话，等当前操作实际结束后再打开。")
        if self.review_active and self._review_id is None:
            raise ReviewError("busy", "审阅正在取消清理。")
        if self._review_reader is None:
            raise ReviewError("unavailable", "当前组合没有审阅能力。")

    async def open_review(self) -> ReviewSnapshot:
        """同步准入、占住页面 lease 后派发一个受拥有读取；刷新不排队。"""
        self._check_review_admission()
        if not self.review_active:
            self._ownership.claim("review")
        self._review_targets = {}
        review_id = uuid.uuid4().hex
        self._review_id = review_id
        self._review_snapshot = None
        token = CancelToken()
        try:
            snapshot = await self._review_io(self._review_reader.capture(review_id, token), token)
            if self._review_id != review_id or self._closing:
                raise ReviewError("inactive", "审阅已关闭。")
            snapshot = self._apply_check_state(snapshot)
            self._review_snapshot = snapshot
            return snapshot
        except BaseException:
            self.cancel_review()
            raise

    def _apply_check_state(self, snapshot: ReviewSnapshot) -> ReviewSnapshot:
        """C8 三状态: 无 plan 未启用; 有 plan 无本进程 outcome 尚无记录; 有记录
        登记 checks/工件目标。恢复的旧回执不进入, 也不获得当前权威。"""
        if self._config.plan is None or self._service is None:
            return replace(snapshot, checks_state="not_enabled")
        if self._last_outcome is None:
            return replace(snapshot, checks_state="not_recorded")
        checks_target_id = uuid.uuid4().hex
        self._review_targets[checks_target_id] = _ReviewTarget(kind="checks")
        artifact_target_id = None
        evidence = self._last_outcome.evidence
        if (evidence is not None and evidence.artifact_path is not None
                and evidence.artifact_sha256 is not None):
            artifact_target_id = uuid.uuid4().hex
            self._review_targets[artifact_target_id] = _ReviewTarget(
                kind="artifact", path=evidence.artifact_path,
                expected_sha256=evidence.artifact_sha256, evidence=evidence)
        return replace(snapshot, checks_state="available", checks_target_id=checks_target_id)

    async def read_review(self, review_id: str, target_id: str) -> ReviewPreview | ReviewCheckDetails:
        snapshot = self._review_snapshot
        if not self.review_active or self._review_id != review_id or snapshot is None:
            raise ReviewError("inactive", "审阅观察已失效，请刷新。")
        target = self._review_targets.get(target_id)
        if target is None and not any(item.item_id == target_id for item in snapshot.items):
            raise ReviewError("invalid_target", "未知审阅目标。")
        if self._review_task is not None and not self._review_task.done():
            raise ReviewError("busy", "上一项读取尚未结束。")
        token = CancelToken()
        if target is None:
            operation = self._review_reader.read_file(snapshot, target_id, token)
        elif target.kind == "checks":
            operation = self._read_check_details(snapshot, target_id, token)
        else:
            operation = self._read_artifact_preview(snapshot, target_id, target, token)
        result = await self._review_io(operation, token)
        if self._review_id != review_id or self._review_snapshot is not snapshot or self._closing:
            raise ReviewError("inactive", "审阅已关闭或刷新。")
        return result

    async def _read_check_details(self, snapshot: ReviewSnapshot, target_id: str,
                                  token: CancelToken) -> ReviewCheckDetails:
        """检查详情只投影验证事实: refresh 重算 freshness, 最近工件有界核验。

        不调用 check/finalize/decide_delivery; refresh 沿既有 snapshotter 更新
        单调观察账本, 是原应用验证语义。整体读取共用 10s 预算, 超限如实报错。
        """
        assert self._service is not None and self._review_reader is not None
        assert self._config.plan is not None
        try:
            async with asyncio.timeout(REVIEW_TIMEOUT_S):
                view = await self._service.refresh(token)
                rows = tuple([await self._check_row(state, token) for state in view.checks])
        except TimeoutError:
            raise ReviewError("unavailable", "检查详情读取超出预算，请重试。") from None
        evidence = self._last_outcome.evidence if self._last_outcome is not None else None
        artifact_target_id = next((tid for tid, registered in self._review_targets.items()
                                   if registered.kind == "artifact"), None)
        return ReviewCheckDetails(
            review_id=snapshot.review_id, target_id=target_id,
            verification_run_id=view.verification_run_id,
            snapshot_ref=view.snapshot_ref, scope_id=view.scope_id,
            rows=rows, process_uncertain=view.process_uncertain,
            blocking_reasons=view.blocking_reasons,
            artifact_target_id=artifact_target_id,
            artifact_label=_evidence_label(evidence),
        )

    async def _check_row(self, state, token: CancelToken) -> ReviewCheckRow:
        """单检查行: 无回执即尚未运行, 不伪造 execution/verification 状态。"""
        definition = next(
            (c for c in self._config.plan.checks if c.check_id == state.check_id), None)
        command = definition.command if definition is not None else state.check_id
        receipt = state.receipt
        if receipt is None:
            return ReviewCheckRow(state.check_id, None, command, None, None,
                                  state.freshness, "not_recorded", None)
        integrity, reason = "not_recorded", None
        if receipt.output_artifact_path is not None:
            if receipt.artifact_error is not None:
                integrity, reason = "unknown", receipt.artifact_error[:256]
            elif receipt.output_artifact_sha256 is None:
                integrity, reason = "unknown", "输出工件未记录 hash，无法核验"
            else:
                integrity, reason = await self._verify_check_artifact(receipt, token)
        return ReviewCheckRow(state.check_id, receipt.receipt_id, command,
                              receipt.execution_status, receipt.verification_status,
                              state.freshness, integrity, reason)

    async def _verify_check_artifact(self, receipt, token: CancelToken) -> tuple[str, str | None]:
        """复用 verify_receipt_artifacts 的规则口径, 经 reader 有界核验不阻塞 loop。"""
        assert self._review_reader is not None
        preview = await self._review_reader.read_artifact(
            receipt.output_artifact_path, receipt.output_artifact_sha256, token)
        if preview.integrity == "verified":
            return "verified", None
        notice = preview.notice or "unknown"
        if notice == "missing":
            return "missing", "输出工件缺失或不可读"
        if notice == "mismatch":
            return "mismatch", "输出工件与写入时记录的 hash 不符"
        if notice == "over_limit":
            return "unknown", "输出工件超过全量核验限额"
        if notice == "changed":
            return "unknown", "输出工件在核验期间发生变化"
        return "unknown", f"输出工件核验未完成（{notice}）"

    async def _read_artifact_preview(self, snapshot: ReviewSnapshot, target_id: str,
                                     target: _ReviewTarget, token: CancelToken) -> ReviewPreview:
        """累计变更工件: 完整 hash 核验与有界正文同一次有界读取; 只显示历史身份。

        启动基线→生成时 snapshot/scope 进标题, 无观察完成时间不显示;
        正文整体呈现, 不解析为当前文件清单, 也不获得当前交付权威。
        """
        assert self._review_reader is not None
        assert target.path is not None and target.expected_sha256 is not None
        read = await self._review_reader.read_artifact(target.path, target.expected_sha256, token)
        evidence = target.evidence
        assert evidence is not None
        comparison = (f"启动基线 {evidence.baseline_snapshot_ref or '未知'} → 生成时快照 "
                      f"{evidence.snapshot_ref or '未知'} · scope {evidence.scope_id or '未知'}")
        notes = ["累计历史工件，不随当前修改刷新",
                 "生成时完整" if evidence.complete
                 else f"生成时不完整 · {evidence.failure_kind or 'unknown'}"]
        if read.notice:
            notes.append(_ARTIFACT_NOTICES.get(read.notice, read.notice))
        return ReviewPreview(snapshot.review_id, target_id, "累计变更工件", comparison,
                             read.state, read.text, " · ".join(notes), None, read.integrity)

    async def _review_io(self, operation, token: CancelToken):
        task = asyncio.create_task(operation)
        self._review_task = task
        self._review_token = token
        task.add_done_callback(self._review_io_settled)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancel_review()
            raise

    def _review_io_settled(self, task: asyncio.Task) -> None:
        # 取走迟到异常；即使 caller 被取消，reader/worker 仍由应用持有。
        if not task.cancelled():
            task.exception()
        if self._review_task is task:
            self._review_token = None
            if self._review_id is None:
                self._ownership.release("review")

    def cancel_review(self) -> None:
        self._review_id = None
        self._review_snapshot = None
        self._review_targets = {}
        if self._review_token is not None:
            self._review_token.cancel()
        if self._review_task is None or self._review_task.done():
            self._ownership.release("review")

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
    def busy_operation(self) -> str | None:
        """当前持有会话独占权的事务 (P0-3); None = 没有事务在途.

        这是"谁在改会话"的唯一答案; UI 能否提交仍以 state 与自身闸门为准。
        """
        return self._ownership.owner

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
            user_name=self._user_name,
        )

    @property
    def session_generation(self) -> int:
        return self._session_generation

    @property
    def restore_warnings(self) -> tuple[str, ...]:
        """启动机会式清理的告警事实 (C3); 无告警/未启用捕获时为空."""
        return self._restore_warnings

    @property
    def session_management_active(self) -> bool:
        return self._ownership.holds("session")

    @property
    def session_cleanup_failed(self) -> str | None:
        return self._session_cleanup_failed

    @property
    def session_short_title(self) -> str:
        return self._store.recovery_snapshot().short_title if self._store is not None else ""

    def submit_session_name(self, user_name: str | None) -> SessionManagementSubmission:
        try:
            name = validate_session_name(user_name, trim=True)
        except ValueError as exc:
            raise SessionManagementError("invalid-input", str(exc)) from exc
        if self._closing or self._closed:
            raise SessionManagementError("closing", "应用正在关闭。")
        if self._store is None:
            raise SessionManagementError("unavailable", "当前会话未启用保存，无法持久命名。")
        if (self._state != "idle" or self._ownership.owner is not None
                or self._in_flight_workers()):
            raise SessionManagementError("busy", "命名需要空闲会话。")
        store, epoch_id = self._store, self._epoch_id or ""
        loop = asyncio.get_running_loop()
        self._ownership.claim("session")
        self._session_done = loop.create_future()
        token = CancelToken()
        self._session_token = token
        task = loop.create_task(self._set_session_name(store, epoch_id, name, token))
        self._session_task = task
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        self._notify(OperationBusy("session"))
        return SessionManagementSubmission(task, token)

    def submit_new_session(self) -> SessionManagementSubmission:
        return self._submit_session_switch("new")

    def submit_resume_session(self, session_id: str, *,
                              adopt_current_config: bool = False) -> SessionManagementSubmission:
        try:
            if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
                raise ValueError
            if not isinstance(adopt_current_config, bool):
                raise ValueError
        except (ValueError, AttributeError):
            raise SessionManagementError("invalid-input", "恢复需要完整 UUID 与布尔配置选择。") from None
        return self._submit_session_switch("resume", session_id, adopt_current_config)

    def _submit_session_switch(self, operation: str, session_id: str | None = None,
                               adopt_current_config: bool = False) -> SessionManagementSubmission:
        if self._closing or self._closed:
            raise SessionManagementError("closing", "应用正在关闭。")
        factory = self._new_store_factory if operation == "new" else self._resume_store_factory
        if not isinstance(self._store, SessionStoreRoute) or factory is None:
            raise SessionManagementError("unavailable", "当前会话没有保存路由，无法切换会话。")
        if (self._state != "idle" or self._ownership.owner is not None
                or self._in_flight_workers()):
            raise SessionManagementError("busy", "切换需要空闲会话。")
        reason = (self._session_cleanup_failed or self._storage_failed or self._view_only_reason
                  or ("process_uncertain" if self.has_process_uncertain() else None))
        if reason is not None and not (operation == "resume" and session_id == self._store.session_id):
            raise SessionManagementError("blocked", "切换被阻断：" + reason)
        loop = asyncio.get_running_loop()
        self._ownership.claim("session")
        self._session_done = loop.create_future()
        token = CancelToken()
        self._session_token = token
        task = loop.create_task(self._switch_session(
            self._store.capture(), self.session_generation, token, operation, session_id, adopt_current_config))
        self._session_task = task
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        self._notify(OperationBusy("session"))
        for query_token in self._file_queries.values():
            query_token.cancel()
        return SessionManagementSubmission(task, token)

    async def _settle_session_work(self, awaitable, token: CancelToken):
        task = asyncio.ensure_future(awaitable)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                token.cancel()
                if task.done():
                    return task.result()

    async def _switch_session(self, source: SessionStorePort, generation: int,
                              token: CancelToken, operation: str, session_id: str | None,
                              adopt_current_config: bool) -> SessionManagementResult:
        route = self._store
        assert isinstance(route, SessionStoreRoute)
        target = None
        published = False
        try:
            if self._file_queries:
                await self._settle_session_work(
                    asyncio.gather(*tuple(self._file_queries), return_exceptions=True), token)
            token.throw_if_cancelled()
            if self._closing:
                raise asyncio.CancelledError
            if operation == "resume" and session_id == source.session_id:
                return SessionManagementResult("resume", source.session_id, generation, False)
            # 线程结果属于本受拥有任务；重复取消先取得资源，再登记并清理。
            opening = (asyncio.to_thread(self._new_store_factory) if operation == "new" else
                       asyncio.to_thread(self._resume_store_factory, session_id))
            target = await self._settle_session_work(opening, token)
            route.retain(target)
            token.throw_if_cancelled()
            if self._closing:
                raise asyncio.CancelledError
            identity = source.recovery_snapshot().identity
            if identity is None:
                raise SessionManagementError("prepare-failed", "当前配置代次缺少身份。")
            view_only_reason = None
            if operation == "resume":
                before = target.recovery_snapshot()
                if before.identity is None:
                    raise SessionManagementError("prepare-failed", "目标档缺少配置身份。")
                changed = list(conflicting_launch_fields(before.identity.launch, identity.launch))
                if before.identity.tool_fingerprint != identity.tool_fingerprint:
                    changed.append("tool_fingerprint")
                if changed and not adopt_current_config:
                    raise SessionManagementError("config-conflict", "目标配置冲突：" + ", ".join(changed))
                view_only_reason = recovery._admit_persistent_resume(
                    PersistenceConfig(identity=identity.launch, session_id=session_id,
                                      adopt_current_config=True), before)
                if adopt_current_config:
                    sources = {**identity.launch.sources, "resume_config": "current"}
                    if changed:
                        sources["changed_fields"] = ",".join(changed)
                    identity = replace(identity, launch=replace(identity.launch, sources=sources))
            epoch = await self._settle_session_work(target.start_epoch(identity), token)
            snapshot = target.recovery_snapshot()
            # 空档同样核对共享恢复构建；发布段之前完成所有可能失败的准备。
            proof = recovery._build_prefix_proof(snapshot).proof if self._projection is not None else None
            blocks = recovery._rebuild_context_projection(snapshot)
            history = [isolate_message(message) for message in snapshot.canonical_history]
            spans = []
            span_start = 0
            for request in snapshot.requests:
                if request.context_generation == snapshot.context_generation:
                    end = span_start + len(request.messages)
                    spans.append((request.seq, span_start, end))
                    span_start = end
            token.throw_if_cancelled()
            if self._closing or source is not route.capture() or generation != self.session_generation:
                raise asyncio.CancelledError
            # 发布无 await；runtime、Verifier、读取依据及进程累计列表保持。
            retired = route.publish(target)
            published = True
            self._epoch_id = epoch
            self._persistence_mode = "resume" if operation == "resume" else "save"
            self._history[:] = history
            self._context_generation = snapshot.context_generation
            self._committed_spans[:] = spans
            self._restored_requests = snapshot.requests
            self._restored_request_count = len(snapshot.requests)
            self._restored_model_call_count = sum(len(r.outcome.model_calls) for r in snapshot.requests)
            self._restored_interrupted = snapshot.interrupted
            self._restored_receipts = snapshot.restored_receipts
            self._restored_summaries = snapshot.summaries
            if view_only_reason is not None:
                self._view_only_reason = f"{target.session_id}: {view_only_reason}"
                if snapshot.recorded_process_uncertain:
                    self._process_uncertain = True
                    if self._service is not None:
                        self._service.mark_process_uncertain(self._view_only_reason)
            self._summary_records.clear()
            self._saved_request_count = 0
            self._last_committed_seq = None
            self._request_id = None
            self._request_store = None
            self._request_epoch = None
            self._last_outcome = None
            self._last_request_model = None
            self._review_targets.clear()
            self._user_name = snapshot.user_name
            if self._projection is not None:
                self._projection.proof = proof
            self._context_projection.adopt(blocks)
            if self._budget_policy is not None:
                self._budget_policy.invalidate()
            self._session_generation += 1
            self._notify(SessionChanged(self.session_generation, self.persistence, snapshot))
            try:
                await settle(route.release(retired))
            except BaseException as exc:
                self._session_cleanup_failed = str(exc)
                raise SessionManagementError("cleanup-failed", "目标会话已生效，旧档关闭失败：" + str(exc),
                                             published=True) from exc
            return SessionManagementResult(operation, target.session_id, self.session_generation, True)
        except (asyncio.CancelledError, SessionManagementError):
            raise
        except Exception as exc:
            raise SessionManagementError("prepare-failed", "切换准备失败：" + str(exc), published=published) from exc
        finally:
            try:
                if target is not None and not published:
                    try:
                        await settle(route.release(target))
                    except BaseException as exc:
                        self._session_cleanup_failed = str(exc)
                        raise SessionManagementError("cleanup-failed", "准备档关闭失败：" + str(exc)) from exc
            finally:
                self._ownership.release("session")
                self._session_token = None
                if self._session_done is not None and not self._session_done.done():
                    self._session_done.set_result(None)

    async def _set_session_name(
        self, store: SessionStorePort, epoch_id: str, name: str | None, token: CancelToken,
    ) -> SessionManagementResult:
        try:
            token.throw_if_cancelled()
            if self._closing or self._closed:
                raise asyncio.CancelledError
            if store.recovery_snapshot().user_name == name:
                return SessionManagementResult("name", store.session_id, self.session_generation, False)
            try:
                submission = store.submit_name(name, expected_epoch_id=epoch_id)
                saved_name = await submission.wait()
            except (StorageWriteError, StorageUnavailableError, ArchiveLimitError) as exc:
                self._latch_storage_failed(str(exc))
                raise SessionManagementError("write-failed", "名称保存失败：" + str(exc)) from exc
            except (PersistenceError, ValueError) as exc:
                raise SessionManagementError("prepare-failed", str(exc)) from exc
            self._user_name = saved_name
            self._notify(SessionNameChanged(store.session_id, saved_name, self.session_generation))
            return SessionManagementResult("name", store.session_id, self.session_generation, True)
        finally:
            self._ownership.release("session")
            self._session_token = None
            if self._session_done is not None and not self._session_done.done():
                self._session_done.set_result(None)

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
    def cost(self) -> CostSnapshot | None:
        """本进程费用事实；切换会话或配置不重置，也不从存档恢复。"""
        return self._config.cost_control.snapshot if self._config.cost_control is not None else None

    def diagnostics_report(self) -> dict:
        """隔离的只读报告；完成时事实不会因导出变成当前检查权威。"""
        return build_report(config=self._config, persistence=self.persistence,
            generation=self.session_generation, outcome=self._last_outcome,
            request_model=self._last_request_model,
            model_identity=self._model_route.binding.identity if self._model_route is not None else None,
            ordinary_calls=self.process_model_calls,
            summary_calls=self.process_summary_calls, cost=self.cost, restored_receipts=self.restored_receipts)

    def export_diagnostics(self, path: Path) -> Path:
        """空闲或可靠关闭后同步写一次有界元数据，不建后台生产者。"""
        if (self._state not in ('idle', 'closed') or (self._closing and not self._closed)
                or self._ownership.owner is not None
                or self._in_flight_workers()):
            raise ReportError('diagnostic export requires a settled session')
        destination = self._config.workspace.expanduser().resolve() / path.expanduser()
        write_report(destination, self.diagnostics_report())
        return destination

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

    @property
    def cache_observation(self) -> CacheRequestObservation | None:
        """当前普通路由的可选只读发送观察 (缓存 03, 契约 C7); 无能力/未观察为 None.

        只读当前普通 binding 的模型能力, 不拿摘要实例; 换模型/强度的新 binding
        观察从空开始, 恢复进程不持久观察, clear/压缩由下一次实际比较表达替换。
        """
        route = self._model_route
        model = route.binding.model if route is not None else None
        if isinstance(model, CacheDiagnosticsPort):
            return model.cache_observation
        return None

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
        if isinstance(event, ModelCallCompleted):
            self._process_model_calls.append(event)
        self._notify(AgentEventReceived(event))
        if isinstance(event, RunStarted):
            self._run_id = event.run_id
            self._notify(RequestPhaseChanged(event.run_id, "executing"))
        elif isinstance(event, RunFinished):
            # 内核结束 != 应用空闲; 收尾由 execute 的 finalizing 段完成
            self._notify(RequestPhaseChanged(event.run_id, "finalizing"))

    # --- 审批与执行档 (P0-1) ------------------------------------------------------

    @property
    def execution_mode(self) -> str:
        return self._mode_controller.mode

    def set_execution_mode(self, mode: str) -> bool:
        """会话内切换执行档; 返回是否变化. 只影响本进程, 不写存档."""
        changed = self._mode_controller.set_mode(mode)
        if changed:
            self._notify(ExecutionModeChanged(self._mode_controller.mode))
        return changed

    def enable_approval_ui(self) -> None:
        """TUI 组合层打开会话后绑定审批 UI; 未绑定时审批档一律 deny."""
        self._mode_controller.broker.enable(self._publish_approval)

    def _publish_approval(self, request: ApprovalRequest) -> None:
        self._notify(ApprovalRequested(
            request_id=request.request_id,
            tool_name=request.tool_name,
            command=request.command,
        ))

    def resolve_approval(self, request_id: str, decision: Decision) -> bool:
        """UI 决定回传; 无待决请求或身份不符返回 False."""
        return self._mode_controller.broker.resolve(request_id, decision)

    # --- 执行 ----------------------------------------------------------------------

    @property
    def model_options(self) -> tuple[ModelOption, ...]:
        return self._model_selection.options if self._model_selection is not None else ()

    @property
    def model_catalog_revision(self) -> str | None:
        return getattr(self._model_selection, 'revision', None)

    @property
    def current_model_option_id(self) -> str | None:
        route = self._model_route
        return (route.binding.option_id or route.binding.identity.model) if route else None

    def _validate_binding_cost(self, binding: ModelBinding) -> None:
        if self._config.cost_control is not None:
            self._config.cost_control.validate_identity(binding.identity)
            self._config.cost_control.validate_identity(binding.summary_identity)

    def _reasoning_capability(self) -> ReasoningEffortPort | None:
        """最新 binding 的普通模型强度能力 (C5); 每次从当前路由取, 不缓存过期 provider。"""
        route = self._model_route
        model = route.binding.model if route is not None else None
        return model if isinstance(model, ReasoningEffortPort) else None

    @property
    def reasoning_effort(self) -> ReasoningEffortState:
        """当前思考强度状态 (03, C5): 只读, current 从实际安全身份取值。

        think 关闭 (或身份无档位) 为 None; 无能力 provider 为 enabled=False,
        不以 UI 猜测。
        """
        capability = self._reasoning_capability()
        if capability is None or self._model_route is None:
            return ReasoningEffortState((), None, enabled=False)
        identity = self._model_route.binding.identity
        current = dict(identity.options).get("reasoning_effort") if identity.think is True else None
        levels = tuple(capability.reasoning_efforts)
        return ReasoningEffortState(levels, current, enabled=bool(levels))

    def _check_model_transaction_admission(self, blocked: str) -> None:
        """select_model 与 set_reasoning_effort 共用的独占事务准入 (C6)."""
        owner = self._ownership.owner
        if owner in ("review", "rewind"):
            raise ModelSelectionError("review is in progress")
        if owner == "login":
            raise ModelSelectionError("login configuration is in progress")
        if self._closing or self._closed or self._state != "idle":
            raise ModelSelectionError(f"{blocked} requires an idle, open session")
        if owner is not None:
            raise ModelSelectionError("another operation is in progress")
        if self._view_only_reason or self._storage_failed or self._process_uncertain:
            raise ModelSelectionError(f"{blocked} is blocked by the session state")

    async def select_model(self, option_id: str, *, catalog_revision: str | None = None,
                           think: bool | None = None) -> None:
        self._check_model_transaction_admission("model selection")
        if self._model_selection is None or self._model_route is None:
            raise ModelSelectionError("model selection is unavailable for this provider")
        if option_id not in {option.id for option in self.model_options}:
            raise ModelSelectionError("unknown model option")
        catalog = self._model_selection if getattr(self._model_selection, 'is_catalog', False) else None
        if catalog_revision is not None and catalog_revision != self.model_catalog_revision:
            raise ModelSelectionError('模型目录已变化，请重新打开 /model')
        current = self._model_route.binding
        if option_id == self.current_model_option_id and think is None:
            if catalog is None or current.catalog_revision == catalog.revision:
                return

        def prepare() -> ModelBinding:
            return (catalog.prepare_model(option_id, think=think) if catalog is not None
                    else self._model_selection.prepare_model(option_id))

        await self._run_model_change("model", prepare)

    async def set_reasoning_effort(self, level: str) -> None:
        """从当前绑定本地准备思考强度，经同一模型事务可靠提交与发布。"""
        self._check_model_transaction_admission("reasoning effort selection")
        capability = self._reasoning_capability()
        if capability is None or self._model_route is None:
            raise ModelSelectionError("reasoning effort is unavailable for this provider")
        if not isinstance(level, str) or level not in capability.reasoning_efforts:
            raise ModelSelectionError("unknown reasoning effort level")
        previous = self.reasoning_effort
        if previous.current == level:
            return  # 同值已启用：不置忙碌、不建任务、不发事件。

        def prepare() -> ModelBinding:
            binding = capability.prepare_reasoning_effort(level)
            current = self._model_route.binding
            return replace(binding, option_id=current.option_id, label=current.label,
                           catalog_revision=current.catalog_revision)

        await self._run_model_change("reasoning_effort", prepare, previous)

    async def _run_model_change(
        self, kind: Literal["model", "reasoning_effort"], prepare: Callable[[], ModelBinding],
        previous_effort: ReasoningEffortState | None = None,
    ) -> None:
        """两种模型变更共用的提交/取消/发布流程；prepare 同步且无 I/O。"""
        self._ownership.claim("model")
        self._model_done = asyncio.get_running_loop().create_future()
        try:
            try:
                binding = prepare()
                self._validate_binding_cost(binding)
                binding.validate_request(ModelRequest(
                    messages=self._context_projection.project(self._history),
                    tools=self._request_tool_specs(), system_prompt=self._config.system_prompt,
                ))
            except ValueError as exc:
                if isinstance(exc, CostError):
                    raise ModelSelectionError('费用准入拒绝：' + str(exc)) from exc
                if kind == "reasoning_effort":
                    raise ModelSelectionError(
                        "reasoning effort change rejected by configuration or history") from exc
                if not getattr(self._model_selection, 'is_catalog', False):
                    raise ModelSelectionError("model selection rejected by configuration or history") from exc
                if isinstance(exc, UserConfigError):
                    raise ModelSelectionError('目标配置或凭据不可用，请打开 /login 检查') from exc
                raise ModelSelectionError('配置、费用或历史不兼容；可在 /model 关闭目标思考后切换，或 /new 新建会话') from exc
            persistence = self._config.persistence
            identity = None
            if persistence is not None:
                safe = binding.identity
                source = "model" if kind == "model" else "reasoning_effort"
                sources = {**persistence.identity.sources, source: "selection"}
                if kind == "reasoning_effort":
                    assert previous_effort is not None
                    if previous_effort.current is None:
                        sources["think"] = "selection"
                identity = replace(persistence.identity, provider=safe.provider, model=safe.model,
                    base_url=safe.base_url, think=safe.think, options=dict(safe.options),
                    max_request_bytes=safe.max_request_bytes, sources=sources)
                persistence = replace(persistence, identity=identity)
            new_config = replace(self._config, model_label=binding.label or binding.identity.model,
                                 budget=binding.budget, persistence=persistence)
            self._notify(OperationBusy(operation="model"))
            operation = "model selection" if kind == "model" else "reasoning effort change"
            if self._store is None:
                if self._closing or self._closed:
                    raise ModelSelectionError(f"application is closing; {operation} was not accepted")
                await self._finish_model_change(
                    binding, new_config, None, kind=kind, previous_effort=previous_effort)
                return
            # 接受前唯一取消点；close 封口后不能再排入存储事务。
            await asyncio.sleep(0)
            if self._closing or self._closed:
                raise ModelSelectionError(f"application is closing; {operation} was not accepted")
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
                operation = "model configuration" if kind == "model" else "reasoning effort change"
                raise ModelSelectionError(f"{operation} was not accepted by storage") from exc
            # 强引用覆盖真实提交和内存发布；重复取消不提前释放忙碌或关闭等待。
            self._model_task = asyncio.create_task(self._finish_model_change(
                binding, new_config, submission, kind=kind, previous_effort=previous_effort))
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
            self._ownership.release("model")
            if self._model_done is not None and not self._model_done.done():
                self._model_done.set_result(None)

    async def _finish_model_change(
        self, binding: ModelBinding, config: ApplicationConfig,
        submission: ModelChangeSubmission | None, *, kind: Literal["model", "reasoning_effort"],
        previous_effort: ReasoningEffortState | None,
    ) -> None:
        epoch_id = self._epoch_id
        if submission is not None:
            try:
                epoch_id = await submission.wait()
            except PersistenceError as exc:
                self._latch_storage_failed(str(exc))
                message = ("model configuration failed to persist; current model is unchanged"
                           if kind == "model" else
                           "reasoning effort change failed to persist; previous effort is unchanged")
                raise ModelSelectionError(message) from exc
        previous_label = self._config.model_label
        self._publish_binding(binding, config, epoch_id)
        if kind == "model":
            self._notify(ModelChanged(previous_label, config.model_label, epoch_id))
        else:
            assert previous_effort is not None
            self._notify(ReasoningEffortChanged(previous_effort, self.reasoning_effort, epoch_id))

    def _publish_binding(
        self, binding: ModelBinding, config: ApplicationConfig, epoch_id: str | None
    ) -> None:
        """无 await 同步发布 (C7): 路由/配置/预算/最新准备源在同一段一次落定."""
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
        policy.token_counter = binding.token_counter
        policy.summary_token_counter = binding.summary_token_counter
        policy.invalidate()
        # 04 (C6): /model 与 /login 从最新接受绑定准备, 不恢复启动缓存对象
        if not getattr(self._model_selection, 'is_catalog', False):
            self._model_selection = binding.model if isinstance(binding.model, ModelSelectionPort) else None
        self._model_configuration = (binding.model
            if isinstance(binding.model, ModelConfigurationPort) else None)

    async def execute(self, prompt: str) -> RequestOutcome:
        """唯一编码请求入口: 准入即占住会话所有权, 直到提交与发布结束 (P0-3)."""
        self._admit_request(prompt)
        try:
            return await self._run_request(prompt)
        finally:
            self._ownership.release("request")

    def _admit_request(self, prompt: str) -> None:
        """请求准入; 所有拒绝都在占槽之前, 通过后请求持有所有权到收尾."""
        if self._session_cleanup_failed is not None:
            raise RuntimeError("session cleanup failed: " + self._session_cleanup_failed)
        owner = self._ownership.owner
        if owner in FOREIGN_OWNERS:
            raise RuntimeError(busy_refusal(owner, "execute"))
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
        if owner == "request":
            # 前一请求仍在 begin 等待窗 (state 尚为 idle); 非持久会话同样适用
            raise RuntimeError(
                "a previous request is still being started; "
                "new requests are refused until it settles"
            )
        if owner == "clear":
            raise RuntimeError(
                "a persistent clear is in progress; new requests are refused until it settles"
            )
        if owner == "compaction":
            # 02/PA05: execute/clear/compact 三向互斥 (应用层, 不依赖 UI 屏障);
            # 压缩在途时新请求必须等待, 否则其提交会与摘要切换交错
            raise RuntimeError(
                "a compaction is in progress; new requests are refused until it settles"
            )
        if self._store is not None and self._storage_failed is not None:
            # 可靠提交请求开始 intent; 失败零模型调用并锁存 storage_failed (D2)。
            # 等待受拥有: 外层取消不把已派发事务当作未发生, 取得明确结果后才收尾
            raise RuntimeError(
                f"storage_failed: earlier session save failed ({self._storage_failed}); "
                "new coding requests are refused"
            )
        self._ownership.claim("request")

    async def _run_request(self, prompt: str) -> RequestOutcome:
        # 请求边界接缝 (D1/PA02): 准入通过后、首次预算预估之前调用一次,
        # 建立本次 execute 尝试的预算状态范围; begin 不再调用或清零
        if self._budget_policy is not None:
            self._budget_policy.note_request_start(len(self._history))
        # 03: 请求 token 提前到接缝处建立——begin 前自动压缩期间 session.cancel()
        # 即取消本次尝试 (零 begin/零模型调用); begin 后取消语义不变
        token = CancelToken()
        self._cancel_token = token
        # 所有权已由 execute 占住: 预估/自动压缩/begin 窗都算请求在途
        self._request_store = self._store.capture() if isinstance(self._store, SessionStoreRoute) else self._store
        self._request_epoch = self._epoch_id
        self._request_generation = self.session_generation
        self._request_receipts_before = frozenset(
            receipt.receipt_id for receipt in self._service.receipts
        ) if self._service is not None and self._request_store is not None else frozenset()
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
                self._request_store.begin_request(self._request_epoch or "", prompt)
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
        # 非持久会话无 begin: 接缝同为首次预估前 (准入通过后), 预算拒绝零副作用
        # (D9); 自动压缩需持久会话, 回调内判定后原样拒绝
        loop = asyncio.get_running_loop()
        self._state = "executing"
        self._run_id = None
        # token 已在接缝处建立 (03): 覆盖预估/自动压缩/begin/全部轮次
        self._active_done = loop.create_future()
        self._notify(RequestPhaseChanged(None, "executing"))
        restore_opened = False
        try:
            # 请求级快照与撤回 01 (C2): 捕获上下文在 begin 成功后建立 (request_id、
            # epoch_id、session_id); 建立失败即拒绝本次请求并诊断, 不静默降级为
            # 无捕获请求, 也经同一 closing 路径收尾。非持久/关闭捕获时零副作用。
            restore_opened = self._open_restore_capture()
            result = await self._app.agent.run(prompt, token, history=tuple(self._history))
            # 内核结束 != 应用空闲: 显式进入收尾状态, 完成前不接受新请求 (T07);
            # closed 是终态, close 监督超时后的迟到完成不得改写它 (02-R-F1)
            if self._state != "closed":
                self._state = "finalizing"
            outcome = await self._finalize_request(result)
        except BaseException:
            # 异常路径也不得把应用留在"可执行"状态: 内核/UI 异常一律经关闭路径处理
            await self._close_restore_capture(restore_opened, interrupted=True)
            if self._state != "closed":
                self._state = "closing"
                self._closing = True
            if self._active_done is not None and not self._active_done.done():
                self._active_done.set_result(None)
            raise
        # 先更新隔离的 state/history, 再通知订阅者; 应用把后缀追加一次
        # 捕获上下文在 commit 收尾后清除 (C2): note_commit 已在上一步取到真实 seq;
        # 提交未成功 (落盘故障) 时按中断如实标注来源, 已捕获部分仍可还原。放 finally:
        # 提交路径自身抛错也不能让上下文跨请求残留 (残留会让下一请求被误拒)。
        interrupted = True
        completion_failed = True
        try:
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
                interrupted = (
                    outcome.persistence is not None and not outcome.persistence.saved
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
            completion_failed = False
        finally:
            await self._close_restore_capture(restore_opened, interrupted=interrupted)
            if completion_failed and self._active_done is not None and not self._active_done.done():
                self._active_done.set_result(None)
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
        # C8: 在既有发布位置保存最近本进程 outcome 的隔离只读引用 (02);
        # 不从 CheckProjection/_last_completed 反取权威, 也不从 restored_requests 恢复
        self._last_outcome = _isolate_outcome(outcome)
        self._last_request_model = self._config.model_label
        self._notify(RequestCompleted(outcome))
        # 请求拥有持久提交、seq补记和捕获收尾的真实IO；内核结束不释放此信号。
        if self._active_done is not None and not self._active_done.done():
            self._active_done.set_result(None)
        return outcome

    def _open_restore_capture(self) -> bool:
        """begin 成功后建立本请求的捕获上下文 (C2); 失败即拒绝请求并诊断.

        只有持久会话 + 捕获开启时才存在捕获汇; 未安装时本方法零副作用并返回 False,
        edit/write 行为与既有逐位一致。上下文建立失败属编程错误 (未建立), 不静默
        降级为无捕获请求, 也不清除他人上下文。返回是否由本次调用建立。
        """
        store = self._restore_store
        if store is None:
            return False
        session_id = self._store.session_id if self._store is not None else None
        try:
            store.begin_request(self._request_id or "", self._request_epoch or "",
                                session_id or "")
        except Exception as exc:
            raise RuntimeError(
                "cannot open the restore capture context for this request; "
                f"the request was refused before any model call: {exc}"
            ) from exc
        return True

    async def _close_restore_capture(self, opened: bool, interrupted: bool = False) -> None:
        """清除本次请求建立的捕获上下文 (幂等); interrupted 时如实标注该点来源.

        未由本次请求建立 (opened=False) 时不动上下文, 避免误清他人状态; 清除失败
        不阻断请求收尾: 捕获上下文只是编辑准入的守卫, 会话提交事实与规范历史不
        取决于它。
        """
        if not opened or self._restore_store is None:
            return
        task = asyncio.create_task(asyncio.to_thread(
            self._restore_store.end_request, interrupted=interrupted))
        try:
            while True:
                try:
                    await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    continue  # 实际 IO 落定后才清除请求归属并允许关闭
        except Exception:
            pass
        self._restore_warnings += self._restore_store.take_warnings()

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
        finalized 错误记录。预估不含锚 (锚状态在 BudgetPolicy), 检查模式 system
        面用 RunPolicy.preflight_system_prompt 同源生成, 并按同源 cap 加最大尾部
        允许量 (缓存 02, 契约 C4); 不再逐旧结果添加 freshness allowance。
        """
        policy = self._budget_policy
        config = self._config.budget
        if policy is None or config is None:
            return
        # 01-F10/PA02: 预估计量面 = 当前 ContextProjection 的有效消息
        # ([摘要块] + 保留后缀) + 本次 prompt; 旧规范历史不再参与 (压缩后重估
        # 必须能回到限内)。
        messages = (
            *self._context_projection.project(self._history),
            UserMessage(text=prompt),
        )
        tail_allowance = 0
        if self._config.plan is not None:
            # plan 存在而策略实例缺失的人工组合明确拒绝, 不回退常量漏算 (契约 C1)
            run_policy = self._policy
            if run_policy is None:
                raise RuntimeError(
                    "check mode is configured without a RunPolicy instance; "
                    "the preflight budget estimate cannot be sourced"
                )
            system_prompt = run_policy.preflight_system_prompt(self._config.system_prompt)
            tail_allowance = (
                math.ceil(run_policy.request_context_max_bytes / 3)
                + config.message_template_overhead
            )
        else:
            system_prompt = self._config.system_prompt
            if self._cancel_token is not None and self._cancel_token.cancelled:
                raise asyncio.CancelledError()
            report = policy.assess_preflight(ModelRequest(
                messages=messages, tools=self._request_tool_specs(), system_prompt=system_prompt,
            ))
            if self._cancel_token is not None and self._cancel_token.cancelled:
                raise asyncio.CancelledError()
            if report.total_estimate > config.num_ctx:
                raise RuntimeError(f"request refused before start: {budget_exceeded_text(report)}")
            return
        input_estimate, components = estimate_input_components(
            system_prompt, self._request_tool_specs(), messages, config
        )
        if tail_allowance:
            components = components + (
                ComponentEstimate(
                    "verification_context_allowance",
                    tail_allowance,
                    "estimated",
                    note="capacity placeholder estimate for the RunPolicy verification context tail",
                ),
            )
            input_estimate += tail_allowance
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
            if self._budget_policy is None or not await self._auto_compact_for_request():
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
            finalized = await service.finalize_current(token)
            view, evidence = finalized.view, finalized.evidence
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
        assert self._request_id is not None and self._request_store is not None
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
            seq = await self._request_store.commit_request(
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
        if self._restore_store is not None:
            # C2 取值裁决: commit 取得真实 seq 后补记 (幂等元数据更新, 不重写 entries);
            # 补记失败不改写会话提交事实, 点保持 seq=null 如实呈现
            try:
                await asyncio.to_thread(self._restore_store.note_commit, seq)
            except Exception:
                pass
        return RequestPersistenceFacts(request_id=self._request_id, saved=True, seq=seq)

    def _new_receipts(self, outcome: RequestOutcome) -> tuple[StoredReceipt, ...]:
        """本请求新产生的检查回执 (尚未持久化的); 与请求记录同事务落库 (03)."""
        view = outcome.view
        if self._config.plan is None or self._request_epoch is None or self._request_store is None:
            return ()
        receipts = self._service.receipts if self._service is not None else (view.receipts if view else ())
        return tuple(
            StoredReceipt(
                verification_run_id=receipt.verification_run_id,
                receipt_id=receipt.receipt_id,
                epoch_id=self._request_epoch,
                request_id=self._request_id or "",
                definition_fingerprint=recovery._receipt_fingerprint(
                    receipt.check_id, receipt.command, receipt.timeout_s
                ),
                execution_status=receipt.execution_status,
                exit_code=receipt.exit_code,
                timed_out=receipt.timed_out,
                cancelled=receipt.cancelled,
                output_complete=receipt.output_complete,
                failure_kind=receipt.failure_kind,
            )
            for receipt in receipts
            if receipt.receipt_id not in self._persisted_receipt_ids
            and self._receipt_owners.get(receipt.receipt_id) == (
                self._request_store.session_id, self._request_id, self._request_epoch)
        )

    def record_check_receipts(self) -> None:
        """工具正常/异常落定时绑定真实账本差集，Verifier 不读取路由。"""
        if (self._request_store is not None and self._request_id is not None
                and self._request_epoch is not None and self._request_generation == self.session_generation
                and self._service is not None):
            for receipt in self._service.receipts:
                if receipt.receipt_id not in self._request_receipts_before:
                    self._receipt_owners.setdefault(receipt.receipt_id, (
                        self._request_store.session_id, self._request_id, self._request_epoch))

    @property
    def tool_names(self) -> tuple[str, ...]:
        """当前组装解析出的工具名 (只读显示出口; 不含 schema 与实现)。"""
        return tuple(spec.name for spec in self._request_tool_specs())

    @property
    def process_model_calls(self) -> tuple[ModelCallCompleted, ...]:
        return tuple(self._process_model_calls)

    @property
    def process_summary_calls(self) -> tuple[SummaryOutcome, ...]:
        return tuple(self._process_summary_calls)

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
        owner = self._ownership.owner
        if owner in FOREIGN_OWNERS:
            raise RuntimeError(busy_refusal(owner, "clear"))
        if self._store is not None:
            raise RuntimeError(
                "persistent sessions must use clear_context(); the synchronous entry is refused "
                "to keep the clear boundary reliable"
            )
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        if owner is not None:
            # 请求 begin 等待窗: state 仍是 idle, 但请求已经占住互斥
            raise RuntimeError(busy_refusal(owner, "clear"))
        self._history.clear()
        if self._budget_policy is not None:
            self._budget_policy.invalidate()
        self._notify(HistoryCleared())

    async def clear_context(self) -> None:
        """持久清空: 先可靠提交清空边界, 成功后才清内存历史; 失败内存不变.

        无持久化模式委托旧同步入口; storage_failed 不阻止清空尝试, 失败如实报错.
        清空进行中 (owner=clear) 与 execute 互斥, 双向拒绝 (契约 §3).
        """
        owner = self._ownership.owner
        if owner in FOREIGN_OWNERS:
            raise RuntimeError(busy_refusal(owner, "clear"))
        if self._store is None:
            self.clear_history()
            return
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        if owner == "clear":
            raise RuntimeError("a persistent clear is already in progress")
        if owner == "compaction":
            # 02: 压缩在途时清空被拒——clear 推进代次会让在途摘要提交撞关系校验,
            # 且其失败会被误归存储故障; 等压缩落定后再清 (三向互斥, 审查 02-F8)
            raise RuntimeError("a compaction is in progress; clear is refused until it settles")
        if owner == "request":
            # 请求 begin 等待窗内 state 仍为 idle; 该窗缺口的 clear 变体在基线已存在
            raise RuntimeError(
                "a request start is being persisted; clear is refused until it settles"
            )
        self._ownership.claim("clear")
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
            self._ownership.release("clear")
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
        owner = self._ownership.owner
        if owner in FOREIGN_OWNERS:
            raise RuntimeError(busy_refusal(owner, "compact"))
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
        if owner == "compaction":
            raise RuntimeError("a compaction is already in progress")
        if owner == "clear":
            # 三向互斥 (审查 02-F8): 清空在途时压缩拒绝, 与 execute/clear 同一面
            raise RuntimeError(
                "a persistent clear is in progress; compaction is refused until it settles"
            )
        if owner == "request":
            # 请求 begin 等待窗内 state 仍为 idle; 否则压缩可与在途请求交错
            raise RuntimeError(
                "a request start is being persisted; compaction is refused until it settles"
            )
        if self._compact_sealed:
            raise RuntimeError("application is closing; compaction is sealed")
        # 可压缩区间与摘要输入预算在占槽前判定: 拒绝不留下 busy/屏障 (D1)
        token = CancelToken()
        plan = self._compaction_plan(token)
        self._ownership.claim("compaction")
        self._compact_done = asyncio.get_running_loop().create_future()
        self._compact_token = token
        self._notify(OperationBusy(operation="compact"))
        try:
            return await self._run_compaction(
                plan, reason="manual", cancel_token=self._compact_token
            )
        finally:
            self._ownership.release("compaction")
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
        if self._compact_sealed:
            return None
        self._check_summary_admission(cancel_token)
        report = self._budget_policy.assess_summary(plan.request)
        self._check_summary_admission(cancel_token)
        if report.total_estimate > report.limit:
            raise RuntimeError("compaction unavailable: " + budget_exceeded_text(report))
        try:
            outcome = await generate_summary(channel, plan.request, cancel_token)
        except asyncio.CancelledError:
            self._process_summary_calls.append(SummaryOutcome(ok=False, failure="aborted"))
            raise
        self._process_summary_calls.append(outcome)
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

    async def _auto_compact_for_request(self) -> bool:
        """自动压缩回调 (D7, 03): 发送前准入自动压缩的应用侧实现.

        属于已有 execute, 不重复获取会自锁的互斥 (execute 已排除并发
        compact/clear; 本方法置位同一屏障供 close 监督与封口)。全部准入条件
        (锁存/互斥/可压缩区间) 通过后才消费本次请求额度, 失败不返还; 成功返回
        True (投影已切换, 调用方对有效投影重估), 其余返回 False, 压缩失败不
        中断请求 (按原准入路径收尾)。不发 OperationBusy: 自动属于请求
        生命周期, UI 反馈随请求事件。
        """
        if self._store is None:
            return False
        policy = self._budget_policy
        if policy is None or self._config.budget is None or policy.summarize_channel is None:
            return False
        if self._closing or self._closed or self._compact_sealed:
            return False
        if self._ownership.holds("compaction") or self._ownership.holds("clear"):
            return False
        if self._view_only_reason is not None:
            # 两种锁存状态下零摘要调用 (D9): view-only 不把历史发往模型
            return False
        if self._storage_failed is not None:
            # storage_failed 锁存 (含请求中途由活动包装器锁存): 零摘要调用
            return False
        token = self._cancel_token
        assert token is not None, "auto compaction belongs to an execute attempt"
        try:
            plan = self._compaction_plan(token)
        except RuntimeError:
            return False
        if not policy.try_consume_admission_auto_attempt():
            return False
        # 请求内部的子阶段: 临时占住压缩槽, 让 clear/新请求看到"压缩在途"
        self._ownership.claim("compaction", inside="request")
        self._compact_done = asyncio.get_running_loop().create_future()
        self._compact_token = token
        try:
            compacted = await self._run_compaction(plan, reason="auto", cancel_token=token)
            return compacted is not None
        except RuntimeError:
            # 摘要失败/提交失败: 投影不变; 真实落盘故障已在 _run_compaction 锁存
            return False
        finally:
            self._ownership.release("compaction")
            self._compact_token = None
            if self._compact_done is not None and not self._compact_done.done():
                self._compact_done.set_result(None)

    def _check_summary_admission(self, cancel_token: CancelToken | None) -> None:
        if self._compact_sealed or self._closing or self._closed or (cancel_token is not None and cancel_token.cancelled):
            raise RuntimeError("summary generation failed (aborted); the model projection is unchanged")

    def _compaction_plan(self, cancel_token: CancelToken | None = None) -> _CompactionPlan:
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
        chosen_span: tuple[int, int, int] | None = None
        for span in reversed(eligible):
            self._check_summary_admission(cancel_token)
            request = build_summary_request(summary_input_messages(
                tuple(self._history[start : span[2]])
            ))
            report = self._budget_policy.assess_summary(request)
            self._check_summary_admission(cancel_token)
            if report.total_estimate <= config.num_ctx:
                chosen_span = span
                break
        if chosen_span is None:
            raise RuntimeError(
                "compaction unavailable: even the oldest single request does not fit "
                "the summary request budget"
            )
        return _CompactionPlan(
            request=request,
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

    # --- 用户默认配置 ---------------------------------------------------------------

    @property
    def login_busy(self) -> bool:
        return self._ownership.holds("login")

    @property
    def login_status(self) -> tuple[str, ...]:
        """来源是最近用户文件观察，当前模型/思考来自有效绑定；零凭据 I/O。"""
        view = self._login_view
        if view is None:
            return ()
        lines = (f"默认凭据来源（最近观察）: {view.credential_source}",)
        if self._model_route is not None:
            identity = self._model_route.binding.identity
            if identity.provider == 'deepseek':
                effort = dict(identity.options)['reasoning_effort'] if identity.think else 'off'
                lines += (f"当前有效模型/思考: {identity.model}/{effort}",)
        return lines

    def _check_login_admission(self) -> None:
        """login 编辑窗是唯一允许在 owner=login 时继续的入口 (保存/应用)."""
        if (self._ownership.owner not in (None, "login")
                or self._closing or self._closed or self._state != 'idle'):
            raise UserConfigError('login', 'operation-in-progress')

    def login_view(self, provider_id: str | None = None) -> LoginView:
        """同步占住编辑窗口后才观察；失败释放，关闭/取消不改变任何会话事实。"""
        self._check_login_admission()
        if self._ownership.holds("login"):
            raise UserConfigError('login', 'operation-in-progress')
        if self._login_store_factory is None:
            raise UserConfigError('login', 'unavailable')
        self._ownership.claim("login")
        try:
            if self._login_store is None:
                self._login_store = self._login_store_factory()
            catalog = self._model_selection if getattr(self._model_selection, 'is_catalog', False) else None
            environment = (self._login_environment() if self._login_environment and catalog is None else None)
            if catalog is not None:
                selected = provider_id or (self._model_route.binding.identity.provider if self._model_route else None)
                view = self._login_store.read_view(provider_id=selected)
                catalog.reload(self._login_store.read_profiles(),
                    default_id='/'.join(self._login_store.read_default_reference()))
            else:
                view = self._login_store.read_view(environment)
            apply_offered = bool(catalog is not None and not self._view_only_reason
                                 and not self._storage_failed and not self._process_uncertain)
            if self._model_route is not None:
                identity = self._model_route.binding.identity
                if catalog is None and identity.provider == 'deepseek':
                    options = dict(identity.options)
                    view = replace(view, model=identity.model, think=identity.think,
                        reasoning_effort=options['reasoning_effort'], num_ctx=options['num_ctx'],
                        max_output_tokens=options['max_tokens'])
                    # 04 (D7): 正常空闲 DeepSeek 会话才提供保存并应用; 阻断锁存只保存
                    apply_offered = (
                        self._model_configuration is not None
                        and not self._view_only_reason and not self._storage_failed
                        and not self._process_uncertain
                    )
            if apply_offered:
                view = replace(view, actions=('save-and-apply', 'defaults-only'),
                               credentials_only=False,
                               effect='保存并应用：立即更新当前会话凭据与思考设置；仅保存：下次新会话生效')
            else:
                view = replace(view, actions=('defaults-only',), credentials_only=False,
                               effect='保存（下次新会话生效）；当前会话配置保持')
            self._login_view = view
            self._notify(OperationBusy('login'))
            return view
        except BaseException:
            self._ownership.release("login")
            raise

    def cancel_login(self) -> None:
        """编辑取消；保存接受后由生产者发布真实结果。"""
        if self._login_task is not None and not self._login_task.done():
            return
        if self._ownership.holds("login"):
            self._ownership.release("login")
            self._notify(LoginCompleted(None))

    async def configure_login(self, submission: LoginSubmission) -> LoginResult:
        self._check_login_admission()
        if (not self._ownership.holds("login") or self._login_view is None or self._login_store is None or
                submission.revision != self._login_view.revision or
                (self._login_task is not None and not self._login_task.done())):
            raise UserConfigError('login', 'operation-in-progress')
        # 应用再次核对目标: 只允许本次视图实际提供的动作, 不能只信任 UI (D7)
        if submission.target not in self._login_view.actions:
            raise UserConfigError('login', 'unsupported-target')
        self._login_cancel_requested = False
        self._login_task = asyncio.create_task(self._finish_login(submission))
        # 保存任务是应用拥有的生产者；外层取消不能把已发布事实丢掉。
        while True:
            try:
                return await asyncio.shield(self._login_task)
            except asyncio.CancelledError:
                self._cancel_login_save()

    def _cancel_login_save(self) -> None:
        self._login_cancel_requested = True
        if self._login_save_task is not None and not self._login_save_task.done():
            self._login_save_task.cancel()

    async def _finish_login(self, submission: LoginSubmission) -> LoginResult:
        result = LoginResult(False, False, 'closing' if self._closing else 'not-requested', 'cancelled')
        try:
            if not self._closing and not self._login_cancel_requested:
                self._login_save_task = asyncio.create_task(self._login_store.save(submission))
                try:
                    result = await self._login_save_task
                except asyncio.CancelledError:
                    pass  # 尚未起跑的 worker 零写入；已起跑的 Adapter 返回真实结果。
                except Exception:
                    result = LoginResult(False, False, 'not-requested', 'write-failed')
            if result.defaults_saved and getattr(self._model_selection, 'is_catalog', False):
                try:
                    default_id = '/'.join(self._login_store.read_default_reference())
                    self._model_selection.reload(self._login_store.read_profiles(), default_id=default_id)
                except (ValueError, OSError):
                    result = replace(result, application_outcome='blocked' if submission.target == 'save-and-apply'
                                     else 'not-requested', error_code='configuration-changed')
                    return result
            if submission.target == 'save-and-apply':
                if result.defaults_saved:
                    result = await self._apply_login_configuration(submission, result)
                elif result.application_outcome == 'not-requested':
                    # 保存失败零应用 (D6); 应用结果与保存结果分开报告
                    result = replace(result, application_outcome='failed')
            return result
        finally:
            self._login_save_task = None
            self._ownership.release("login")
            self._notify(LoginCompleted(result))

    async def _apply_login_configuration(
        self, submission: LoginSubmission, saved: LoginResult
    ) -> LoginResult:
        """defaults 发布点之后的会话应用段 (04, C6/C7); 已保存与未应用分开报告。

        接受前取消/关闭/锁存零 epoch; 接受后沿模型选择事务等明确结果再无 await
        发布, close 经 _login_task 等同一生产者才释放 store/client。
        """
        if self._closing or self._closed:
            return replace(saved, application_outcome='closing', error_code='closing')
        if self._login_cancel_requested:
            return replace(saved, application_outcome='blocked', error_code='cancelled')
        if self._view_only_reason or self._storage_failed or self._process_uncertain:
            return replace(saved, application_outcome='blocked', error_code='session-blocked')
        route = self._model_route
        configuration = self._model_configuration
        catalog = self._model_selection if getattr(self._model_selection, 'is_catalog', False) else None
        if route is None or (catalog is None and (configuration is None
                or route.binding.identity.provider != 'deepseek')):
            return replace(saved, application_outcome='blocked', error_code='unsupported-target')
        current = route.binding.identity
        try:
            if catalog is not None:
                chosen_id = f'{submission.defaults.provider}/{submission.defaults.model}'
                proposed = submission.provider_profile
                if proposed is not None:
                    observed, observed_model = catalog.lookup(chosen_id)
                    proposed_model = next(m for m in proposed.models if m.key == submission.defaults.model)
                    if (observed.protocol, observed.base_url, observed.auth, observed_model.model_id) != (
                            proposed.protocol, proposed.base_url, proposed.auth, proposed_model.model_id):
                        return replace(saved, application_outcome='blocked', error_code='configuration-changed')
                binding = catalog.prepare_model(
                    chosen_id,
                    credential=submission.credential if submission.credential_action == 'replace' else None)
                if (submission.credential_action == 'keep' and binding.identity == current
                        and binding.catalog_revision == route.binding.catalog_revision):
                    return replace(saved, application_outcome='unchanged')
            else:
                target = replace(current, model=submission.defaults.model,
                    think=submission.defaults.think,
                    options=(('num_ctx', submission.defaults.num_ctx),
                             ('max_tokens', submission.defaults.max_output_tokens),
                             ('reasoning_effort', submission.defaults.reasoning_effort)))
                if submission.credential_action == 'keep' and target == current:
                    return replace(saved, application_outcome='unchanged')
                binding = configuration.prepare_configuration(
                    ConfigurationRequest(target, submission.credential_action, submission.credential))
            self._validate_binding_cost(binding)
        except CostError as exc:
            return replace(saved, application_outcome='blocked', error_code='cost-admission: ' + str(exc))
        except ValueError:  # UserConfigError 同为 ValueError; 固定分类, 不回显细节
            return replace(saved, application_outcome='blocked', error_code='invalid-model-settings')
        try:
            # 当前有效投影的历史资格核对 (D7): 不兼容保留旧绑定, 不改写历史
            binding.validate_request(ModelRequest(
                messages=self._context_projection.project(self._history),
                tools=self._request_tool_specs(), system_prompt=self._config.system_prompt,
            ))
        except ValueError:
            return replace(saved, application_outcome='blocked', error_code='history-incompatible')
        persistence = self._config.persistence
        identity = None
        if persistence is not None:
            safe = binding.identity
            sources = dict(persistence.identity.sources)
            if submission.credential_action == 'replace':
                # 固定安全来源标签; ref/key/hash 不进入 epoch (C6)
                sources['credentials'] = 'login'
            identity = replace(persistence.identity, provider=safe.provider, model=safe.model,
                base_url=safe.base_url, think=safe.think, options=dict(safe.options),
                max_request_bytes=safe.max_request_bytes, sources=sources)
            persistence = replace(persistence, identity=identity)
        new_config = replace(self._config, model_label=binding.label or binding.identity.model,
                             budget=binding.budget, persistence=persistence)
        if self._store is None:
            self._publish_binding(binding, new_config, self._epoch_id)
            self._notify(ConfigurationChanged(binding.identity, self._epoch_id))
            return LoginResult(True, True, 'applied')
        # 接受前唯一取消点; close 封口后不能把新事务排进队列
        await asyncio.sleep(0)
        if self._closing or self._closed:
            return replace(saved, application_outcome='closing', error_code='closing')
        if self._login_cancel_requested:
            return replace(saved, application_outcome='blocked', error_code='cancelled')
        snapshot = self._store.recovery_snapshot()
        assert identity is not None and snapshot.identity is not None
        try:
            handle = self._store.submit_model_change(
                StoredIdentity(identity, snapshot.identity.tool_fingerprint),
                expected_epoch_id=self._epoch_id or '',
            )
        except PersistenceError as exc:
            if isinstance(exc, (StorageUnavailableError, StorageWriteError, ArchiveLimitError)):
                self._latch_storage_failed(str(exc))
            return replace(saved, application_outcome='failed', error_code='storage-failed')
        try:
            epoch_id = await handle.wait()
        except PersistenceError as exc:
            # 已接受提交的真实存储失败: 保留旧绑定并沿原 storage_failed 锁存 (C7)
            self._latch_storage_failed(str(exc))
            return replace(saved, application_outcome='failed', error_code='storage-failed')
        self._publish_binding(binding, new_config, epoch_id)
        self._notify(ConfigurationChanged(binding.identity, epoch_id))
        return LoginResult(True, True, 'applied')

    async def set_default_model(self, option_id: str, *, catalog_revision: str | None = None) -> LoginResult:
        self._check_model_transaction_admission('default model')
        catalog = self._model_selection
        if not getattr(catalog, 'is_catalog', False):
            raise ModelSelectionError('当前目录不支持保存默认模型')
        if catalog_revision is not None and catalog_revision != catalog.revision:
            raise ModelSelectionError('模型目录已变化，请重新打开 /model')
        provider, model = catalog.lookup(option_id)
        view = self.login_view(provider_id=provider.id)
        try:
            if catalog_revision is not None and catalog_revision != catalog.revision:
                raise ModelSelectionError('模型目录已变化，请重新打开 /model')
            # login_view 可能观察到外部编辑，提交只能使用同一最新修订的配置。
            provider, model = catalog.lookup(option_id)
            defaults = UserDefaults(provider=provider.id, model=model.key, think=model.think,
                reasoning_effort=model.reasoning_effort, num_ctx=model.context_window,
                max_output_tokens=model.max_output_tokens)
            submission = LoginSubmission(defaults, 'keep', view.revision,
                provider_profile=replace(provider, credential_ref=None), set_default=True)
            return await self.configure_login(submission)
        except BaseException:
            self.cancel_login()
            raise

    # --- 关闭 ----------------------------------------------------------------------

    async def aclose(self) -> None:
        """幂等关闭; 重复调用共享同一次关闭任务和结果, 不重复 stop."""
        if self._close_task is None:
            self._closing = True
            if self._catalog_token is not None:
                self._catalog_token.cancel()
            if self._session_token is not None:
                self._session_token.cancel()
            for query_token in self._file_queries.values():
                query_token.cancel()
            self.cancel_review()
            self.cancel_rewind()
            self._cancel_login_save()
            self.cancel_login()
            self._close_task = asyncio.get_running_loop().create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        # 关闭意图已锁存 (调用方 UI 应先行恢复其输入/光标模式); closing 不可逆
        self._closing = True
        self._mode_controller.broker.close()
        if self._state == "idle":
            self._state = "closing"
        deadline = time.monotonic() + CLOSE_SUPERVISION_S
        self._close_deadline = deadline
        self._close_latch.set()
        if self._catalog_task is not None:
            # 目录 worker 的实际 finally 先于 runtime/store 清理，无监督替代。
            await asyncio.gather(self._catalog_task, return_exceptions=True)
        if self._file_queries:
            await asyncio.gather(*tuple(self._file_queries), return_exceptions=True)
        if self._login_task is not None and not self._login_task.done():
            # 不以监督超时代替文件 worker 的真实终结，不提前关闭资源。
            await asyncio.shield(self._login_task)
        # 02/PA05: 永久封住未接受的摘要提交并协作取消摘要生成;
        # 已被 store 接受的提交由其生产者等明确结果 (PA03), 不被取消掩盖
        self._compact_sealed = True
        if self._compact_token is not None:
            self._compact_token.cancel()
        # 取消活动请求并等待现有请求/清空/压缩生产者终结 (含持久提交与活动写入),
        # 共享同一截止时间; execute/clear/compact 互斥, 至多一个在途
        producers = [
            done for done in (self._active_done, self._clear_done, self._compact_done, self._model_done, self._review_task, self._rewind_task)
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
            except asyncio.CancelledError:
                if producer not in (self._review_task, self._rewind_task) or not producer.done():
                    raise
                # reader 的协作取消是正常页面收尾，不取消应用 close。
            except Exception:
                if producer not in (self._review_task, self._rewind_task):
                    raise
                self._close_problems.append("review reading failed during close")
        # 配置接受后不能在 COMMIT 与内存发布之间释放 store/client 或观察者。
        # 监督超时记录诊断, 不能替代事务明确结果; model 生产者仍受拥有。
        if self._model_done is not None and not self._model_done.done():
            await asyncio.shield(self._model_done)
        # 命名提交已接受时，取得真实结果并发布后才清理存储和观察者。
        if self._session_done is not None and not self._session_done.done():
            await asyncio.shield(self._session_done)
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
