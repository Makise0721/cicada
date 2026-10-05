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
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

from cicada.boot import App, BootError, bootstrap
from cicada.core.agent import RunResult
from cicada.core.cancel import CancelToken
from cicada.core.events import Event, RunFinished, RunStarted, ToolCompleted
from cicada.core.messages import Message, ToolResultMessage, isolate_message, isolate_tool_result
from cicada.persistence.codec import dumps_strict, encode_messages
from cicada.persistence.contracts import (
    ArchiveLimitError,
    InterruptedRequest,
    LaunchIdentity,
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
    conflicting_launch_fields,
    receipt_terminal_trustworthy,
)
from cicada.process_activity import ActivityCoordinator, ProcessActivityWrapper
from cicada.run_policy import (
    HistoricalPrefixProof,
    HistoricalProjection,
    PrefixEntry,
    RunPolicy,
    verification_policy,
)
from cicada.plugins.session_store import SQLiteSessionStore, open_session_store, session_store_plugin
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
from cicada.prompting import PromptInfo
from cicada.reporting import DeliveryDecision, decide_delivery
from cicada.runtime.plugin import PluginDefinition

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
    workspace: Path, model_plugin: PluginDefinition, plan: VerificationPlan | None = None
) -> list[PluginDefinition]:
    """内置 Coding 能力 + 模型插件 (+ 检查模式验证服务) 的启动名单."""
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


@dataclass(frozen=True)
class PersistenceConfig:
    """持久会话入口配置; session_id=None 表示创建新会话, 否则按精确 ID 恢复.

    identity 是启动层给出的有效配置身份 (默认值已解析); 04 前恢复要求与
    存档完全一致, 冲突在 bootstrap 前拒绝.
    """

    session_id: str | None
    identity: LaunchIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LaunchIdentity):
            raise ValueError("PersistenceConfig.identity must be a LaunchIdentity")
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise ValueError("PersistenceConfig.session_id must be a str or None")


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
    """非请求操作进行中 (当前: 持久清空提交).

    只读补充原请求 phase: 请求生命周期已有 RequestPhaseChanged, 清空等操作
    经本事件让 UI 输入屏障可消费 (05 接线); 完成由 HistoryCleared 标记.
    """

    operation: Literal["clear"]


@dataclass(frozen=True)
class ApplicationClosed:
    """关闭完成或明确清理失败; error 非空不代表所有资源均已成功释放."""

    error: str | None = None


ApplicationUpdate = (
    AgentEventReceived | RequestPhaseChanged | RequestCompleted | HistoryCleared
    | OperationBusy | ApplicationClosed
)


class ApplicationOpenError(RuntimeError):
    """启动组装或检查基线建立失败; 显示文本即异常文本."""


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
        view_only_reason: str | None = None,
        projection: HistoricalProjection | None = None,
    ) -> None:
        self._app = app
        self._config = config
        self._service = service
        self._policy = policy
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
        self._context_generation = restored_context_generation
        self._request_id: str | None = None
        self._saved_request_count = 0
        self._last_committed_seq: int | None = None
        self._storage_failed: str | None = None
        self._state: SessionStateName = "idle"
        self._run_id: str | None = None
        self._observers: list[Callable[[ApplicationUpdate], None]] = []
        self.observer_errors: list[BaseException] = []
        self._cancel_token: CancelToken | None = None
        self._active_done: asyncio.Future | None = None
        self._clear_done: asyncio.Future | None = None  # 清空生产者完成标记 (close 监督对象)
        self._clearing = False  # 持久清空进行中: 与 execute 互斥 (契约 §3)
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
    def view_only_reason(self) -> str | None:
        """非空 = 恢复准入发现未证终结活动, 仅可查看; 新编码请求被拒绝 (契约 D3).

        清空不解锁此状态; 普通/检查模式及配置覆盖都不能绕过 (A15).
        """
        return self._view_only_reason

    @property
    def active_request_id(self) -> str | None:
        """当前 (或最近一次) 请求的持久 ID; 活动包装器经协调器读取."""
        return self._request_id

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

    async def execute(self, prompt: str) -> RequestOutcome:
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
        loop = asyncio.get_running_loop()
        self._state = "executing"
        self._run_id = None
        token = CancelToken()
        self._cancel_token = token
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
        if self._store is not None:
            raise RuntimeError(
                "persistent sessions must use clear_context(); the synchronous entry is refused "
                "to keep the clear boundary reliable"
            )
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        self._history.clear()
        self._notify(HistoryCleared())

    async def clear_context(self) -> None:
        """持久清空: 先可靠提交清空边界, 成功后才清内存历史; 失败内存不变.

        无持久化模式委托旧同步入口; storage_failed 不阻止清空尝试, 失败如实报错.
        清空进行中 (_clearing) 与 execute 互斥, 双向拒绝 (契约 §3).
        """
        if self._store is None:
            self.clear_history()
            return
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        if self._clearing:
            raise RuntimeError("a persistent clear is already in progress")
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
            self._notify(HistoryCleared())
            if self._projection is not None:
                # 前缀证明与内存历史同步重置; 失败路径不会到达这里 (契约 §5)
                self._projection.reset()
            if clear_cancelled:
                raise asyncio.CancelledError()
        finally:
            self._clearing = False
            if self._clear_done is not None and not self._clear_done.done():
                self._clear_done.set_result(None)

    # --- 关闭 ----------------------------------------------------------------------

    async def aclose(self) -> None:
        """幂等关闭; 重复调用共享同一次关闭任务和结果, 不重复 stop."""
        if self._close_task is None:
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
        # 取消活动请求并等待现有请求/清空生产者终结 (含持久提交与活动写入),
        # 共享同一截止时间; execute 与 clear 互斥, 至多一个在途
        producers = [
            done for done in (self._active_done, self._clear_done)
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
    policy_state: dict[str, object] = {}
    # 恢复前缀证明容器 (03): 检查模式 + 持久会话时由 RunPolicy 每次发送前核对
    history_projection: HistoricalProjection | None = None

    def _model_policy(model, runtime):
        service = runtime.capability(VERIFICATION_CAPABILITY)
        wrapped = verification_policy(model, service, history=history_projection)
        policy_state["policy"] = wrapped
        policy_state["service"] = service
        return wrapped

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

    definitions: list[PluginDefinition] = []
    if store is not None:
        definitions.append(session_store_plugin(store))
        definitions.extend(_bind_store_last_cleanup(
            default_definitions(config.workspace, config.model_plugin, plan)
        ))
    else:
        definitions.extend(default_definitions(config.workspace, config.model_plugin, plan))

    if store is not None:
        assert snapshot is not None and config.persistence is not None
        try:
            view_only_reason = _admit_persistent_resume(config.persistence, snapshot)
            await _run_preflight(preflight)
            app: App = await bootstrap(
                definitions,
                tool_capabilities=CHECK_TOOL_CAPABILITIES if plan is not None else TOOL_CAPABILITIES,
                system_prompt=config.system_prompt,
                model_policy=_model_policy if plan is not None else None,
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
                tool_capabilities=CHECK_TOOL_CAPABILITIES if plan is not None else TOOL_CAPABILITIES,
                system_prompt=config.system_prompt,
                model_policy=_model_policy if plan is not None else None,
                max_turns=config.max_turns,
            )
        except BootError as exc:
            raise ApplicationOpenError(f"boot failed: {exc}") from exc

    if store is not None and snapshot is not None and config.persistence is not None:
        # 指纹核对/启动代次/会话构造的每条失败路径都必须释放 app (store 随 runtime 清理)
        try:
            fingerprint = _tool_fingerprint(app)
            saved_identity = snapshot.identity
            if saved_identity is not None and saved_identity.tool_fingerprint != fingerprint:
                raise ApplicationOpenError(
                    "restored session tool declarations differ from the current launch "
                    "(tool_fingerprint)"
                )
            try:
                epoch_id = await store.start_epoch(
                    StoredIdentity(launch=config.persistence.identity, tool_fingerprint=fingerprint)
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
                view_only_reason=view_only_reason,
                projection=history_projection,
            )
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
