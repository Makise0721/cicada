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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from cicada.boot import App, BootError, bootstrap
from cicada.core.agent import RunResult
from cicada.core.cancel import CancelToken
from cicada.core.events import Event, RunFinished, RunStarted, ToolCompleted
from cicada.core.messages import Message, isolate_message, isolate_tool_result
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
from cicada.run_policy import RunPolicy, verification_policy
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

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, Path):
            raise ValueError("workspace must be a Path")
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a str")
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if self.plan is not None and not isinstance(self.plan, VerificationPlan):
            raise ValueError("plan must be a VerificationPlan or None")


@dataclass(frozen=True)
class RequestOutcome:
    """单个用户请求的权威终结; 复用现有记录, 不另造第二份检查回执."""

    result: RunResult
    view: VerificationView | None
    evidence: ChangeEvidence | None
    decision: DeliveryDecision | None
    exit_code: int


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
class ApplicationClosed:
    """关闭完成或明确清理失败; error 非空不代表所有资源均已成功释放."""

    error: str | None = None


ApplicationUpdate = (
    AgentEventReceived | RequestPhaseChanged | RequestCompleted | HistoryCleared | ApplicationClosed
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
    ) -> None:
        self._app = app
        self._config = config
        self._service = service
        self._policy = policy
        self._history: list[Message] = []
        self._state: SessionStateName = "idle"
        self._run_id: str | None = None
        self._observers: list[Callable[[ApplicationUpdate], None]] = []
        self.observer_errors: list[BaseException] = []
        self._cancel_token: CancelToken | None = None
        self._active_done: asyncio.Future | None = None
        self._process_uncertain = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task | None = None
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
        if self._config.plan is not None and self._process_uncertain:
            raise RuntimeError(
                "process_uncertain: new coding requests are refused; only status, clear or exit remain"
            )
        loop = asyncio.get_running_loop()
        self._state = "executing"
        self._run_id = None
        token = CancelToken()
        self._cancel_token = token
        self._active_done = loop.create_future()
        self._notify(RequestPhaseChanged(None, "executing"))
        try:
            result = await self._app.agent.run(prompt, token, history=tuple(self._history))
            # 内核结束 != 应用空闲: 显式进入收尾状态, 完成前不接受新请求 (T07)
            self._state = "finalizing"
            outcome = await self._finalize_request(result)
        except BaseException:
            # 异常路径也不得把应用留在"可执行"状态: 内核/UI 异常一律经关闭路径处理
            self._state = "closing"
            self._closing = True
            raise
        finally:
            if self._active_done is not None and not self._active_done.done():
                self._active_done.set_result(None)
        # 先更新隔离的 state/history, 再通知订阅者; 应用把后缀追加一次
        self._history.extend(isolate_message(m) for m in result.messages)
        if outcome.view is not None and outcome.view.process_uncertain:
            self._process_uncertain = True
        if self._closing:
            self._state = "closing"
        else:
            self._state = "idle"
            self._notify(RequestPhaseChanged(result.run_id, "idle"))
        self._notify(RequestCompleted(outcome))
        return outcome

    async def _finalize_request(self, result: RunResult) -> RequestOutcome:
        """验证收尾: 扫描末次终结事实 -> refresh -> finalize -> 工件核对 -> 交付判定.

        建立拟定 15 秒监督窗口; 超时取消收尾任务并返回明确不可交付结果.
        """
        if self._config.plan is None or self._service is None or self._policy is None:
            decision = None
            view = None
            evidence = None
        else:
            deadline = time.monotonic() + REQUEST_FINALIZE_SUPERVISION_S
            view, evidence, decision = await self._run_finalization(result, deadline)
        return RequestOutcome(
            result=result,
            view=view,
            evidence=evidence,
            decision=decision,
            exit_code=_outcome_exit_code(result, decision),
        )

    async def _run_finalization(
        self, result: RunResult, deadline: float
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

        task = asyncio.create_task(work())
        remaining = max(0.0, deadline - time.monotonic())
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.TimeoutError:
            token.cancel()
            task.cancel()
            done, pending = await asyncio.wait([task], timeout=_UNDEAD_GRACE_S)
            if pending:
                # 监督到点仍未终结的任务保留强引用与诊断, 不丢给隐式全量取消
                self._undead_tasks.append(task)
            reason = (
                "verification finalization did not complete within the "
                f"{REQUEST_FINALIZE_SUPERVISION_S:g}s supervision window"
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

    # --- 取消 / 清空 ---------------------------------------------------------------

    def cancel(self) -> None:
        """请求取消活动 token; 幂等; 不伪造完成, 不直接杀 UI."""
        if self._state == "executing":
            self._state = "cancelling"
            self._notify(RequestPhaseChanged(self._run_id, "cancelling"))
        if self._cancel_token is not None:
            self._cancel_token.cancel()

    def clear_history(self) -> None:
        """仅空闲允许; 清除模型历史, 保留 App/Agent、基线/账本/锁存和已有工件."""
        if self._closing or self._closed or self._state != "idle":
            raise RuntimeError(f"history can only be cleared while idle and open (state={self._state})")
        self._history.clear()
        self._notify(HistoryCleared())

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
        # 取消活动请求并等待现有请求路径 (含 P4 收尾) 终结, 共享同一截止时间
        if self._active_done is not None and not self._active_done.done():
            if self._cancel_token is not None:
                self._cancel_token.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._active_done),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                self._close_problems.append(
                    "the active request did not finalize before the close supervision deadline"
                )
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


async def open_application(config: ApplicationConfig) -> ApplicationSession:
    """输入/preflight/boot/initialize 成功才返回; 部分启动失败关闭已取得资源."""
    plan = config.plan
    policy_state: dict[str, object] = {}

    def _model_policy(model, runtime):
        service = runtime.capability(VERIFICATION_CAPABILITY)
        wrapped = verification_policy(model, service)
        policy_state["policy"] = wrapped
        policy_state["service"] = service
        return wrapped

    try:
        app: App = await bootstrap(
            default_definitions(config.workspace, config.model_plugin, plan),
            tool_capabilities=CHECK_TOOL_CAPABILITIES if plan is not None else TOOL_CAPABILITIES,
            system_prompt=config.system_prompt,
            model_policy=_model_policy if plan is not None else None,
            max_turns=config.max_turns,
        )
    except BootError as exc:
        raise ApplicationOpenError(f"boot failed: {exc}") from exc

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
    return session
