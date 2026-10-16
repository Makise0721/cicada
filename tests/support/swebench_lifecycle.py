"""PV02 生产生命周期 launcher: R1/R2 → 关闭 → 新进程恢复 → 摘要 → R3。

编排只走生产接缝 (``open_application`` / ``execute`` / ``compact_context`` /
``aclose`` 与真实恢复入口), 不拷贝模型循环、不改写规范历史; 只注入 transport
(生产 ``deepseek_plugin`` 已公开的 ``transport`` 参数) 与只读事件观察。

两条通道都经 ``AttemptBudget``: 普通 (thinking enabled / cap 8192) 与摘要
(thinking disabled / cap 2048) 在真实 transport 发送前都必须先预留成功, 否则
**零发送**。请求形态用旧栈的 ``classify_request`` 资格事实判定, 不重写一份。

诚实边界: 本模块只证明"两个已提交只读请求 → 正常关闭 → 新进程真实恢复 → 摘要
→ R3"这一生命周期; 不证明模型能力, 也不产生真实网络请求 (inner 由调用方传入:
离线为 MockTransport, live 需另获授权)。摘要通道是否独立于普通通道, 由模型是否
实现 ``ModelSelectionPort`` 决定 —— 真实 DeepSeek 适配器是独立的 (think/max_tokens
各不同), 而 FakeModel 会与普通通道共用同一实例。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import contextlib
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx

from cicada.application import (
    ApplicationConfig,
    ApplicationClosed,
    ContextCompacted,
    PersistenceConfig,
    RequestCompleted,
    open_application,
    verification_checks_signature,
)
from cicada.budget import BudgetConfig
from cicada.core.ports import ModelMetrics
from cicada.persistence.contracts import LaunchIdentity
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import VerificationPlan
from cicada.plugins.deepseek.plugin import deepseek_plugin
from cicada.plugins.deepseek.protocol import DeepSeekConfig
from cicada.prompting import PromptInfo
from tests.support.swebench_budget import (
    PROCESS_1,
    PROCESS_2,
    AttemptBudget,
    AttemptBudgetError,
)
from tests.support.swebench_cost import classify_request

# 离线演练用的占位凭据: 永远不会离开本机 (inner 是 MockTransport)。
# 名字刻意显眼, 避免被误当成真实密钥。
OFFLINE_CREDENTIAL = "offline-rehearsal-placeholder-not-a-real-key"

# --- 冻结 prompt 模板 (规格 §2) ---

WORKSPACE_NOTE = (
    "上述 /app 指本次 Windows 工作区, 源码与官方基础版本一致。请自行读取与定位问题, "
    "只修改允许的业务文件。先用 check-1 确认初始未通过, 修改后用 check-1 重检; "
    "检查在固定 Linux 容器运行。不得改测试、样本、依赖或项目指令, 不查找参考修复, "
    "不联网、不提交。不要在宿主机对工作区执行 python 或导入其代码, 验证只用 check-1。"
    "普通环境是 Windows/PowerShell。"
)

R1_SUFFIX = (
    "本请求仅阅读相关源码、定位问题并执行一次指定 check-1, 最后简述观察与下一步; "
    "禁止编辑、写入业务文件或实施修复。"
)
R2_SUFFIX = (
    "仅依据本会话已有观察, 陈述待改文件路径、预期行为及完成后应重新执行的指定检查; "
    "本请求只作简短说明, 不调用工具、不编辑、不实施修复。"
)
R3_SUFFIX = (
    "继续此前同一公开任务, 独立实施所需修复; 只修改允许的业务路径, 完成后重新执行"
    "本进程指定 check-1, 并如实报告结果。仍遵守原始公开题意、禁止联网和禁止修改"
    "测试/依赖/项目指令的纪律。"
)


def build_prompts(instruction: str) -> dict:
    """展开三个最终 UTF-8 prompt 并记录 SHA256 (真实运行不动态追加指导)。"""
    prompts = {
        "R1": f"{instruction}\n{WORKSPACE_NOTE}\n{R1_SUFFIX}",
        "R2": f"{instruction}\n{WORKSPACE_NOTE}\n{R2_SUFFIX}",
        "R3": f"{instruction}\n{WORKSPACE_NOTE}\n{R3_SUFFIX}",
    }
    return {"prompts": prompts,
            "sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                       for name, text in prompts.items()}}


# --- 预算闸 transport ---


@dataclass
class GateObservation:
    call_id: str
    kind: str
    output_cap: int
    http_status: int | None = None
    usage: dict | None = None
    done_seen: bool = False
    eof_seen: bool = False
    closed: bool = False
    transport_error: str | None = None
    resolved: bool = False
    # 结算需要**两侧证据齐备**: transport 侧 (2xx + usage + DONE + EOF + 流已关闭)
    # 与**生产侧** (模型自己的终结事件, 且其计量与观察到的 usage 逐位一致)。
    # 规格 §3 要求"完整合法 usage、成功终局、DONE、EOF、响应关闭和生产确认
    # 全部具备后才结算一次"; 缺生产确认就保留 pending 并封口, 不当作已结算。
    transport_complete: bool = False
    production_seen: bool = False
    production_terminal: str | None = None
    production_ok: bool = False


class AttemptGateTransport(httpx.AsyncBaseTransport):
    """发送前经 ``AttemptBudget`` 预留; 两侧证据齐备才结算。

    结算条件刻意严格: transport 事实与生产确认缺一不可, 任一缺失都保留 pending
    并封口 (不把无法确认的发送写成已结算)。
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, budget: AttemptBudget, *,
                 phase: str, session: str | None = None,
                 deadline_monotonic: float | None = None) -> None:
        self.inner = inner
        self.budget = budget
        self.phase = phase
        self.session = session
        self.deadline = deadline_monotonic
        self.observations: dict[str, GateObservation] = {}
        self.handoffs = 0
        self.settled_ids: list[str] = []
        self.resolved_ids: list[str] = []
        self.network_sealed_reason: str | None = None
        self.active_call_id: str | None = None
        self.admitting = False

    def seal_network(self, reason: str) -> None:
        self.network_sealed_reason = self.network_sealed_reason or reason

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.network_sealed_reason:
            raise AttemptBudgetError(f"network sealed: {self.network_sealed_reason}")
        if self.active_call_id is not None or self.admitting:
            self.seal_network("concurrent-send-refused")
            raise AttemptBudgetError("concurrent send refused")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.seal_network("run-timeout")
            raise AttemptBudgetError("run timeout")
        self.admitting = True
        try:
            body = await request.aread()
            try:
                kind, cap, _model = classify_request(body)
            except Exception as exc:  # noqa: BLE001 - 形态越界即拒发
                self.budget.seal("request shape outside the frozen contract")
                raise AttemptBudgetError(
                    f"request identity outside the frozen contract: {exc}") from None
            call_id = uuid.uuid4().hex
            try:
                self.budget.reserve(call_id=call_id, phase=self.phase, kind=kind,
                                    output_cap=cap, session=self.session)
            except AttemptBudgetError:
                self.seal_network("attempt-budget-refused")
                raise
        finally:
            self.admitting = False

        observation = GateObservation(call_id=call_id, kind=kind, output_cap=cap)
        self.observations[call_id] = observation
        self.active_call_id = call_id
        try:
            self.handoffs += 1
            response = await self.inner.handle_async_request(request)
        except BaseException as exc:
            observation.transport_error = type(exc).__name__
            self.seal_network("http-handoff-failure")
            self.budget.seal(f"handoff failed for {call_id}")
            self.active_call_id = None
            raise
        observation.http_status = response.status_code
        response.stream = _ObservedStream(response.stream, observation, self)
        return response

    def _finish(self, observation: GateObservation) -> None:
        """流耗尽: 记录 transport 侧事实; 不完整即**立刻封口**, 完整则等生产确认。

        幂等: httpx 可能在迭代结束与显式关闭时各调一次 ``aclose``。第二次进入
        直接返回, 而不是把"已结算"当矛盾抛出去 (那会把正常收尾误报成失败)。
        """
        if observation.transport_complete or observation.resolved:
            return
        self.active_call_id = None
        observation.transport_complete = (
            observation.transport_error is None
            and isinstance(observation.http_status, int)
            and 200 <= observation.http_status < 300
            and observation.done_seen
            and observation.eof_seen
            and observation.closed
            and isinstance(observation.usage, dict))
        if not observation.transport_complete:
            # 规格: 部分异常/usage 缺失/关闭失败一律**保留 pending 并封口**。
            # 不回退成"留着不封口"—— 那会允许下一笔发送与一条未决预留并存。
            self._resolve(observation, complete=False)
            return
        self._maybe_settle(observation)

    def confirm(self, call_id: str, terminal) -> None:
        """生产侧确认: 模型自己的终结事件必须成功, 且计量与观察到的 usage 一致。

        由 ``observe_model_stream`` 在模型 stream 耗尽后调用 (与旧栈同一接缝)。
        """
        observation = self.observations.get(call_id)
        if observation is None or observation.resolved:
            return
        observation.production_seen = True
        observation.production_terminal = None if terminal is None else terminal.stop_reason
        ok = (terminal is not None and terminal.stop_reason in ("stop", "tool_use")
              and terminal.error is None and terminal.metrics is not None)
        if ok and isinstance(observation.usage, dict):
            usage = observation.usage
            try:
                ok = (terminal.metrics.input_tokens == int(usage["prompt_tokens"])
                      and terminal.metrics.output_tokens == int(usage["completion_tokens"]))
            except (KeyError, TypeError, ValueError):
                ok = False
        observation.production_ok = bool(ok)
        if not observation.production_ok:
            # 生产确认失败同样是"部分异常": 立刻封口并保留 pending。
            self._resolve(observation, complete=False)
            return
        self._maybe_settle(observation)

    def _maybe_settle(self, observation: GateObservation) -> None:
        """两侧证据齐备才结算一次; 未齐备则保持未决 (不猜测)。"""
        if observation.resolved:
            return
        if not (observation.transport_complete and observation.production_seen):
            return
        self._resolve(observation, complete=observation.production_ok)

    def _resolve(self, observation: GateObservation, *, complete: bool) -> None:
        observation.resolved = True
        metrics = None
        if complete and isinstance(observation.usage, dict):
            usage = observation.usage
            try:
                metrics = ModelMetrics(input_tokens=int(usage["prompt_tokens"]),
                                       output_tokens=int(usage["completion_tokens"]))
            except (KeyError, TypeError, ValueError):
                metrics = None
        # 结算可能返回 False (封口保留 pending); 分开记"已了结"与"真结算", 否则
        # 把未结算的也报成 settled 会让人误以为金额已释放 (审查 P3-1)。
        settled_ok = bool(self.budget.settle(observation.call_id, metrics,
                                             complete=complete,
                                             closed=observation.closed))
        self.resolved_ids.append(observation.call_id)
        if settled_ok:
            self.settled_ids.append(observation.call_id)


class _ObservedStream(httpx.AsyncByteStream):
    """只读观察 SSE: usage 帧 / [DONE] / EOF / close。"""

    def __init__(self, inner: httpx.AsyncByteStream, observation: GateObservation,
                 transport: AttemptGateTransport) -> None:
        self.inner = inner
        self.observation = observation
        self.transport = transport
        self.buffer = b""

    async def __aiter__(self):
        try:
            async for chunk in self.inner:
                self._scan(chunk)
                yield chunk
        except BaseException as exc:
            # 流中途异常必须留痕: 否则"已见 usage/DONE"会把一条失败的流误判为完整
            # (审查 P1-A 实测: DONE 之后连接重置曾被当作完整结算)。
            self.observation.transport_error = type(exc).__name__
            raise
        else:
            # 只有**自然耗尽**才算 EOF; 消费者提前 break 时 eof_seen 保持 False
            # (保守: 未证完整就不结算)。
            self.observation.eof_seen = True

    def _scan(self, chunk: bytes) -> None:
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.observation.done_seen = True
                continue
            try:
                frame = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            usage = frame.get("usage")
            if isinstance(usage, dict):
                self.observation.usage = usage

    async def aclose(self) -> None:
        try:
            await self.inner.aclose()
        finally:
            self.observation.closed = True
            self.transport._finish(self.observation)


# --- 生产确认: 包装模型 stream 取得终结事实 ---


def observe_model_stream(original, gate):
    """包装 ``DeepSeekModel.stream``, 把模型自己的终结事件交给闸做生产确认。

    与旧栈同一接缝: 一次 stream 期间新出现的预订必须恰好一条, 否则视为归因不明
    并封口 (宁可停在未结算, 也不把别人的 usage 记到这笔上)。事件与请求内容不变。
    """
    from cicada.core.ports import StreamDone

    async def observed(model, request, cancel):
        before = len(gate.observations)
        terminal = None
        completed = False
        try:
            async for event in original(model, request, cancel):
                if isinstance(event, StreamDone):
                    terminal = event
                else:
                    yield event
            completed = True
        finally:
            new_ids = list(gate.observations)[before:]
            if len(new_ids) != 1:
                gate.seal_network("ambiguous-call-correlation")
                gate.budget.seal("ambiguous call correlation")
            else:
                gate.confirm(new_ids[0], terminal if completed else None)
        if terminal is not None:
            yield terminal

    return observed


@contextlib.contextmanager
def production_confirmation(gate):
    """在会话使用期间把生产确认接到 ``DeepSeekModel.stream`` 上, 退出时严格还原。"""
    from cicada.plugins.deepseek.model import DeepSeekModel

    original = DeepSeekModel.stream
    DeepSeekModel.stream = observe_model_stream(original, gate)
    try:
        yield
    finally:
        DeepSeekModel.stream = original


# --- 生产启动配置 ---


@dataclass(frozen=True)
class LifecycleSpec:
    """一次生命周期运行所需的冻结输入 (来自 task manifest 与 prepare 产物)。"""

    workspace: Path
    instruction: str
    checks: tuple = ()
    model_doc: dict = field(default_factory=dict)
    max_turns: int = 50
    restore_capture: bool = False
    # 生产进程账本 (cicada.costs.CostLedger); None = 本次不安装进程级费用控制。
    # 与 AttemptBudget 是两本账: 前者按产品规则"重启新建额度", 后者是 attempt 级总闸。
    cost_control: object | None = None


def _deepseek_config(spec: LifecycleSpec) -> DeepSeekConfig:
    normal = spec.model_doc["normal"]
    return DeepSeekConfig(model=normal["model"], think=bool(normal["thinking"] == "enabled"),
                          reasoning_effort=normal["reasoning_effort"],
                          num_ctx=int(normal["num_ctx"]),
                          max_tokens=int(normal["max_tokens"]))


def launch_identity(spec: LifecycleSpec, prompt_info: PromptInfo,
                    config: DeepSeekConfig) -> LaunchIdentity:
    safe = config.identity
    return LaunchIdentity(
        provider=safe.provider, model=safe.model, base_url=safe.base_url, think=safe.think,
        options=dict(safe.options), max_request_bytes=safe.max_request_bytes,
        max_turns=spec.max_turns, prompt_version=prompt_info.version,
        prompt_sha256=prompt_info.prompt_sha256,
        instructions_sha256=prompt_info.instructions_sha256,
        checks_signature=verification_checks_signature(spec.checks) if spec.checks else None,
        sources={"harness": "pv02"})


def _plan(spec: LifecycleSpec) -> VerificationPlan | None:
    if not spec.checks:
        return None
    root = Path(str(spec.workspace))
    return VerificationPlan(verification_run_id=generate_verification_run_id(),
                            root=root, checks=tuple(spec.checks))


@dataclass
class _EventLog:
    completed: int = 0
    compacted: list = field(default_factory=list)
    events: list = field(default_factory=list)
    closed_events: int = 0
    close_problem: str | None = None

    def __call__(self, update) -> None:
        self.events.append(update)
        if isinstance(update, RequestCompleted):
            self.completed += 1
        elif isinstance(update, ContextCompacted):
            self.compacted.append(update)
        elif isinstance(update, ApplicationClosed):
            self.closed_events += 1
            self.close_problem = "runtime stop failed or close incomplete" if update.error else None


async def close_process(session, log: _EventLog, facts: dict) -> None:
    await session.aclose()
    facts["closed_ok"] = log.closed_events == 1 and log.close_problem is None
    facts["close_problem"] = log.close_problem


async def open_process(spec: LifecycleSpec, transport, *, session_id: str | None,
                       system_prompt: str, prompt_info: PromptInfo):
    """按生产配置打开一个进程的 ApplicationSession (新建或按 ID 恢复)。"""
    config = _deepseek_config(spec)
    app_config = ApplicationConfig(
        workspace=spec.workspace,
        system_prompt=system_prompt,
        prompt_info=prompt_info,
        model_plugin=deepseek_plugin(config, api_key=OFFLINE_CREDENTIAL,
                                     transport=transport),
        plan=_plan(spec),
        max_turns=spec.max_turns,
        model_label=config.model,
        persistence=PersistenceConfig(
            session_id=session_id,
            identity=launch_identity(spec, prompt_info, config)),
        budget=BudgetConfig(config.num_ctx, config.max_tokens),
        restore_capture=spec.restore_capture,
        cost_control=spec.cost_control,
    )
    session = await open_application(app_config)
    log = _EventLog()
    session.subscribe(log)
    return session, log


def _request_facts(name: str, outcome) -> dict:
    facts = {"name": name, "exit_code": outcome.exit_code,
             "stop_reason": outcome.result.stop_reason}
    persistence = outcome.persistence
    if persistence is not None:
        facts.update({"saved": persistence.saved, "request_id": persistence.request_id,
                      "seq": persistence.seq, "save_error": persistence.error})
    return facts


async def run_process_1(spec: LifecycleSpec, transport, *, system_prompt: str,
                        prompt_info: PromptInfo) -> dict:
    """R1 + R2 两个只读请求, 逐一自然终局后关闭; 返回可核对事实。"""
    session, log = await open_process(spec, transport, session_id=None,
                                      system_prompt=system_prompt,
                                      prompt_info=prompt_info)
    from tests.support import swebench_bridge as bridge

    git = bridge.GitCLI(spec.workspace)
    manifest_before = bridge.workspace_manifest(spec.workspace, git)
    facts: dict = {"requests": [], "manifest_unchanged": None,
                   "session_id": None, "compacted": 0}
    prompts = build_prompts(spec.instruction)["prompts"]
    try:
        facts["session_id"] = session.persistence.session_id
        # 生产确认必须在场: 每次发送的结算要求模型自己的终结事件与观察到的 usage 一致。
        with production_confirmation(transport):
            for name in ("R1", "R2"):
                outcome = await session.execute(prompts[name])
                facts["requests"].append(_request_facts(name, outcome))
                # 每个请求结束都检查业务 manifest 未变 (只读纪律)。
                manifest_after = bridge.workspace_manifest(spec.workspace, git)
                if manifest_after.to_json() != manifest_before.to_json():
                    facts["manifest_unchanged"] = False
                    facts["dirty_after"] = name
                    break
            else:
                facts["manifest_unchanged"] = True
        facts["completed_events"] = log.completed
        facts["state"] = session.persistence.mode
    finally:
        await close_process(session, log, facts)
    return facts


async def run_process_2(spec: LifecycleSpec, transport, budget: AttemptBudget, *,
                        session_id: str, system_prompt: str, prompt_info: PromptInfo) -> dict:
    """真实恢复同一 session → ``compact_context`` 摘要 → R3。"""
    session, log = await open_process(spec, transport, session_id=session_id,
                                      system_prompt=system_prompt,
                                      prompt_info=prompt_info)
    facts: dict = {"restored_requests": len(session.restored_requests),
                   "restored_receipts": len(session.restored_receipts),
                   "view_only_reason": session.view_only_reason,
                   "restore_warnings": list(session.restore_warnings)}
    prompts = build_prompts(spec.instruction)["prompts"]
    try:
        restored_ids = [r.request_id for r in session.restored_requests]
        facts["restored_request_ids"] = restored_ids
        # 恢复出的旧回执只是历史: 记录其只读暴露, 不作为当前权威。
        facts["historical_receipts_have_current_authority"] = False
        # 摘要与 R3 的发送同样需要生产确认在场才结算。
        with production_confirmation(transport):
            try:
                compacted = await session.compact_context()
            except RuntimeError as exc:
                # 摘要失败时附上模型侧的终结事实, 否则错误只剩一个 "error" 分类。
                from cicada.core.events import RunFinished
                errors = [e.event.error for e in log.events
                          if isinstance(e, RunFinished) and e.event.error]
                exc.add_note(f"summary terminal errors: {errors[-2:]} "
                             f"gate={getattr(transport, 'network_sealed_reason', None)}")
                raise
            facts["compacted"] = None if compacted is None else {
                "summary_id": compacted.summary_id,
                "first_request_seq": compacted.first_request_seq,
                "last_request_seq": compacted.last_request_seq,
                "message_count": compacted.message_count,
                "end_message_index": compacted.end_message_index,
                "reason": compacted.reason}
            facts["context_compacted_events"] = len(log.compacted)
            facts["projection_blocks"] = len(session._context_projection.blocks)
            if compacted is None or not log.compacted or transport.network_sealed_reason:
                raise AttemptBudgetError("summary did not commit; refusing R3")
            if facts["restored_requests"] != 2 or session.view_only_reason:
                raise AttemptBudgetError("restore did not admit two committed requests")
            outcome = await session.execute(prompts["R3"])
        facts["requests"] = [_request_facts("R3", outcome)]
        facts["completed_events"] = log.completed
    finally:
        await close_process(session, log, facts)
    return facts

# --- 离线 fixture (MockTransport) ---

# 响应编排: 索引即"第几次模型请求"。R1 是多轮 (读+检查 → 收尾), R2 单轮,
# 摘要一轮, R3 多轮 (重读 → 写+检查 → 收尾)。
#
# R3 前必须先重读: 写工具要求"已有文件必须先用 read 建立**进程内**版本依据"
# (file_access.py: "existing file must be read with the read tool before write"),
# 恢复不导入读取依据 (与 ADR「恢复不导入依据」一致), 所以第二进程里直接写会被拒。
_FIXTURE_SEQUENCE = ("read-check", "text", "text", "summary", "read",
                     "write-check", "text")

# coding 场景 (new-multifile): 单进程, 无恢复/压缩; 先读定位, 再写并检查, 最后收尾。
_CODING_SEQUENCE = ("read-check", "write-check", "text")


def _sse(frames: list[dict]) -> httpx.Response:
    """把若干帧编成 SSE; 末帧带 usage, 并补 [DONE] (闸要求 DONE+EOF+usage 齐备)。"""
    body = b""
    for frame in frames:
        body += ("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode("utf-8")
    body += b"data: [DONE]\n\n"
    return httpx.Response(200, stream=httpx.ByteStream(body))


def _call_frame(index: int, calls: list[tuple[str, dict]], *, finish: str) -> dict:
    return {"choices": [{"index": 0, "delta": {
        "reasoning_content": "offline rehearsal fixture",
        "tool_calls": [{"index": i, "id": f"fixture-{index}-{i}", "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(args, ensure_ascii=False)}}
                       for i, (name, args) in enumerate(calls)]},
        "finish_reason": finish}],
        "usage": {"prompt_tokens": 100 + index, "completion_tokens": 10}}


def _text_frame(index: int, text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100 + index, "completion_tokens": 10}}


def learned_solution(workspace: Path, base_commit: str, gold_patch: Path,
                     allowed_paths) -> dict:
    """从参考修复学到"正确内容": 在临时 clone 上应用 gold patch 后读取业务文件。

    这样 fixture 写出的修改是**真的能让 check-1 通过**的解, 而不是自说自话的文本。
    """
    scratch = Path(tempfile.mkdtemp(prefix="pv02-fixture-", dir=workspace.parent))
    try:
        clone = scratch / "clone"
        subprocess.run(["git", "clone", "--no-checkout", "--quiet",
                        str(workspace), str(clone)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", base_commit],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "apply", str(Path(gold_patch).resolve())],
                       check=True, capture_output=True)
        return {name: (clone / name).read_text(encoding="utf-8")
                for name in allowed_paths}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def base_solution(workspace: Path, base_commit: str, allowed_paths) -> dict:
    """基线上**已存在**的业务文件内容; 基线里没有的路径不出现在结果里。

    不存在的路径意味着参考解要新建该文件, 只能用 write (edit 要求先存在)。
    """
    scratch = Path(tempfile.mkdtemp(prefix="pv02-base-", dir=workspace.parent))
    try:
        clone = scratch / "clone"
        subprocess.run(["git", "clone", "--no-checkout", "--quiet",
                        str(workspace), str(clone)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", base_commit],
                       check=True, capture_output=True)
        return {name: (clone / name).read_text(encoding="utf-8")
                for name in allowed_paths if (clone / name).is_file()}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def unique_edits(base_text: str, fixed_text: str, *, min_context: int = 3,
                 max_context: int = 4000) -> list[dict]:
    """把 base→fixed 拆成 ``edit`` 工具能接受的 old_text/new_text 对。

    为什么不用 ``write`` 全量覆盖: write 的 content 上限是 50 KiB, 而 new-multifile
    的 basic.py 是 86 KiB —— 全量写会被工具直接拒绝 (实测踩过)。真实模型也是用
    edit 改局部, 这里保持一致。

    old_text 必须**唯一**匹配 (edit 工具的硬要求), 故按需扩大上下文, 直到唯一或
    覆盖整文件。
    """
    import difflib

    base_lines = base_text.splitlines(keepends=True)
    fixed_lines = fixed_text.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, base_lines, fixed_lines, autojunk=False)
    edits: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        context = min_context
        while True:
            lo = max(0, i1 - context)
            hi = min(len(base_lines), i2 + context)
            old = "".join(base_lines[lo:hi])
            new = ("".join(base_lines[lo:i1]) + "".join(fixed_lines[j1:j2])
                   + "".join(base_lines[i2:hi]))
            covers_all = lo == 0 and hi == len(base_lines)
            if covers_all or base_text.count(old) == 1 or context >= max_context:
                break
            context *= 2
        edits.append({"old_text": old, "new_text": new})
    # 自检: 这些编辑必须真的把 base 变成 fixed, 否则 fixture 会写出错误的解。
    for edit in edits:
        if base_text.count(edit["old_text"]) != 1:
            raise ValueError("fixture edit is not uniquely applicable")
    produced = base_text
    for edit in edits:
        produced = produced.replace(edit["old_text"], edit["new_text"])
    if produced != fixed_text:
        raise ValueError("fixture edits do not reproduce the reference solution")
    return edits


def learned_edits(workspace: Path, base_commit: str, gold_patch: Path,
                  allowed_paths) -> dict:
    """{路径: 工具调用规格}; 已自检"应用后等于参考解"。

    每个路径按基线里是否已存在来选择工具:
      - 基线不存在 (参考解新建它, 如 new-multifile 的 locale.py) → ``write``;
      - 已存在 → ``edit`` 最小替换 (write 的 50 KiB 上限容不下 basic.py 的 86 KiB)。
    """
    fixed = learned_solution(workspace, base_commit, gold_patch, allowed_paths)
    existing = base_solution(workspace, base_commit, allowed_paths)
    plan: dict = {}
    for name in allowed_paths:
        if name not in existing:
            plan[name] = {"tool": "write", "content": fixed[name]}
        else:
            plan[name] = {"tool": "edit",
                          "edits": unique_edits(existing[name], fixed[name])}
    return plan


def offline_fixture(*, edits: dict, allowed_paths, check_id: str = "check-1",
                    sequence=_FIXTURE_SEQUENCE, start: int = 0) -> httpx.MockTransport:
    """离线响应: 读+检查 → 说明 → 摘要 → 重读 → 编辑+检查 → 收尾。

    只读纪律由编排体现: R1 只发 read/check, R2 与摘要不发工具, 修改只出现在实施轮。

    ``edits`` 是 {路径: [edit, ...]}; 用 ``edit`` 而非 ``write``, 因为 write 的
    content 上限 50 KiB 而 basic.py 有 86 KiB (实测), 且真实模型也是改局部。

    ``start`` 是**跨进程**的序列起点: 两个进程各自计数器独立但剧本连续, 第二进程
    必须从第一进程消耗处继续, 否则摘要会拿到给 R1 的响应 (实测踩过)。
    """
    counter = {"n": start}

    def respond(request: httpx.Request) -> httpx.Response:
        index = counter["n"]
        counter["n"] += 1
        if index >= len(sequence):
            raise ValueError(f"offline fixture exhausted at request {index + 1}")
        step = sequence[index]
        if step == "read-check":
            calls = [("read", {"path": name}) for name in allowed_paths]
            calls.append(("check", {"action": "run", "check_id": check_id}))
            return _sse([_call_frame(index, calls, finish="tool_calls")])
        if step == "read":
            calls = [("read", {"path": name}) for name in allowed_paths]
            return _sse([_call_frame(index, calls, finish="tool_calls")])
        if step == "write-check":
            calls = []
            for name in allowed_paths:
                plan = edits[name]
                if plan["tool"] == "write":
                    calls.append(("write", {"path": name, "content": plan["content"]}))
                else:
                    calls.append(("edit", {"path": name, "edits": plan["edits"]}))
            calls.append(("check", {"action": "run", "check_id": check_id}))
            return _sse([_call_frame(index, calls, finish="tool_calls")])
        if step == "summary":
            return _sse([_text_frame(index, "较早请求的摘要: 定位与说明。")])
        return _sse([_text_frame(index, "offline rehearsal complete.")])

    return httpx.MockTransport(respond)

# --- 离线全路径演练 (preview_runner run 的离线入口) ---


def _runner_command(manifest_path: Path, prepared_root: Path, workspace: Path) -> str:
    """check-1 的生产命令。

    所有路径**必须绝对化**: check 工具以 workspace 为 cwd 运行该命令, 相对路径会
    落到工作区里 (实测踩过: 报 "manifest missing ...\\workspace\\.scratch\\...")。
    """
    import sys as _sys

    runner = (Path(__file__).resolve().parents[1] / "e2e" / "preview_runner.py").resolve()
    manifest_path = Path(manifest_path).resolve()
    prepared_root = Path(prepared_root).resolve()
    workspace = Path(workspace).resolve()
    return (f"& '{_sys.executable}' '{runner}' check "
            f"--manifest '{manifest_path}' --run-root '{prepared_root}' "
            f"--workspace '{workspace}'")


def source_sha256(path: Path) -> str:
    """Python 源码按 LF 冻结, 兼容 Git 检出时的 CRLF 转换。"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def runtime_identity() -> dict:
    """产出准入的 harness 与生产源码身份。

    为什么要它: admission 若只记题目身份与限额, 换掉 harness 后旧 admission 仍然
    "看起来有效"。把实际产出它的源码 hash 记进准入, 漂移才能被发现 (旧栈同一做法)。
    """
    project = Path(__file__).resolve().parents[2]
    tracked = [
        project / "tests" / "e2e" / "preview_runner.py",
        project / "tests" / "support" / "swebench_budget.py",
        project / "tests" / "support" / "swebench_lifecycle.py",
        project / "tests" / "support" / "swebench_isolation.py",
        project / "tests" / "support" / "swebench_tasks.py",
        project / "tests" / "support" / "swebench_bridge.py",
        project / "tests" / "support" / "swebench_cost.py",
    ]
    tracked.extend(sorted((project / "tests" / "support").glob("swebench_live*.py")))
    tracked.extend(sorted((project / "tests" / "support").glob("swebench_authorization.py")))
    identity = {p.relative_to(project).as_posix(): source_sha256(p)
                for p in tracked if p.is_file()}
    identity.update({p.relative_to(project).as_posix(): source_sha256(p)
                     for p in sorted((project / "src").rglob("*.py"))})
    return identity


def rehearsal_plan(prepared_root: Path, manifest_path: Path) -> dict:
    """离线演练的静态输入核对与 prompt hash 冻结 (不执行任何模型请求)。"""
    prepared_root = Path(prepared_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    lock = json.loads((prepared_root / "lock.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instruction = (prepared_root / "public" / "instruction.md").read_text(encoding="utf-8")
    prompts = build_prompts(instruction)
    return {
        "manifest": manifest,
        "lock": lock,
        "allowed_paths": tuple(lock["instance"]["allowed_paths"]),
        "base_commit": lock["instance"]["base_commit"],
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "prompt_sha256": {**prompts["sha256"], "coding": hashlib.sha256(
            build_coding_prompt(instruction).encode("utf-8")).hexdigest()},
        "model_doc": manifest["model"],
        "check_timeout_s": float(manifest["limits"].get("check_timeout_s", 300)),
        "runner_command": _runner_command(
            manifest_path, prepared_root, prepared_root / "workspace"),
    }


def _ledger_facts(ledger) -> dict | None:
    if ledger is None:
        return None
    snapshot = ledger.snapshot
    return {"scope": snapshot.scope, "calls": snapshot.calls,
            "max_calls": snapshot.max_calls, "limit_cny": snapshot.limit_cny,
            "settled_upper_cny": snapshot.settled_upper_cny,
            "pending_upper_cny": snapshot.pending_upper_cny,
            "sealed_reason": snapshot.sealed_reason,
            "profile_sha256": snapshot.profile_sha256}


def run_rehearsal(*, prepared_root: Path, manifest_path: Path, attempt_root: Path,
                  profile: CostProfile, limit_cent: int = 450,
                  max_calls: int = 24) -> dict:
    """完整离线演练 + 跨进程总闸 + 双账本; 零真实网络请求。

    按 manifest 的 scenario 分派:
      - ``restore-compact-continue``: 两进程生命周期 (R1/R2 → 关闭 → 恢复 → 摘要 → R3);
      - ``coding``: 单进程编码路径 (读定位 → 写并检查 → 收尾), 不做恢复/压缩。

    返回可用于准入判定的全部事实。任何一步不成立都如实记录, 不粉饰。
    """
    import asyncio
    import shutil
    import subprocess

    from cicada.prompting import CheckPrompt, DEFAULT_TOOLS, compose_system_prompt
    from tests.support.swebench_budget import PROCESS_1, PROCESS_2, AttemptBudget

    plan = rehearsal_plan(prepared_root, manifest_path)
    workspace = prepared_root / "workspace"
    allowed = plan["allowed_paths"]
    manifest = plan["manifest"]

    # 干净工作区: 上一轮残留会让 R3 的写入依据失真。
    subprocess.run(["git", "-C", str(workspace), "checkout", "--quiet", "--", "."],
                   capture_output=True)
    subprocess.run(["git", "-C", str(workspace), "clean", "-qfd"], capture_output=True)

    edits = learned_edits(workspace, plan["base_commit"],
                               prepared_root / "private" / "gold_patch.diff", allowed)
    check_command = plan["runner_command"]
    from cicada.plugins.coding.verification_contracts import CheckDefinition

    checks = (CheckDefinition(check_id="check-1", command=check_command,
                              timeout_s=plan["check_timeout_s"]),)
    system_prompt, prompt_info = compose_system_prompt(
        workspace, None, tools=DEFAULT_TOOLS + ("check",),
        checks=(CheckPrompt("check-1", check_command, plan["check_timeout_s"]),))
    spec = LifecycleSpec(workspace=workspace,
                         instruction=(prepared_root / "public" / "instruction.md")
                         .read_text(encoding="utf-8"),
                         checks=checks, model_doc=manifest["model"],
                         max_turns=50, restore_capture=False)

    if attempt_root.exists() and any(attempt_root.iterdir()):
        # 不覆盖既有 attempt: 先失败后成功时同一路径只留成功那份, 等于掩盖失败
        # (审查 P1-C 实测过既有 admission 被无声删除)。
        raise AttemptBudgetError(
            f"attempt root already exists and is not empty: {attempt_root}; "
            "refusing to overwrite existing attempt evidence")
    attempt_root.mkdir(parents=True, exist_ok=True)
    # 生产进程账本: 每个进程各一本 (产品规则"重启新建额度"), 与 attempt 总闸分开记。
    from decimal import Decimal

    from cicada.costs import CostLedger

    limits = manifest.get("limits", {})
    execution_limit = Decimal(str(limits.get("budget_cny_execution", "4.50")))
    call_limit = int(limits.get("http_limit", 24))

    def process_ledger(name: str) -> CostLedger:
        return CostLedger(attempt_root / "cost" / name, limit=execution_limit,
                          max_calls=call_limit, profile=profile)

    spec = replace(spec, cost_control=process_ledger("process-1"))
    process_ledgers: dict = {}
    identity = {"provider": "deepseek", "model": manifest["model"]["normal"]["model"]}
    budget = AttemptBudget(
        attempt_root, attempt_id="pv02-" + manifest["task_key"],
        manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        prepared_lock_hash=hashlib.sha256(
            (prepared_root / "lock.json").read_bytes()).hexdigest(),
        profile=profile, model_identity=identity,
        limit_cent=limit_cent, max_calls=max_calls)

    deadline = time.monotonic() + float(manifest["limits"].get("run_timeout_s", 1500))
    facts: dict = {"task_key": manifest["task_key"],
                   "scenario": manifest["scenario"],
                   "prompt_sha256": plan["prompt_sha256"],
                   "instruction_sha256": plan["instruction_sha256"],
                   "allowed_paths": list(allowed),
                   "profile_sha256": profile.sha256,
                   "manifest_sha256": hashlib.sha256(
                       Path(manifest_path).resolve().read_bytes()).hexdigest(),
                   "prepared_lock_sha256": hashlib.sha256(
                       (prepared_root / "lock.json").read_bytes()).hexdigest(),
                   "runtime_sha256": runtime_identity(),
                   "live": "not-run"}

    if manifest["scenario"] == "coding":
        return _run_coding(spec=spec, facts=facts, budget=budget, allowed=allowed,
                           edits=edits, deadline=deadline,
                           prompt_info=prompt_info, system_prompt=system_prompt,
                           process_ledgers=process_ledgers, attempt_root=attempt_root,
                           manifest=manifest)
    if manifest["scenario"] != "restore-compact-continue":
        facts["outcome"] = "offline-failed"
        facts["problems"] = [f"unsupported scenario {manifest['scenario']!r}"]
        return facts

    budget.acquire(PROCESS_1)
    gate1 = AttemptGateTransport(
        offline_fixture(edits=edits, allowed_paths=allowed), budget,
        phase=PROCESS_1, deadline_monotonic=deadline)
    facts["process_1"] = asyncio.run(run_process_1(
        spec, gate1, system_prompt=system_prompt, prompt_info=prompt_info))
    facts["process_1"]["gate"] = {"handoffs": gate1.handoffs,
                                  "settled": len(gate1.settled_ids),
                                  "resolved": len(gate1.resolved_ids),
                                  "sealed": gate1.network_sealed_reason}
    process_ledgers["process-1"] = _ledger_facts(spec.cost_control)
    # 第一进程用剩余工作时长记账; checkpoint 只记事实。
    # 前置必须在**起第二进程之前**强制 —— 否则会带着一个未提交的 attempt 继续花真钱
    # (审查 P2-1)。三个事实都要: 两次请求都提交、无存档错误、模型正常终局。
    p1_requests = facts["process_1"].get("requests") or []
    if facts["process_1"]["manifest_unchanged"] is not True:
        facts["outcome"] = "offline-failed"
        facts["reason"] = "R1/R2 changed the workspace or did not both commit"
        budget.release()
        return facts
    unmet = [r["name"] for r in p1_requests
             if not r.get("saved") or r.get("save_error") or r.get("stop_reason") != "stop"]
    if len(p1_requests) != 2 or unmet:
        facts["outcome"] = "offline-failed"
        facts["reason"] = ("process-1 did not finish two cleanly committed requests: "
                           f"{unmet or 'missing requests'}")
        budget.release()
        return facts
    checkpoint = budget.close_checkpoint(process_identity={"pid": "process-1"})
    budget.release()

    second = AttemptBudget(
        attempt_root, attempt_id=budget.identity["attempt_id"],
        manifest_hash=budget.identity["manifest_hash"],
        prepared_lock_hash=budget.identity["prepared_lock_hash"],
        profile=profile, model_identity=identity,
        limit_cent=limit_cent, max_calls=max_calls)
    second.acquire(PROCESS_2)
    continuation = second.verify_continuation(expect_phase=PROCESS_1)
    started = facts["process_1"]["gate"]["handoffs"]
    spec2 = replace(spec, cost_control=process_ledger("process-2"))
    gate2 = AttemptGateTransport(
        offline_fixture(edits=edits, allowed_paths=allowed, start=started),
        second, phase=PROCESS_2, deadline_monotonic=deadline)
    facts["process_2"] = asyncio.run(run_process_2(
        spec2, gate2, second, session_id=facts["process_1"]["session_id"],
        system_prompt=system_prompt, prompt_info=prompt_info))
    facts["process_2"]["gate"] = {"handoffs": gate2.handoffs,
                                  "settled": len(gate2.settled_ids),
                                  "resolved": len(gate2.resolved_ids),
                                  "sealed": gate2.network_sealed_reason}
    process_ledgers["process-2"] = _ledger_facts(spec2.cost_control)
    facts["process_ledgers"] = process_ledgers
    facts["checkpoint_after_process_1"] = checkpoint
    facts["continuation_verified"] = continuation["totals"]
    facts["attempt_ledger"] = second.totals()
    facts["attempt_calls"] = [{"kind": c["kind"], "state": c["state"],
                               "reserve_cent": c["reserve_cent"],
                               "settled_cent": c.get("settled_cent")}
                              for c in second.recorded_calls()]
    second.release()

    # 判定: 生命周期成立 + 闸未封口 + 无残留预留 + 官方评分通过。
    p1, p2 = facts["process_1"], facts["process_2"]
    problems: list[str] = []
    if p1.get("completed_events") != 2:
        problems.append("R1/R2 did not both commit")
    if p2.get("restored_requests") != 2:
        problems.append("restore did not bring back exactly two committed requests")
    if p2.get("view_only_reason"):
        problems.append(f"restored session is view-only: {p2['view_only_reason']}")
    if p2.get("compacted") is None:
        problems.append("no summary was produced")
    if p2.get("context_compacted_events", 0) < 1:
        problems.append("ContextCompacted was not published")
    if p1["gate"]["sealed"] is not None or p2["gate"]["sealed"] is not None:
        problems.append("the attempt gate sealed the network")
    if facts["attempt_ledger"]["pending_cent"] != 0:
        problems.append("a reservation is still pending")
    r3 = (p2.get("requests") or [{}])[0]
    if r3.get("exit_code") != 0:
        problems.append(f"R3 did not pass its own check (exit={r3.get('exit_code')})")
    facts["problems"] = problems
    facts["outcome"] = "offline-passed-review-pending" if not problems else "offline-failed"
    return facts


def build_coding_prompt(instruction: str) -> str:
    """coding 场景的单请求 prompt (无恢复/压缩语义)。"""
    return (f"{instruction}\n{WORKSPACE_NOTE}\n请自行阅读相关源码定位问题, 只修改允许的"
            "业务路径, 完成后执行一次指定 check-1 并如实报告结果。")


async def run_process_coding(spec: LifecycleSpec, transport, *, system_prompt: str,
                             prompt_info: PromptInfo) -> dict:
    """coding 场景: 单进程读定位 → 写并检查 → 收尾 (不持久化, 不压缩)。"""
    config = _deepseek_config(spec)
    app_config = ApplicationConfig(
        workspace=spec.workspace, system_prompt=system_prompt, prompt_info=prompt_info,
        model_plugin=deepseek_plugin(config, api_key=OFFLINE_CREDENTIAL,
                                     transport=transport),
        plan=_plan(spec), max_turns=spec.max_turns, model_label=config.model,
        # coding 场景不需要跨进程延续, 故不建持久会话 (compact_context 也就不可用)。
        persistence=None,
        budget=BudgetConfig(config.num_ctx, config.max_tokens),
        restore_capture=spec.restore_capture, cost_control=spec.cost_control)
    session = await open_application(app_config)
    log = _EventLog()
    session.subscribe(log)
    facts: dict = {"requests": [], "compacted": 0}
    try:
        with production_confirmation(transport):
            outcome = await session.execute(build_coding_prompt(spec.instruction))
        facts["requests"].append(_request_facts("coding", outcome))
        facts["completed_events"] = log.completed
    finally:
        await close_process(session, log, facts)
    return facts


def _run_coding(*, spec, facts, budget, allowed, edits, deadline, prompt_info,
                system_prompt, process_ledgers, attempt_root, manifest) -> dict:
    """coding 场景的离线演练与判定。"""
    import asyncio

    from tests.support.swebench_budget import PROCESS_1

    budget.acquire(PROCESS_1)
    gate = AttemptGateTransport(
        offline_fixture(edits=edits, allowed_paths=allowed,
                        sequence=_CODING_SEQUENCE),
        budget, phase=PROCESS_1, deadline_monotonic=deadline)
    facts["process_1"] = asyncio.run(run_process_coding(
        spec, gate, system_prompt=system_prompt, prompt_info=prompt_info))
    facts["process_1"]["gate"] = {"handoffs": gate.handoffs,
                                  "settled": len(gate.settled_ids),
                                  "resolved": len(gate.resolved_ids),
                                  "sealed": gate.network_sealed_reason}
    process_ledgers["process-1"] = _ledger_facts(spec.cost_control)
    facts["process_ledgers"] = process_ledgers
    facts["attempt_ledger"] = budget.totals()
    facts["attempt_calls"] = [{"kind": c["kind"], "state": c["state"],
                               "reserve_cent": c["reserve_cent"],
                               "settled_cent": c.get("settled_cent")}
                              for c in budget.recorded_calls()]
    budget.release()

    problems: list[str] = []
    exit_code = (facts["process_1"].get("requests") or [{}])[0].get("exit_code")
    if exit_code != 0:
        problems.append(f"coding request did not pass its own check (exit={exit_code})")
    if facts["process_1"]["gate"]["sealed"] is not None:
        problems.append("the attempt gate sealed the network")
    if facts["attempt_ledger"]["pending_cent"] != 0:
        problems.append("a reservation is still pending")
    facts["problems"] = problems
    facts["outcome"] = ("offline-passed-review-pending" if not problems
                        else "offline-failed")
    return facts
