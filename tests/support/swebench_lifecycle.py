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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from cicada.application import (
    ApplicationConfig,
    ContextCompacted,
    PersistenceConfig,
    RequestCompleted,
    open_application,
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


class AttemptGateTransport(httpx.AsyncBaseTransport):
    """发送前经 ``AttemptBudget`` 预留; 只有完整 usage + DONE + EOF + 2xx 才结算。

    结算条件刻意严格: 任何一项缺失都保留 pending 并封口 (不把无法确认的发送
    写成已结算)。
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
        """流耗尽后的结算点; 条件不齐即封口且保留 pending。

        幂等: httpx 可能在迭代结束与显式关闭时各调一次 ``aclose``。结算必须
        只发生一次 —— 第二次进入直接返回, 而不是把"已结算"当矛盾抛出去
        (那会把正常收尾误报成摘要失败)。
        """
        if observation.resolved:
            return
        observation.resolved = True
        self.active_call_id = None
        complete = (
            observation.transport_error is None
            and isinstance(observation.http_status, int)
            and 200 <= observation.http_status < 300
            and observation.done_seen
            and observation.eof_seen
            and observation.closed
            and isinstance(observation.usage, dict))
        metrics = None
        if complete:
            usage = observation.usage
            try:
                metrics = ModelMetrics(input_tokens=int(usage["prompt_tokens"]),
                                       output_tokens=int(usage["completion_tokens"]))
            except (KeyError, TypeError, ValueError):
                metrics = None
        self.budget.settle(observation.call_id, metrics, complete=complete,
                           closed=observation.closed)
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
        finally:
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
        checks_signature=None, sources={"harness": "pv02"})


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

    def __call__(self, update) -> None:
        self.events.append(update)
        if isinstance(update, RequestCompleted):
            self.completed += 1
        elif isinstance(update, ContextCompacted):
            self.compacted.append(update)


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
        facts["closed_ok"] = True
    finally:
        await session.aclose()
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
        try:
            compacted = await session.compact_context()
        except RuntimeError as exc:
            # 摘要失败时把模型侧的 error 终结事实附上, 否则错误只剩一个 "error" 分类。
            from cicada.core.events import RunFinished
            errors = [e.event.error for e in log.events
                      if isinstance(e, RunFinished) and e.event.error]
            exc.add_note(f"summary terminal errors: {errors[-2:]} "
                         f"gate={getattr(transport, 'network_sealed_reason', None)}")
            raise
        facts["compacted"] = None if compacted is None else {            "summary_id": compacted.summary_id,
            "first_request_seq": compacted.first_request_seq,
            "last_request_seq": compacted.last_request_seq,
            "message_count": compacted.message_count,
            "end_message_index": compacted.end_message_index,
            "reason": compacted.reason}
        facts["context_compacted_events"] = len(log.compacted)
        facts["projection_blocks"] = len(session._context_projection.blocks)
        outcome = await session.execute(prompts["R3"])
        facts["requests"] = [_request_facts("R3", outcome)]
        facts["completed_events"] = log.completed
        facts["closed_ok"] = True
    finally:
        await session.aclose()
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

    这样 fixture 写出的补丁是**真的能让 check-1 通过**的解, 而不是自说自话的文本。
    """
    scratch = Path(tempfile.mkdtemp(prefix="pv02-fixture-"))
    try:
        clone = scratch / "clone"
        subprocess.run(["git", "clone", "--no-checkout", "--quiet",
                        str(workspace), str(clone)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", base_commit],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "apply", str(gold_patch)],
                       check=True, capture_output=True)
        return {name: (clone / name).read_text(encoding="utf-8")
                for name in allowed_paths}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def offline_fixture(*, replacements: dict, allowed_paths, check_id: str = "check-1",
                    sequence=_FIXTURE_SEQUENCE, start: int = 0) -> httpx.MockTransport:
    """离线响应: 读+检查 → 说明 → 摘要 → 写+检查 → 收尾。

    只读纪律由编排体现: R1 只发 read/check, R2 与摘要不发工具, 写入只出现在 R3。

    ``start`` 是**跨进程**的序列起点: 第一进程与第二进程是两个真实进程, 各自的
    计数器独立, 但整条剧本是一条连续流 —— 第二进程必须从第一进程消耗处继续,
    否则摘要会拿到给 R1 的响应 (实测踩过这个坑)。
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
            calls = [("write", {"path": name, "content": replacements[name]})
                     for name in allowed_paths]
            calls.append(("check", {"action": "run", "check_id": check_id}))
            return _sse([_call_frame(index, calls, finish="tool_calls")])
        if step == "summary":
            return _sse([_text_frame(index, "较早请求的摘要: 定位与说明。")])
        return _sse([_text_frame(index, "offline rehearsal complete.")])

    return httpx.MockTransport(respond)
