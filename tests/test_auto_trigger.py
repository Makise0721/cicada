"""发送前自动压缩、错误零重试与持久会话取消/投影验收。"""

from __future__ import annotations

import asyncio

import pytest

from cicada.application import (
    ApplicationConfig,
    ApplicationSession,
    ContextCompacted,
)
from cicada.boot import bootstrap
from cicada.budget import BudgetConfig
from cicada.context_policy import (
    BudgetPolicy,
    ContextProjection,
    SummaryBlock,
    canonical_prefix_sha256,
    summary_block_message,
    summary_input_messages,
    summary_prompt_text,
)
from cicada.core.cancel import CancelToken
from cicada.core.messages import AssistantMessage, ToolResult, ToolResultMessage, UserMessage
from cicada.core.ports import (
    ModelMetrics,
    ModelRequest,
    StreamDone,
    TextDelta,
    ToolCallEvent,
)
from cicada.persistence.contracts import (
    RequestOutcomeFacts,
    StoredIdentity,
)
from cicada.plugins.coding.verification_contracts import VerificationView
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.session_store import open_session_store
from cicada.run_policy import verification_policy

from test_application import _prompt_info
from test_persistence_application import _identity

from tests.support.budget import admission_total


def _budget(num_ctx: int = 1_000_000) -> BudgetConfig:
    return BudgetConfig(num_ctx=num_ctx, num_predict=2048)


# 服务端错误原样发布，发送后不自动压缩或重试。
OVERFLOW_ERROR = (
    "provider protocol error: request (302 tokens) exceeds the available context size "
    "(256 tokens), try increasing it"
)
OTHER_ERROR = "provider http 500: backend failed"


class _CountingModel:
    """记录请求并按剧本逐次响应的最小 ModelPort 替身 (发送次数/消息内容断言用)."""

    def __init__(self, script) -> None:
        self._script = [list(entry) for entry in script]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest, cancel: CancelToken):
        self.requests.append(request)
        entry = self._script.pop(0) if self._script else [StreamDone("stop")]
        for event in entry:
            if cancel.cancelled:
                yield StreamDone("aborted")
                return
            await asyncio.sleep(0)
            yield event


class _AutoCompactStub:
    """自动压缩回调桩: 记录额度请求; 消费对应额度后按 outcome 返回.

    与 open_application 接线的应用侧回调同构: 条件判定与额度消费在回调内,
    失败不返还; 成功即给投影追加一个摘要块 (切换投影)。
    """

    def __init__(self, policy: BudgetPolicy, *, outcome: bool = True, block_text: str = "摘要正文") -> None:
        self.policy = policy
        self.outcome = outcome
        self.block_text = block_text
        self.calls: list[str] = []
        self.cancel_token: CancelToken | None = None  # 取消场景由测试注入

    async def __call__(self) -> bool:
        self.calls.append("admission")
        if self.cancel_token is not None:
            self.cancel_token.cancel()
        if not self.policy.try_consume_admission_auto_attempt():
            return False
        if not self.outcome:
            return False
        block = SummaryBlock(
            message=summary_block_message(
                first_request_seq=1, last_request_seq=1, message_count=1, text=self.block_text,
            ),
            end_message_index=1,
            first_request_seq=1,
            last_request_seq=1,
            message_count=1,
            prefix_sha256=canonical_prefix_sha256((UserMessage(text="x"),)),
        )
        self.policy.projection.apply_summary(block, canonical_length=1)
        return True


def _window_request() -> tuple[ModelRequest, BudgetConfig]:
    """请求 + 恰好卡在 full>num_ctx≥projected 窗口内的配置."""
    big = UserMessage(text="x" * 4000)
    prompt = UserMessage(text="please")
    config = _budget(1)
    full = admission_total("sys", (big, prompt), config)
    projected = admission_total("sys", (summary_block_message(
        first_request_seq=1, last_request_seq=1, message_count=1, text="摘要正文",
    ), prompt), config)
    config = BudgetConfig(num_ctx=projected + (full - projected) // 2, num_predict=2048)
    assert admission_total("sys", (big, prompt), config) > config.num_ctx
    return ModelRequest(messages=(big, prompt), tools=(), system_prompt="sys"), config


def _admitted_request() -> tuple[ModelRequest, BudgetConfig]:
    """预估可过但服务端仍报超限的请求 (现实场景: 启发式低估).

    返回 (请求, 宽裕配置): 准入通过，服务端终结错误保持原样。
    """
    request = ModelRequest(
        messages=(UserMessage(text="x" * 4000), UserMessage(text="please")),
        tools=(), system_prompt="sys",
    )
    config = _budget()
    assert admission_total("sys", request.messages, config) <= config.num_ctx
    return request, config


async def _collect(policy: BudgetPolicy, request: ModelRequest, token: CancelToken | None = None):
    token = token or CancelToken()
    return [event async for event in policy.stream(request, token)], token


def _done(events) -> StreamDone:
    assert events and isinstance(events[-1], StreamDone)
    return events[-1]


# --- §1 额度语义 ---------------------------------------------------------------------------


def test_admission_quota_resets_per_seam_and_fails_closed():
    """准入额度; 接缝重置; 未接缝的 begin 语义不重置; 消费至多一次."""
    model = _CountingModel([])
    policy = BudgetPolicy(model, _budget())
    policy.note_request_start(0)
    assert policy.try_consume_admission_auto_attempt()
    assert not policy.try_consume_admission_auto_attempt()
    # begin 不调用接缝: 额度不重置
    assert not policy.try_consume_admission_auto_attempt()
    policy.note_request_start(4)
    assert policy.try_consume_admission_auto_attempt(), "下一次 execute 重新建立额度"


async def test_admission_auto_attempt_success_continues_to_send():
    """准入失败 → 消费额度 → 压缩切换投影 → 重估通过 → 正常发送 (D7)."""
    request, config = _window_request()
    model = _CountingModel([[StreamDone("stop")]])
    policy = BudgetPolicy(model, config)
    stub = _AutoCompactStub(policy)
    policy.set_auto_compact(stub)
    policy.note_request_start(2)
    events, _token = await _collect(policy, request)
    assert stub.calls == ["admission"]
    assert _done(events).stop_reason == "stop"
    assert len(model.requests) == 1
    # 发送的是投影后消息: [摘要块] + 当前请求
    assert model.requests[0].messages[0].text.startswith("[程序生成的对话摘要]")
    assert model.requests[0].messages[1] == request.messages[1]
    assert policy.try_consume_admission_auto_attempt() is False, "额度已消费且不返还"


async def test_admission_auto_attempt_failure_keeps_refusal_and_quota():
    """压缩失败 (回调 False): 原样拒绝; 额度不返还, 同 execute 内不再尝试."""
    request, config = _window_request()
    model = _CountingModel([])
    policy = BudgetPolicy(model, config)
    stub = _AutoCompactStub(policy, outcome=False)
    policy.set_auto_compact(stub)
    policy.note_request_start(2)
    events, _token = await _collect(policy, request)
    assert stub.calls == ["admission"]
    done = _done(events)
    assert done.stop_reason == "error" and "context budget exceeded" in done.error
    assert model.requests == [], "拒绝发送点零适配器调用"
    # 同一次 execute 内重复准入失败: 额度已消费, 不再调回调
    events_again, _ = await _collect(policy, request)
    assert stub.calls == ["admission"]
    assert _done(events_again).stop_reason == "error"


async def test_admission_without_callback_refuses_immediately():
    """未接线回调 (非持久/01 基线): 直接拒绝, 行为与 01 一致."""
    request, config = _window_request()
    model = _CountingModel([])
    policy = BudgetPolicy(model, config)
    policy.note_request_start(2)
    events, _token = await _collect(policy, request)
    done = _done(events)
    assert done.stop_reason == "error" and "context budget exceeded" in done.error
    assert model.requests == []


async def test_admission_cancel_during_compaction_ends_aborted():
    """begin 后轮次内取消: 该轮以 StreamDone("aborted") 收尾, 不按 error (D7)."""
    request, config = _window_request()
    model = _CountingModel([])
    policy = BudgetPolicy(model, config)
    token = CancelToken()
    stub = _AutoCompactStub(policy, outcome=False)
    stub.cancel_token = token
    policy.set_auto_compact(stub)
    policy.note_request_start(2)
    events, _ = await _collect(policy, request, token)
    assert _done(events).stop_reason == "aborted"
    assert model.requests == [], "取消后零主模型发送"


# --- §1 服务端错误零重试 -------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix_events",
    [
        [TextDelta("")],  # 空片段同样禁止透明重发 (A10 边界)
        [ToolCallEvent("c1", "read", "{}")],
    ],
    ids=["empty-text-delta", "tool-call-event"],
)
async def test_transparent_resend_banned_after_any_output(prefix_events):
    """透明重发禁令: 该发送点出现任何 TextDelta/ToolCallEvent 后的上下文错误不重试."""
    request, config = _admitted_request()
    model = _CountingModel([[*prefix_events, StreamDone("error", OVERFLOW_ERROR)]])
    policy = BudgetPolicy(
        model, config,
    )
    stub = _AutoCompactStub(policy)
    policy.set_auto_compact(stub)
    policy.note_request_start(2)
    events, _token = await _collect(policy, request)
    assert stub.calls == [], "任何可见输出后不触发重试"
    assert len(model.requests) == 1
    done = _done(events)
    assert done.stop_reason == "error" and OVERFLOW_ERROR in done.error, "保留原错误"


async def test_provider_failure_passes_through_once():
    """取消/网络/其他服务端错误形态: 原样发布, 不消费额度."""
    request, config = _admitted_request()
    model = _CountingModel([[StreamDone("error", OTHER_ERROR)]])
    policy = BudgetPolicy(
        model, config,
    )
    stub = _AutoCompactStub(policy)
    policy.set_auto_compact(stub)
    policy.note_request_start(2)
    events, _token = await _collect(policy, request)
    assert stub.calls == []
    assert len(model.requests) == 1
    assert _done(events).error == OTHER_ERROR


async def test_context_error_is_not_retried():
    """服务端上下文超限原样发布，不调用自动压缩或重试。"""
    request, config = _admitted_request()
    model = _CountingModel([[StreamDone("error", OVERFLOW_ERROR)]])
    policy = BudgetPolicy(model, config)
    stub = _AutoCompactStub(policy)
    policy.set_auto_compact(stub)
    policy.note_request_start(2)
    events, _token = await _collect(policy, request)
    assert stub.calls == []
    assert len(model.requests) == 1
    done = _done(events)
    assert done.stop_reason == "error" and OVERFLOW_ERROR in done.error


# --- §2 持久会话组合 (A09/A11 + PA03/PA05 复验) ----------------------------------------------


from tests.support.summary import GatedSummary, ScriptedSummary as _ScriptedSummary


def _GatedSummary(text: str = "门控摘要正文") -> GatedSummary:
    """门控摘要通道 (本文件历史行为: 终结事件不携带计量)。"""
    return GatedSummary(text=text, metrics=None)


class _LatchingModel(FakeModel):
    """首个模型请求前锁存 storage_failed (模拟活动包装器请求中途锁存)."""

    def __init__(self, script) -> None:
        super().__init__(script)
        self.session: ApplicationSession | None = None
        self._latched = False

    async def stream(self, request, cancel):
        if self.session is not None and not self._latched:
            self._latched = True
            self.session.latch_storage_failure("activity write failed mid-request")
        async for event in super().stream(request, cancel):
            yield event


def _seed_pair() -> tuple[tuple, tuple]:
    """种子请求: 请求1 (大, 可摘要) + 请求2 (中等, 保留后缀主体).

    约束: 请求2 + 当前 prompt 必须重于摘要系统提示, 否则"摘要输入可装下而
    原始请求超限"的窗口不存在 (压缩永远无法救回)。
    """
    req1 = (UserMessage(text="第一问 " + "x" * 3000), AssistantMessage(text="回答一"))
    req2 = (UserMessage(text="第二问 " + "y" * 1500), AssistantMessage(text="回答二"))
    return req1, req2


def _window_config_auto() -> tuple[BudgetConfig, tuple, tuple]:
    """Class A (begin 前触发): summary_input ≤ num_ctx < total(H + 小 prompt)."""
    req1, req2 = _seed_pair()
    base = _budget(1)
    block = summary_block_message(
        first_request_seq=1, last_request_seq=1, message_count=len(req1),
        text="较早请求的摘要正文",
    )
    summary_input = admission_total(summary_prompt_text(), summary_input_messages(req1), base)
    full = admission_total("test system prompt", (*req1, *req2, UserMessage(text="第三问")), base)
    assert summary_input < full, "摘要输入窗口必须窄于预估失败的完整请求"
    config = _budget(summary_input + (full - summary_input) // 2)
    projected = admission_total("test system prompt", (block, *req2, UserMessage(text="第三问")), config)
    assert admission_total("test system prompt", (*req1, *req2, UserMessage(text="第三问")), config) > config.num_ctx
    assert projected <= config.num_ctx, "压缩后投影可容纳"
    return config, req1, req2


def _tool_turn(text_len: int = 600) -> list:
    """轮 0 剧本: 助手可见文本 + 未知工具调用 (工具结果入历史, 抬高轮 1 请求)."""
    return [
        TextDelta("t" * text_len),
        ToolCallEvent("c1", "missing_tool", "{}"),
        StreamDone("tool_use"),
    ]


def _window_config_per_turn() -> tuple[BudgetConfig, tuple, tuple]:
    """Class B (begin 后触发): 预估 (H+prompt) 通过, 轮 1 (H+prompt+轮0增量) 超限."""
    req1, req2 = _seed_pair()
    base = _budget(1)
    prompt = UserMessage(text="第三问")
    block = summary_block_message(
        first_request_seq=1, last_request_seq=1, message_count=len(req1),
        text="较早请求的摘要正文",
    )
    turn0 = (
        AssistantMessage(text="t" * 600, tool_calls=()),
        ToolResultMessage(result=ToolResult(
            call_id="c1", name="missing_tool", content="unknown tool: missing_tool",
            is_error=True,
        )),
    )
    summary_input = admission_total(summary_prompt_text(), summary_input_messages(req1), base)
    admitted = admission_total("test system prompt", (*req1, *req2, prompt), base)
    next_turn = admission_total("test system prompt", (*req1, *req2, prompt, *turn0), base)
    assert summary_input <= admitted < next_turn, "预估通过而下一轮超限的窗口必须存在"
    config = _budget(admitted + (next_turn - admitted) // 2)
    assert admission_total("test system prompt", (*req1, *req2, prompt), config) <= config.num_ctx
    projected = admission_total("test system prompt", (block, *req2, prompt, *turn0), config)
    assert projected <= config.num_ctx, "压缩后轮 1 可容纳"
    return config, req1, req2


async def _seed_committed_requests(session: ApplicationSession, store, req1: tuple, req2: tuple) -> None:
    """真实提交两个完整请求到 store, 并同步内存历史与区间簿记.

    自动压缩的摘要提交经真实 store commit_summary, 其关系校验以 store 已提交
    请求为准——只注内存历史会撞关系冲突, 因此种子也走真实 begin/commit。
    """
    facts = RequestOutcomeFacts(
        run_id="seed-run", stop_reason="stop", error=None, exit_code=0,
        finalized=True, finalization_notes=(), model_calls=(),
    )
    start = 0
    for messages in (req1, req2):
        request_id = await store.begin_request(session._epoch_id, messages[0].text)
        seq = await store.commit_request(request_id, messages=tuple(messages), outcome=facts)
        session._committed_spans.append((seq, start, start + len(messages)))
        session._history.extend(messages)
        start += len(messages)


async def _auto_session(
    tmp_path, *, script, channel, num_ctx: int, model=None,
):
    """持久会话 + 手动组装 (open_application 无法注入独立通道, 02 同型);
    与 open_application 一致地绑定自动压缩回调."""
    store = open_session_store(
        directory=tmp_path / ".cicada" / "sessions", session_id=None, workspace=tmp_path
    )
    epoch_id = await store.start_epoch(StoredIdentity(launch=_identity(), tool_fingerprint="fp"))
    main_model = model if model is not None else FakeModel(script)
    config = ApplicationConfig(
        workspace=tmp_path,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(tmp_path),
        model_plugin=fake_model_plugin(main_model),
        max_turns=6,
        model_label="fake",
        budget=BudgetConfig(num_ctx=num_ctx, num_predict=2048),
    )
    context_projection = ContextProjection()
    policy = BudgetPolicy(
        main_model, config.budget,
        summarize=channel, projection=context_projection,
    )
    app = await bootstrap(
        [fake_model_plugin(main_model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save",
        budget_policy=policy, context_projection=context_projection,
    )
    session._budget_policy.set_auto_compact(session._auto_compact_for_request)
    return session, main_model, channel, store, policy


async def test_pre_estimate_auto_compaction_switches_projection_and_proceeds(tmp_path):
    """begin 前预估失败 → 消费准入额度 → 自动压缩 (先持久后切换) → 重估通过 →
    请求正常完成并提交 (完整请求生产者屏障: commit_request 落在 execute 内)。"""
    config_seed, req1, req2 = _window_config_auto()
    events: list = []
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path,
        script=[[StreamDone("stop")]],
        channel=_ScriptedSummary(metrics=ModelMetrics(input_tokens=9, output_tokens=2)),
        num_ctx=config_seed.num_ctx,
    )
    session.subscribe(events.append)
    await _seed_committed_requests(session, store, req1, req2)
    try:
        outcome = await session.execute("第三问")
        assert outcome.result.stop_reason == "stop"
        # 摘要块成为发送消息首条 (投影切换真实发生)
        assert main_model.requests[0].messages[0].text.startswith("[程序生成的对话摘要]")
        # 摘要记录持久化, 原因=auto, 计量保留 (D7)
        snapshot = store.recovery_snapshot()
        assert len(snapshot.summaries) == 1
        record = snapshot.summaries[0]
        assert record.reason == "auto"
        assert record.metrics is not None and record.metrics.input_tokens == 9
        # 事件: ContextCompacted(auto); 自动路径不发 OperationBusy (属于请求生命周期)
        kinds = [type(e).__name__ for e in events]
        assert "ContextCompacted" in kinds
        compacted = next(e for e in events if isinstance(e, ContextCompacted))
        assert compacted.reason == "auto"
        assert "OperationBusy" not in kinds
        # 完整请求生产者屏障: 摘要完成不替代请求完成——commit_request 已落库
        assert session.persistence.saved_request_count == 1
        assert len(session._committed_spans) == 3, "压缩切点簿记随新请求推进"
        # 本次 execute 的准入额度已消费
        assert policy.try_consume_admission_auto_attempt() is False
    finally:
        await session.aclose()
        await store.aclose()


async def test_pre_estimate_still_over_refuses_but_keeps_auto_summary(tmp_path):
    """压缩后重估仍超限: begin 前口径拒绝 (零 begin/零模型); 摘要记录与计量按
    事实保留; 下一次请求 C 是新范围 (额度重置, 对新区间再次尝试自动压缩)。"""
    config_seed, req1, req2 = _window_config_auto()
    req3 = (UserMessage(text="第三问种子"), AssistantMessage(text="回答三"))
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path,
        script=[],
        channel=_ScriptedSummary(
            text="太长的摘要正文" + "s" * 3000,  # 摘要块自身超限
            metrics=ModelMetrics(input_tokens=9, output_tokens=2),
        ),
        num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    facts = RequestOutcomeFacts(
        run_id="seed-run", stop_reason="stop", error=None, exit_code=0,
        finalized=True, finalization_notes=(), model_calls=(),
    )
    request_id = await store.begin_request(session._epoch_id, req3[0].text)
    seq3 = await store.commit_request(request_id, messages=req3, outcome=facts)
    session._committed_spans.append(
        (seq3, len(session._history), len(session._history) + len(req3))
    )
    session._history.extend(req3)
    try:
        with pytest.raises(RuntimeError, match="context budget exceeded"):
            await session.execute("第四问")
        # PA01: 无 begin/无 intent/无 finalized; 零主模型发送
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 3, "只有种子请求, 本次尝试零 begin"
        assert snapshot.interrupted == ()
        assert main_model.requests == []
        assert session.state == "idle"
        assert session.persistence.storage_failed is None
        # D7: 已调用摘要通道, 摘要记录与计量按实际事实保留
        assert len(snapshot.summaries) == 1
        assert snapshot.summaries[0].reason == "auto"
        assert snapshot.summaries[0].metrics is not None
        assert snapshot.summaries[0].metrics.input_tokens == 9, "摘要计量随记录保留 (D7)"
        assert len(session._context_projection.blocks) == 1
        # C: 新 execute 范围, 准入额度重置 → 对新区间再次自动压缩 (第二次摘要调用)
        with pytest.raises(RuntimeError, match="context budget exceeded"):
            await session.execute("第五问")
        summaries = store.recovery_snapshot().summaries
        assert len(summaries) == 2, "C 有自己的准入额度"
        # C 只追加更晚的新区间 (不重写旧摘要); 容器对区间连续性结构性校验
        assert summaries[1].first_request_seq > summaries[0].last_request_seq
        assert summaries[1].end_message_index > summaries[0].end_message_index
        assert len(session._context_projection.blocks) == 2
    finally:
        await session.aclose()
        await store.aclose()


async def test_begin_does_not_reset_admission_quota_within_execute(tmp_path):
    """B 预估压缩成功后 begin 不清零: 后续轮次再超限不能再次准入压缩, 按每轮
    权威口径拒绝并落 finalized 错误记录 (A09 跨预估反例)。"""
    config_seed, req1, req2 = _window_config_auto()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path,
        script=[_tool_turn(2700)],
        channel=_ScriptedSummary(),
        num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    try:
        outcome = await session.execute("第三问")
        # 轮 0: 预估压缩成功后放行 (tool_use 轮); 轮 1: 超限且额度已消费 → 拒绝
        assert outcome.result.stop_reason == "error"
        assert "context budget exceeded" in (outcome.result.error or "")
        assert len(channel.requests) == 1, "begin 后不再第二次摘要调用"
        assert len(main_model.requests) == 1, "轮 1 拒绝零发送"
        # finalized 错误记录落库 (PA01 begin 后口径)
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 3, "两个种子 + 本次请求"
        assert snapshot.requests[-1].outcome.stop_reason == "error"
    finally:
        await session.aclose()
        await store.aclose()


async def test_cancel_before_begin_during_auto_compaction_ends_attempt(tmp_path):
    """begin 前取消: 结束 execute 尝试, 零主模型/零 intent/零 finalized,
    不伪造 Agent run 或 StreamDone; 下一次请求重新建立范围。"""
    config_seed, req1, req2 = _window_config_auto()
    channel = _GatedSummary()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[], channel=channel, num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    try:
        task = asyncio.ensure_future(session.execute("第三问"))
        while not channel.requests:
            await asyncio.sleep(0.01)
        session.cancel()  # state 仍 idle: 经请求 token 取消 (03 接线)
        channel.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert session.state == "idle"
        assert main_model.requests == [], "零主模型调用"
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 2 and snapshot.interrupted == ()
        assert snapshot.summaries == (), "提交前取消零写入"
        assert session._context_projection.blocks == ()
        assert session.persistence.storage_failed is None
        # 下一次请求重新建立范围: 自动压缩 (gate 已放行) 后成功
        outcome = await session.execute("轻量问题")
        assert outcome.result.stop_reason == "stop"
        assert len(store.recovery_snapshot().summaries) == 1
    finally:
        await session.aclose()
        await store.aclose()


async def test_cancel_after_begin_during_per_turn_compaction_ends_aborted(tmp_path):
    """begin 后轮次内取消: 该轮以 aborted 收尾 (不按 error), 提交前取消零写入。"""
    config_seed, req1, req2 = _window_config_per_turn()
    channel = _GatedSummary()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[_tool_turn()], channel=channel, num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    try:
        task = asyncio.ensure_future(session.execute("第三问"))
        while len(channel.requests) < 1:
            await asyncio.sleep(0.01)
        session.cancel()
        channel.gate.set()
        outcome = await asyncio.wait_for(task, timeout=10)
        assert outcome.result.stop_reason == "aborted"
        assert len(main_model.requests) == 1
        snapshot = store.recovery_snapshot()
        assert snapshot.summaries == (), "提交前取消零摘要写入"
        assert session._context_projection.blocks == ()
        assert snapshot.requests[-1].outcome.stop_reason == "aborted"
    finally:
        await session.aclose()
        await store.aclose()


async def test_close_during_pre_estimate_auto_compaction_seals_and_settles(tmp_path):
    """close 组合 (PA05 复验): 封口 + 协作取消 begin 前自动压缩; execute 尝试
    结束; 零提交; close 干净完成, 重开无 interrupted。"""
    config_seed, req1, req2 = _window_config_auto()
    channel = _GatedSummary()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[], channel=channel, num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    directory = tmp_path / ".cicada" / "sessions"
    try:
        task = asyncio.ensure_future(session.execute("第三问"))
        while not channel.requests:
            await asyncio.sleep(0.01)
        close_task = asyncio.ensure_future(session.aclose())
        await asyncio.sleep(0.05)  # close 同步前缀: 封口 + 取消请求 token
        channel.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        await asyncio.wait_for(close_task, timeout=10)
        assert session.state == "closed"
        assert store.recovery_snapshot().summaries == (), "封口先于接受 → 零提交"
        assert session._close_problems == [], "生产者在截止时间内落定"
        await store.aclose()
        reopened = open_session_store(directory=directory, session_id=None, workspace=tmp_path)
        try:
            assert reopened.recovery_snapshot().interrupted == (), "零 begin: 无中断 intent"
        finally:
            await reopened.aclose()
    finally:
        if not session._closed:
            await session.aclose()
        if not store._closed:
            await store.aclose()


async def test_close_during_per_turn_compaction_supervises_auto_barrier(tmp_path):
    """close 组合 (begin 后窗口): 自动压缩屏障 (_compact_done) 纳入监督, 请求以
    aborted 收尾并完成提交, close 干净完成。"""
    config_seed, req1, req2 = _window_config_per_turn()
    channel = _GatedSummary()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[_tool_turn()], channel=channel, num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    try:
        task = asyncio.ensure_future(session.execute("第三问"))
        while len(channel.requests) < 1:
            await asyncio.sleep(0.01)
        close_task = asyncio.ensure_future(session.aclose())
        await asyncio.sleep(0.05)
        channel.gate.set()
        outcome = await asyncio.wait_for(task, timeout=15)
        assert outcome.result.stop_reason == "aborted"
        await asyncio.wait_for(close_task, timeout=10)
        assert session.state == "closed"
        assert store.recovery_snapshot().summaries == (), "封口零提交"
        assert session._close_problems == []
        assert session.persistence.storage_failed is None
    finally:
        if not session._closed:
            await session.aclose()
        if not store._closed:
            await store.aclose()


async def test_accepted_summary_commit_survives_cancel_during_request(tmp_path):
    """已接受事务 + 请求期取消 (D6/PA03 自动组合): 等明确结果 → 成功先切换并
    发布事件; 取消经内核边界落定为 aborted 请求, 摘要及计量保留。"""
    config_seed, req1, req2 = _window_config_per_turn()
    channel = _ScriptedSummary(metrics=ModelMetrics(input_tokens=9, output_tokens=2))
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[_tool_turn()], channel=channel, num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    original_commit = store.commit_summary
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_commit(record):
        await original_commit(record)  # store 已接受且事务完成
        entered.set()
        await release.wait()  # completion 尚未送达调用方
        return None

    object.__setattr__(store, "commit_summary", gated_commit)
    events: list = []
    session.subscribe(events.append)
    try:
        task = asyncio.ensure_future(session.execute("第三问"))
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        release.set()
        outcome = await asyncio.wait_for(task, timeout=10)
        # 明确结果先落定: 摘要持久 + 投影切换 + 事件; 请求以 aborted 收尾
        assert outcome.result.stop_reason == "aborted"
        snapshot = store.recovery_snapshot()
        assert len(snapshot.summaries) == 1 and snapshot.summaries[0].reason == "auto"
        assert snapshot.summaries[0].metrics is not None
        assert len(session._context_projection.blocks) == 1
        assert any(isinstance(e, ContextCompacted) for e in events)
        assert session.persistence.storage_failed is None
    finally:
        release.set()
        await session.aclose()
        await store.aclose()


async def test_operations_refused_while_auto_compaction_in_flight(tmp_path):
    """begin 前自动压缩在途: 手动压缩/清空/新请求互斥拒绝, 不自锁 (D7)。"""
    config_seed, req1, req2 = _window_config_auto()
    channel = _GatedSummary()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[], channel=channel, num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    try:
        task = asyncio.ensure_future(session.execute("第三问"))
        while not channel.requests:
            await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="a compaction is already in progress"):
            await session.compact_context()
        with pytest.raises(RuntimeError, match="a compaction is in progress"):
            await session.clear_context()
        with pytest.raises(RuntimeError, match="a compaction is in progress"):
            await session.execute("另一问")
        session.cancel()
        channel.gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
    finally:
        await session.aclose()
        await store.aclose()


async def test_mid_request_storage_failed_latch_blocks_auto_summary(tmp_path):
    """请求中途锁存 storage_failed 后, 后续轮次准入失败零摘要调用, 按每轮口径
    拒绝 (D9; 中途锁存入口 process_activity.latch_storage_failure 可达)."""
    config_seed, req1, req2 = _window_config_per_turn()
    model = _LatchingModel([_tool_turn()])
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[_tool_turn()], channel=_ScriptedSummary(),
        num_ctx=config_seed.num_ctx, model=model,
    )
    model.session = session
    await _seed_committed_requests(session, store, req1, req2)
    try:
        outcome = await session.execute("第三问")
        assert outcome.result.stop_reason == "error"
        assert "context budget exceeded" in (outcome.result.error or "")
        assert channel.requests == [], "storage_failed 锁存下零摘要调用"
        assert len(main_model.requests) == 1, "轮 1 拒绝零发送"
        assert session.persistence.storage_failed is not None
        assert store.recovery_snapshot().summaries == ()
    finally:
        await session.aclose()
        await store.aclose()


async def test_latched_states_yield_zero_summary_calls_in_auto_callback(tmp_path):
    """自动回调的条件顺序: 两种锁存状态直接返回 False 且零摘要调用, 额度不被
    无谓消费; 锁存解除后同一额度仍可用并正常压缩 (A09 条件在消费之前)."""
    config_seed, req1, req2 = _window_config_auto()
    session, main_model, channel, store, policy = await _auto_session(
        tmp_path, script=[], channel=_ScriptedSummary(), num_ctx=config_seed.num_ctx,
    )
    await _seed_committed_requests(session, store, req1, req2)
    try:
        session._view_only_reason = "restored session has unproven activity (probe)"
        assert await session._auto_compact_for_request() is False
        assert channel.requests == []
        session._view_only_reason = None
        session.latch_storage_failure("store closed")
        assert await session._auto_compact_for_request() is False
        assert channel.requests == []
        session._storage_failed = None
        # 直接探针不在 execute 内: 补一个请求 token 满足屏障归属断言
        session._cancel_token = CancelToken()
        assert await session._auto_compact_for_request() is True
        assert len(channel.requests) == 1
        assert len(session._context_projection.blocks) == 1
    finally:
        await session.aclose()
        await store.aclose()


async def test_a11_terminal_scan_covers_canonical_history_through_compacted_projection(tmp_path):
    """A11 复核: 压缩投影组合下, 上游 RunPolicy 的终结扫描仍针对完整
    规范历史——被摘要覆盖的未证终结事实仍锁存; 摘要块不携带验证权威。"""
    canonical = (
        UserMessage(text="第一问"),
        ToolResultMessage(result=ToolResult(
            call_id="c1", name="powershell", content="超时被杀", is_error=True, details=None,
        )),
        UserMessage(text="第二问"),
    )
    view = VerificationView(
        verification_run_id="run-1", snapshot_ref=None, scope_id=None,
        checks=(), receipts=(), process_uncertain=False, blocking_reasons=(),
    )

    class _RecordingService:
        def __init__(self, view) -> None:
            self._view = view
            self.reasons: list[str] = []

        def mark_process_uncertain(self, reason: str) -> None:
            self.reasons.append(reason)

        async def refresh(self, cancel):
            return self._view

    service = _RecordingService(view)
    inner = _CountingModel([
        [StreamDone("stop")],
    ])
    config = _budget()
    policy = BudgetPolicy(
        inner, config,
    )

    block = SummaryBlock(
        message=summary_block_message(
            first_request_seq=1, last_request_seq=1, message_count=2, text="摘要",
        ),
        end_message_index=2, first_request_seq=1, last_request_seq=1, message_count=2,
        prefix_sha256=canonical_prefix_sha256(canonical[:2]),
    )
    policy.projection.apply_summary(block, canonical_length=len(canonical))
    wrapped = verification_policy(policy, service)
    request = ModelRequest(messages=canonical, tools=(), system_prompt="sys")
    events = [event async for event in wrapped.stream(request, CancelToken())]
    dones = [event for event in events if isinstance(event, StreamDone)]
    assert len(dones) == 1 and dones[0].stop_reason == "stop"
    assert service.reasons, "被摘要覆盖的未证终结事实仍被扫描锁存 (A11)"
    assert len(inner.requests) == 1
    assert inner.requests[0].messages[0].text.startswith("[程序生成的对话摘要]")
    assert "验证权威" in inner.requests[0].messages[0].text, "摘要块不携带验证权威"
