"""手动压缩集成测试 (02, A05/A06/A08 + D4/D5/D6/PA03/PA05).

真实 SQLite store + fake 主模型 + 独立摘要通道替身 (B-F4: 不消费共享剧本队列)。
取消/关闭组合按接受点分阶段断言: 派发前取消零写入; 接受后取消等明确结果且
成功先切换再传播; close 封口零提交, 迟到响应丢弃不误锁 storage_failed。
"""

from __future__ import annotations

import asyncio
import hashlib
import math

import pytest

from cicada.application import (
    ApplicationConfig,
    ApplicationSession,
    ContextCompacted,
    OperationBusy,
    PersistenceConfig,
    open_application,
)
from cicada.boot import bootstrap
from cicada.budget import BudgetConfig
from cicada.context_policy import (
    BudgetPolicy,
    ContextProjection,
    canonical_prefix_sha256,
    summary_block_message,
)
from cicada.core.events import ModelCallCompleted  # noqa: F401
from cicada.core.messages import AssistantMessage, UserMessage
from cicada.core.ports import ModelMetrics, StreamDone, TextDelta
from cicada.persistence.codec import encode_messages
from cicada.persistence.contracts import LaunchIdentity, StoredIdentity
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.session_store import open_session_store

from test_application import _prompt_info
from test_persistence_application import _identity


def _budget(num_ctx: int = 1_000_000) -> BudgetConfig:
    return BudgetConfig(num_ctx=num_ctx, num_predict=2048)


def _config(tmp_path, model: FakeModel, *, budget: BudgetConfig | None = None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=tmp_path,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(tmp_path),
        model_plugin=fake_model_plugin(model),
        max_turns=6,
        model_label="fake",
        budget=budget,
    )


class _ScriptedSummary:
    """无门控摘要通道: 恒定正文 + 可得计量 (正常路径用)."""

    def __init__(self, text: str = "较早请求的摘要正文", metrics=None) -> None:
        self._text = text
        self._metrics = metrics
        self.requests = []

    async def stream(self, request, cancel):
        self.requests.append(request)
        yield TextDelta(self._text)
        yield StreamDone("stop", metrics=self._metrics)


class _GatedSummary:
    """门控摘要通道: 放行前挂起, 可选忽略协作取消 (PA05 反例用)."""

    def __init__(self, *, text: str = "门控摘要正文", ignore_cancel: bool = False) -> None:
        self.gate = asyncio.Event()
        self.cancel_seen = False
        self.requests = []
        self._text = text
        self._ignore_cancel = ignore_cancel

    async def stream(self, request, cancel):
        self.requests.append(request)
        await self.gate.wait()
        if cancel.cancelled:
            self.cancel_seen = True
            if not self._ignore_cancel:
                yield StreamDone("aborted")
                return
        yield TextDelta(self._text)
        yield StreamDone("stop", metrics=ModelMetrics(input_tokens=9, output_tokens=2))


async def _manual_persistent_session(
    tmp_path, *, script: list | None = None, channel=None, num_ctx: int = 1_000_000,
):
    """直接组装: 真实 store + 独立摘要通道 (open_application 无法注入独立通道)."""
    store = open_session_store(
        directory=tmp_path / ".cicada" / "sessions", session_id=None, workspace=tmp_path
    )
    epoch_id = await store.start_epoch(StoredIdentity(launch=_identity(), tool_fingerprint="fp"))
    main_model = FakeModel(script or [])
    config = _config(tmp_path, main_model, budget=_budget(num_ctx))
    summary_channel = channel if channel is not None else _ScriptedSummary()
    # 策略与会话必须持同一投影容器 (与 open_application 接线一致)
    context_projection = ContextProjection()
    policy = BudgetPolicy(
        main_model, config.budget,
        summarize=summary_channel, projection=context_projection,
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
    return session, main_model, summary_channel, store


async def _two_committed_requests(session):
    await session.execute("第一问")
    await session.execute("第二问")
    assert len(session._committed_spans) == 2


async def test_manual_compaction_happy_path_switches_projection_and_persists(tmp_path):
    events: list = []
    session, main_model, summary_channel, store = await _manual_persistent_session(
        tmp_path,
        script=[[StreamDone("stop")], [StreamDone("stop")]],
        channel=_ScriptedSummary(metrics=ModelMetrics(input_tokens=9, output_tokens=2)),
    )
    session.subscribe(events.append)
    try:
        await _two_committed_requests(session)
        history_before = session.history
        compacted = await session.compact_context()
        assert isinstance(compacted, ContextCompacted)
        assert compacted.first_request_seq == session._committed_spans[0][0]
        assert compacted.last_request_seq == compacted.first_request_seq
        assert compacted.message_count == 2
        assert compacted.end_message_index == 2
        # 摘要记录持久化: 覆盖前缀/计量/原因/身份入档
        snapshot = store.recovery_snapshot()
        assert len(snapshot.summaries) == 1
        record = snapshot.summaries[0]
        assert record.reason == "manual"
        assert record.metrics is not None and record.metrics.input_tokens == 9
        assert record.prompt_version == "1" and len(record.prompt_sha256) == 64
        # 前一投影身份 = 切换前的恒等投影 (完整规范历史) 指纹
        assert record.previous_projection_identity == canonical_prefix_sha256(history_before)
        # 投影切换: 有效消息 = [摘要块] + 保留后缀; 结构合法 (切点不切工具组)
        blocks = session._context_projection.blocks
        assert len(blocks) == 1
        expected_block = summary_block_message(
            first_request_seq=record.first_request_seq,
            last_request_seq=record.last_request_seq,
            message_count=record.message_count,
            text="较早请求的摘要正文",
        )
        assert blocks[0].message == expected_block
        effective = session._context_projection.project(session._history)
        assert effective[0] == expected_block
        assert effective[1:] == session.history[2:]
        from cicada.core.messages import validate_history_structure

        validate_history_structure(effective)  # 切点落在请求边界, 工具组零切断
        # 摘要请求形态: 无工具 + 专用系统提示 (D5)
        summary_request = summary_channel.requests[0]
        assert summary_request.tools == ()
        assert len(summary_channel.requests) == 1
        # 事件序列: busy → compacted
        kinds = [type(e).__name__ for e in events]
        assert "OperationBusy" in kinds and "ContextCompacted" in kinds
        busy = next(e for e in events if isinstance(e, OperationBusy))
        assert busy.operation == "compact"
        # 二次压缩: 保留边界前的请求已摘要
        with pytest.raises(RuntimeError, match="already summarized"):
            await session.compact_context()
    finally:
        await session.aclose()
        await store.aclose()


def _isolate(messages):
    from cicada.core.messages import isolate_message

    return tuple(isolate_message(m) for m in messages)


async def test_compaction_requires_two_committed_requests_and_channel(tmp_path):
    session, _main, _channel, store = await _manual_persistent_session(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="at least two committed full requests"):
            await session.compact_context()
    finally:
        await session.aclose()
        await store.aclose()


async def test_compaction_refusals_view_only_storage_failed_and_no_channel(tmp_path):
    from cicada.persistence.contracts import StorageUnavailableError

    # 非持久: 明确拒绝
    main_model = FakeModel([[StreamDone("stop")]])
    config = _config(tmp_path, main_model, budget=_budget())
    policy = BudgetPolicy(main_model, config.budget)  # 无摘要通道
    app = await bootstrap(
        [fake_model_plugin(main_model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None, budget_policy=policy
    )
    try:
        with pytest.raises(RuntimeError, match="requires a persistent session"):
            await session.compact_context()
    finally:
        await session.aclose()
    # 持久 + 预算未启用 (无 policy): 明确拒绝
    store = open_session_store(
        directory=tmp_path / "s2", session_id=None, workspace=tmp_path
    )
    epoch_id = await store.start_epoch(StoredIdentity(launch=_identity(), tool_fingerprint="fp"))
    plain_model = FakeModel([[StreamDone("stop")]])
    plain_config = _config(tmp_path, plain_model, budget=None)
    plain_app = await bootstrap(
        [fake_model_plugin(plain_model)], tool_capabilities=(),
        system_prompt="test system prompt",
    )
    plain_session = ApplicationSession(
        app=plain_app, config=plain_config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save",
    )
    try:
        with pytest.raises(RuntimeError, match="requires an enabled budget"):
            await plain_session.compact_context()
        # storage_failed 锁存: 压缩拒绝
        plain_session.latch_storage_failure("store closed")
        with pytest.raises(RuntimeError, match="storage_failed"):
            await plain_session.compact_context()
    finally:
        await plain_session.aclose()
        await store.aclose()


async def test_compaction_shrinks_range_to_fit_summary_budget(tmp_path):
    """摘要输入自身须满足预算: 最长最旧前缀区间; 收缩从新端开始 (D4)."""
    from cicada.budget import assemble_report, estimate_input_components
    from cicada.context_policy import summary_prompt_text

    # 三个请求; num_ctx 卡在"覆盖前两个请求装不下、单个请求可装"的窗口内
    canonical_two = (
        UserMessage(text="第一问"), AssistantMessage(text="回答1"),
        UserMessage(text="第二问"), AssistantMessage(text="回答2"),
    )

    def _total(messages):
        input_estimate, _components = estimate_input_components(
            summary_prompt_text(), (), messages, _budget(1)
        )
        return input_estimate + 2048 + max(256, math.ceil(0.10 * input_estimate))

    total_one = _total(canonical_two[:2])
    total_two = _total(canonical_two)
    assert total_two > total_one
    num_ctx = total_one + (total_two - total_one) // 2
    session, _main, _channel, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")]] * 3, num_ctx=num_ctx,
    )
    try:
        await session.execute("第一问")
        await session.execute("第二问")
        await session.execute("第三问")
        compacted = await session.compact_context()
        assert compacted is not None
        assert compacted.end_message_index == 2, "只覆盖第一个请求 (收缩从新端)"
        assert compacted.first_request_seq == session._committed_spans[0][0]
        assert compacted.first_request_seq == compacted.last_request_seq
        assert compacted.message_count == 2
    finally:
        await session.aclose()
        await store.aclose()


async def test_compaction_refuses_when_oldest_single_request_does_not_fit(tmp_path):
    session, _main, _channel, store = await _manual_persistent_session(
        tmp_path, num_ctx=40
    )
    try:
        # 直接注入长历史与区间簿记: 只测切点拒绝 (执行路径另测)
        session._history.append(UserMessage(text="x" * 6000))
        session._history.append(UserMessage(text="y" * 20))
        session._committed_spans.append((1, 0, 1))
        session._committed_spans.append((2, 1, 2))
        with pytest.raises(RuntimeError, match="does not fit"):
            session._compaction_plan()
    finally:
        await session.aclose()
        await store.aclose()


# --- PA03: 分阶段取消 -------------------------------------------------------------------


async def test_cancel_before_dispatch_leaves_zero_writes(tmp_path):
    channel = _GatedSummary()
    session, _main, _channel_obj, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]], channel=channel,
    )
    observer_events: list = []
    session.subscribe(observer_events.append)
    try:
        await _two_committed_requests(session)
        task = asyncio.ensure_future(session.compact_context())
        while not channel.requests:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # 零写入、旧投影、屏障落定、无 storage_failed
        assert store.recovery_snapshot().summaries == ()
        assert session._context_projection.blocks == ()
        assert session._compact_done is not None and session._compact_done.done()
        assert session.persistence.storage_failed is None
        assert not any(isinstance(e, ContextCompacted) for e in observer_events)
        # 下一次压缩重新建立并成功
        channel.gate.set()
        await asyncio.sleep(0.05)
        channel2_requests = len(channel.requests)
        session._compact_token = None
        compacted = await session.compact_context()
        assert compacted is not None
        assert len(channel.requests) == channel2_requests + 1
    finally:
        channel.gate.set()
        await session.aclose()
        await store.aclose()


async def test_cancel_after_store_accept_settles_then_switches_and_propagates(tmp_path):
    """事务已被 store 接受且完成, completion 送达前取消: 等明确结果 → 成功先切换
    → 再传播 CancelledError (PA03); 摘要计量与记录保留。"""
    store = open_session_store(
        directory=tmp_path / "sessions", session_id=None, workspace=tmp_path
    )
    epoch_id = await store.start_epoch(StoredIdentity(launch=_identity(), tool_fingerprint="fp"))
    original_commit = store.commit_summary
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_commit(record):
        await original_commit(record)  # store 已接受且事务完成
        entered.set()
        await release.wait()  # completion 尚未送达调用方
        return None

    main_model = FakeModel([[StreamDone("stop")], [StreamDone("stop")]])
    config = _config(tmp_path, main_model, budget=_budget())
    channel = _ScriptedSummary()
    policy = BudgetPolicy(main_model, config.budget, summarize=channel)
    app = await bootstrap(
        [fake_model_plugin(main_model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save",
        budget_policy=policy,
    )
    object.__setattr__(store, "commit_summary", gated_commit)
    try:
        await _two_committed_requests(session)
        task = asyncio.ensure_future(session.compact_context())
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # 已接受事务的成功不能被取消掩盖: 摘要落库 + 投影已切换 + 事件已发布
        assert len(store.recovery_snapshot().summaries) == 1
        assert len(session._context_projection.blocks) == 1
        assert session._compact_done.done()
        assert session.persistence.storage_failed is None
    finally:
        release.set()
        await session.aclose()
        await store.aclose()


# --- PA05: 关闭组合 ---------------------------------------------------------------------


async def test_close_seals_unaccepted_commit_and_discards_late_response(tmp_path):
    """close 封口 + 协作取消生成: 尊重取消的通道返回 aborted; 迟到合法文本零提交。"""
    channel = _GatedSummary()  # 尊重协作取消
    session, _main, _channel_obj, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]], channel=channel,
    )
    try:
        await _two_committed_requests(session)
        compact_task = asyncio.ensure_future(session.compact_context())
        while not channel.requests:
            await asyncio.sleep(0.01)
        close_task = asyncio.ensure_future(session.aclose())
        # close 封口 + 协作取消; 放行后通道返回 aborted → 归关闭取消, 静默放弃
        await asyncio.sleep(0.05)
        channel.gate.set()
        result = await asyncio.wait_for(compact_task, timeout=10)
        assert result is None, "封口路径静默放弃, 不产生成功事件"
        assert store.recovery_snapshot().summaries == ()
        assert session._context_projection.blocks == ()
        assert session.persistence.storage_failed is None
        await asyncio.wait_for(close_task, timeout=10)
        assert session.state == "closed"
    finally:
        channel.gate.set()


async def test_close_seal_with_late_valid_text_zero_commit(tmp_path):
    """模型忽略协作取消而迟到返回合法文本: 零提交、旧投影、不误锁 storage_failed."""
    channel = _GatedSummary(ignore_cancel=True)
    session, _main, _channel_obj, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]], channel=channel,
    )
    try:
        await _two_committed_requests(session)
        compact_task = asyncio.ensure_future(session.compact_context())
        while not channel.requests:
            await asyncio.sleep(0.01)
        close_task = asyncio.ensure_future(session.aclose())
        await asyncio.sleep(0.05)
        channel.gate.set()  # 迟到合法响应
        result = await asyncio.wait_for(compact_task, timeout=10)
        assert result is None
        assert store.recovery_snapshot().summaries == (), "封口先于接受: 零提交"
        assert session._context_projection.blocks == ()
        assert session.persistence.storage_failed is None
        await asyncio.wait_for(close_task, timeout=10)
        assert session.state == "closed"
    finally:
        channel.gate.set()


async def test_close_waits_for_accepted_summary_commit_before_store_release(tmp_path, monkeypatch):
    """事务已被 store 接受但 completion 未送达时 close: 生产者屏障等明确结果。"""
    import cicada.application as app_module

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 5.0)
    store = open_session_store(
        directory=tmp_path / "sessions", session_id=None, workspace=tmp_path
    )
    epoch_id = await store.start_epoch(StoredIdentity(launch=_identity(), tool_fingerprint="fp"))
    original_commit = store.commit_summary
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_commit(record):
        await original_commit(record)
        entered.set()
        await release.wait()
        return None

    main_model = FakeModel([[StreamDone("stop")], [StreamDone("stop")]])
    config = _config(tmp_path, main_model, budget=_budget())
    channel = _ScriptedSummary()
    policy = BudgetPolicy(main_model, config.budget, summarize=channel)
    app = await bootstrap(
        [fake_model_plugin(main_model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save",
        budget_policy=policy,
    )
    object.__setattr__(store, "commit_summary", gated_commit)
    try:
        await _two_committed_requests(session)
        compact_task = asyncio.ensure_future(session.compact_context())
        await asyncio.wait_for(entered.wait(), timeout=5)
        close_task = asyncio.ensure_future(session.aclose())
        await asyncio.sleep(0.1)
        assert not close_task.done(), "接受先于封口: close 等待摘要生产者明确结果"
        release.set()
        await asyncio.wait_for(close_task, timeout=10)
        assert len(store.recovery_snapshot().summaries) == 1
        assert session.state == "closed"
        compact_task.cancel()
        try:
            await compact_task
        except asyncio.CancelledError:
            pass
    finally:
        release.set()
        await store.aclose()


async def test_close_supervision_timeout_records_problem_and_does_not_fake_done(tmp_path, monkeypatch):
    import cicada.application as app_module

    monkeypatch.setattr(app_module, "CLOSE_SUPERVISION_S", 0.2)
    channel = _GatedSummary(ignore_cancel=True)  # 持续在途且忽略取消
    session, _main, _channel_obj, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]], channel=channel,
    )
    closed_updates: list = []
    try:
        await _two_committed_requests(session)
        compact_task = asyncio.ensure_future(session.compact_context())
        while not channel.requests:
            await asyncio.sleep(0.01)
        session.subscribe(closed_updates.append)
        close_task = asyncio.ensure_future(session.aclose())
        await asyncio.wait_for(close_task, timeout=10)
        assert session.state == "closed"
        closed = next(e for e in closed_updates if type(e).__name__ == "ApplicationClosed")
        assert closed.error is not None and "close supervision deadline" in closed.error
        assert not session._compact_done.done(), "不伪造 _compact_done 完成"
        # 迟到生成任务保持强引用 (本测试持有 compact_task); 释放后归关闭取消
        channel.gate.set()
        result = await asyncio.wait_for(compact_task, timeout=5)
        assert result is None
        assert session.state == "closed", "迟到收尾不恢复 idle、不破坏 closed"
    finally:
        channel.gate.set()
        await store.aclose()


# --- A08: 恢复同投影与篡改拒绝 ------------------------------------------------------------


async def test_restore_rebuilds_identical_projection_and_rejects_tamper(tmp_path, monkeypatch):
    session, _main, _channel_obj, store = await _manual_persistent_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]],
        channel=_ScriptedSummary(text="恢复一致性摘要正文"),
    )
    try:
        await _two_committed_requests(session)
        compacted = await session.compact_context()
        assert compacted is not None
        pre_close_blocks = session._context_projection.blocks
        pre_close_effective = session._context_projection.project(session._history)
        session_id = store.session_id
        await session.aclose()
        await store.aclose()

        # open_application 恢复: 投影确定性重建, 块文本逐位一致
        identity = LaunchIdentity(
            provider="fake", model="fake", base_url=None, think=None, options={},
            max_turns=6, prompt_version="test", prompt_sha256="a" * 64,
            instructions_sha256=None, checks_signature=None, sources={},
        )
        resumed_model = FakeModel([[StreamDone("stop")]])
        config = _config(tmp_path, resumed_model, budget=_budget())
        config = type(config)(
            workspace=config.workspace, system_prompt=config.system_prompt,
            prompt_info=config.prompt_info, model_plugin=config.model_plugin,
            max_turns=config.max_turns, model_label=config.model_label,
            # adopt: 手工 bootstrap 的工具集与 open_application 不同, 指纹差异被采用
            persistence=PersistenceConfig(
                session_id=session_id, identity=identity, adopt_current_config=True
            ),
            budget=config.budget,
        )
        resumed = await open_application(config)
        try:
            blocks = resumed._context_projection.blocks
            assert len(blocks) == len(pre_close_blocks)
            assert blocks[0].message.text == pre_close_blocks[0].message.text
            assert blocks[0].prefix_sha256 == pre_close_blocks[0].prefix_sha256
            assert (
                resumed._context_projection.project(resumed.history)
                == pre_close_effective
            )
            # 恢复后首请求: 模型所见 = [摘要块] + 保留后缀 + 新 prompt (A08)
            await resumed.execute("恢复后的追问")
            sent = resumed_model.requests[0]
            assert sent.messages[0].text == blocks[0].message.text
            assert sent.messages[0].text == pre_close_blocks[0].message.text
        finally:
            await resumed.aclose()

        # 篡改: 覆盖前缀指纹与规范历史不符 → ApplicationOpenError, 模型零请求
        import sqlite3

        db_path = tmp_path / ".cicada" / "sessions" / f"{session_id}.sqlite3"
        connection = sqlite3.connect(str(db_path))
        try:
            row = connection.execute(
                "SELECT summary_json FROM summaries"
            ).fetchone()
            tampered = row[0].replace("b" * 64, "c" * 64)  # prefix_sha256 不存在时退化为改 seq
            if tampered == row[0]:
                connection.execute("UPDATE summaries SET first_request_seq = 12345")
            else:
                connection.execute("UPDATE summaries SET summary_json = ?", (tampered,))
                connection.execute(
                    "UPDATE summaries SET payload_sha256 = ?",
                    (hashlib.sha256(tampered.encode("utf-8")).hexdigest(),),
                )
            connection.commit()
        finally:
            connection.close()
        from cicada.application import ApplicationOpenError

        # store 列交叉核对或应用侧指纹核对任一层拒绝均可 (fail-closed 双层)
        with pytest.raises(ApplicationOpenError, match="(relation error|not match)"):
            await open_application(config)
    finally:
        pass


async def test_terminal_scan_still_covers_canonical_after_projection(tmp_path):
    """A11 预备: 压缩投影替换发生在 RunPolicy 下游, 终结扫描仍针对完整规范历史.

    被摘要覆盖的未证终结 powershell 结果仍触发保守锁存; 下游 BudgetPolicy 所见
    请求首条已是摘要块 (替换真实发生), 证明扫描面不随投影收缩。
    """
    from cicada.core.messages import ToolResult, ToolResultMessage
    from cicada.core.ports import ModelRequest
    from cicada.plugins.coding.verification_contracts import VerificationView
    from cicada.run_policy import verification_policy

    class _RecordingService:
        def __init__(self, view) -> None:
            self._view = view
            self.reasons: list[str] = []

        def mark_process_uncertain(self, reason: str) -> None:
            self.reasons.append(reason)

        async def refresh(self, cancel):
            return self._view

    view = VerificationView(
        verification_run_id="run-1", snapshot_ref=None, scope_id=None,
        checks=(), receipts=(), process_uncertain=False, blocking_reasons=(),
    )
    service = _RecordingService(view)
    inner = FakeModel([])
    policy = BudgetPolicy(inner, _budget())
    canonical = (
        UserMessage(text="第一问"),
        ToolResultMessage(result=ToolResult(
            call_id="c1", name="powershell", content="超时被杀", is_error=True, details=None,
        )),
        UserMessage(text="第二问"),  # 保留后缀起点 (第二请求)
    )
    from cicada.context_policy import SummaryBlock

    policy.projection.apply_summary(
        SummaryBlock(
            message=summary_block_message(
                first_request_seq=1, last_request_seq=1, message_count=2, text="摘要",
            ),
            end_message_index=2, first_request_seq=1, last_request_seq=1, message_count=2,
            prefix_sha256=canonical_prefix_sha256(canonical[:2]),
        ),
        canonical_length=len(canonical),
    )
    wrapped = verification_policy(policy, service)
    request = ModelRequest(messages=canonical, tools=(), system_prompt="sys")
    from cicada.core.cancel import CancelToken
    _ = [event async for event in wrapped.stream(request, CancelToken())]
    assert service.reasons, "被摘要覆盖的未证终结事实仍被扫描锁存 (A11)"
    assert inner.requests, "请求已发出 (投影替换后正常放行)"
    assert inner.requests[0].messages[0].text.startswith("[程序生成的对话摘要]")


# --- 阶段收口审查处置 (STG-F1): begin 等待窗的应用层互斥 ----------------------------------


async def _begin_gated_execute(session, store, prompt: str):
    """派发一个停在 begin 等待窗的 execute; 返回 (task, release)。

    store.begin_request 经实例替换挂闸: entered 置位即已到窗内 (state 仍 idle)。
    """
    release = asyncio.Event()
    entered = asyncio.Event()
    original_begin = store.begin_request

    async def gated_begin(epoch_id, prompt_text):
        entered.set()
        await release.wait()
        return await original_begin(epoch_id, prompt_text)

    store.begin_request = gated_begin  # type: ignore[method-assign]
    task = asyncio.create_task(session.execute(prompt))
    await asyncio.wait_for(entered.wait(), timeout=5)
    return task, release


async def test_compact_refused_during_begin_window(tmp_path):
    """STG-F1: begin 等待窗内 state 仍为 idle, 压缩必须被同步旗标拒绝 (不依赖 UI 屏障)."""
    session, _m, _c, store = await _manual_persistent_session(
        tmp_path,
        script=[[StreamDone("stop")], [StreamDone("stop")], [StreamDone("stop")]],
    )
    try:
        await _two_committed_requests(session)
        task, release = await _begin_gated_execute(session, store, "第三问")
        assert session.state == "idle"
        assert not session._compacting and not session._clearing
        try:
            await session.compact_context()
            raise AssertionError("begin 窗内的压缩必须被拒绝")
        except RuntimeError as exc:
            assert "being persisted" in str(exc)
        assert session._context_projection.blocks == (), "拒绝不产生投影切换"
        release.set()
        await asyncio.wait_for(task, timeout=15)
        assert session.state == "idle", "窗关闭后请求正常完成"
    finally:
        await session.aclose()
        await store.aclose()


async def test_clear_refused_during_begin_window(tmp_path):
    """STG-F1: begin 等待窗内清空同样拒绝 (基线已有的 clear 变体一并闭合)."""
    session, _m, _c, store = await _manual_persistent_session(
        tmp_path,
        script=[[StreamDone("stop")], [StreamDone("stop")]],
    )
    try:
        await session.execute("第一问")
        task, release = await _begin_gated_execute(session, store, "第二问")
        try:
            await session.clear_context()
            raise AssertionError("begin 窗内的清空必须被拒绝")
        except RuntimeError as exc:
            assert "being persisted" in str(exc)
        assert session.persistence.context_generation == 0, "拒绝不推进代次"
        release.set()
        await asyncio.wait_for(task, timeout=15)
    finally:
        await session.aclose()
        await store.aclose()


async def test_second_execute_refused_during_begin_window(tmp_path):
    """STG-F1: begin 等待窗内新请求拒绝, 避免 double begin 与交错提交."""
    session, _m, _c, store = await _manual_persistent_session(
        tmp_path,
        script=[[StreamDone("stop")], [StreamDone("stop")]],
    )
    try:
        task, release = await _begin_gated_execute(session, store, "第一问")
        try:
            await session.execute("第二问")
            raise AssertionError("begin 窗内的新请求必须被拒绝")
        except RuntimeError as exc:
            assert "still being started" in str(exc)
        release.set()
        await asyncio.wait_for(task, timeout=15)
        assert session.state == "idle"
    finally:
        await session.aclose()
        await store.aclose()
