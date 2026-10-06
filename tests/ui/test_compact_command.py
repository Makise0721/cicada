"""票据 02 UI 组合检查: /compact 命令、输入屏障、确认条目与指引/失败路径.

真实共享应用 (手工组装独立摘要通道, B-F4) + test_conversation_view 的 Harness;
屏障语义与持久清空同型 (命令处置时同步置位, 落定释放)。
"""

from __future__ import annotations

import asyncio

import pytest

from test_conversation_view import Harness, _prompt_info, _wait_until

from cicada.application import ApplicationConfig, ApplicationSession
from cicada.budget import BudgetConfig
from cicada.boot import bootstrap
from cicada.context_policy import BudgetPolicy, ContextProjection
from cicada.core.ports import ModelMetrics, StreamDone, TextDelta
from cicada.persistence.contracts import LaunchIdentity, StoredIdentity
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.session_store import open_session_store
from cicada.ui.commands import SLASH_COMMANDS
from cicada.ui.main_screen import strip_terminal_sequences


def _identity() -> LaunchIdentity:
    return LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=6, prompt_version="test", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=None, sources={},
    )


def _budget() -> BudgetConfig:
    return BudgetConfig(num_ctx=1_000_000, num_predict=2048)


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

    def __init__(self, text: str = "较早请求的摘要正文") -> None:
        self._text = text
        self.requests = []

    async def stream(self, request, cancel):
        self.requests.append(request)
        yield TextDelta(self._text)
        yield StreamDone("stop", metrics=ModelMetrics(input_tokens=9, output_tokens=2))


class _GatedSummary:
    """门控摘要通道: 放行前挂起, 尊重协作取消。"""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.requests = []

    async def stream(self, request, cancel):
        self.requests.append(request)
        await self.gate.wait()
        if cancel.cancelled:
            yield StreamDone("aborted")
            return
        yield TextDelta("门控摘要正文")
        yield StreamDone("stop", metrics=ModelMetrics(input_tokens=9, output_tokens=2))


_UNSET = object()


async def _manual_session(tmp_path, *, script=None, channel=None, budget=_UNSET, store=True):
    if store:
        real_store = open_session_store(
            directory=tmp_path / ".cicada" / "sessions", session_id=None, workspace=tmp_path
        )
        epoch_id = await real_store.start_epoch(
            StoredIdentity(launch=_identity(), tool_fingerprint="fp")
        )
    else:
        real_store, epoch_id = None, None
    main_model = FakeModel(script or [])
    effective_budget = _budget() if budget is _UNSET else budget
    config = _config(tmp_path, main_model, budget=effective_budget)
    summary_channel = channel if channel is not None else _ScriptedSummary()
    # 策略与会话必须持同一投影容器 (与 open_application 接线一致)
    context_projection = ContextProjection() if config.budget is not None else None
    policy = (
        BudgetPolicy(
            main_model, config.budget,
            summarize=summary_channel, projection=context_projection,
        )
        if config.budget is not None
        else None
    )
    app = await bootstrap(
        [fake_model_plugin(main_model)], tool_capabilities=(),
        system_prompt="test system prompt",
        model_policy=(lambda inner, runtime: policy) if policy is not None else None,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=real_store, epoch_id=epoch_id,
        persistence_mode="save" if store else None,
        budget_policy=policy, context_projection=context_projection,
    )
    return session, main_model, summary_channel, real_store


async def test_compact_barriers_input_until_settled_and_confirms(tmp_path):
    """A05: /compact 置压缩屏障; 期间 Enter 拒绝且草稿保留; 成功追加确认条目."""
    session, _main, channel, store = await _manual_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]], channel=_GatedSummary()
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        assert harness.submit("第二问")
        await harness.wait_request_done()

        assert harness.submit("/compact")
        assert harness.view._gate.compacting, "命令处置时屏障已同步置位"
        harness.bundle["editor"].set_text("压缩中的新请求")
        assert not harness.submit("压缩中的新请求"), "屏障内 Enter 不提交"
        assert "压缩中的新请求" in harness.bundle["editor"].get_text(), "草稿保留"
        channel.gate.set()
        assert await _wait_until(lambda: not harness.view._gate.compacting), "落定后释放屏障"
        rendered = chr(10).join(harness.lines(80))
        assert "已完成压缩" in rendered
        assert "原始记录保留" in rendered
        assert harness.submit("压缩后的新请求")
        await harness.wait_request_done()
        # 模型所见 = [摘要块] + 保留后缀 + 新 prompt (A08)
        sent = harness.session._app.agent._model.inner.requests[-1]
        assert sent.messages[0].text.startswith("[程序生成的对话摘要]")
    finally:
        channel.gate.set()
        await session.aclose()
        if store is not None:
            await store.aclose()


async def test_compact_guidance_for_non_persistent_and_no_budget(tmp_path):
    session, _main, _channel, _store = await _manual_session(
        tmp_path, script=[[StreamDone("stop")]], budget=None, store=False
    )

    harness = Harness(session)
    try:
        assert harness.submit("/compact")
        rendered = chr(10).join(harness.lines(80))
        assert "需要持久会话" in rendered
        assert not harness.view._gate.compacting
    finally:
        await session.aclose()

    session2, _m2, _c2, store2 = await _manual_session(
        tmp_path, script=[[StreamDone("stop")]], budget=None, store=True
    )
    harness2 = Harness(session2)
    try:
        assert harness2.submit("/compact")
        rendered = chr(10).join(harness2.lines(80))
        assert "需要启用预算" in rendered
        assert not harness2.view._gate.compacting
    finally:
        await session2.aclose()
        if store2 is not None:
            await store2.aclose()


async def test_help_and_menu_list_compact_with_accurate_description(tmp_path):
    """04: /help 与斜线菜单含 /compact, 描述与实际准入行为一致 (需持久会话)."""
    assert any(
        item.command == "/compact" and "持久会话" in item.description
        for item in SLASH_COMMANDS
    )
    session, _m, _c, _s = await _manual_session(tmp_path, script=[[StreamDone("stop")]])
    harness = Harness(session)
    try:
        assert harness.submit("/help")
        rendered = strip_terminal_sequences(chr(10).join(harness.lines(90)))
        assert "/compact" in rendered
        assert "摘要较早对话以释放上下文（需持久会话）" in rendered
    finally:
        await session.aclose()


async def test_compact_failure_shows_error_and_releases_barrier(tmp_path):
    """摘要生成失败 → 屏障释放 + 错误如实成条, 投影不变 (D5/PA03)."""
    class _FailingSummary:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request, cancel):
            self.requests.append(request)
            yield StreamDone("length")  # 截断 → 失败分类

    session, _main, _channel, store = await _manual_session(
        tmp_path, script=[[StreamDone("stop")], [StreamDone("stop")]],
        channel=_FailingSummary(),
    )
    harness = Harness(session)
    try:
        assert harness.submit("第一问")
        await harness.wait_request_done()
        assert harness.submit("第二问")
        await harness.wait_request_done()
        assert harness.submit("/compact")
        assert await _wait_until(lambda: not harness.view._gate.compacting), "失败后屏障释放"
        rendered = chr(10).join(harness.lines(80))
        assert "压缩失败" in rendered
        assert "length" in rendered
        assert session._context_projection.blocks == ()
        assert session.persistence.storage_failed is None
    finally:
        await session.aclose()
        if store is not None:
            await store.aclose()
