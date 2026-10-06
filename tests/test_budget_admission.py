"""应用层预算准入集成测试 (01, A02/A04 + D1 四态 + D9 启用条件).

两道口分别钉住: 请求前预估 (零 begin/零模型调用/不锁存 storage_failed) 与
BudgetPolicy 每轮权威准入 (零发送 + 既有错误路径收尾 + 持久模式落 finalized
错误记录)。请求边界接缝的调用时机与参数值用记录策略直接断言。fake 直建
配置 (budget=None) 的既有基线不本文件覆盖——既有全量套件即两叉验收的 a 叉。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cicada.application import (
    ApplicationConfig,
    ApplicationOpenError,
    ApplicationSession,
    PersistenceConfig,
    open_application,
)
from cicada.boot import bootstrap
from cicada.budget import (
    BudgetConfig,
    assemble_report,
    budget_exceeded_text,
    estimate_input_components,
)
from cicada.context_policy import BudgetPolicy
from cicada.core.messages import UserMessage
from cicada.core.ports import ModelMetrics, StreamDone, ToolCallEvent
from cicada.persistence.contracts import StoredIdentity
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.session_store import open_session_store

from test_application import _prompt_info, _repo
from test_persistence_application import _identity

STOP_WITH_METRICS = [StreamDone("stop", metrics=ModelMetrics(input_tokens=42, output_tokens=1))]


def _config(root: Path, model: FakeModel, *, budget: BudgetConfig | None = None,
            plan: VerificationPlan | None = None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(model),
        plan=plan,
        max_turns=6,
        model_label="fake",
        budget=budget,
    )


def _generous() -> BudgetConfig:
    return BudgetConfig(num_ctx=1_000_000, num_predict=1)


def _plan(root: Path) -> VerificationPlan:
    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(os.path.realpath(root)),
        checks=(CheckDefinition("check-1", "exit 0", timeout_s=30.0),),
    )


class _BeginSpyStore:
    """SessionStorePort 最小替身: 记录 begin/commit 调用, 供零 begin 断言."""

    session_id = "spy-session"

    def __init__(self) -> None:
        self.begins: list[str] = []
        self.commits: list[str] = []

    def recovery_snapshot(self):  # pragma: no cover - 本组测试不消费快照
        raise AssertionError("snapshot not expected")

    async def start_epoch(self, identity):
        return "epoch-1"

    async def begin_request(self, epoch_id, prompt):
        self.begins.append(prompt)
        return "req-1"

    async def commit_request(self, request_id, *, messages, outcome, receipts=()):
        self.commits.append(request_id)
        return 1

    async def begin_process(self, request_id, *, tool_name, call_id, arguments):
        return "act-1"

    async def end_process(self, activity_id, terminal):
        return None

    async def commit_clear(self, epoch_id):
        return 1

    async def aclose(self):
        return None


# --- 请求前预估 (A02 上道口) -----------------------------------------------------------


async def test_pre_estimate_rejects_persistent_before_begin(tmp_path):
    store = _BeginSpyStore()
    model = FakeModel([])
    config = _config(tmp_path, model, budget=BudgetConfig(num_ctx=50, num_predict=1))
    policy = BudgetPolicy(model, config.budget)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id="epoch-1", persistence_mode="save", budget_policy=policy,
    )
    try:
        with pytest.raises(RuntimeError, match="context budget exceeded"):
            await session.execute("x" * 500)
        assert store.begins == [], "预估拒绝在 begin_request 之前: 无新 intent"
        assert store.commits == []
        assert model.requests == [], "零模型调用"
        assert session.state == "idle"
        assert session.persistence.storage_failed is None, "预算拒绝不是存储故障"
    finally:
        await session.aclose()


async def test_pre_estimate_rejection_leaves_no_intent_in_real_store(tmp_path):
    """01 裁决: 预估拒绝不落库; 重开后无新 intent (start_epoch 后 interrupted 仍为空)."""
    directory = tmp_path / ".cicada" / "sessions"
    store = open_session_store(directory=directory, session_id=None, workspace=tmp_path)
    epoch_id = await store.start_epoch(
        StoredIdentity(launch=_identity(), tool_fingerprint="fp")
    )
    model = FakeModel([])
    config = _config(tmp_path, model, budget=BudgetConfig(num_ctx=50, num_predict=1))
    policy = BudgetPolicy(model, config.budget)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save", budget_policy=policy,
    )
    try:
        with pytest.raises(RuntimeError, match="context budget exceeded"):
            await session.execute("x" * 500)
    finally:
        await session.aclose()
        await store.aclose()
    reopened = open_session_store(directory=directory, session_id=store.session_id, workspace=tmp_path)
    try:
        await reopened.start_epoch(
            StoredIdentity(launch=_identity(), tool_fingerprint="fp")
        )
        snapshot = reopened.recovery_snapshot()
        assert snapshot.requests == (), "没有已提交请求"
        assert snapshot.interrupted == (), "没有遗留 intent (begin 从未发生)"
    finally:
        await reopened.aclose()


async def test_pre_estimate_rejects_nonpersistent_with_zero_side_effects(tmp_path):
    model = FakeModel([])
    session = await open_application(
        _config(tmp_path, model, budget=BudgetConfig(num_ctx=50, num_predict=1))
    )
    try:
        with pytest.raises(RuntimeError, match="request refused before start"):
            await session.execute("x" * 500)
        assert model.requests == []
        assert session.state == "idle"
    finally:
        await session.aclose()


async def test_single_request_over_limit_names_composition(tmp_path):
    """无历史可压缩的单请求自身超限: 明确失败且带超限构成说明 (A02/C-F6)."""
    model = FakeModel([])
    config = _config(tmp_path, model, budget=BudgetConfig(num_ctx=50, num_predict=1))
    policy = BudgetPolicy(model, config.budget)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None, budget_policy=policy,
    )
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await session.execute("x" * 500)
        text = str(excinfo.value)
        assert "message:0:user" in text
        assert "system_prompt" in text
        assert "output_reserve" in text or "safety_margin" in text
    finally:
        await session.aclose()


def test_config_rejects_wrong_budget_type(tmp_path):
    with pytest.raises(ValueError, match="budget must be a BudgetConfig"):
        _config(tmp_path, FakeModel([]), budget="not a config")  # type: ignore[arg-type]


# --- 每轮权威准入 (A02 下道口) ---------------------------------------------------------


async def _tight_manual_session(tmp_path, *, store=None, epoch_id=None):
    """num_ctx=10 恰好容纳首轮 (system 6 + 消息 2 + 预留 1 + 余量 1), 第二轮必超."""
    model = FakeModel([
        [ToolCallEvent(id="c1", name="nope", arguments_json="{}"), StreamDone("tool_use")],
    ])
    config = _config(
        tmp_path, model,
        budget=BudgetConfig(
            num_ctx=10, num_predict=1,
            safety_margin_min=1, safety_margin_ratio=0, message_template_overhead=1,
        ),
    )
    policy = BudgetPolicy(model, config.budget)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id=epoch_id, persistence_mode="save" if store is not None else None,
        budget_policy=policy,
    )
    return session, model, policy


async def test_per_turn_rejection_persists_finalized_error(tmp_path):
    directory = tmp_path / ".cicada" / "sessions"
    store = open_session_store(directory=directory, session_id=None, workspace=tmp_path)
    epoch_id = await store.start_epoch(
        StoredIdentity(launch=_identity(), tool_fingerprint="fp")
    )
    session, model, _policy = await _tight_manual_session(tmp_path, store=store, epoch_id=epoch_id)
    try:
        outcome = await session.execute("q")
        assert outcome.result.stop_reason == "error"
        assert "context budget exceeded" in outcome.result.error
        assert outcome.exit_code == 1, "普通模式错误路径 exit 1"
        assert len(model.requests) == 1, "第二轮被拦, 适配器只见首轮请求"
        assert outcome.persistence.saved is True
        snapshot = store.recovery_snapshot()
        assert len(snapshot.requests) == 1
        persisted = snapshot.requests[0].outcome
        assert persisted.finalized is True
        assert persisted.stop_reason == "error"
        assert "context budget exceeded" in persisted.error
    finally:
        await session.aclose()
        await store.aclose()


async def test_per_turn_rejection_nonpersistent_exit_one(tmp_path):
    session, model, _policy = await _tight_manual_session(tmp_path)
    try:
        outcome = await session.execute("q")
        assert outcome.result.stop_reason == "error"
        assert "context budget exceeded" in outcome.result.error
        assert outcome.exit_code == 1
        assert len(model.requests) == 1
    finally:
        await session.aclose()


# --- D1 四态与检查模式控制段 ------------------------------------------------------------


async def test_budget_only_boot_has_no_run_policy(tmp_path):
    """非检查+预算: 仅 budget_policy(inner), 无 RunPolicy、不取 verification capability."""
    model = FakeModel([list(STOP_WITH_METRICS)])
    session = await open_application(_config(tmp_path, model, budget=_generous()))
    try:
        outcome = await session.execute("hello")
        assert outcome.exit_code == 0
        sent = model.requests[0]
        assert sent.system_prompt == "test system prompt", "无控制段"
        assert "<cicada-verification-state>" not in sent.system_prompt
        snapshot = session.budget
        assert snapshot is not None
        assert snapshot.limit == 1_000_000
        assert snapshot.report is not None
        assert snapshot.report.source == "estimated"
    finally:
        await session.aclose()


async def test_check_mode_budget_covers_control_section(tmp_path):
    """检查+预算: BudgetPolicy 所见请求含控制段, 计量覆盖动态验证段最终形态 (A01/D1)."""
    root = _repo(tmp_path)
    model = FakeModel([list(STOP_WITH_METRICS)])
    session = await open_application(
        _config(root, model, budget=_generous(), plan=_plan(root))
    )
    try:
        outcome = await session.execute("hello")
        # 检查模式交付判定: 模型 stop 但检查未运行 -> can_deliver=False -> exit 3 (既有语义)
        assert outcome.exit_code == 3
        sent = model.requests[0]
        assert "<cicada-verification-state>" in sent.system_prompt
        report = session.budget.report
        assert report is not None
        tools = tuple(tool.spec for tool in session._app.agent._tools.values())
        raw, _components = estimate_input_components(
            "test system prompt", tools, (UserMessage(text="hello"),), session.config.budget
        )
        assert report.input_estimate > raw, "预算计量包含控制段增量"
        assert report.source == "estimated"
    finally:
        await session.aclose()


# --- 请求边界接缝 (D1) -----------------------------------------------------------------


class _SeamRecorder(BudgetPolicy):
    """记录接缝调用 (committed_count, 接缝时适配器已见请求数) 的策略."""

    def __init__(self, inner, config, model: FakeModel):
        super().__init__(inner, config)
        self._model = model
        self.seams: list[tuple[int, int]] = []

    def note_request_start(self, committed_count: int) -> None:
        self.seams.append((committed_count, len(self._model.requests)))
        super().note_request_start(committed_count)


async def test_request_boundary_seam_timing_and_arguments(tmp_path):
    model = FakeModel([list(STOP_WITH_METRICS), list(STOP_WITH_METRICS)])
    config = _config(tmp_path, model, budget=_generous())
    policy = _SeamRecorder(model, config.budget, model)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None, budget_policy=policy,
    )
    try:
        await session.execute("q1")
        await session.execute("q2")
        # q1 后规范历史 = [user q1, assistant] -> 长度 2; 接缝时适配器仅见首轮请求
        assert policy.seams == [(0, 0), (2, 1)], (
            "每请求一次、参数=调用时规范历史长度、先于 agent.run (适配器零新增)"
        )
    finally:
        await session.aclose()


async def test_clear_drops_anchor_through_seam(tmp_path):
    model = FakeModel([
        list(STOP_WITH_METRICS),
        [StreamDone("stop", metrics=ModelMetrics(input_tokens=55, output_tokens=1))],
        [StreamDone("stop")],
    ])
    session = await open_application(_config(tmp_path, model, budget=_generous()))
    try:
        await session.execute("q1")
        assert session.budget.anchor.measured_input_tokens == 42
        await session.execute("q2")
        assert session.budget.report.source == "anchored", "请求 2 = 请求 1 的前缀延长"
        session.clear_history()
        await session.execute("q3")
        assert session.budget.report.source == "estimated", "接缝 committed_count=0 丢弃锚"
        assert session.budget.anchor is None, "q3 无计量, 锚保持空"
    finally:
        await session.aclose()


# --- wire 字节守卫独立语义 (A04) --------------------------------------------------------


def test_budget_rejection_distinct_from_wire_byte_guard():
    """预算超限与 P4 §8 wire 超限是两条可区分信息: 文本不同源, 语义互不替代."""
    config = BudgetConfig(num_ctx=10, num_predict=1)
    report = assemble_report(config, input_estimate=1000, input_components=(), source="estimated")
    assert report.total_estimate > report.limit
    text = budget_exceeded_text(report)
    assert text.startswith("context budget exceeded")
    assert "num_ctx" in text
    # wire 守卫是 adapter 级另一条信息 (ollama request too large ...), 不在预算文本中
    assert "ollama request too large" not in text
