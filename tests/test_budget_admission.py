"""应用层预算准入集成测试 (01, A02/A04 + D1 四态 + D9 启用条件).

两道口分别钉住: 请求前预估 (零 begin/零模型调用/不锁存 storage_failed) 与
BudgetPolicy 每轮权威准入 (零发送 + 既有错误路径收尾 + 持久模式落 finalized
错误记录)。请求边界接缝的调用时机与参数值用记录策略直接断言。fake 直建
配置 (budget=None) 的既有基线不本文件覆盖——既有全量套件即两叉验收的 a 叉。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
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
from cicada.core.ports import ModelMetrics, ModelRequest, StreamDone, ToolCallEvent
from cicada.persistence.contracts import StorageUnavailableError, StoredIdentity
from cicada.plugins.coding.verification import generate_verification_run_id
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.deepseek import DeepSeekModel, DeepSeekConfig
from cicada.plugins.session_store import open_session_store

from test_application import _prompt_info, _repo
from test_persistence_application import _identity

STOP_WITH_METRICS = [StreamDone("stop", metrics=ModelMetrics(input_tokens=42, output_tokens=1))]


def _config(root: Path, model: FakeModel, *, budget: BudgetConfig | None = None,
            plan: VerificationPlan | None = None,
            persistence: PersistenceConfig | None = None) -> ApplicationConfig:
    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(model),
        plan=plan,
        max_turns=6,
        model_label="fake",
        persistence=persistence,
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

    def recovery_snapshot(self):
        return SimpleNamespace(user_name=None, short_title="")

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
    """检查+预算: BudgetPolicy 所见请求含固定规则与程序尾部, 计量覆盖验证段最终形态 (缓存 02 C4)."""
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
        assert "<cicada-verification-rules>" in sent.system_prompt, "固定信任规则在 system"
        assert sent.request_context.startswith('<cicada-verification-context version="1">'), (
            "当前验证观察在独立尾部"
        )
        report = session.budget.report
        assert report is not None
        tools = tuple(tool.spec for tool in session._app.agent._tools.values())
        raw, _components = estimate_input_components(
            "test system prompt", tools, (UserMessage(text="hello"),), session.config.budget
        )
        assert report.input_estimate > raw, "预算计量包含固定规则与尾部增量"
        assert any(c.label == "request_context" for c in report.components), "尾部单独组成"
        assert report.source == "estimated"
    finally:
        await session.aclose()


async def test_complete_counter_two_admission_gates_and_finalized_qualification_error(tmp_path):
    from cicada.budget import LocalTokenCount

    class Counter:
        def __init__(self):
            self.requests = []
        def count_request(self, request):
            self.requests.append(request)
            if request.request_context:
                raise ValueError("safe qualification rejection")
            return LocalTokenCount(10000, "test-profile")

    model = FakeModel([])
    session = await open_application(_config(tmp_path, model, budget=BudgetConfig(500, 1)))
    counter = Counter()
    session._budget_policy.token_counter = counter
    try:
        with pytest.raises(RuntimeError, match="context budget exceeded"):
            await session.execute("small user")
        assert len(counter.requests) == 1
        assert counter.requests[0].messages[-1] == UserMessage("small user")
        assert model.requests == [] and session.state == "idle"
    finally:
        await session.aclose()
    root = _repo(tmp_path)
    persistence = PersistenceConfig(identity=_identity(), session_id=None)
    session = await open_application(_config(root, model, budget=_generous(), plan=_plan(root), persistence=persistence))
    counter = Counter()
    session._budget_policy.token_counter = counter
    try:
        outcome = await session.execute("checked user")
        assert len(counter.requests) == 1  # 容量预检不调用完整 counter
        assert counter.requests[0].request_context
        assert outcome.result.stop_reason == "error" and outcome.result.error == "safe qualification rejection"
        assert model.requests == [] and session.state == "idle"
        assert outcome.persistence.saved is True
        stored = session._store.recovery_snapshot().requests[-1].outcome
        assert stored.finalized is True and stored.stop_reason == "error"
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
        assert session.budget.report is None and session.budget.anchor is None, "成功清空立即失效"
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
    # wire 守卫是 adapter 级另一条信息 (request exceeds byte limit ...), 不在预算文本中
    assert "request exceeds byte limit" not in text


# --- 审查处置 (01 独立审查 2026-10-06) --------------------------------------------------


class _FailingBeginStore(_BeginSpyStore):
    """begin_request 恒抛 StorageUnavailableError: 验证 storage_failed 锁存面."""

    async def begin_request(self, epoch_id, prompt):
        raise StorageUnavailableError("store closed")


async def test_storage_failed_latch_precedes_budget_pre_estimate(tmp_path):
    """锁存 storage_failed 的会话: 超大 prompt 仍先报 storage_failed, 不被预算文本覆盖 (01-F4)."""
    model = FakeModel([])
    config = _config(tmp_path, model, budget=_generous())
    policy = BudgetPolicy(model, config.budget)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=_FailingBeginStore(), epoch_id="epoch-1", persistence_mode="save",
        budget_policy=policy,
    )
    try:
        with pytest.raises(RuntimeError, match="session save failed before the request started"):
            await session.execute("q")
        assert session.persistence.storage_failed is not None
        # 锁存后即使超大 prompt 也走 storage_failed 门 (先于预算预估)
        with pytest.raises(RuntimeError, match="storage_failed: earlier session save failed") as exc_info:
            await session.execute("x" * 1_000_000)
        assert "context budget" not in str(exc_info.value)
        assert model.requests == []
    finally:
        await session.aclose()


async def test_check_mode_per_turn_rejection_model_stopped_false(tmp_path):
    """检查模式每轮准入拒绝: 既有错误路径收尾, 交付判定 model_stopped=False (A02, 01-F8/F12)."""
    root = _repo(tmp_path)
    (root / "big.txt").write_text("x" * 20_000, encoding="utf-8")
    plan = _plan(root)
    # 开一次 (不执行请求) 取实际工具声明与同源预估面, 据此精确算出请求前预估门槛
    warm = await open_application(_config(root, FakeModel([]), budget=_generous(), plan=plan))
    try:
        tools = tuple(tool.spec for tool in warm._app.agent._tools.values())
        assert warm._policy is not None
        system_for_estimate = warm._policy.preflight_system_prompt("test system prompt")
        tail_cap = warm._policy.request_context_max_bytes
    finally:
        await warm.aclose()
    shape = BudgetConfig(num_ctx=1, num_predict=2048)
    raw, _components = estimate_input_components(
        system_for_estimate, tools, (UserMessage(text="q"),), shape
    )
    input_pre = raw + math.ceil(tail_cap / 3) + shape.message_template_overhead  # verification_context_allowance
    threshold = input_pre + 2048 + max(256, math.ceil(0.10 * input_pre))
    # num_ctx 恰在预估门槛之上: 预估与首轮都放行; 首轮 read 20KB 结果使次轮必超
    model = FakeModel([
        [ToolCallEvent(id="c1", name="read", arguments_json='{"path": "big.txt"}'),
         StreamDone("tool_use")],
    ])
    session = await open_application(
        _config(root, model, budget=BudgetConfig(num_ctx=threshold + 50, num_predict=2048), plan=plan)
    )
    try:
        outcome = await session.execute("q")
        assert outcome.result.stop_reason == "error"
        assert "context budget exceeded" in outcome.result.error
        assert len(model.requests) == 1, "次轮被拦, 适配器只见首轮请求"
        assert outcome.decision is not None
        assert outcome.decision.model_stopped is False, "预算拒绝不是模型 stop"
        assert outcome.exit_code == 1
    finally:
        await session.aclose()


async def test_restored_session_starts_without_anchor(tmp_path):
    """恢复后首轮: 新策略实例无锚, 恢复首请求纯启发式 (A03 矩阵, 01-F7)."""
    model1 = FakeModel([list(STOP_WITH_METRICS)])
    config1 = _config(
        tmp_path, model1, budget=_generous(),
        persistence=PersistenceConfig(session_id=None, identity=_identity()),
    )
    session1 = await open_application(config1)
    await session1.execute("q1")
    assert session1.budget.anchor is not None
    saved_id = session1.persistence.session_id
    await session1.aclose()

    model2 = FakeModel([[StreamDone("stop")]])
    config2 = _config(
        tmp_path, model2, budget=_generous(),
        persistence=PersistenceConfig(session_id=saved_id, identity=_identity()),
    )
    session2 = await open_application(config2)
    try:
        assert session2.budget is not None
        assert session2.budget.anchor is None, "恢复打开新策略实例, 不继承旧锚"
        await session2.execute("q2")
        assert session2.budget.report.source == "estimated", "恢复首请求无锚可用"
    finally:
        await session2.aclose()


def _wire_model(limit: int | None) -> DeepSeekModel:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be reached: guards fire before send")

    config = DeepSeekConfig(think=False, max_request_bytes=limit or 2 * 1024 * 1024)
    return DeepSeekModel(httpx.AsyncClient(transport=httpx.MockTransport(handler)), config, "offline-key")


async def test_budget_and_wire_guards_coexist_and_stay_distinct():
    """预算启用组合下两道守卫共存且信息可区分 (A04, 01-F5): 预算在策略层, wire 在适配层."""
    from cicada.core.cancel import CancelToken

    request = ModelRequest(messages=(UserMessage(text="x" * 3000),), tools=(), system_prompt="sys")
    token = CancelToken()

    tight = BudgetPolicy(_wire_model(None), BudgetConfig(num_ctx=100, num_predict=1))
    events = [event async for event in tight.stream(request, token)]
    assert events[-1].stop_reason == "error"
    assert events[-1].error is not None and "context budget exceeded" in events[-1].error
    assert "request exceeds byte limit" not in events[-1].error

    loose = BudgetPolicy(_wire_model(10), BudgetConfig(num_ctx=1_000_000, num_predict=1))
    events = [event async for event in loose.stream(request, token)]
    assert events[-1].stop_reason == "error"
    assert events[-1].error is not None and "request exceeds byte limit" in events[-1].error
    assert "context budget" not in events[-1].error


# --- PA02: 接缝生命周期 (02 迁移: 准入后、首次预估前) ------------------------------------


class _OrderSeamPolicy(BudgetPolicy):
    """记录接缝调用的策略; 与 store 的 begin 记录共享同一事件列表."""

    def __init__(self, inner, config, events):
        super().__init__(inner, config)
        self._events = events

    def note_request_start(self, committed_count: int) -> None:
        self._events.append(("seam", committed_count))
        super().note_request_start(committed_count)


class _OrderSpyStore(_BeginSpyStore):
    def __init__(self, events):
        super().__init__()
        self._events = events

    async def begin_request(self, epoch_id, prompt):
        self._events.append(("begin", prompt))
        return await super().begin_request(epoch_id, prompt)


async def _order_session(tmp_path, events, *, script=None, budget=None):
    model = FakeModel(script if script is not None else [list(STOP_WITH_METRICS)])
    config = _config(tmp_path, model, budget=budget or _generous())
    policy = _OrderSeamPolicy(model, config.budget, events)
    store = _OrderSpyStore(events)
    app = await bootstrap(
        [fake_model_plugin(model)], tool_capabilities=(),
        system_prompt="test system prompt", model_policy=lambda inner, runtime: policy,
    )
    session = ApplicationSession(
        app=app, config=config, service=None, policy=None,
        store=store, epoch_id="epoch-1", persistence_mode="save", budget_policy=policy,
    )
    return session, model, store


async def test_seam_precedes_begin_and_recurs_per_attempt(tmp_path):
    """PA02: 接缝在首次预估前建立范围 (begin 之前), 每次尝试重新建立, begin 不再调用."""
    events: list = []
    session, _model, _store = await _order_session(tmp_path, events)
    try:
        await session.execute("q1")
        await session.execute("q2")
        assert events[:4] == [("seam", 0), ("begin", "q1"), ("seam", 2), ("begin", "q2")], (
            "接缝先于 begin, 参数=该时刻规范历史消息数 (含 user prompt 与 assistant 回答)"
        )
        # 预估拒绝的尝试同样先建立范围; 下一次请求重新建立 (PA02)
        events.clear()
        tight = await _order_session(tmp_path, events, budget=BudgetConfig(num_ctx=1, num_predict=1))
        try:
            with pytest.raises(RuntimeError, match="context budget exceeded"):
                await tight[0].execute("x" * 500)
            assert [kind for kind, _value in events] == ["seam"], "拒绝尝试只建立范围, 不 begin"
            assert tight[0]._budget_policy._last_committed_count == 0
        finally:
            await tight[0].aclose()
    finally:
        await session.aclose()


async def test_seam_exception_rejects_request_and_next_attempt_recovers(tmp_path):
    events: list = []
    session, _model, store = await _order_session(tmp_path, events)
    try:
        # 首次接缝调用抛异常: execute 失败, 零 begin, 会话保持可用
        policy = session._budget_policy
        original = policy.note_request_start
        calls = {"n": 0}

        def flaky_seam(count: int) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("seam exploded")
            original(count)

        policy.note_request_start = flaky_seam  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="seam exploded"):
            await session.execute("q1")
        assert store.begins == [], "接缝异常在 begin 之前, 无 intent"
        assert session.state == "idle"
        assert session.persistence.storage_failed is None
        # 下一次请求重新建立并成功
        outcome = await session.execute("q1")
        assert outcome.result.stop_reason == "stop"
        assert ("seam", 0) in events and ("begin", "q1") in events
    finally:
        await session.aclose()
