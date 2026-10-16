"""PV02 跨进程总额度闸与离线 fixture 的最小关键回归。

只覆盖关键风险 (预算准入、持久化/跨进程契约、封口语义), 不做状态笛卡尔积:
金额与次数上限的越界、重复预留/重复结算的结构性检出、封口保留 pending、
refused-before-send 保留尝试号、第二进程续跑核对与各类矛盾拒绝, 以及
fixture 编辑计划的自检。全部离线, 不调用 Docker、不发模型请求。
"""
import json
from pathlib import Path

import pytest

from cicada.core.ports import ModelMetrics
from cicada.costs import CostProfile
from tests.support import swebench_budget as budget_mod
from tests.support import swebench_lifecycle as life

# 冻结单价: 普通 cap 8192 的预留 = ceil((1_100_000×2 + 8192×8)/1e6) = 227 分;
# 摘要 cap 2048 = 222 分。这两个数字是本模块多处断言的基础。
NORMAL_RESERVE_CENT = 227
SUMMARY_RESERVE_CENT = 222


def _profile(tmp_path: Path) -> CostProfile:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
        "checked_at": "2026-10-15",
        "models": {"deepseek-flash": {"input_cny_per_million": "2",
                                      "output_cny_per_million": "8"}}},
        ensure_ascii=False), encoding="utf-8")
    return CostProfile.load(path)


def _budget(tmp_path: Path, *, limit_cent: int = 450, max_calls: int = 24,
            attempt_id: str = "attempt-1", manifest_hash: str = "m" * 64):
    return budget_mod.AttemptBudget(
        tmp_path / "attempt", attempt_id=attempt_id, manifest_hash=manifest_hash,
        prepared_lock_hash="l" * 64, profile=_profile(tmp_path),
        model_identity={"provider": "deepseek", "model": "deepseek-flash"},
        limit_cent=limit_cent, max_calls=max_calls)


def test_reserve_uses_the_production_integer_ceiling(tmp_path):
    """预留金额必须等于生产整数向上取分算法的结果, 不是旧固定公式。"""
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    record = budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                            output_cap=8192)
    assert record["reserve_cent"] == NORMAL_RESERVE_CENT
    summary = budget.reserve(call_id="c2", phase=budget_mod.PROCESS_1, kind="summary",
                             output_cap=2048)
    assert summary["reserve_cent"] == SUMMARY_RESERVE_CENT
    assert budget.totals()["pending_cent"] == NORMAL_RESERVE_CENT + SUMMARY_RESERVE_CENT


def test_no_send_without_a_price_profile(tmp_path):
    """没有显式价格 profile 就不存在总闸, 必须拒绝构造。"""
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget_mod.AttemptBudget(
            tmp_path / "attempt", attempt_id="a", manifest_hash="m",
            prepared_lock_hash="l", profile=None,
            model_identity={"model": "deepseek-flash"})


def test_reserve_requires_the_lock_and_the_matching_phase(tmp_path):
    budget = _budget(tmp_path)
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                       output_cap=8192)
    budget.acquire(budget_mod.PROCESS_1)
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget.reserve(call_id="c1", phase=budget_mod.PROCESS_2, kind="normal",
                       output_cap=8192)


def test_amount_limit_refuses_and_seals_without_sending(tmp_path):
    """金额越界必须零发送并封口; 227 分下 450 分只容得下一次预留。"""
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                   output_cap=8192)
    with pytest.raises(budget_mod.AttemptBudgetError, match="allowance"):
        budget.reserve(call_id="c2", phase=budget_mod.PROCESS_1, kind="normal",
                       output_cap=8192)
    assert budget.sealed_reason is not None
    # 封口后不得再有新预留
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget.reserve(call_id="c3", phase=budget_mod.PROCESS_1, kind="summary",
                       output_cap=2048)


def test_call_limit_refuses_the_25th_call(tmp_path):
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    for index in range(24):
        budget.reserve(call_id=f"c{index}", phase=budget_mod.PROCESS_1,
                       kind="summary", output_cap=2048)
        budget.settle(f"c{index}", ModelMetrics(input_tokens=0, output_tokens=0),
                      complete=True, closed=True)
    assert budget.totals()["calls"] == 24
    with pytest.raises(budget_mod.AttemptBudgetError, match="limit reached"):
        budget.reserve(call_id="c25", phase=budget_mod.PROCESS_1, kind="summary",
                       output_cap=2048)


def test_sealing_keeps_the_reservation_pending(tmp_path):
    """缺 usage 不得当 0 结算: 保留 pending 并封口, 不自动释放。"""
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                   output_cap=8192)
    assert budget.settle("c1", None, complete=True, closed=True) is False
    totals = budget.totals()
    assert totals["pending_cent"] == NORMAL_RESERVE_CENT
    assert totals["settled_cent"] == 0
    assert budget.sealed_reason is not None


def test_settlement_above_the_reservation_is_refused(tmp_path):
    """实测超预留不得静默超支: 封口并保留 pending。"""
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                   output_cap=8192)
    assert budget.settle("c1", ModelMetrics(input_tokens=10, output_tokens=999_999),
                         complete=True, closed=True) is False
    assert budget.totals()["pending_cent"] == NORMAL_RESERVE_CENT


def test_duplicate_reservation_and_settlement_are_detected(tmp_path):
    """append-only 事件流让"重复预留/重复结算"成为结构性矛盾而非被覆盖抹掉。"""
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                   output_cap=8192)
    budget.settle("c1", ModelMetrics(input_tokens=100, output_tokens=10),
                  complete=True, closed=True)
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget.settle("c1", ModelMetrics(input_tokens=1, output_tokens=1),
                      complete=True, closed=True)
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                       output_cap=8192)


def test_refused_before_send_releases_amount_but_keeps_the_attempt_number(tmp_path):
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                   output_cap=8192)
    budget.refuse_before_send("c1", "process ledger refused")
    totals = budget.totals()
    assert totals["pending_cent"] == 0
    assert totals["calls"] == 1, "尝试号必须保守保留, 不可重用"
    with pytest.raises(budget_mod.AttemptBudgetError):
        budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                       output_cap=8192)


def test_corrupt_record_line_is_rejected_not_repaired(tmp_path):
    budget = _budget(tmp_path)
    budget.acquire(budget_mod.PROCESS_1)
    budget.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                   output_cap=8192)
    with budget.calls_path.open("a", encoding="utf-8") as handle:
        handle.write('{"call_id": "c2", "event": "res')
    with pytest.raises(budget_mod.AttemptBudgetError, match="truncated"):
        budget.totals()


def test_second_process_verifies_and_continues_the_same_attempt(tmp_path):
    """第一进程 checkpoint → 第二进程身份全等、重算一致后才可续跑。"""
    first = _budget(tmp_path)
    first.acquire(budget_mod.PROCESS_1)
    first.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                  output_cap=8192)
    first.settle("c1", ModelMetrics(input_tokens=500, output_tokens=50),
                 complete=True, closed=True)
    checkpoint = first.close_checkpoint(process_identity={"pid": "p1"})
    first.release()
    assert checkpoint["calls"] == 1 and checkpoint["pending_cent"] == 0

    second = _budget(tmp_path)
    second.acquire(budget_mod.PROCESS_2)
    verified = second.verify_continuation(expect_phase=budget_mod.PROCESS_1)
    assert verified["totals"]["calls"] == 1
    # 续跑继承已结算金额, 且可以继续预留
    assert verified["totals"]["settled_cent"] == 1
    second.reserve(call_id="c2", phase=budget_mod.PROCESS_2, kind="normal",
                   output_cap=8192)
    assert second.totals()["calls"] == 2


def test_second_process_refuses_on_identity_drift(tmp_path):
    first = _budget(tmp_path)
    first.acquire(budget_mod.PROCESS_1)
    first.close_checkpoint(process_identity={"pid": "p1"})
    first.release()
    drifted = _budget(tmp_path, manifest_hash="D" * 64)
    drifted.acquire(budget_mod.PROCESS_2)
    with pytest.raises(budget_mod.AttemptBudgetError, match="identity drift"):
        drifted.verify_continuation(expect_phase=budget_mod.PROCESS_1)


def test_second_process_refuses_a_pending_reservation_without_releasing_it(tmp_path):
    first = _budget(tmp_path)
    first.acquire(budget_mod.PROCESS_1)
    first.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                  output_cap=8192)
    first.write_checkpoint({"phase": budget_mod.PROCESS_1})
    first.release()
    second = _budget(tmp_path)
    second.acquire(budget_mod.PROCESS_2)
    with pytest.raises(budget_mod.AttemptBudgetError, match="pending"):
        second.verify_continuation(expect_phase=budget_mod.PROCESS_1)
    assert second.sealed_reason is not None
    assert second.totals()["pending_cent"] == NORMAL_RESERVE_CENT, "不得自动释放"


def test_second_process_refuses_a_checkpoint_that_contradicts_the_records(tmp_path):
    first = _budget(tmp_path)
    first.acquire(budget_mod.PROCESS_1)
    first.reserve(call_id="c1", phase=budget_mod.PROCESS_1, kind="normal",
                  output_cap=8192)
    first.write_checkpoint({"phase": budget_mod.PROCESS_1, "pending_cent": 0, "calls": 7})
    first.release()
    second = _budget(tmp_path)
    second.acquire(budget_mod.PROCESS_2)
    with pytest.raises(budget_mod.AttemptBudgetError, match="contradicts"):
        second.verify_continuation(expect_phase=budget_mod.PROCESS_1)


def test_live_lock_holder_is_not_preempted(tmp_path):
    holder = _budget(tmp_path)
    holder.acquire(budget_mod.PROCESS_1)
    other = _budget(tmp_path)
    with pytest.raises(budget_mod.AttemptBudgetError, match="live"):
        other.acquire(budget_mod.PROCESS_2)


# --- 离线 fixture 的编辑计划 ---


def test_unique_edits_reproduce_the_reference_and_reject_ambiguity():
    base = "alpha\nbeta\ngamma\ndelta\nbeta\n"
    fixed = "alpha\nBETA\ngamma\ndelta\nbeta\n"
    edits = life.unique_edits(base, fixed)
    produced = base
    for edit in edits:
        assert base.count(edit["old_text"]) == 1, "old_text 必须唯一"
        produced = produced.replace(edit["old_text"], edit["new_text"])
    assert produced == fixed


def test_unique_edits_expand_context_until_unique():
    """两处相同文本时, 必须扩大上下文才能唯一匹配。"""
    base = "x\ny\nsame\nz\nx\ny\nsame\nw\n"
    fixed = "x\ny\nsame\nz\nx\ny\nCHANGED\nw\n"
    edits = life.unique_edits(base, fixed)
    for edit in edits:
        assert base.count(edit["old_text"]) == 1
    produced = base
    for edit in edits:
        produced = produced.replace(edit["old_text"], edit["new_text"])
    assert produced == fixed


def test_offline_fixture_rejects_being_driven_past_its_script():
    """剧本耗尽必须报错, 不能悄悄回退成"成功"响应。"""
    import httpx

    transport = life.offline_fixture(edits={}, allowed_paths=(), sequence=())
    with pytest.raises(ValueError, match="exhausted"):
        transport.handle_request(httpx.Request("POST", "https://api.deepseek.com/chat/completions"))
