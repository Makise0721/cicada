"""live 父子进程边界的关键回归; inner 全部为 MockTransport。"""
import asyncio
import json
import time
import threading
from pathlib import Path

import httpx
import pytest

from cicada.costs import CostProfile
from tests.support import swebench_lifecycle as lifecycle
from tests.support.swebench_budget import AttemptBudget, AttemptBudgetError, PROCESS_1
from tests.support.swebench_live import (
    WATCH_MAX_FAILURES,
    Broker,
    RemoteGate,
    _supervise_child,
    check_gate,
    watch_parent,
)


def test_go_broker_uses_frozen_route_and_stops_at_physical_send_limit(tmp_path):
    from cicada.core.ports import ModelMetrics, StreamDone
    from tests.support.swebench_batch import CallBudget, MODEL, classify_go

    seen = []

    def respond(request):
        seen.append((str(request.url), request.headers["x-opencode-session"], request.headers["Authorization"]))
        return lifecycle._sse([lifecycle._text_frame(0, "fixture")])

    async def run():
        budget = CallBudget(tmp_path / "calls.json", 1)
        gate = lifecycle.AttemptGateTransport(httpx.MockTransport(respond), budget,
            phase=PROCESS_1, request_classifier=classify_go)
        broker = Broker(gate=gate, check=lambda: None, deadline=time.monotonic() + 10,
            credential="parent-only-key", model_url=MODEL["base_url"] + "/chat/completions",
            model_headers={"x-opencode-session": "frozen-session"}, trace_dir=tmp_path / "model-trace")
        body = json.dumps({"model": MODEL["model"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": MODEL["max_tokens"],
            "thinking": {"type": "disabled"}, "messages": []}).encode()
        try:
            _, headers, _ = await broker.dispatch("/model", body)
            assert budget.calls[0]["state"] == "pending"
            gate.confirm(headers["X-Call-Id"], StreamDone("stop", metrics=ModelMetrics(100, 10)))
            with pytest.raises(AttemptBudgetError, match="physical call budget"):
                await broker.dispatch("/model", body)
            assert gate.handoffs == 1
            assert budget.calls[0]["state"] == "complete"
            assert seen == [(MODEL["base_url"] + "/chat/completions", "frozen-session", "Bearer parent-only-key")]
            assert json.loads((tmp_path / "calls.json").read_text())["cost_cny"] is None
            assert json.loads((tmp_path / "model-trace/turn-0001.request.json").read_text())["model"] == MODEL["model"]
            assert "fixture" in (tmp_path / "model-trace/turn-0001.response.sse").read_text()
            assert all("parent-only-key" not in p.read_text() for p in (tmp_path / "model-trace").iterdir())
        finally:
            await broker.close()
    asyncio.run(run())


def test_check_feedback_is_opt_in_bounded_and_keeps_private_results_parent_side(tmp_path):
    names = ["long-test-" + "x" * 600, *(f"test-{i}" for i in range(6))]
    summary = {"result": "failed", "required_passed": 3, "required_total": 10,
               "patch_empty": False, "grade_id": "2", "cleanup": "confirmed",
               "not_passed_tests": names, "raw_logs": "PRIVATE_ASSERTION_AND_SOURCE",
               "snapshot": {"private": "PRIVATE_SNAPSHOT"}}

    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(lambda r: lifecycle._sse([])))
        baseline = Broker(gate=gate, check=lambda: (1, summary), deadline=time.monotonic() + 10)
        candidate = Broker(gate=gate, check=lambda: (1, summary), deadline=time.monotonic() + 10,
                           check_feedback="required-tests-v1")
        try:
            _, _, old_bytes = await baseline.dispatch("/check", b"")
            assert set(json.loads(old_bytes)["summary"]) == {
                "result", "required_passed", "required_total", "patch_empty", "grade_id", "cleanup"}
            _, _, new_bytes = await candidate.dispatch("/check", b"")
            visible = json.loads(new_bytes)["summary"]
            assert visible["not_passed_tests"] == [name[:512] for name in names[:5]]
            assert visible["not_passed_tests_total"] == 7
            assert visible["not_passed_tests_truncated"] is True
            assert b"PRIVATE_" not in new_bytes
            assert candidate.checks[0]["not_passed_tests"] == names
        finally:
            await baseline.close()
            await candidate.close()
            gate.budget.release()
    asyncio.run(run())


def test_response_trace_write_failure_closes_response_and_seals_pending_call(tmp_path, monkeypatch):
    original = Path.write_bytes

    def write(path, data):
        if path.suffix == ".sse":
            raise OSError("trace disk failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", write)

    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(
            lambda r: lifecycle._sse([lifecycle._text_frame(0, "fixture")])))
        broker = Broker(gate=gate, check=lambda: None, deadline=time.monotonic() + 10,
                        trace_dir=tmp_path / "model-trace")
        try:
            with pytest.raises(OSError, match="trace disk failure"):
                await broker.dispatch("/model", _body())
            assert gate.handoffs == 1
            assert next(iter(gate.observations.values())).closed
            assert gate.budget.sealed_reason
            assert gate.budget.totals()["pending_cent"] > 0
        finally:
            await broker.close()
            gate.budget.release()
    asyncio.run(run())


def test_go_missing_usage_remains_unknown_and_cannot_send_again(tmp_path):
    from tests.support.swebench_batch import CallBudget, MODEL, classify_go

    async def run():
        frame = lifecycle._text_frame(0, "fixture")
        frame.pop("usage")
        budget = CallBudget(tmp_path / "calls.json", 2)
        gate = lifecycle.AttemptGateTransport(httpx.MockTransport(lambda r: lifecycle._sse([frame])),
            budget, phase=PROCESS_1, request_classifier=classify_go)
        body = json.dumps({"model": MODEL["model"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 8192,
            "thinking": {"type": "disabled"}}).encode()
        response = await gate.handle_async_request(httpx.Request("POST", MODEL["base_url"], content=body))
        await response.aread()
        await response.aclose()
        with pytest.raises(AttemptBudgetError):
            await gate.handle_async_request(httpx.Request("POST", MODEL["base_url"], content=body))
        assert gate.handoffs == 1
        assert budget.calls[0]["state"] == "pending"
        assert budget.calls[0]["input_tokens"] is None
    asyncio.run(run())


def _gate(tmp_path, inner):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema_version": 1, "source": "https://example.org/pricing",
        "checked_at": "2026-10-16", "models": {"deepseek-flash": {
            "input_cny_per_million": "2", "output_cny_per_million": "8"}}}))
    profile = CostProfile.load(path)
    budget = AttemptBudget(tmp_path / "budget", attempt_id="test", manifest_hash="m",
        prepared_lock_hash="p", profile=profile,
        model_identity={"model": "deepseek-flash"})
    budget.acquire(PROCESS_1)
    return lifecycle.AttemptGateTransport(inner, budget, phase=PROCESS_1)


def _body():
    return json.dumps({"model": "deepseek-flash", "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": 2048,
        "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "x"}]}).encode()


def test_broker_remote_transport_requires_production_confirmation_and_keeps_key_parent_side(tmp_path):
    seen = []

    def respond(request):
        seen.append(request.headers["Authorization"])
        return lifecycle._sse([lifecycle._text_frame(0, "fixture")])

    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(respond))
        broker = Broker(gate=gate, check=lambda: None, deadline=time.monotonic() + 10,
                        credential="parent-only-secret")
        endpoint = broker.start()
        remote = RemoteGate(endpoint, broker.token)
        try:
            response = await remote.handle_async_request(httpx.Request("POST",
                "https://api.deepseek.com/chat/completions", content=_body(),
                headers={"Authorization": "Bearer child-placeholder"}))
            await response.aread()
            await response.aclose()
            assert seen == ["Bearer parent-only-secret"]
            assert gate.budget.totals()["pending_cent"] > 0
            call_id = next(iter(remote.observations))
            from cicada.core.ports import ModelMetrics, StreamDone
            await asyncio.to_thread(remote.confirm, call_id,
                StreamDone("stop", metrics=ModelMetrics(input_tokens=100, output_tokens=10)))
            assert gate.budget.totals()["pending_cent"] == 0
            assert gate.budget.totals()["settled_cent"] == 1
        finally:
            await remote.aclose()
            await broker.close()
            gate.budget.release()
    asyncio.run(run())


def test_broker_does_not_send_again_without_confirmation_or_after_deadline(tmp_path):
    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(
            lambda request: lifecycle._sse([lifecycle._text_frame(0, "fixture")])))
        broker = Broker(gate=gate, check=lambda: None, deadline=time.monotonic() + 10)
        try:
            await broker.dispatch("/model", _body())
            with pytest.raises(AttemptBudgetError, match="confirmation"):
                await broker.dispatch("/model", _body())
            assert gate.handoffs == 1
            assert gate.budget.totals()["pending_cent"] > 0
            broker.deadline = time.monotonic() - 1
            with pytest.raises(AttemptBudgetError, match="deadline"):
                await broker.dispatch("/model", _body())
            assert gate.handoffs == 1
        finally:
            gate.budget.release()
    asyncio.run(run())


def test_supervisor_seals_before_cancel_and_confirms_os_exit(tmp_path, monkeypatch):
    from tests.support import swebench_live as live

    events = []

    class Child:
        pid = 123
        def __init__(self, *args):
            self.n = 0
        def poll(self):
            self.n += 1
            return None if self.n < 4 else 2
        def terminate_tree(self):
            events.append("terminate")
        def close(self):
            events.append("close")

    monkeypatch.setattr(live.isolation, "LowProcess", Child)

    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(lambda req: None))
        broker = Broker(gate=gate, check=lambda: None, deadline=time.monotonic() - 1)
        original = broker.stop
        def stop(reason):
            events.append("sealed")
            original(reason)
        broker.stop = stop
        try:
            facts = await _supervise_child({"child_root": str(tmp_path), "phase": PROCESS_1},
                tmp_path / "config.json", broker, time.monotonic() - 1)
            assert facts["timed_out"] and facts["os_exit_confirmed"]
            assert events[0] == "sealed" and events[-1] == "close"
            assert gate.handoffs == 0 and gate.budget.sealed_reason
        finally:
            gate.budget.release()
    asyncio.run(run())


def test_close_records_production_runtime_failure_even_when_aclose_returns(tmp_path, monkeypatch):
    """真实 ApplicationClosed 诊断不能被正常 aclose 返回/OS exit 掩盖。"""
    from cicada.prompting import compose_system_prompt

    async def run():
        spec = lifecycle.LifecycleSpec(workspace=tmp_path, instruction="fixture",
            model_doc={"normal": {"model": "deepseek-flash", "thinking": "enabled",
                "reasoning_effort": "low", "num_ctx": 262144, "max_tokens": 8192}})
        prompt, info = compose_system_prompt(tmp_path, None)
        session, log = await lifecycle.open_process(spec, httpx.MockTransport(lambda r: None),
            session_id=None, system_prompt=prompt, prompt_info=info)
        stop = session._app.runtime.stop
        async def failed_stop():
            await stop()
            raise RuntimeError("injected stop failure")
        monkeypatch.setattr(session._app.runtime, "stop", failed_stop)
        facts = {}
        await lifecycle.close_process(session, log, facts)
        assert facts["closed_ok"] is False
        assert "runtime stop failed" in facts["close_problem"]
    asyncio.run(run())


def test_broker_close_waits_for_evaluator_thread_to_finish(tmp_path):
    async def run():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def check():
            loop.call_soon_threadsafe(entered.set)
            release.wait(5)
            return 1, {"result": "failed"}
        gate = _gate(tmp_path, httpx.MockTransport(lambda req: None))
        broker = Broker(gate=gate, check=check, deadline=time.monotonic() + 10)
        checking = asyncio.create_task(broker.dispatch("/check", b"{}"))
        await entered.wait()
        closing = asyncio.create_task(broker.close())
        await asyncio.sleep(0)
        try:
            assert not closing.done()
        finally:
            release.set()
            await asyncio.gather(checking, closing)
            gate.budget.release()
        assert not broker.checking and len(broker.checks) == 1
    asyncio.run(run())


async def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


def test_watch_tolerates_transient_state_failures_and_obeys_stop(tmp_path):
    """子进程轮询失败是存活抖动, 不是父进程失联。

    首次 live 就是因为**单次**轮询失败即 task.cancel() 而整轮中止, 生产账本留下未决
    预留, 父进程只看到 "child-sealed"。这里锁定: 连续失败到阈值才取消, 明确 stopped
    立刻取消。
    """
    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(lambda request: None))
        broker = Broker(gate=gate, check=lambda: None, deadline=time.monotonic() + 30)
        endpoint = broker.start()
        original = broker.dispatch
        state = {"fail": 0}

        async def dispatch(route, body):
            if route == "/state" and state["fail"]:
                state["fail"] -= 1
                raise AttemptBudgetError("injected transient state failure")
            return await original(route, body)

        broker.dispatch = dispatch  # type: ignore[method-assign]
        try:
            task = asyncio.create_task(asyncio.sleep(30))
            diagnostics: dict = {}
            state["fail"] = WATCH_MAX_FAILURES - 1
            watcher = asyncio.create_task(
                watch_parent(endpoint, broker.token, task, diagnostics))
            assert await _until(
                lambda: diagnostics.get("watch", {}).get("failures") == WATCH_MAX_FAILURES - 1)
            assert not task.cancelled() and not task.done()
            state["fail"] = 1
            assert await _until(lambda: task.cancelled() or task.done(), timeout=5)
            await asyncio.gather(watcher, task, return_exceptions=True)
            assert task.cancelled()
            assert diagnostics["watch"]["failures"] == WATCH_MAX_FAILURES
            assert diagnostics["watch"]["stop_seen"] is False
            # 失败形态必须可读 (首题"不知道是超时还是别的 httpx 错误"的盲点)。
            assert diagnostics["watch"]["last_failure"].startswith("KeyError")

            stopped_task = asyncio.create_task(asyncio.sleep(30))
            stopped_diagnostics: dict = {}
            broker.stopped = True
            await asyncio.gather(watch_parent(endpoint, broker.token, stopped_task,
                                              stopped_diagnostics), return_exceptions=True)
            await asyncio.gather(stopped_task, return_exceptions=True)
            assert stopped_task.cancelled()
            # 新的 watcher 实例从零起算: total_failures 不跨实例累计。
            assert stopped_diagnostics["watch"] == {"stop_seen": True, "failures": 0,
                                                    "total_failures": 0}
        finally:
            await broker.close()
            gate.budget.release()
    asyncio.run(run())


def test_check_gate_separates_offline_ordering_from_live_outcome():
    """离线要求首次检查是空补丁失败(我们的契约); live 只要求存在通过检查(模型行为自由)。"""
    edited_first = {"code": 0, "result": "passed", "patch_empty": False,
                    "required_passed": 11, "required_total": 11}
    empty_failure = {"code": 1, "result": "failed", "patch_empty": True,
                     "required_passed": 0, "required_total": 11}
    assert check_gate([empty_failure], offline=True) is None
    assert check_gate([empty_failure], offline=False) is not None      # 还没有通过检查
    assert check_gate([], offline=True) is not None
    # live 首题实测形态: 模型先改后检, 唯一一次检查是 11/11 通过 -> 必须放行
    assert check_gate([edited_first], offline=False) is None
    # 生命周期题实测形态: 首次是空补丁失败(p1), 通过检查出现在 p2 的 R3 -> 必须放行
    # (这一条正是审查抓到的 P1: live 判若放在 p1 之后, 这个序列会被误拒)
    assert check_gate([empty_failure, edited_first], offline=False) is None
    assert check_gate([empty_failure, edited_first], offline=True) is None
    # 离线同样的"先改后检"序列必须拒绝(证明检查落在未修改的工作区上)
    assert "empty failing patch" in check_gate([edited_first], offline=True)
    # 部分通过(未达全覆盖)不算达标
    partial = {**edited_first, "required_passed": 9}
    assert check_gate([partial], offline=False) is not None


def test_broker_audit_trail_records_route_decisions(tmp_path):
    """每个路由的决策与原因必须落盘: 首题 live 没有这份轨迹, 根因无法判定。"""
    async def run():
        gate = _gate(tmp_path, httpx.MockTransport(
            lambda request: lifecycle._sse([lifecycle._text_frame(0, "fixture")])))
        audit = tmp_path / "broker-audit.jsonl"
        broker = Broker(gate=gate, check=lambda: (1, {"result": "failed"}),
                        deadline=time.monotonic() + 10, credential="parent-only-secret",
                        audit=audit)
        try:
            code, headers, _body_bytes = await broker.dispatch("/model", _body())
            assert code == 200 and headers["X-Call-Id"]
            await broker.dispatch("/check", b"{}")
            await broker.dispatch("/seal", b"{}")
            with pytest.raises(AttemptBudgetError, match="deadline"):
                await broker.dispatch("/model", _body())
            # 只经真实 HTTP 才会走到 handler 两级拒绝: 这一行是"为什么没成"的唯一落痕。
            endpoint = broker.start()
            # 必须用 async client: 同步 post 会阻塞 loop, 而 Handler 线程正等 loop 跑 dispatch。
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                rejected = await client.post(endpoint + "/model",
                                             headers={"X-PV-Token": "wrong"})
                assert rejected.status_code == 403
                refused = await client.post(endpoint + "/model",
                                            headers={"X-PV-Token": broker.token})
                assert refused.status_code == 503
        finally:
            gate.budget.release()
        raw = audit.read_text(encoding="utf-8")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        events = [record["event"] for record in records]
        assert "model-ok" in events and "check-ok" in events
        assert "seal" in events and "stop" in events
        assert "handler-rejected" in events and "handler-refused" in events
        # 首个封口原因只记一次, 后续 stop 不再重复写行 (首因优先)。
        assert events.count("stop") == 1
        refused_record = next(r for r in records if r["event"] == "handler-refused")
        assert refused_record["path"] == "/model" and "detail" in refused_record
        model_ok = next(record for record in records if record["event"] == "model-ok")
        assert model_ok["status"] == 200 and model_ok["call_id"] == headers["X-Call-Id"]
        stop = next(record for record in records if record["event"] == "stop")
        assert stop["reason"] == "child-sealed"
        # 审计只记事实, 不把父进程凭据写进去。
        assert "parent-only-secret" not in raw
    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "confirm", "seal"])
def test_ambiguous_correlation_seal_is_recorded_in_diagnostics(failure):
    """封口语义不变, 但"为什么封口"必须留痕 (新预订数/终局/错误分类)。"""
    class FakeBudget:
        sealed_reason = None

        def seal(self, reason):
            self.sealed_reason = self.sealed_reason or reason
            if failure == "seal":
                raise ConnectionError("parent unavailable")

    class FakeGate:
        def __init__(self):
            self.observations = {}
            self.network_sealed_reason = None
            self.budget = FakeBudget()

        def seal_network(self, reason):
            self.network_sealed_reason = self.network_sealed_reason or reason

        def confirm(self, call_id, terminal):
            assert failure == "confirm"
            raise ConnectionError("parent unavailable")

    from cicada.core.ports import StreamDone

    gate = FakeGate()
    diagnostics: dict = {}

    async def original(model, request, cancel):
        if failure == "confirm":
            gate.observations["call-1"] = True
        yield StreamDone("stop")

    async def run():
        observed = lifecycle.observe_model_stream(original, gate, diagnostics)
        if failure is None:
            events = [event async for event in observed(None, None, None)]
            assert [event.stop_reason for event in events] == ["stop"]
        else:
            with pytest.raises(ConnectionError, match="parent unavailable"):
                _ = [event async for event in observed(None, None, None)]

    asyncio.run(run())
    network_reason = None if failure == "confirm" else "ambiguous-call-correlation"
    attempt_reason = None if failure == "confirm" else "ambiguous call correlation"
    assert gate.network_sealed_reason == network_reason
    assert gate.budget.sealed_reason == attempt_reason
    expected = {
        "new_correlations": int(failure == "confirm"), "terminal": "stop", "terminal_error": None,
        "completed": True, "network_sealed": network_reason,
        "attempt_sealed": attempt_reason}
    if failure is not None:
        expected["decision_error"] = "ConnectionError"
    assert diagnostics["streams"] == [expected]
