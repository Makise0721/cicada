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
from tests.support.swebench_live import Broker, RemoteGate, _supervise_child


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
