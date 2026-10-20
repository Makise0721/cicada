"""私有基准 harness 的补丁/评测安全边界；不调用 Docker 或模型。"""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.support.swebench_bridge import (
    GitCLI, PatchScopeError, export_patch, grade_from_artifacts,
)
from tests.support.swebench_owner import EvaluationOwner


def repository(tmp_path: Path):
    root = tmp_path / "repository"
    root.mkdir()
    git = GitCLI(root)
    git.run("init", check=True)
    git.run("config", "user.email", "test@example.invalid", check=True)
    git.run("config", "user.name", "test", check=True)
    git.run("config", "core.autocrlf", "false", check=True)
    (root / "source.txt").write_text("base\n", encoding="utf-8")
    git.run("add", ".", check=True)
    git.run("commit", "-m", "base", check=True)
    return root, git, git.run("rev-parse", "HEAD", check=True).stdout.strip()


def test_large_patch_new_file_and_deletion_round_trip(tmp_path):
    root, git, base = repository(tmp_path)
    index = (root / ".git/index").read_bytes()
    (root / "source.txt").unlink()
    content = "".join(f"line {i}: payload\n" for i in range(8000))
    (root / "new.txt").write_text(content, encoding="utf-8", newline="\n")
    outputs = root / ".cicada/outputs"
    outputs.mkdir(parents=True)
    (outputs / "check.txt").write_text("runtime evidence", encoding="utf-8")
    exported = export_patch(root, base, ("source.txt", "new.txt"), tmp_path, git)
    assert len(exported.patch_bytes) > 20000
    assert b".cicada" not in exported.patch_bytes
    assert (root / ".git/index").read_bytes() == index
    patch = tmp_path / "export.patch"
    patch.write_bytes(exported.patch_bytes)
    # 只回退本测试拥有的源码，验证完整补丁能重建最终内容。
    (root / "new.txt").unlink()
    git.run("checkout", "--", "source.txt", check=True)
    git.run("apply", str(patch), check=True)
    assert not (root / "source.txt").exists()
    assert (root / "new.txt").read_text(encoding="utf-8") == content


def test_outside_scope_including_ignored_file_is_rejected(tmp_path):
    root, git, base = repository(tmp_path)
    (root / ".git/info/exclude").write_text("secret.txt\n", encoding="utf-8")
    (root / "secret.txt").write_text("unexplained", encoding="utf-8")
    with pytest.raises(PatchScopeError):
        export_patch(root, base, ("source.txt",), tmp_path, git)


def test_python_bytecode_caches_are_exempted_and_recorded(tmp_path):
    root, git, base = repository(tmp_path)
    cache = root / "pkg/__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (root / "source.txt").write_text("changed\n", encoding="utf-8")
    exported = export_patch(root, base, ("source.txt",), tmp_path, git)
    assert exported.exempted_paths == ["pkg/__pycache__/mod.cpython-312.pyc"]
    assert exported.to_json()["exempted_paths"] == exported.exempted_paths
    assert b"__pycache__" not in exported.patch_bytes


def test_official_reward_cannot_replace_required_coverage():
    output = json.dumps({"tests": [{"name": "regression", "status": "PASSED"}]}).encode()
    assert not grade_from_artifacts(0, b"1", output, ["required"], ["regression"])["grader_pass"]
    assert not grade_from_artifacts(0, b"1", output, [], ["regression"])["grader_pass"]
    assert not grade_from_artifacts(0, b"1", b"not-json", ["required"], [])["grader_pass"]
    assert grade_from_artifacts(1, b"0", None, ["required"], [])["parse_failed"]
    assert grade_from_artifacts(0, b"1", output, ["regression"], [])["grader_pass"]


def test_owner_seals_and_cleans_after_checker_is_killed(tmp_path):
    started = threading.Event()
    stopped = threading.Event()
    resource = tmp_path / "owned-resource"

    def evaluate(cancel):
        resource.write_text("active", encoding="utf-8")
        started.set()
        if cancel.wait(10):
            stopped.set()
        return 2, {"result": "anomaly"}

    def cleanup():
        resource.unlink(missing_ok=True)
        return [{"outcome": "confirmed"}]

    with EvaluationOwner(tmp_path, evaluate, cleanup) as owner:
        code = "from pathlib import Path; from tests.support.swebench_owner import request_check; import sys; request_check(Path(sys.argv[1]))"
        checker = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert started.wait(10)
            checker.kill()
            checker.wait(timeout=5)
            assert stopped.wait(5)
            assert owner.stopping.is_set()
        finally:
            if checker.poll() is None:
                checker.kill()
            checker.communicate(timeout=5)
    assert not resource.exists()
    assert not owner.thread.is_alive() and not owner.watch.is_alive()


def test_cleanup_accepts_racing_removal_only_after_successful_absence_query(tmp_path):
    from tests.support.swebench_bridge import ContainerSupervisor, CommandRecord, BridgeError

    class Docker:
        def run(self, *args, **kwargs):
            code = 1 if args[0] in ("stop", "rm") else 0
            return CommandRecord(list(args), code, "", "", 0)

    supervisor = ContainerSupervisor(tmp_path, Docker())
    assert supervisor.stop_and_remove("owned-label", "owned-id") == "confirmed"

    class BrokenQuery(Docker):
        def run(self, *args, **kwargs):
            return CommandRecord(list(args), 1, "", "", 0)

    with pytest.raises(BridgeError):
        ContainerSupervisor(tmp_path, BrokenQuery()).stop_and_remove("owned-label", "owned-id")


# 费用/流终结没有既有覆盖：使用实际 DeepSeekModel + MockTransport，验证先预留、
# 最终帧与生产 finally 联合结算，异常不能靠旧 usage 释放额度。
def completion_bytes(*, usage=None, finish="stop", done=True, suffix=b""):
    usage = {"prompt_tokens": 100, "completion_tokens": 10} if usage is None else usage
    payload = {"choices": [{"index": 0, "delta": {"content": "offline"},
                             "finish_reason": finish}], "usage": usage}
    return ("data: " + json.dumps(payload) + "\n\n" +
            ("data: [DONE]\n\n" if done else "")).encode() + suffix


async def model_call(transport, *, summary=False):
    import httpx
    from cicada.core.cancel import CancelToken
    from cicada.core.messages import UserMessage
    from cicada.core.ports import ModelRequest
    from cicada.plugins.deepseek.model import DeepSeekModel
    from cicada.plugins.deepseek.protocol import DeepSeekConfig
    from tests.support.swebench_cost import observe_model_stream

    async with httpx.AsyncClient(transport=transport) as client:
        model = DeepSeekModel(client, DeepSeekConfig(model="deepseek-flash", think=not summary,
            reasoning_effort="low", num_ctx=262144, max_tokens=2048 if summary else 8192), "offline-key")
        wrapped = observe_model_stream(DeepSeekModel.stream, transport)
        return [event async for event in wrapped(model, ModelRequest((UserMessage("offline"),), ()), CancelToken())]


def ledger_at(root):
    from tests.support.swebench_cost import CostLedger
    ledger = CostLedger(root)
    ledger.acquire({"mode": "offline-test"})
    return ledger


@pytest.mark.parametrize("summary", [False, True])
async def test_cost_transport_persists_reserve_and_settles_once_after_production_close(tmp_path, summary):
    import httpx
    from tests.support.swebench_cost import CostTransport, LedgerError
    ledger = ledger_at(tmp_path)
    seen = []
    def response(request):
        disk = json.loads(ledger.budget_path.read_text(encoding="utf-8"))
        assert disk["calls"] == 1 and disk["unresolved_reserve"] == ("2.22" if summary else "2.27")
        seen.append(request.content)
        return httpx.Response(200, stream=httpx.ByteStream(completion_bytes()))
    transport = CostTransport(httpx.MockTransport(response), ledger)
    try:
        events = await model_call(transport, summary=summary)
        assert events[-1].stop_reason == "stop" and len(seen) == 1
        assert ledger.budget["settled_upper"] == "0.01" and ledger.budget["unresolved_reserve"] == "0.00"
        assert ledger.calls[0]["kind"] == ("summary" if summary else "normal")
        assert ledger.calls[0]["transport_events"]["closed"]
        with pytest.raises(LedgerError):
            ledger.settle(ledger.calls[0]["call_id"], None)
    finally:
        ledger.release_lock()


@pytest.mark.parametrize("mode", ["eof", "late-invalid", "missing-usage", "http", "length", "close", "cache"])
async def test_cost_unresolved_keeps_reserve_and_seals(tmp_path, mode):
    import httpx
    from tests.support.swebench_cost import CostTransport, LedgerError
    ledger = ledger_at(tmp_path)
    data = completion_bytes()
    if mode == "eof":
        data = completion_bytes(done=False)
    elif mode == "late-invalid":
        invalid = json.dumps({"choices": [], "usage": {"prompt_tokens": -1, "completion_tokens": 0}})
        data = completion_bytes(done=False) + ("data: " + invalid + "\n\ndata: [DONE]\n\n").encode()
    elif mode == "missing-usage":
        data = completion_bytes(usage={})
    elif mode == "length":
        data = completion_bytes(finish="length")
    elif mode == "cache":
        data = completion_bytes(usage={"prompt_tokens": 100, "completion_tokens": 10,
            "prompt_cache_hit_tokens": 99, "prompt_cache_miss_tokens": 2})
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield data
        async def aclose(self):
            if mode == "close":
                raise RuntimeError("offline close failure")
    transport = CostTransport(httpx.MockTransport(lambda request: httpx.Response(
        500 if mode == "http" else 200, stream=Stream())), ledger)
    try:
        with pytest.raises(LedgerError):
            await model_call(transport)
        assert ledger.sealed and transport.network_sealed
        assert ledger.budget["unresolved_reserve"] == "2.27" and ledger.budget["settled_upper"] == "0.00"
    finally:
        ledger.release_lock()


@pytest.mark.parametrize("kind", ["equal", "over", "limit", "persist"])
async def test_cost_send_gates_before_handoff(tmp_path, monkeypatch, kind):
    import httpx
    from tests.support.swebench_cost import CostTransport, LedgerError, MAX_HTTP_CALLS
    ledger = ledger_at(tmp_path)
    if kind in ("equal", "over"):
        ledger.budget["settled_upper"] = "2.23" if kind == "equal" else "2.24"
        ledger._persist()
    elif kind == "limit":
        ledger.budget["calls"] = MAX_HTTP_CALLS
    else:
        def fail():
            raise OSError("offline disk failure")
        monkeypatch.setattr(ledger, "_persist", fail)
    sends = []
    transport = CostTransport(httpx.MockTransport(lambda request: (
        sends.append(True) or httpx.Response(200, stream=httpx.ByteStream(completion_bytes())))), ledger)
    try:
        if kind == "equal":
            await model_call(transport)
            assert sends == [True]
        else:
            events = await model_call(transport)
            assert events[-1].stop_reason == "error"
            assert sends == [] and transport.network_sealed
    finally:
        ledger.release_lock()


def test_cost_attempt_cannot_resume_in_another_ledger(tmp_path):
    from tests.support.swebench_cost import CostLedger, LedgerError
    first = ledger_at(tmp_path)
    first.release_lock()
    second = CostLedger(tmp_path)
    try:
        with pytest.raises(LedgerError):
            second.acquire({"mode": "resume"})
    finally:
        second.release_lock()



def tool_completion_bytes(calls):
    fragments = [{"index": i, "id": f"offline-{i}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(arguments)}}
                 for i, (name, arguments) in enumerate(calls)]
    payload = {"choices": [{"index": 0, "delta": {"reasoning_content": "offline fixture",
               "tool_calls": fragments}, "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    return ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode()


def cli_fixture(tmp_path, monkeypatch):
    import tests.support.swebench_run as runner
    root = tmp_path / "attempt"
    root.mkdir()
    workspace, git, _ = repository(root)
    workspace.rename(root / "workspace")
    # 用户级运行状态跟着用例走且必须在被检查工作区之外。
    home = root.parent / "test-user-home"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    original = runner.cli_arguments
    def arguments(run_root, prompt):
        args = original(run_root, prompt)
        args[args.index("--check-command") + 1] = "exit 0"
        return args
    monkeypatch.setattr(runner, "cli_arguments", arguments)
    return root


def test_real_main_uses_cost_transport_and_restores_bindings(tmp_path, monkeypatch):
    import httpx
    import cicada.__main__ as cli
    from cicada.plugins.deepseek.model import DeepSeekModel
    from tests.support.swebench_cost import CostTransport
    from tests.support.swebench_run import run_cli
    root = cli_fixture(tmp_path, monkeypatch)
    ledger = ledger_at(root / "evidence/cost")
    bindings = (cli.deepseek_plugin, cli._run, cli.open_application, DeepSeekModel.stream, sys.argv)
    sends = []
    def response(request):
        sends.append(request.content)
        data = tool_completion_bytes([("check", {"action": "run", "check_id": "check-1"})]) if len(sends) == 1 else completion_bytes()
        return httpx.Response(200, stream=httpx.ByteStream(data))
    transport = CostTransport(httpx.MockTransport(response), ledger)
    try:
        facts = run_cli(root, transport, "offline-key", "offline fixture")
        assert facts["main_exit_code"] == 0 and facts["application_closed"]
        assert facts["delivery"]["decision"]["can_deliver"]
        assert len(sends) == 2 and ledger.budget["unresolved_reserve"] == "0.00"
        assert bindings == (cli.deepseek_plugin, cli._run, cli.open_application, DeepSeekModel.stream, sys.argv)
        assert "offline-key" not in "".join(p.read_text(encoding="utf-8") for p in (root / "evidence").rglob("*.json"))
    finally:
        ledger.release_lock()


@pytest.mark.parametrize("trigger", ["deadline", "external"])
def test_actual_production_task_timeout_seals_before_cancellation(tmp_path, monkeypatch, trigger):
    import asyncio
    import httpx
    from tests.support.swebench_cost import CostTransport
    from tests.support.swebench_run import run_cli
    root = cli_fixture(tmp_path, monkeypatch)
    ledger = ledger_at(root / "evidence/cost")
    cancelled_after_seal = []
    async def response(request):
        if trigger == "external":
            parent = next(t for t in asyncio.all_tasks() if t.get_coro().__qualname__.endswith("supervised"))
            asyncio.get_running_loop().call_soon(parent.cancel)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_after_seal.append(transport.network_sealed and ledger.sealed)
            raise
    transport = CostTransport(httpx.MockTransport(response), ledger)
    try:
        facts = run_cli(root, transport, "offline-key", "offline timeout", timeout_s=2)
        assert facts["timed_out"] == (trigger == "deadline")
        assert facts["application_closed"]
        assert cancelled_after_seal == [True] and transport.handoffs == 1
        assert ledger.sealed and ledger.budget["unresolved_reserve"] == "2.27"
    finally:
        ledger.release_lock()


async def test_async_body_cannot_race_single_flight_or_send_after_seal(tmp_path):
    import asyncio
    import httpx
    from tests.support.swebench_cost import CostTransport, LedgerSealed
    ledger = ledger_at(tmp_path)
    started, release = asyncio.Event(), asyncio.Event()
    body = json.dumps({"model": "deepseek-flash", "stream": True, "max_tokens": 8192,
                       "thinking": {"type": "enabled"}, "reasoning_effort": "low"}).encode()
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await release.wait()
            yield body
    transport = CostTransport(httpx.MockTransport(lambda request: pytest.fail("must not handoff")), ledger)
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions", stream=Body())
    task = asyncio.create_task(transport.handle_async_request(request))
    try:
        await started.wait()
        with pytest.raises(LedgerSealed):
            await transport.handle_async_request(httpx.Request("POST", str(request.url), content=body))
        release.set()
        with pytest.raises(LedgerSealed):
            await task
        assert transport.handoffs == 0 and ledger.budget["calls"] == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        ledger.release_lock()


async def test_settlement_persist_failure_never_releases_reservation(tmp_path, monkeypatch):
    import httpx
    from tests.support.swebench_cost import CostTransport, LedgerError
    ledger = ledger_at(tmp_path)
    def response(request):
        def fail():
            raise OSError("offline settlement disk failure")
        monkeypatch.setattr(ledger, "_persist", fail)
        return httpx.Response(200, stream=httpx.ByteStream(completion_bytes()))
    transport = CostTransport(httpx.MockTransport(response), ledger)
    try:
        with pytest.raises(LedgerError):
            await model_call(transport)
        assert ledger.sealed and transport.network_sealed
        assert ledger.budget["unresolved_reserve"] == "2.27" and ledger.budget["settled_upper"] == "0.00"
        disk = json.loads(ledger.budget_path.read_text(encoding="utf-8"))
        assert disk["unresolved_reserve"] == "2.27"
    finally:
        ledger.release_lock()


def test_preflight_failure_receipt_does_not_overwrite_previous_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests.e2e import swebench_runner as h
    from tests.support import swebench_attempt as attempt
    root = tmp_path / "old-attempt"
    root.mkdir()
    (root / "result.json").write_text("old evidence", encoding="utf-8")
    def rejected(*args, **kwargs):
        raise attempt.PreflightRejected("offline admission is stale")
    monkeypatch.setattr(attempt, "execute_attempt", rejected)
    monkeypatch.setattr(h, "PROJECT", tmp_path)
    args = SimpleNamespace(live=True, prepared_root=str(tmp_path / "prepared"), run_root=str(root))
    assert h.cmd_run(args) == 2
    receipts = list((tmp_path / ".scratch/live-coding-under-5-rmb/runs/preflight-rejections").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["http_handoffs"] == 0
    assert (root / "result.json").read_text(encoding="utf-8") == "old evidence"
