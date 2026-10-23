"""真实应用/Coding/SQLite，仅 HTTP 响应受控；逐请求体核对思考回传。"""

import json
from dataclasses import replace

import httpx
import pytest

from cicada.application import ApplicationConfig, PersistenceConfig, open_application
from cicada.core.messages import AssistantMessage, ReasoningContent
from cicada.persistence.contracts import LaunchIdentity
from cicada.plugins.deepseek import DeepSeekConfig, deepseek_plugin
from test_application import _repo, _prompt_info


def response(reasoning, text="answer", calls=()):
    delta = {"content": text}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if calls:
        delta["tool_calls"] = [{"index": i, "id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
            for i, (id, name, args) in enumerate(calls)]
    payload = {"choices": [{"index": 0, "delta": delta,
                           "finish_reason": "tool_calls" if calls else "stop"}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode()


def config(root, transport, session_id=None, model_config=None):
    model_config = model_config or DeepSeekConfig()
    safe = model_config.identity
    info = _prompt_info(root)
    identity = LaunchIdentity(provider=safe.provider, model=safe.model, base_url=safe.base_url,
        think=safe.think, options=dict(safe.options), max_request_bytes=safe.max_request_bytes,
        max_turns=8, prompt_version=info.version, prompt_sha256=info.prompt_sha256,
        instructions_sha256=None, checks_signature=None, sources={})
    return ApplicationConfig(workspace=root, system_prompt="test", prompt_info=info,
        model_plugin=deepseek_plugin(model_config, api_key="offline-test-key", transport=transport),
        persistence=PersistenceConfig(identity=identity, session_id=session_id), max_turns=8)


async def test_process_cost_control_is_shared_across_compaction_clear_and_session_switch(tmp_path):
    from cicada.costs import CostLedger

    root = _repo(tmp_path)
    bodies = []
    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, content=response('thought' if body['thinking']['type'] == 'enabled' else None))
    ledger = CostLedger(root / '.cicada' / 'cost', max_calls=5)
    session = await open_application(replace(config(root, httpx.MockTransport(handle)), cost_control=ledger))
    try:
        original = session.persistence.session_id
        await session.execute('first')
        await session.execute('second')
        await session.compact_context()
        assert bodies[-1]['thinking']['type'] == 'disabled' and session.cost.calls == 3
        await session.clear_context()
        await session.execute('after clear')
        await session.select_model('deepseek-v4-pro')
        await session.set_reasoning_effort('low')
        await session.submit_new_session().wait()
        assert session.cost.calls == 4
        await session.execute('new session')
        await session.submit_resume_session(original, adopt_current_config=True).wait()
        outcome = await session.execute('limit must survive')
        assert outcome.exit_code == 1 and 'call limit' in outcome.result.error
        assert len(bodies) == 5 and session.cost.calls == 5
    finally:
        await session.aclose()
    assert 'offline-test-key' not in ledger.path.read_text(encoding='utf-8')


async def test_tools_new_user_restart_and_nonthinking_summary_use_real_bindings(tmp_path):
    root = _repo(tmp_path)
    bodies = []
    answers = iter([
        response(" 原始\n思考 ", calls=(("server-write-1", "write", {"path": "one.txt", "content": "甲"}),
            ("server-write-2", "write", {"path": "two.txt", "content": "乙"}))),
        response("复查", calls=(("server-read", "read", {"path": "one.txt"}),)),
        response("", "第一答"), response("第二轮", "第二答"),
        response(None, "旧记录摘要"), response("恢复后", "第三答"),
    ])
    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=next(answers))
    session = await open_application(config(root, httpx.MockTransport(handle)))
    try:
        first = await session.execute("写两个文件并核对")
        assert first.exit_code == 0 and first.persistence.saved
        assert (root / "one.txt").read_text(encoding="utf-8") == "甲"
        assert (root / "two.txt").read_text(encoding="utf-8") == "乙"
        assert bodies[1]["messages"][2]["reasoning_content"] == " 原始\n思考 "
        assert [m["tool_call_id"] for m in bodies[1]["messages"] if m["role"] == "tool"] == ["server-write-1", "server-write-2"]
        assert bodies[2]["messages"][5]["reasoning_content"] == "复查"
        await session.execute("下一用户轮")
        assistants = [m for m in bodies[3]["messages"] if m["role"] == "assistant"]
        assert [m["reasoning_content"] for m in assistants] == [" 原始\n思考 ", "复查", ""]
        session_id = session.persistence.session_id
        expected = session.history
    finally:
        await session.aclose()
    # 恢复先续写，完整规范历史逐字段进入下一实际 HTTP 请求。
    answers = iter([response("恢复后", "第三答"), response(None, "旧记录摘要")])
    restored = await open_application(config(root, httpx.MockTransport(handle), session_id))
    try:
        assert restored.history == expected
        await restored.execute("重启续写")
        retained = [m for m in bodies[4]["messages"] if m["role"] == "assistant"]
        assert [m["reasoning_content"] for m in retained] == [" 原始\n思考 ", "复查", "", "第二轮"]
        await restored.compact_context()
        summary_body = bodies[-1]
        assert summary_body["thinking"] == {"type": "disabled"}
        assert summary_body["max_tokens"] == 4096 and "tools" not in summary_body
        assert "reasoning_effort" not in summary_body
        record = restored.summary_records[-1]
        assert record.call_identity.think is False
        assert dict(record.call_identity.options)["max_tokens"] == 4096
        assert record.provider == "deepseek" and record.model == "deepseek-flash"
        assert restored.config.budget.effective_output_reserve == 32768
    finally:
        await restored.aclose()
    again = await open_application(config(root, httpx.MockTransport(handle), session_id))
    try:
        assert again.summary_records[-1].call_identity == record.call_identity
        assert b"offline-test-key" not in again._store.capture().db_path.read_bytes()
    finally:
        await again.aclose()


async def test_cache_observation_tracks_normal_route_and_resets_on_restore(tmp_path):
    """缓存 03 (契约 C7): 应用只读能力接普通路由; 摘要通道不覆盖; 恢复进程从空开始."""
    root = _repo(tmp_path)
    answers = iter([response("", "第一答"), response("", "第二答"), response(None, "旧记录摘要"),
                    response("", "压缩后一答"), response("", "恢复后一答")])
    def handle(req):
        return httpx.Response(200, content=next(answers))
    session = await open_application(config(root, httpx.MockTransport(handle)))
    try:
        assert session.cache_observation is None, "未发送前无观察"
        await session.execute("第一问")
        first = session.cache_observation
        assert (first.sequence, first.changes, first.preserved_messages) == (
            1, ("first_request",), None)
        await session.execute("第二问")
        second = session.cache_observation
        assert (second.sequence, second.changes, second.preserved_messages) == (
            2, ("messages_extended",), 1), "稳定历史保留, 延长如实表达"
        await session.compact_context()
        assert session.cache_observation is second, "摘要调用不覆盖普通路由观察"
        await session.execute("压缩后一问")
        replaced = session.cache_observation
        assert replaced.sequence == 3 and "messages_replaced" in replaced.changes, (
            "压缩后的连续性由下一次实际比较表达替换, 不保存虚构连续")
        session_id = session.persistence.session_id
    finally:
        await session.aclose()
    restored = await open_application(config(root, httpx.MockTransport(handle), session_id))
    try:
        assert restored.cache_observation is None, "恢复进程不持久观察"
        await restored.execute("恢复后一问")
        after = restored.cache_observation
        assert (after.sequence, after.changes) == (1, ("first_request",)), "新普通路由从首调用起"
    finally:
        await restored.aclose()


@pytest.mark.parametrize("failure", ["error", "length", "missing-tool"])
async def test_incompatible_archive_refused_before_reliable_begin(tmp_path, failure):
    root = _repo(tmp_path)
    from cicada.plugins.fake_model import FakeModel, fake_model_plugin
    from cicada.core.ports import ReasoningDelta, StreamDone, ToolCallEvent
    events = [StreamDone("error", "failed")] if failure == "error" else [ReasoningDelta("partial"), StreamDone("length")]
    if failure == "missing-tool":
        events = [ToolCallEvent("old", "read", '{"path":"missing"}'), StreamDone("tool_use")]
    initial = config(root, httpx.MockTransport(lambda _: None))
    old = await open_application(replace(initial, model_plugin=fake_model_plugin(FakeModel([events])), budget=None))
    try:
        await old.execute("old")
        session_id = old.persistence.session_id
    finally:
        await old.aclose()
    sent = []
    def handle(req):
        sent.append(req)
        return httpx.Response(200, content=response("", "done"))
    restored = await open_application(config(root, httpx.MockTransport(handle), session_id))
    try:
        before = restored._store.recovery_snapshot()
        with pytest.raises(ValueError, match="thinking"):
            await restored.execute("refuse")
        after = restored._store.recovery_snapshot()
        assert after.requests == before.requests and after.interrupted == before.interrupted
        assert sent == [] and restored.state == "idle"
    finally:
        await restored.aclose()


async def test_check_mode_keeps_same_deepseek_output_cap_and_verification_authority(tmp_path):
    from cicada.application import verification_checks_signature
    from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
    root = _repo(tmp_path)
    bodies = []
    answers = iter([
        response("写文件", calls=(("svc-write", "write", {"path": "result.txt", "content": "done"}),)),
        response("执行声明检查", calls=(("svc-check", "check", {"action": "run", "check_id": "check-1"}),)),
        response("完成", "done"),
    ])
    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=next(answers))
    initial = config(root, httpx.MockTransport(handle))
    plan = VerificationPlan(verification_run_id="deepseek-offline-check", root=root,
        checks=(CheckDefinition(check_id="check-1", command="exit 0", timeout_s=10),))
    launch = replace(initial.persistence.identity, checks_signature=verification_checks_signature(plan.checks))
    session = await open_application(replace(initial, plan=plan,
        persistence=replace(initial.persistence, identity=launch)))
    try:
        outcome = await session.execute("写文件并通过检查")
        assert outcome.exit_code == 0 and outcome.persistence.saved
        assert all(body["max_tokens"] == 32768 for body in bodies)
        assert all(body["thinking"] == {"type": "enabled"} for body in bodies)
        assert session.budget.report.output_reserve == 32768
        assert (root / "result.txt").read_text(encoding="utf-8") == "done"
    finally:
        await session.aclose()
