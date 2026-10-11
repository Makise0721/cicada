"""票据01: 思考的独立收集、失败事实、预算与真实应用/SQLite闭环."""

from dataclasses import FrozenInstanceError, replace
import asyncio
import json
import hashlib
import sqlite3
import time

import pytest

from cicada.application import ApplicationConfig, ApplicationOpenError, PersistenceConfig, open_application
from cicada.budget import BudgetConfig, BUDGET_ESTIMATOR_VERSION, estimate_message_tokens
from cicada.context_policy import BudgetPolicy, canonical_prefix_sha256
from cicada.core.agent import Agent
from cicada.core.cancel import CancelToken
from cicada.core.events import AssistantTextDelta
from cicada.core.messages import AssistantMessage, ReasoningContent, ToolResultMessage, UserMessage
from cicada.core.ports import ModelRequest, ReasoningDelta, StreamDone, TextDelta, ToolCallEvent
from cicada.persistence.codec import CodecError, decode_messages, encode_messages
from cicada.persistence.contracts import SummaryRecord
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from test_application import _repo, _prompt_info
from test_persistence_application import _identity, _check_config


def _config(root, model, session_id=None):
    return ApplicationConfig(
        workspace=root, system_prompt="test", prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(model), max_turns=8, model_label="fake",
        persistence=PersistenceConfig(session_id=session_id, identity=_identity()),
        budget=BudgetConfig(num_ctx=100_000, num_predict=1),
    )


async def test_thinking_tools_new_user_sqlite_restore_next_request(tmp_path):
    root = _repo(tmp_path)
    first = FakeModel([
        [ReasoningDelta(" 原始\n"), ReasoningDelta("思考 "), TextDelta("写文件"),
         ToolCallEvent("service-id", "write", json.dumps({"path": "result.txt", "content": "完成"})),
         StreamDone("tool_use")],
        [ReasoningDelta("检查文件"), ToolCallEvent("service-id", "read", '{"path":"result.txt"}'),
         StreamDone("tool_use")],
        [ReasoningDelta(""), TextDelta("第一答"), StreamDone("stop")],
        [ReasoningDelta("下一问"), TextDelta("第二答"), StreamDone("stop")],
    ])
    session = await open_application(_config(root, first))
    try:
        result = await session.execute("完成文件")
        assert result.exit_code == 0 and result.persistence.saved
        assert (root / "result.txt").read_text(encoding="utf-8") == "完成"
        assert first.requests[1].messages[1].reasoning == ReasoningContent(" 原始\n思考 ", True)
        assert first.requests[2].messages[3].reasoning == ReasoningContent("检查文件", True)
        assert [m.result.call_id for m in session.history if isinstance(m, ToolResultMessage)] == [
            "service-id", "service-id",
        ]
        assert not any(m.result.is_error for m in session.history if isinstance(m, ToolResultMessage))
        await session.execute("新用户轮")
        assert first.requests[3].messages[:-1] == session.history[:-2]
        expected = session.history
        session_id = session.persistence.session_id
    finally:
        await session.aclose()
    restored_model = FakeModel([[ReasoningDelta("恢复后"), TextDelta("第三答"), StreamDone("stop")]])
    restored = await open_application(_config(root, restored_model, session_id))
    try:
        assert restored.history == expected
        assert [m.reasoning for m in restored.history if isinstance(m, AssistantMessage)] == [
            ReasoningContent(" 原始\n思考 ", True), ReasoningContent("检查文件", True),
            ReasoningContent("", True), ReasoningContent("下一问", True),
        ]
        await restored.execute("恢复续写")
        assert restored_model.requests[0].messages[:-1] == expected
        assert (root / "result.txt").read_text(encoding="utf-8") == "完成"
    finally:
        await restored.aclose()


@pytest.mark.parametrize("ending", ["stop", "tool_use", "length", "error", "aborted", "eof", "raise", "cancel"])
async def test_reasoning_completion_uses_effective_terminal(ending):
    class Model:
        async def stream(self, request, cancel):
            yield ReasoningDelta("片段")
            yield TextDelta("答案")
            if ending == "raise":
                raise RuntimeError("stream failure")
            if ending == "cancel":
                raise asyncio.CancelledError
            if ending != "eof":
                yield StreamDone(ending)
    events = []
    agent = Agent(model=Model(), tools={})
    agent.subscribe(events.append)
    result = await agent.run("q")
    assistant = result.messages[-1]
    assert assistant.reasoning == ReasoningContent("片段", ending == "stop")
    assert assistant.text == "答案"
    assert [e.text for e in events if isinstance(e, AssistantTextDelta)] == ["答案"]
    if ending == "tool_use":
        assert assistant.stop_reason == result.stop_reason == "error"


async def test_cancel_after_terminal_consumption_marks_reasoning_partial():
    class Model:
        async def stream(self, request, cancel):
            yield ReasoningDelta("")
            cancel.cancel()
            yield StreamDone("stop")
    result = await Agent(model=Model(), tools={}).run("q")
    assert result.stop_reason == "aborted"
    assert result.messages[-1].reasoning == ReasoningContent("", False)


@pytest.mark.parametrize("stop", ["stop", "tool_use"])
async def test_cancel_after_assistant_publication_preserves_completed_response(stop):
    from cicada.core.events import AssistantCompleted
    model = FakeModel([[ReasoningDelta("complete"),
                        *([ToolCallEvent("id", "missing", "{}")] if stop == "tool_use" else []),
                        StreamDone(stop)]])
    token = CancelToken()
    emitted = []
    def observer(event):
        if isinstance(event, AssistantCompleted):
            emitted.append(event.message)
            token.cancel()
    agent = Agent(model=model, tools={})
    agent.subscribe(observer)
    result = await agent.run("q", token)
    assert result.messages[1] == emitted[0]
    assert result.messages[1].reasoning == ReasoningContent("complete", True)
    if stop == "tool_use":
        assert result.stop_reason == "aborted" and result.messages[-1].result.is_error


@pytest.mark.parametrize("stop", ["length", "error", "aborted", "duplicate"])
async def test_failed_call_groups_never_execute_and_preserve_reasoning(tmp_path, stop):
    root = _repo(tmp_path)
    call = ToolCallEvent("id", "write", '{"path":"unexecuted.txt","content":"bad"}')
    model = FakeModel([[ReasoningDelta("partial"), call, *([call] if stop == "duplicate" else []),
                        StreamDone("tool_use" if stop == "duplicate" else stop)]])
    session = await open_application(_config(root, model))
    try:
        outcome = await session.execute("q")
        assert outcome.persistence.saved
        assert not (root / "unexecuted.txt").exists()
        assistant = session.history[1]
        assert assistant.reasoning == ReasoningContent("partial", False)
        if stop == "duplicate":
            assert assistant.tool_calls == ()
        else:
            assert session.history[2].result.is_error
        expected = session.history
        session_id = session.persistence.session_id
    finally:
        await session.aclose()
    restored = await open_application(_config(root, FakeModel([]), session_id))
    try:
        assert restored.history == expected
    finally:
        await restored.aclose()


async def test_no_reasoning_event_keeps_failure_unknown():
    result = await Agent(model=FakeModel([[StreamDone("error", "failure")]]), tools={}).run("q")
    assert result.messages[-1].reasoning is None


@pytest.mark.parametrize("reason", [None, ReasoningContent("", True), ReasoningContent(" raw\n思考 ", False)])
def test_reasoning_codec_optional_bytes_and_hash(reason):
    message = AssistantMessage("answer", reasoning=reason)
    raw = encode_messages((message,))
    assert decode_messages(raw) == (message,)
    if reason is None:
        assert raw == '[{"role":"assistant","text":"answer","tool_calls":[],"stop_reason":"stop","error":null}]'
    else:
        assert canonical_prefix_sha256((message,)) != canonical_prefix_sha256((replace(message, reasoning=None),))
        assert canonical_prefix_sha256((message,)) != canonical_prefix_sha256((
            replace(message, reasoning=replace(reason, complete=not reason.complete)),
        ))
        assert canonical_prefix_sha256((message,)) != canonical_prefix_sha256((
            replace(message, reasoning=replace(reason, text=reason.text + "x")),
        ))
        with pytest.raises(FrozenInstanceError):
            reason.text = "changed"


@pytest.mark.parametrize("raw", [None, {}, {"text": 3, "complete": True}, {"text": "", "complete": 1},
                                  {"text": "", "complete": True, "unknown": 0}])
def test_bad_reasoning_codec_rejected(raw):
    payload = json.loads(encode_messages((AssistantMessage("answer"),)))
    payload[0]["reasoning"] = raw
    with pytest.raises(CodecError):
        decode_messages(json.dumps(payload))


@pytest.mark.parametrize("text,complete", [(3, True), ("", 1), ("", None)])
def test_reasoning_dto_strict_types(text, complete):
    with pytest.raises(ValueError):
        ReasoningContent(text, complete)


def test_budget_counts_raw_reasoning_bytes():
    config = BudgetConfig(num_ctx=1000)
    assert BUDGET_ESTIMATOR_VERSION == "3"  # 缓存 02: 尾部组件进入估计面
    plain = estimate_message_tokens(AssistantMessage(""), config)
    assert estimate_message_tokens(AssistantMessage("", reasoning=ReasoningContent("思考", True)), config) == plain + 2


@pytest.mark.parametrize("text", ["", "output"])
async def test_reasoning_server_error_sends_once(text):
    attempts = []
    async def compact():
        attempts.append("admission")
        return True
    model = FakeModel([[ReasoningDelta(text), StreamDone("error", "overflow")], [StreamDone("stop")]])
    policy = BudgetPolicy(model, BudgetConfig(num_ctx=100_000),
                          auto_compact=compact)
    events = [e async for e in policy.stream(ModelRequest((UserMessage("q"),), ()), CancelToken())]
    assert events[-1].stop_reason == "error"
    assert len(model.requests) == 1 and attempts == []


@pytest.mark.parametrize("field", ["text", "complete"])
async def test_changed_reasoning_breaks_restored_historical_prefix_proof(tmp_path, field):
    root = _repo(tmp_path)
    model = FakeModel([[ReasoningDelta("original"), StreamDone("stop")]])
    config = replace(_check_config(root, ()), model_plugin=fake_model_plugin(model))
    session = await open_application(config)
    try:
        await session.execute("q")
        session_id = session.persistence.session_id
    finally:
        await session.aclose()
    next_model = FakeModel([[StreamDone("stop")]])
    restored = await open_application(replace(_check_config(root, (), session_id=session_id),
                                              model_plugin=fake_model_plugin(next_model)))
    try:
        original = restored._history[1]
        changed = replace(original.reasoning, **({"text": "changed"} if field == "text" else {"complete": False}))
        restored._history[1] = replace(original, reasoning=changed)
        outcome = await restored.execute("next")
        assert outcome.result.stop_reason == "error"
        assert "changed since the proof was built" in outcome.result.error
        assert next_model.requests == []
    finally:
        await restored.aclose()


@pytest.mark.parametrize("field", ["text", "complete"])
async def test_changed_archived_reasoning_breaks_summary_source_proof(tmp_path, field):
    root = _repo(tmp_path)
    model = FakeModel([[ReasoningDelta("original"), TextDelta("answer"), StreamDone("stop")],
                       [StreamDone("stop")]])
    session = await open_application(_config(root, model))
    try:
        await session.execute("q1")
        await session.execute("q2")
        snapshot = session._store.recovery_snapshot()
        request = snapshot.requests[0]
        record = SummaryRecord(
            summary_id="test-summary", epoch_id=request.epoch_id, context_generation=0,
            first_request_seq=request.seq, last_request_seq=request.seq, message_count=2,
            end_message_index=2, prefix_sha256=canonical_prefix_sha256(request.messages),
            text="可追溯任务事实", provider="fake", model="fake", prompt_version="1",
            prompt_sha256="a" * 64, metrics=None, reason="manual", created_at=time.time(),
        )
        await session._store.commit_summary(record)
        session_id = session.persistence.session_id
        path = session._store.db_path
    finally:
        await session.aclose()
    restored = await open_application(_config(root, FakeModel([]), session_id))
    try:
        assert restored.summary_records == (record,)
    finally:
        await restored.aclose()
    with sqlite3.connect(path) as connection:
        seq, raw = connection.execute("SELECT seq, payload_json FROM records WHERE kind='request' ORDER BY seq").fetchone()
        payload = json.loads(raw)
        messages = json.loads(payload["messages"])
        messages[1]["reasoning"][field] = "changed" if field == "text" else False
        payload["messages"] = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        tampered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        connection.execute("UPDATE records SET payload_json=?, payload_sha256=? WHERE seq=?",
                           (tampered, hashlib.sha256(tampered.encode()).hexdigest(), seq))
    with pytest.raises(ApplicationOpenError, match="coverage does not match"):
        await open_application(_config(root, FakeModel([]), session_id))
