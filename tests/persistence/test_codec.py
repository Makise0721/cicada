"""持久化纯契约与 codec 聚焦测试: 严格 roundtrip、拒绝面、身份比对.

覆盖 A02 的序列化基础 (Unicode/嵌套 details/arguments 原文本/工具身份/
stop/error/计量缺失), 但完整请求级恢复由应用与 store 测试证明.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cicada.core.events import ModelCallCompleted
from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ModelMetrics
from cicada.persistence.codec import (
    CodecError,
    decode_identity,
    decode_messages,
    decode_model_calls,
    decode_outcome_facts,
    encode_identity,
    encode_messages,
    encode_model_calls,
    encode_outcome_facts,
    strict_load,
)
from cicada.persistence.contracts import (
    LaunchIdentity,
    RequestOutcomeFacts,
    StoredIdentity,
    conflicting_launch_fields,
)


def _sample_messages() -> tuple:
    return (
        UserMessage(text="第一问 ünïcödé 中文 \n\t 控制串\x1b[31m"),
        AssistantMessage(
            text="回答 ‽",
            tool_calls=(
                ToolCall(id="call-1", name="read", arguments_json='{"path": "a.py", "nest": {"x": 1}}'),
                ToolCall(id="call-2", name="glob", arguments_json="[]"),
            ),
            stop_reason="tool_use",
        ),
        ToolResultMessage(result=ToolResult(
            call_id="call-1", name="read", content="line 1\nline 2", is_error=False,
            details={"path": "a.py", "lines": [{"no": 1, "text": "x"}, {"no": 2, "text": "ü"}],
                     "nested": {"deep": {"deeper": [1, 2, {"leaf": True}]}}, "empty": None},
        )),
        ToolResultMessage(result=ToolResult(
            call_id="call-2", name="glob", content="boom", is_error=True,
            details=None,
        )),
        AssistantMessage(text="done", stop_reason="error", error="provider exploded"),
    )


def test_message_roundtrip_preserves_unicode_details_and_identity():
    messages = _sample_messages()
    decoded = decode_messages(encode_messages(messages))
    assert decoded == messages
    # arguments_json 保持模型原文本, 不重排不重编码
    assert decoded[1].tool_calls[0].arguments_json == '{"path": "a.py", "nest": {"x": 1}}'
    # 嵌套 details 结构与值完整
    assert decoded[2].result.details["nested"]["deep"]["deeper"][2] == {"leaf": True}
    # 解码产物与源对象隔离: 改动不回流
    decoded[2].result.details["tampered"] = True
    assert "tampered" not in messages[2].result.details


def test_decode_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(CodecError, match="duplicate"):
        strict_load('{"a": 1, "a": 2}')
    for bad in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(CodecError, match="non-finite"):
            strict_load(f'{{"x": {bad}}}')
    with pytest.raises(CodecError, match="invalid JSON"):
        strict_load("{not json")


def test_decode_rejects_excess_depth():
    ok = '{"a": ' * 32 + "1" + "}" * 32
    assert strict_load(ok) is not None
    too_deep = '{"a": ' * 33 + "1" + "}" * 33
    with pytest.raises(CodecError, match="depth"):
        strict_load(too_deep)


def test_decode_rejects_unknown_keys_roles_and_wrong_types():
    base = json.loads(encode_messages((UserMessage(text="hi"),)))
    with pytest.raises(CodecError, match="unexpected keys"):
        decode_messages(json.dumps([{**base[0], "extra": 1}]))
    with pytest.raises(CodecError, match="role"):
        decode_messages(json.dumps([{"role": "system", "text": "hi"}]))
    assistant = json.loads(encode_messages((AssistantMessage(text="x"),)))[0]
    with pytest.raises(CodecError, match="stop_reason"):
        decode_messages(json.dumps([{**assistant, "stop_reason": "bogus"}]))
    tool = json.loads(encode_messages((ToolResultMessage(result=ToolResult("c", "n", "x")),)))[0]
    with pytest.raises(CodecError, match="is_error"):
        decode_messages(json.dumps([{**tool, "result": {**tool["result"], "is_error": "yes"}}]))


def test_decode_rejects_bool_masquerading_as_int():
    call = ModelCallCompleted(run_id="run-1", turn_index=0, stop_reason="stop",
                              metrics=None, elapsed_s=0.5)
    raw = json.loads(encode_model_calls((call,)))
    raw[0]["turn_index"] = True
    with pytest.raises(CodecError, match="turn_index"):
        decode_model_calls(json.dumps(raw))


def test_encode_rejects_unsupported_message_and_nonfinite_details():
    with pytest.raises(CodecError, match="not a supported message"):
        encode_messages(("not a message",))  # type: ignore[list-item]
    with pytest.raises(CodecError, match="must be a tuple"):
        encode_messages([UserMessage(text="hi")])  # type: ignore[arg-type]
    poisoned = ToolResultMessage(result=ToolResult("c", "n", "x", details={"f": float("nan")}))
    with pytest.raises(CodecError):
        encode_messages((poisoned,))


def test_model_calls_roundtrip_with_unknown_and_known_metrics():
    calls = (
        ModelCallCompleted(run_id="run-1", turn_index=0, stop_reason="stop",
                           metrics=None, elapsed_s=0.0),
        ModelCallCompleted(run_id="run-2", turn_index=3, stop_reason="aborted",
                           metrics=ModelMetrics(input_tokens=12, output_tokens=None,
                                                provider_duration_s=1.25),
                           elapsed_s=1.5),
    )
    decoded = decode_model_calls(encode_model_calls(calls))
    assert decoded == calls
    assert decoded[1].metrics is not None and decoded[1].metrics.output_tokens is None


def _identity(**overrides) -> LaunchIdentity:
    fields = dict(
        provider="fake", model="fake", base_url=None, think=None,
        options={"num_ctx": 32768}, max_request_bytes=None, max_turns=6,
        prompt_version="builtin:p4-v1", prompt_sha256="a" * 64,
        instructions_sha256=None, checks_signature=None,
        sources={"num_ctx": "default"},
    )
    fields.update(overrides)
    return LaunchIdentity(**fields)


def test_identity_roundtrip_and_conflict_detection():
    stored = StoredIdentity(launch=_identity(), tool_fingerprint="fp-1")
    decoded = decode_identity(encode_identity(stored))
    assert decoded == stored
    assert conflicting_launch_fields(decoded.launch, _identity()) == ()
    conflicts = conflicting_launch_fields(
        decoded.launch, _identity(model="other", max_turns=9, options={"num_ctx": 8192})
    )
    assert set(conflicts) == {"model", "max_turns", "options"}


def test_identity_validation_rejects_wrong_types():
    with pytest.raises(ValueError, match="provider"):
        LaunchIdentity(provider=1, model="m", base_url=None, think=None, max_turns=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_turns"):
        LaunchIdentity(provider="p", model="m", base_url=None, think=None, max_turns=True)
    with pytest.raises(ValueError, match="max_turns"):
        LaunchIdentity(provider="p", model="m", base_url=None, think=None, max_turns=0)


def test_outcome_facts_roundtrip():
    facts = RequestOutcomeFacts(
        run_id="run-9", stop_reason="length", error="cut", exit_code=1,
        finalized=False, finalization_notes=("finalization timed out",),
        model_calls=(ModelCallCompleted(run_id="run-9", turn_index=0, stop_reason="length",
                                        metrics=ModelMetrics(input_tokens=1, output_tokens=2,
                                                             provider_duration_s=0.5),
                                        elapsed_s=0.5),),
    )
    assert decode_outcome_facts(encode_outcome_facts(facts)) == facts


# --- 02: 进程活动终结事实与开始记录 -----------------------------------------------------


def test_terminal_facts_roundtrip_all_kinds():
    from cicada.persistence.codec import decode_terminal_facts, encode_terminal_facts
    from cicada.persistence.contracts import ProcessTerminalFacts

    for facts in (
        ProcessTerminalFacts(kind="proven_terminal", exit_code=0, output_complete=True),
        ProcessTerminalFacts(kind="proven_terminal", exit_code=3, output_complete=True,
                             receipt_id="receipt-1"),
        ProcessTerminalFacts(kind="not_started", reason="pwsh executable not found on PATH"),
        ProcessTerminalFacts(kind="unproven_terminal", timed_out=True,
                             exit_code=None, output_complete=False, reason="timed out before EOF"),
        ProcessTerminalFacts(kind="unproven_terminal", cancelled=True),
    ):
        encoded = encode_terminal_facts(facts)
        assert decode_terminal_facts(json.loads(json.dumps(encoded))) == facts


def test_terminal_facts_decode_rejects_unknown_keys_and_kinds():
    from cicada.persistence.codec import decode_terminal_facts

    with pytest.raises(CodecError, match="unexpected keys"):
        decode_terminal_facts({"kind": "proven_terminal", "extra": 1})
    with pytest.raises(CodecError, match="kind"):
        decode_terminal_facts({"kind": "nonsense"})


def test_activity_begin_roundtrip_and_digest_validation():
    from cicada.persistence.codec import decode_activity_begin, encode_activity_begin

    encoded = json.loads(encode_activity_begin('{"command": "x"}', "a" * 64, 12.5))
    assert decode_activity_begin(encoded) == ('{"command": "x"}', "a" * 64, 12.5)
    with pytest.raises(CodecError, match="sha256"):
        decode_activity_begin({"description": "d", "arguments_digest": "zz", "started_at": 1.0})
    with pytest.raises(CodecError, match="unexpected keys"):
        decode_activity_begin({"description": "d", "arguments_digest": "a" * 64, "extra": 1})


def test_dumps_strict_rejects_unserializable_objects():
    from cicada.persistence.codec import dumps_strict

    with pytest.raises(CodecError, match="serializable"):
        dumps_strict({"command": object()})


def test_cache_metrics_old_bytes_and_both_codecs_strictness():
    from cicada.persistence.codec import decode_summary_record, encode_summary_record
    from cicada.persistence.contracts import SummaryRecord
    from cicada.models import SafeModelIdentity
    metrics = ModelMetrics(10, 2, 0.5)
    call = ModelCallCompleted("run-1", 0, "stop", metrics, 0.5)
    record = SummaryRecord(
        summary_id="s-1", epoch_id="e-1", context_generation=0,
        first_request_seq=1, last_request_seq=2, message_count=4, end_message_index=4,
        prefix_sha256="b" * 64, text="正文", provider="fake", model="m",
        prompt_version="1", prompt_sha256="c" * 64, metrics=metrics,
        reason="manual", created_at=1.0, previous_projection_identity=None,
        call_identity=SafeModelIdentity("fake", "m", None, None),
    )
    for encode, decode in (
        (lambda m: encode_model_calls((replace(call, metrics=m),)),
         lambda raw: decode_model_calls(raw)[0].metrics),
        (lambda m: encode_summary_record(replace(record, metrics=m)),
         lambda raw: decode_summary_record(strict_load(raw)).metrics),
    ):
        old = encode(metrics)
        assert "cache_" not in old
        assert encode(decode(old)) == old
        for cache in (ModelMetrics(10, 2, 0.5, 7, 3), ModelMetrics(cache_hit_input_tokens=0)):
            assert decode(encode(cache)) == cache
        payload = strict_load(old)
        raw_metrics = payload[0]["metrics"] if isinstance(payload, list) else payload["metrics"]
        for key, bad in (("cache_hit_input_tokens", True), ("cache_miss_input_tokens", -1),
                         ("cache_hit_input_tokens", 1.5), ("unexpected", 3)):
            raw_metrics[key] = bad
            with pytest.raises(CodecError):
                decode(json.dumps(payload))
            del raw_metrics[key]
        raw_metrics.update(cache_hit_input_tokens=7, cache_miss_input_tokens=4)
        with pytest.raises(CodecError, match="inconsistent"):
            decode(json.dumps(payload))
        with pytest.raises(CodecError):
            encode(ModelMetrics(10, 2, 0.5, True, 3))
        raw_metrics.update(cache_hit_input_tokens=None, cache_miss_input_tokens=None)
        assert encode(decode(json.dumps(payload))) == old
