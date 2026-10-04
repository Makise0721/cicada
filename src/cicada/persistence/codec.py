"""受限 codec: 规范消息/计量/身份/结果事实 ↔ 严格 JSON (会话保存恢复 v1).

不做 pickle、任意对象恢复、NaN/Infinity 或 default=str; 拒绝 bool 冒充 int;
未知键与重复键拒绝; 嵌套深度 ≤ MAX_JSON_DEPTH (编码与解码双向).
字节大小限额由 store 在编解码前核对; 结构闭合校验由调用方用
core.messages.validate_history_structure 完成.
"""

from __future__ import annotations

import json
from typing import Any

from cicada.core.events import ModelCallCompleted
from cicada.core.messages import (
    AssistantMessage,
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ModelMetrics
from cicada.persistence.contracts import (
    MAX_JSON_DEPTH,
    LaunchIdentity,
    PersistenceError,
    ProcessTerminalFacts,
    RequestOutcomeFacts,
    StoredIdentity,
)

_STOP_REASONS = frozenset(("stop", "tool_use", "length", "error", "aborted"))


class CodecError(PersistenceError):
    """严格编解码失败: 编码端是数据不可序列化, 解码端视为存档损坏的依据."""


# --- 严格 JSON 基础 ---------------------------------------------------------------------


def _reject_constant(name: str) -> None:
    raise CodecError(f"non-finite JSON constant {name!r} is not accepted")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CodecError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_load(raw: str) -> Any:
    """解析严格 JSON: 拒绝重复键与 NaN/Infinity, 深度超过 MAX_JSON_DEPTH 拒绝."""
    try:
        parsed = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)
    except CodecError:
        raise
    except ValueError as exc:
        raise CodecError(f"invalid JSON: {exc}") from exc
    _ensure_depth(parsed, 0)
    return parsed


def _ensure_depth(value: Any, depth: int) -> None:
    if depth > MAX_JSON_DEPTH:
        raise CodecError(f"JSON nesting exceeds the v1 depth limit ({MAX_JSON_DEPTH})")
    if isinstance(value, dict):
        for item in value.values():
            _ensure_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _ensure_depth(item, depth + 1)


def _exact_keys(obj: Any, keys: frozenset[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise CodecError(f"{where} must be a JSON object")
    unexpected = set(obj) - keys
    if unexpected:
        raise CodecError(f"{where} has unexpected keys: {sorted(unexpected)}")


def _take_str(obj: dict[str, Any], key: str, where: str) -> str:
    value = obj[key]
    if not isinstance(value, str):
        raise CodecError(f"{where}.{key} must be a string")
    return value


def _take_opt_str(obj: dict[str, Any], key: str, where: str) -> str | None:
    value = obj[key]
    if value is not None and not isinstance(value, str):
        raise CodecError(f"{where}.{key} must be a string or null")
    return value


def _take_bool(obj: dict[str, Any], key: str, where: str) -> bool:
    value = obj[key]
    if not isinstance(value, bool):
        raise CodecError(f"{where}.{key} must be a boolean")
    return value


def _take_opt_bool(obj: dict[str, Any], key: str, where: str) -> bool | None:
    value = obj[key]
    if value is not None and not isinstance(value, bool):
        raise CodecError(f"{where}.{key} must be a boolean or null")
    return value


def _take_int(obj: dict[str, Any], key: str, where: str) -> int:
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodecError(f"{where}.{key} must be an integer")
    return value


def _take_opt_int(obj: dict[str, Any], key: str, where: str) -> int | None:
    value = obj[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodecError(f"{where}.{key} must be an integer or null")
    return value


def _take_float(obj: dict[str, Any], key: str, where: str) -> float:
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, float):
        raise CodecError(f"{where}.{key} must be a finite float")
    return value


def _take_stop_reason(obj: dict[str, Any], key: str, where: str) -> StopReason:
    value = obj[key]
    if value not in _STOP_REASONS:
        raise CodecError(f"{where}.{key} must be one of {sorted(_STOP_REASONS)}")
    return value  # type: ignore[return-value]


def dumps_strict(payload: object) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise CodecError(f"value is not strictly JSON-serializable: {exc}") from exc


# --- 规范消息 ---------------------------------------------------------------------------


def encode_messages(messages: Any) -> str:
    """编码消息序列; 非规范消息类型或不可序列化 details 抛 CodecError."""
    if isinstance(messages, tuple):
        items = messages
    else:
        raise CodecError("messages must be a tuple of messages")
    payload: list[dict[str, Any]] = []
    for index, message in enumerate(items):
        where = f"messages[{index}]"
        if isinstance(message, UserMessage):
            payload.append({"role": "user", "text": message.text})
        elif isinstance(message, AssistantMessage):
            payload.append({
                "role": "assistant",
                "text": message.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments_json": call.arguments_json}
                    for call in message.tool_calls
                ],
                "stop_reason": message.stop_reason,
                "error": message.error,
            })
        elif isinstance(message, ToolResultMessage):
            payload.append({
                "role": "tool",
                "result": {
                    "call_id": message.result.call_id,
                    "name": message.result.name,
                    "content": message.result.content,
                    "is_error": message.result.is_error,
                    "details": message.result.details,
                },
            })
        else:
            raise CodecError(f"{where} is not a supported message type")
    return dumps_strict(payload)


def decode_messages(raw: str) -> tuple[Message, ...]:
    """从严格 JSON 解码消息序列; 逐条校验类型与键, 解析产物天然与源对象隔离."""
    parsed = strict_load(raw)
    if not isinstance(parsed, list):
        raise CodecError("messages payload must be a JSON list")
    messages: list[Message] = []
    for index, item in enumerate(parsed):
        messages.append(_decode_message(item, f"messages[{index}]"))
    return tuple(messages)


def _decode_message(item: Any, where: str) -> Message:
    if not isinstance(item, dict):
        raise CodecError(f"{where} must be a JSON object")
    role = item.get("role")
    if role == "user":
        _exact_keys(item, frozenset({"role", "text"}), where)
        return UserMessage(text=_take_str(item, "text", where))
    if role == "assistant":
        _exact_keys(item, frozenset({"role", "text", "tool_calls", "stop_reason", "error"}), where)
        calls_raw = item["tool_calls"]
        if not isinstance(calls_raw, list):
            raise CodecError(f"{where}.tool_calls must be a list")
        calls = tuple(
            ToolCall(
                id=_take_str(call, "id", f"{where}.tool_calls[{i}]"),
                name=_take_str(call, "name", f"{where}.tool_calls[{i}]"),
                arguments_json=_take_str(call, "arguments_json", f"{where}.tool_calls[{i}]"),
            )
            for i, call in enumerate(calls_raw)
        )
        return AssistantMessage(
            text=_take_str(item, "text", where),
            tool_calls=calls,
            stop_reason=_take_stop_reason(item, "stop_reason", where),
            error=_take_opt_str(item, "error", where),
        )
    if role == "tool":
        _exact_keys(item, frozenset({"role", "result"}), where)
        result = item["result"]
        result_where = f"{where}.result"
        _exact_keys(result, frozenset({"call_id", "name", "content", "is_error", "details"}), result_where)
        return ToolResultMessage(
            result=ToolResult(
                call_id=_take_str(result, "call_id", result_where),
                name=_take_str(result, "name", result_where),
                content=_take_str(result, "content", result_where),
                is_error=_take_bool(result, "is_error", result_where),
                details=result["details"],
            )
        )
    raise CodecError(f"{where}.role must be one of user/assistant/tool")


# --- 计量事实 ---------------------------------------------------------------------------


def encode_model_calls(calls: tuple[ModelCallCompleted, ...]) -> str:
    payload = [
        {
            "run_id": call.run_id,
            "turn_index": call.turn_index,
            "stop_reason": call.stop_reason,
            "metrics": None if call.metrics is None else {
                "input_tokens": call.metrics.input_tokens,
                "output_tokens": call.metrics.output_tokens,
                "provider_duration_s": call.metrics.provider_duration_s,
            },
            "elapsed_s": call.elapsed_s,
        }
        for call in calls
    ]
    return dumps_strict(payload)


def decode_model_calls(raw: str) -> tuple[ModelCallCompleted, ...]:
    parsed = strict_load(raw)
    if not isinstance(parsed, list):
        raise CodecError("model_calls payload must be a JSON list")
    calls: list[ModelCallCompleted] = []
    for index, item in enumerate(parsed):
        where = f"model_calls[{index}]"
        _exact_keys(item, frozenset({"run_id", "turn_index", "stop_reason", "metrics", "elapsed_s"}), where)
        metrics_raw = item["metrics"]
        metrics: ModelMetrics | None = None
        if metrics_raw is not None:
            metrics_where = f"{where}.metrics"
            _exact_keys(
                metrics_raw,
                frozenset({"input_tokens", "output_tokens", "provider_duration_s"}),
                metrics_where,
            )
            metrics = ModelMetrics(
                input_tokens=_take_opt_int(metrics_raw, "input_tokens", metrics_where),
                output_tokens=_take_opt_int(metrics_raw, "output_tokens", metrics_where),
                provider_duration_s=_take_opt_float(metrics_raw, "provider_duration_s", metrics_where),
            )
        calls.append(ModelCallCompleted(
            run_id=_take_str(item, "run_id", where),
            turn_index=_take_int(item, "turn_index", where),
            stop_reason=_take_stop_reason(item, "stop_reason", where),
            metrics=metrics,
            elapsed_s=_take_float(item, "elapsed_s", where),
        ))
    return tuple(calls)


def _take_opt_float(obj: dict[str, Any], key: str, where: str) -> float | None:
    value = obj[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, float):
        raise CodecError(f"{where}.{key} must be a finite float or null")
    return value


# --- 结果事实与身份 ---------------------------------------------------------------------


def encode_outcome_facts(facts: RequestOutcomeFacts) -> str:
    payload = {
        "run_id": facts.run_id,
        "stop_reason": facts.stop_reason,
        "error": facts.error,
        "exit_code": facts.exit_code,
        "finalized": facts.finalized,
        "finalization_notes": list(facts.finalization_notes),
        "model_calls": json.loads(encode_model_calls(facts.model_calls)),
    }
    return dumps_strict(payload)


def decode_outcome_facts(raw: str) -> RequestOutcomeFacts:
    parsed = strict_load(raw)
    where = "outcome"
    _exact_keys(
        parsed,
        frozenset({"run_id", "stop_reason", "error", "exit_code", "finalized",
                   "finalization_notes", "model_calls"}),
        where,
    )
    notes = parsed["finalization_notes"]
    if not isinstance(notes, list) or any(not isinstance(n, str) for n in notes):
        raise CodecError(f"{where}.finalization_notes must be a list of strings")
    return RequestOutcomeFacts(
        run_id=_take_str(parsed, "run_id", where),
        stop_reason=_take_stop_reason(parsed, "stop_reason", where),
        error=_take_opt_str(parsed, "error", where),
        exit_code=_take_int(parsed, "exit_code", where),
        finalized=_take_bool(parsed, "finalized", where),
        finalization_notes=tuple(notes),
        model_calls=decode_model_calls(dumps_strict(parsed["model_calls"])),
    )


# --- 进程活动事实 (02) ------------------------------------------------------------------


_TERMINAL_KINDS = frozenset(("proven_terminal", "not_started", "unproven_terminal"))

_TERMINAL_KEYS = frozenset({
    "kind", "exit_code", "output_complete", "timed_out", "cancelled", "receipt_id", "reason",
})

_ACTIVITY_BEGIN_KEYS = frozenset({"description", "arguments_digest", "started_at"})


def encode_terminal_facts(facts: ProcessTerminalFacts) -> dict[str, Any]:
    """终结事实 -> 严格 JSON dict (store 落 end_json 前再经字节限额核对)."""
    payload: dict[str, Any] = {
        "kind": facts.kind,
        "exit_code": facts.exit_code,
        "output_complete": facts.output_complete,
        "timed_out": facts.timed_out,
        "cancelled": facts.cancelled,
        "receipt_id": facts.receipt_id,
        "reason": facts.reason,
    }
    dumps_strict(payload)
    return payload


def decode_terminal_facts(payload: Any) -> ProcessTerminalFacts:
    """严格 JSON dict -> 终结事实; 未知键/类型不符按存档损坏拒绝."""
    where = "process terminal facts"
    _exact_keys(payload, _TERMINAL_KEYS, where)
    kind = payload["kind"]
    if kind not in _TERMINAL_KINDS:
        raise CodecError(f"{where}.kind must be one of {sorted(_TERMINAL_KINDS)}")
    try:
        return ProcessTerminalFacts(
            kind=kind,
            exit_code=_take_opt_int(payload, "exit_code", where),
            output_complete=_take_opt_bool(payload, "output_complete", where),
            timed_out=_take_bool(payload, "timed_out", where),
            cancelled=_take_bool(payload, "cancelled", where),
            receipt_id=_take_opt_str(payload, "receipt_id", where),
            reason=_take_opt_str(payload, "reason", where),
        )
    except ValueError as exc:
        raise CodecError(f"{where} is inconsistent: {exc}") from exc


def encode_activity_begin(description: str, arguments_digest: str, started_at: float) -> str:
    """活动开始记录 -> 严格 JSON (store 落 begin_json)."""
    return dumps_strict({
        "description": description,
        "arguments_digest": arguments_digest,
        "started_at": started_at,
    })


def decode_activity_begin(payload: Any) -> tuple[str, str, float]:
    """严格 JSON dict -> (description, arguments_digest, started_at)."""
    where = "process activity begin"
    _exact_keys(payload, _ACTIVITY_BEGIN_KEYS, where)
    digest = _take_str(payload, "arguments_digest", where)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise CodecError(f"{where}.arguments_digest must be a sha256 hex digest")
    return (
        _take_str(payload, "description", where),
        digest,
        _take_float(payload, "started_at", where),
    )


_IDENTITY_KEYS = frozenset({
    "provider", "model", "base_url", "think", "options", "max_request_bytes", "max_turns",
    "prompt_version", "prompt_sha256", "instructions_sha256", "checks_signature",
    "sources", "tool_fingerprint",
})


def encode_identity(identity: StoredIdentity) -> str:
    launch = identity.launch
    payload = launch.comparable_dict()
    payload["sources"] = dict(launch.sources)
    payload["tool_fingerprint"] = identity.tool_fingerprint
    return dumps_strict(payload)


def decode_identity(raw: str) -> StoredIdentity:
    parsed = strict_load(raw)
    _exact_keys(parsed, _IDENTITY_KEYS, "identity")
    options = parsed["options"]
    if not isinstance(options, dict):
        raise CodecError("identity.options must be a JSON object")
    sources = parsed["sources"]
    if not isinstance(sources, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in sources.items()
    ):
        raise CodecError("identity.sources must map strings to strings")
    launch = LaunchIdentity(
        provider=_take_str(parsed, "provider", "identity"),
        model=_take_str(parsed, "model", "identity"),
        base_url=_take_opt_str(parsed, "base_url", "identity"),
        think=_take_opt_bool(parsed, "think", "identity"),
        options=options,
        max_request_bytes=_take_opt_int(parsed, "max_request_bytes", "identity"),
        max_turns=_take_int(parsed, "max_turns", "identity"),
        prompt_version=_take_str(parsed, "prompt_version", "identity"),
        prompt_sha256=_take_str(parsed, "prompt_sha256", "identity"),
        instructions_sha256=_take_opt_str(parsed, "instructions_sha256", "identity"),
        checks_signature=_take_opt_str(parsed, "checks_signature", "identity"),
        sources=sources,
    )
    return StoredIdentity(
        launch=launch,
        tool_fingerprint=_take_str(parsed, "tool_fingerprint", "identity"),
    )
