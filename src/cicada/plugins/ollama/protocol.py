"""Ollama /api/chat 协议纯函数: 请求构造与 NDJSON 行解析, 无 IO.

契约 (docs/superpowers/plans/2026-09-30-parallel-p2-ollama.md Part A):
- 请求构造失败 (如历史消息 arguments_json 非法) 以 ValueError 表达, 由 model 层归一为 StreamDone("error").
- 行级解析错误一律以 ParsedChunk.error 返回值表达, 不抛异常.

P3 增补 (docs/superpowers/plans/2026-09-30-parallel-p3-context-observability.md §5):
- request.system_prompt 非空时在最前加一条 system 消息; 空串保持旧请求 JSON 不变.
- 仅有效 done:true 行提取 optional 计量 (prompt_eval_count/eval_count/total_duration);
  非法/缺失映射为 None, 不改变文本/工具/stop 的协议严格性.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cicada.core.messages import AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage
from cicada.core.ports import ModelMetrics, ModelRequest, ToolSpec

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama 适配配置; timeout 为 None 时由 model 层使用默认超时."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:9b"
    think: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)  # 透传 Ollama options (temperature/num_ctx 等)
    timeout: httpx.Timeout | None = None


@dataclass(frozen=True)
class ToolCallChunk:
    """一行 NDJSON 中携带的单个完整工具调用."""

    id: str
    name: str
    arguments: dict[str, Any]  # JSON object


@dataclass(frozen=True)
class ParsedChunk:
    """一行 NDJSON 的结构化结果; error 非 None 时为行级协议错误, 其余字段无效.

    metrics 仅来自有效 done:true 行; done:false 行与其他行恒为 None (不累计).
    """

    text: str = ""
    thinking: str = ""  # 既定语义: model 层丢弃, 不进会话
    tool_calls: tuple[ToolCallChunk, ...] = ()
    done: bool = False
    done_reason: str | None = None
    error: str | None = None
    metrics: ModelMetrics | None = None


def build_request(request: ModelRequest, config: OllamaConfig) -> dict[str, Any]:
    """把 ModelRequest 映射为 /api/chat 请求体; 构造失败抛 ValueError."""
    messages = [_encode_message(message) for message in request.messages]
    if request.system_prompt:
        # P3: 每轮至多一条首位 system 消息; 空串省略, 旧请求 JSON 不变
        messages.insert(0, {"role": "system", "content": request.system_prompt})
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": True,
        "think": config.think,
        "options": dict(config.options),
    }
    if request.tools:
        payload["tools"] = [_encode_tool(spec) for spec in request.tools]
    return payload


def parse_line(raw: bytes) -> ParsedChunk:
    """解析一行 /api/chat NDJSON; 行级错误以 ParsedChunk.error 返回."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return ParsedChunk(error=f"line is not valid UTF-8: {exc}")
    try:
        line = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedChunk(error=f"line is not valid JSON: {exc}")
    if not isinstance(line, dict):
        return ParsedChunk(error="line is not a JSON object")
    if isinstance(line.get("error"), str):
        # 服务端错误行 (如 /api/chat 请求体被拒): 直接携带 error 文本
        return ParsedChunk(error=line["error"])
    message = line.get("message")
    if not isinstance(message, dict):
        return ParsedChunk(error="line is missing a message object")
    done = line.get("done")
    if not isinstance(done, bool):
        return ParsedChunk(error="line is missing a boolean done flag")
    text = message.get("content", "")
    thinking = message.get("thinking", "")
    if not isinstance(text, str) or not isinstance(thinking, str):
        return ParsedChunk(error="message.content/thinking must be strings")
    tool_calls: list[ToolCallChunk] = []
    raw_calls = message.get("tool_calls")
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            return ParsedChunk(error="message.tool_calls is not a list")
        for item in raw_calls:
            outcome = _parse_tool_call(item)
            if isinstance(outcome, str):
                return ParsedChunk(error=outcome)
            tool_calls.append(outcome)
    return ParsedChunk(
        text=text,
        thinking=thinking,
        tool_calls=tuple(tool_calls),
        done=done,
        done_reason=line.get("done_reason"),
        metrics=_extract_metrics(line) if done else None,
    )


def _extract_metrics(line: Mapping[str, Any]) -> ModelMetrics | None:
    """仅从有效 done:true 行提取 optional 计量; 字段非法/缺失映射为 None.

    total_duration 为纳秒, 换秒必须 finite; 巨大整数换秒的 OverflowError
    只使该字段未知, 不破坏控制流. 全部不可得返回 None.
    """
    input_tokens = _metric_count(line.get("prompt_eval_count"))
    output_tokens = _metric_count(line.get("eval_count"))
    provider_duration_s = _metric_duration(line.get("total_duration"))
    if input_tokens is None and output_tokens is None and provider_duration_s is None:
        return None
    return ModelMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_duration_s=provider_duration_s,
    )


def _metric_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _metric_duration(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    try:
        seconds = value / 1e9
    except OverflowError:
        return None
    if not math.isfinite(seconds):
        return None
    return seconds


def _encode_message(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, AssistantMessage):
        encoded: dict[str, Any] = {"role": "assistant", "content": message.text}
        if message.tool_calls:
            # 回显形态实测被服务端接受; arguments 必须是 object, 故回显前 json.loads
            encoded["tool_calls"] = [_encode_call(call) for call in message.tool_calls]
        return encoded
    if isinstance(message, ToolResultMessage):
        result = message.result
        # is_error/details 不上行: 服务端只认 tool_call_id/name/content
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.name,
            "content": result.content,
        }
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def _encode_call(call: ToolCall) -> dict[str, Any]:
    try:
        arguments = json.loads(call.arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool call {call.id!r} arguments_json is not valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"tool call {call.id!r} arguments must decode to a JSON object")
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    }


def _encode_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _parse_tool_call(item: Any) -> ToolCallChunk | str:
    """解析单个 tool call; 返回结构化结果或错误文本 (id 缺省/arguments 非 object 等视为协议错误)."""
    if not isinstance(item, dict):
        return "message.tool_calls item is not an object"
    call_id = item.get("id")
    if not isinstance(call_id, str) or not call_id:
        return "message.tool_calls item is missing id"
    function = item.get("function")
    if not isinstance(function, dict):
        return "message.tool_calls item is missing function object"
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return "message.tool_calls item is missing function.name"
    arguments = function.get("arguments")
    if not isinstance(arguments, dict):
        return "message.tool_calls item arguments is not an object"
    return ToolCallChunk(id=call_id, name=name, arguments=arguments)
