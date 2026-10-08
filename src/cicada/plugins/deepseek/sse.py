"""有界 SSE framing 和单 choice 工具增量收集；不在流终结前交付调用。"""

import codecs
import json
from dataclasses import dataclass

from cicada.core.ports import ModelMetrics, ReasoningDelta, TextDelta, ToolCallEvent

FRAME_LIMIT = 256 * 1024
RESPONSE_LIMIT = 2 * 1024 * 1024
ARGUMENT_LIMIT = 64 * 1024
CALL_LIMIT = 32


class ProtocolError(ValueError):
    """只携带固定的有界协议错误，不包含服务数据。"""


class SSEDecoder:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._line = ""
        self._data: list[str] = []
        self._frame_bytes = 0
        self._total = 0

    def feed(self, chunk: bytes):
        self._total += len(chunk)
        if self._total > RESPONSE_LIMIT:
            raise ProtocolError("response exceeds byte limit")
        try:
            self._line += self._decoder.decode(chunk)
        except UnicodeDecodeError:
            raise ProtocolError("invalid UTF-8") from None
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            self._frame_bytes += len(line.encode("utf-8")) + 1
            if self._frame_bytes > FRAME_LIMIT:
                raise ProtocolError("SSE frame exceeds byte limit")
            line = line.removesuffix("\r")
            if not line:
                if self._data:
                    yield "\n".join(self._data)
                self._data = []
                self._frame_bytes = 0
            elif line.startswith("data:"):
                self._data.append(line[5:].removeprefix(" "))
        if self._frame_bytes + len(self._line.encode("utf-8")) > FRAME_LIMIT:
            raise ProtocolError("SSE frame exceeds byte limit")


@dataclass
class _Call:
    id: str | None = None
    name: str | None = None
    type: str | None = None
    arguments: str = ""


def _strict_json(raw: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError("duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("nonfinite JSON")))
    except (ValueError, RecursionError):
        raise ProtocolError("invalid JSON") from None


class Completion:
    def __init__(self, *, think: bool) -> None:
        self.think = think
        self.reasoning_seen = False
        self.calls: dict[int, _Call] = {}
        self.finish: str | None = None
        self.done = False
        self.metrics: ModelMetrics | None = None

    def consume(self, frame: str):
        if self.done:
            raise ProtocolError("data after DONE")
        if frame == "[DONE]":
            if self.finish is None:
                raise ProtocolError("DONE without finish")
            self.done = True
            return
        payload = _strict_json(frame)
        if not isinstance(payload, dict) or "error" in payload:
            raise ProtocolError("invalid completion chunk")
        usage = payload.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ProtocolError("invalid usage")
            values = []
            for name in ("prompt_tokens", "completion_tokens"):
                value = usage.get(name)
                if value is not None and (type(value) is not int or value < 0):
                    raise ProtocolError("invalid usage count")
                values.append(value)
            self.metrics = ModelMetrics(*values)
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ProtocolError("invalid choices")
        if not choices and usage is not None:
            return
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProtocolError("expected one choice")
        choice = choices[0]
        if type(choice.get("index")) is not int or choice["index"] != 0 or self.finish is not None:
            raise ProtocolError("invalid choice index or chunk after finish")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ProtocolError("invalid delta")
        for field, event_type in (("reasoning_content", ReasoningDelta), ("content", TextDelta)):
            value = delta.get(field)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ProtocolError("invalid text delta")
            if field == "reasoning_content":
                if not self.think and value:
                    raise ProtocolError("reasoning returned in non-thinking mode")
                self.reasoning_seen = True
                yield event_type(value)
            elif value:
                yield event_type(value)
        calls = delta.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ProtocolError("invalid tool delta")
            for fragment in calls:
                self._append_call(fragment)
        finish = choice.get("finish_reason")
        if finish is not None:
            if not isinstance(finish, str):
                raise ProtocolError("invalid finish")
            self.finish = finish

    def _append_call(self, fragment) -> None:
        if not isinstance(fragment, dict):
            raise ProtocolError("invalid tool fragment")
        index = fragment.get("index")
        if type(index) is not int or not 0 <= index < CALL_LIMIT:
            raise ProtocolError("invalid tool index")
        call = self.calls.setdefault(index, _Call())
        function = fragment.get("function", {})
        if not isinstance(function, dict):
            raise ProtocolError("invalid function")
        for field, value in (("id", fragment.get("id")), ("type", fragment.get("type")),
                             ("name", function.get("name"))):
            if value is None:
                continue
            if not isinstance(value, str) or not value or (field == "type" and value != "function"):
                raise ProtocolError("invalid tool identity")
            previous = getattr(call, field)
            if previous is not None:
                raise ProtocolError("repeated tool identity")
            setattr(call, field, value)
        arguments = function.get("arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise ProtocolError("invalid tool arguments delta")
            call.arguments += arguments
            if len(call.arguments.encode("utf-8")) > ARGUMENT_LIMIT:
                raise ProtocolError("tool arguments exceed byte limit")

    def terminal(self) -> tuple[list[ToolCallEvent], str, str | None]:
        if not self.done or self.finish is None:
            raise ProtocolError("stream ended without finish and DONE")
        if self.finish == "length":
            return [], "length", None
        if self.finish == "stop":
            if self.calls:
                raise ProtocolError("stop with tool calls")
            return [], "stop", None
        if self.finish != "tool_calls":
            return [], "error", "deepseek response has an unsuccessful finish"
        if not self.calls or sorted(self.calls) != list(range(len(self.calls))):
            raise ProtocolError("incomplete tool group")
        events = []
        ids = set()
        for call in self.calls.values():
            if not call.id or not call.name or call.type != "function" or call.id in ids:
                raise ProtocolError("incomplete or duplicate tool identity")
            if not isinstance(_strict_json(call.arguments), dict):
                raise ProtocolError("tool arguments must be an object")
            ids.add(call.id)
        for index in sorted(self.calls):
            call = self.calls[index]
            events.append(ToolCallEvent(call.id, call.name, call.arguments))
        if self.think and not self.reasoning_seen:
            return events, "error", "deepseek thinking tool response is missing reasoning"
        return events, "tool_use", None
