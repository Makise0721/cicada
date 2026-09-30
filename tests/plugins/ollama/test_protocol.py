import json

import pytest

from cicada.core.messages import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cicada.core.ports import ModelRequest, ToolSpec
from cicada.plugins.ollama.protocol import (
    OllamaConfig,
    ParsedChunk,
    ToolCallChunk,
    build_request,
    parse_line,
)

CONFIG = OllamaConfig(base_url="http://ollama.test", model="qwen3.5:9b")


def test_user_message_mapping():
    payload = build_request(ModelRequest((UserMessage(text="hello"),), ()), CONFIG)
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["model"] == "qwen3.5:9b"
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["options"] == {}
    assert "tools" not in payload


def test_think_flag_and_options_passthrough():
    config = OllamaConfig(think=True, options={"temperature": 0.7, "num_ctx": 4096})
    payload = build_request(ModelRequest((UserMessage(text="hi"),), ()), config)
    assert payload["think"] is True
    assert payload["options"] == {"temperature": 0.7, "num_ctx": 4096}


def test_assistant_text_without_tool_calls_has_no_tool_calls_key():
    assistant = AssistantMessage(text="done", tool_calls=())
    payload = build_request(ModelRequest((assistant,), ()), CONFIG)
    assert payload["messages"] == [{"role": "assistant", "content": "done"}]


def test_assistant_tool_calls_echo_as_objects():
    assistant = AssistantMessage(
        text="",
        tool_calls=(ToolCall("call_1", "get_weather", '{"city": "Paris"}'),),
    )
    payload = build_request(ModelRequest((assistant,), ()), CONFIG)
    assert payload["messages"][0] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    }


def test_assistant_invalid_arguments_json_raises_value_error():
    bad_json = AssistantMessage(text="", tool_calls=(ToolCall("call_1", "t", "not json"),))
    non_object = AssistantMessage(text="", tool_calls=(ToolCall("call_2", "t", "[1]"),))
    with pytest.raises(ValueError, match="arguments_json"):
        build_request(ModelRequest((bad_json,), ()), CONFIG)
    with pytest.raises(ValueError, match="JSON object"):
        build_request(ModelRequest((non_object,), ()), CONFIG)


def test_tool_result_mapping_drops_is_error_and_details():
    result = ToolResult(
        call_id="call_1",
        name="get_weather",
        content="18C, cloudy",
        is_error=True,
        details={"next_offset": 2},
    )
    payload = build_request(ModelRequest((ToolResultMessage(result=result),), ()), CONFIG)
    assert payload["messages"] == [
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18C, cloudy"}
    ]


def test_tools_included_only_when_present():
    spec = ToolSpec(
        name="get_weather",
        description="Get current weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    payload = build_request(ModelRequest((UserMessage(text="hi"),), (spec,)), CONFIG)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]


def test_multi_turn_history_mapping():
    messages = (
        UserMessage(text="weather?"),
        AssistantMessage(text="", tool_calls=(ToolCall("call_1", "get_weather", '{"city": "Paris"}'),)),
        ToolResultMessage(result=ToolResult(call_id="call_1", name="get_weather", content="18C")),
        AssistantMessage(text="18C"),
    )
    payload = build_request(ModelRequest(messages, ()), CONFIG)
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "tool", "assistant"]


def test_parse_text_chunk():
    raw = b'{"model":"m","message":{"role":"assistant","content":"Hi"},"done":false}'
    assert parse_line(raw) == ParsedChunk(text="Hi")


def test_parse_thinking_chunk():
    raw = b'{"message":{"role":"assistant","content":"","thinking":"Thinking"},"done":false}'
    assert parse_line(raw) == ParsedChunk(text="", thinking="Thinking")


def test_parse_tool_calls_multi_element():
    line = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "function": {"index": 0, "name": "get_weather", "arguments": {"city": "Paris"}}},
                {"id": "call_b", "function": {"name": "get_time", "arguments": {"tz": "UTC"}}},
            ],
        },
        "done": False,
    }
    chunk = parse_line(json.dumps(line).encode("utf-8"))
    assert chunk.tool_calls == (
        ToolCallChunk("call_a", "get_weather", {"city": "Paris"}),
        ToolCallChunk("call_b", "get_time", {"tz": "UTC"}),
    )
    assert chunk.done is False
    assert chunk.error is None


def test_parse_done_line_with_done_reason():
    raw = b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}'
    assert parse_line(raw) == ParsedChunk(done=True, done_reason="stop")
    raw = b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"length"}'
    assert parse_line(raw) == ParsedChunk(done=True, done_reason="length")


def test_parse_server_error_line():
    raw = b'{"error":"model \\"qwen3.5:9b\\" not found"}'
    chunk = parse_line(raw)
    assert chunk.error is not None and "not found" in chunk.error


def test_parse_crlf_and_empty_content():
    chunk = parse_line(b'{"message":{"role":"assistant","content":"a"},"done":false}\r')
    assert chunk.text == "a"


@pytest.mark.parametrize(
    "raw, why",
    [
        (b"not json", "valid JSON"),
        (b"[1, 2]", "JSON object"),
        (b'{"done": false}', "message"),
        (b'{"message": {"role": "assistant", "content": ""}}', "done"),
        (b'{"message": {"role": "assistant", "content": ""}, "done": 1}', "done"),
        (b'{"message": {"content": 5}, "done": false}', "strings"),
        (b'{"message": {"content": "", "tool_calls": {"id": "c"}}, "done": false}', "list"),
        (
            b'{"message": {"content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]}, "done": false}',
            "id",
        ),
        (
            b'{"message": {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": [1]}}]}, "done": false}',
            "arguments",
        ),
        (
            b'{"message": {"content": "", "tool_calls": [{"id": "c1", "function": {"arguments": {}}}]}, "done": false}',
            "function.name",
        ),
    ],
)
def test_bad_lines_report_error_without_raising(raw, why):
    chunk = parse_line(raw)
    assert chunk.error is not None
    assert why in chunk.error
    assert chunk == ParsedChunk(error=chunk.error)


def test_invalid_utf8_line_reports_error():
    chunk = parse_line(b'{"message": {"content": "\xff\xfe"}, "done": false}')
    assert chunk.error is not None
    assert "UTF-8" in chunk.error
