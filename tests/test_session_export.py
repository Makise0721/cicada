"""会话 HTML 导出: 命名、转义、结构与不覆盖 (session-export v1)。

只覆盖纯函数与落盘纪律; 真实 TUI 里的 /export 唤起浏览器不做自动化断言
(webbrowser 依赖宿主), 由人工实机核对。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from cicada.core.messages import AssistantMessage, ReasoningContent, ToolResult, ToolResultMessage
from cicada.core.messages import UserMessage
from cicada.session_export import (
    ExportError,
    default_export_name,
    render_html,
    unique_path,
    write_html,
)


def test_default_name_uses_time_and_session_prefix():
    name = default_export_name(
        session_id="01a1521f-bf26-7000-8423-2bbbaac6ec3",
        now=datetime(2026, 10, 19, 15, 15, 9),
    )
    assert name == "cicada-session-20261019-151509-01a1521f.html"


def test_default_name_marks_missing_session():
    name = default_export_name(session_id=None, now=datetime(2026, 10, 19, 1, 2, 3))
    assert name == "cicada-session-20261019-010203-nosessio.html"


def test_collect_does_not_assign_process_metrics_to_history(tmp_path):
    """恢复/clear 后历史没有调用身份，逐消息计量保持未知。"""
    from types import SimpleNamespace
    from cicada.core.events import ModelCallCompleted
    from cicada.core.ports import ModelMetrics
    from cicada.session_export import collect

    session = SimpleNamespace(
        config=SimpleNamespace(workspace=tmp_path, model_label="fake", max_turns=6,
                               system_prompt="system"),
        persistence=None, tool_names=(),
        reasoning_effort=SimpleNamespace(enabled=False),
        history=(UserMessage("old"), AssistantMessage("restored"),
                 UserMessage("new"), AssistantMessage("current")),
        process_model_calls=(ModelCallCompleted(
            run_id="new-run", turn_index=0, stop_reason="stop",
            metrics=ModelMetrics(input_tokens=987, output_tokens=123), elapsed_s=0.5),),
    )
    payload = collect(session)
    assert all(message["metrics"] is None for message in payload["messages"]
               if message["kind"] == "assistant")
    assert payload["metrics"]["input_tokens"] == 987
    assert payload["metrics"]["calls"] == 1
    session.history = (UserMessage("after clear"), AssistantMessage("answer"))
    assert collect(session)["messages"][-1]["metrics"] is None


def _payload(**overrides) -> dict:
    payload = {
        "file_name": "x.html",
        "generated_at": "2026-10-19 15:00:00",
        "cicada_version": "9.9.9",
        "session": {
            "id": "01a1521f-bf26-7000-8423-2bbbaac6ec3",
            "name": "分层说明",
            "mode": "save",
            "workspace": "C:/work",
            "model": "deepseek-flash",
            "thinking": "high",
            "turns": 8,
        },
        "tools": ["read", "grep"],
        "system_prompt": "You are Cicada.",
        "messages": [],
        "metrics": {
            "calls": 0, "input_tokens": None, "output_tokens": None,
            "cache_hit_input_tokens": None, "cache_miss_input_tokens": None, "elapsed_s": None,
        },
    }
    payload.update(overrides)
    return payload


def test_render_escapes_model_output():
    payload = _payload(messages=[
        {"kind": "user", "text": "普通文本"},
        {"kind": "assistant", "text": "<script>alert(1)</script>",
         "reasoning": None, "stop_reason": "stop", "error": None,
         "tools": [], "metrics": None},
    ])
    html = render_html(payload)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_marks_missing_metrics_as_unknown():
    html = render_html(_payload())
    assert "不可得" in html
    assert "本次会话还没有消息" in html


def test_render_includes_tools_and_system_prompt():
    html = render_html(_payload())
    assert "可用工具 (2)" in html and "read grep" in html
    assert "系统提示" in html and "You are Cicada." in html
    assert "分层说明" in html and "deepseek-flash" in html


def test_render_minimal_markdown_keeps_code_and_lists():
    payload = _payload(messages=[
        {"kind": "assistant", "text": "## 结论\n\n- `core` 内核\n- runtime\n\n**重点** 说明",
         "reasoning": "先看目录", "stop_reason": "stop", "error": None,
         "tools": [], "metrics": None},
    ])
    html = render_html(payload)
    assert "<h3" not in html  # ## -> h4 (级别 +2)
    assert "<h4>结论</h4>" in html
    assert "<ul>" in html and "<li><code>core</code> 内核</li>" in html
    assert "<strong>重点</strong>" in html
    assert "先看目录" in html


def test_tool_failure_marked():
    payload = _payload(messages=[
        {"kind": "assistant", "text": "", "reasoning": None, "stop_reason": "stop",
         "error": None, "metrics": None,
         "tools": [{"name": "grep", "content": "no match", "is_error": True}]},
    ])
    html = render_html(payload)
    assert 'class="tool err"' in html
    assert "grep · 失败" in html


def test_unique_path_never_overwrites(tmp_path):
    first = unique_path(tmp_path, "a.html")
    assert first.name == "a.html"
    first.write_text("x", encoding="utf-8")
    second = unique_path(tmp_path, "a.html")
    assert second.name == "a-2.html"
    second.write_text("y", encoding="utf-8")
    assert unique_path(tmp_path, "a.html").name == "a-3.html"


def test_write_html_is_exclusive(tmp_path):
    target = tmp_path / "out.html"
    write_html(target, "first")
    with pytest.raises(ExportError):
        write_html(target, "second")
    assert target.read_text(encoding="utf-8") == "first"


def test_collect_maps_history_and_metrics():
    from cicada.session_export import collect

    class _Session:
        state = "idle"
        history = (
            UserMessage("问题"),
            AssistantMessage("答", reasoning=ReasoningContent("想", True)),
            ToolResultMessage(ToolResult(call_id="c1", name="read", content="正文")),
        )
        config = type("C", (), {
            "workspace": Path("C:/work"), "model_label": "fake",
            "max_turns": 5, "system_prompt": "sys",
        })()
        persistence = None
        tool_names = ("read",)
        reasoning_effort = type("R", (), {"enabled": False, "current": None})()
        process_model_calls = ()

    blocks = collect(_Session())["messages"]
    assert [block["kind"] for block in blocks] == ["user", "assistant"]
    assert blocks[1]["reasoning"] == "想"
    assert blocks[1]["tools"][0]["name"] == "read"
    assert blocks[1]["metrics"] is None  # 无调用记录: 不伪造计量


def test_export_rejects_busy_session():
    from cicada.session_export import export_session_html

    class _Busy:
        state = "executing"

    with pytest.raises(ExportError):
        export_session_html(_Busy(), open_browser=False)


def test_export_opens_browser_only_when_asked(tmp_path, monkeypatch):
    from cicada import session_export

    calls: list[str] = []
    monkeypatch.setattr(session_export.webbrowser, "open", lambda uri: calls.append(uri) or True)

    class _Session:
        state = "idle"
        history = (UserMessage("hi"),)
        config = type("C", (), {
            "workspace": tmp_path, "model_label": "fake",
            "max_turns": 5, "system_prompt": "sys",
        })()
        persistence = None
        tool_names = ()
        reasoning_effort = type("R", (), {"enabled": False, "current": None})()
        process_model_calls = ()

    quiet = session_export.export_session_html(_Session(), open_browser=False)
    assert quiet.opened is False and quiet.path.exists() and calls == []
    loud = session_export.export_session_html(_Session(), open_browser=True)
    assert loud.opened is True and len(calls) == 1 and calls[0].startswith("file:")
