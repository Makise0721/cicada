"""04 聚焦: rendering.py 的 Rich -> styled fragments 渲染与 T13 显示清理.

真实应用驱动的最终可见输出回归 (prompt_toolkit 输出管线) 在
tests/e2e/test_tui_verification.py; 本文件覆盖单元级格式化与清理语义。
"""

from __future__ import annotations

import io

from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.shortcuts import print_formatted_text

from cicada.ui.contracts import ActivePreview, PresentationRecord, StatusSnapshot
from cicada.ui.rendering import (
    record_fragments,
    sanitize_display,
    status_fragments,
    preview_fragments,
)


def _fragment_text(fragments) -> str:
    return "".join(text for _style, text in fragments)


def _fragment_styles(fragments) -> str:
    return " ".join(style for style, _text in fragments if style)


def _render_to_vt(fragments) -> str:
    """经 prompt_toolkit Vt100 输出管线产出最终可见字节 (与 01 探针同一形态)."""
    stream = io.StringIO()
    output = Vt100_Output(stream, lambda: Size(rows=40, columns=120), enable_cpr=False)
    print_formatted_text(fragments, output=output)
    return stream.getvalue()


# --- sanitize_display ------------------------------------------------------------------------


def test_sanitize_removes_csi_osc_and_control_chars_keeps_newline_tab():
    dirty = "a\x1b[35;46mINJ\x1b[0mb\x1b]8;;http://evil\x1b\\link\x1b]8;;\x1b\\c\x07\x08d\ne\tx"
    clean = sanitize_display(dirty)
    assert "\x1b" not in clean
    assert "\x07" not in clean and "\x08" not in clean
    assert clean == "aINJb\rlinkcd\ne\tx" or clean == "aINJblinkcd\ne\tx"
    assert "\n" in clean and "\t" in clean


def test_sanitize_keeps_visible_text_including_markup_looking_brackets():
    text = "[bold]保留原文[/bold] ?[36m字面"
    assert sanitize_display(text) == text


# --- record_fragments ------------------------------------------------------------------------


def _record(role: str, body: str, status: str = "done") -> PresentationRecord:
    return PresentationRecord(key="k", role=role, title="标题", status=status, body=(body,))


def test_assistant_markdown_rendered_to_styled_fragments_not_plain_ansi_text():
    record = _record("assistant", "### 结论\n\n- 项目一\n- 项目二")
    fragments = record_fragments(record)
    text = _fragment_text(fragments)
    assert "结论" in text and "项目一" in text
    assert "\x1b" not in text, "styled fragments 的文本部分不得残留 ANSI 串"
    assert _fragment_styles(fragments), "Markdown 应产生非默认样式"


def test_untrusted_tool_output_sanitized_and_not_treated_as_markup():
    body = "\x1b[35;46m injected-sgr \n[bold]not markup[/bold]\n?[31m字面问号"
    record = _record("tool", body)
    fragments = record_fragments(record)
    text = _fragment_text(fragments)
    assert "\x1b" not in text, "注入的 CSI 不得进入 fragments 文本"
    assert "[bold]not markup[/bold]" in text, "方括号 markup 应作为字面文本保留"
    assert "?[31m字面问号" in text


def test_untrusted_output_never_yields_terminal_control_in_vt_pipeline():
    record = _record("tool", "\x1b[35;46m injected-sgr \n普通输出")
    stream = _render_to_vt(record_fragments(record))
    assert "\x1b[35;46m" not in stream, "注入 SGR 不得出现在最终可见输出"
    assert "?[35;46m" not in stream, "注入 SGR 不得退化为可见问号控制码"
    assert "普通输出" in stream


def test_styled_output_reaches_vt_pipeline_as_escape_sequences():
    record = _record("system", "状态正文")
    stream = _render_to_vt(record_fragments(record))
    assert "\x1b[" in stream, "程序样式应以真实 SGR 序列输出 (styled fragments 路径)"


def test_tool_diff_body_rendered_with_diff_syntax():
    body = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2"
    record = _record("tool", body)
    text = _fragment_text(record_fragments(record))
    assert "VALUE = 1" in text and "VALUE = 2" in text
    assert "\x1b" not in text


def test_error_record_header_carries_error_style():
    record = _record("error", "失败原因", status="failed")
    fragments = record_fragments(record)
    assert "ansired" in _fragment_styles(fragments)
    assert "失败原因" in _fragment_text(fragments)


def test_record_body_preserves_plain_text_without_rich_interpretation():
    record = _record("user", "普通 <文本> & 符号")
    assert "<文本> & 符号" in _fragment_text(record_fragments(record))


# --- preview / status ------------------------------------------------------------------------


def test_preview_fragments_show_truncation_fact():
    preview = ActivePreview(run_id="run-1", turn_index=0, text="部分文本", truncated=True)
    text = _fragment_text(preview_fragments(preview))
    assert "部分文本" in text and "已截断" in text
    assert "run-1" in text and "turn 0" in text


def test_status_fragments_none_metrics_show_unavailable_not_zero():
    snapshot = StatusSnapshot(model=None, workspace=None, phase="executing")
    text = _fragment_text(status_fragments(snapshot))
    assert "不可得" in text
    assert "累计输入: 不可得" in text and "累计输出: 不可得" in text
    assert "上下文峰值" not in text


def test_status_fragments_label_metrics_as_cumulative():
    snapshot = StatusSnapshot(
        model="fake", workspace="w", phase="idle", turn_index=2,
        input_tokens=350, output_tokens=40,
    )
    text = _fragment_text(status_fragments(snapshot))
    assert "累计输入: 350" in text and "累计输出: 40" in text


def test_status_fragments_normal_mode_shows_check_not_enabled():
    snapshot = StatusSnapshot(model="fake", verification_run_id=None, check_status=None)
    assert "未启用检查" in _fragment_text(status_fragments(snapshot))


def test_status_fragments_check_mode_shows_run_and_scope():
    snapshot = StatusSnapshot(
        model="fake", verification_run_id="vrun", check_status="check-1=passed/current",
        change_scope="累计改动口径: 相对启动基线 1 项变更",
    )
    text = _fragment_text(status_fragments(snapshot))
    assert "vrun" in text and "check-1=passed/current" in text
    assert "累计改动口径" in text
