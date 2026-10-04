"""T13 显示清理单元检查 (rendering.py 随旧渲染链裁剪后仅余 sanitize_display).

真实应用驱动的最终可见输出回归 (VT 屏) 在 tests/e2e/test_tui_verification.py;
显示副本的清理语义在此覆盖。
"""

from __future__ import annotations

from cicada.ui.rendering import sanitize_display


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


def test_sanitize_strips_unterminated_escape_payloads():
    dirty = "ok\x1b[35;46m被截断的参数段\x1b]8;;payload"
    clean = sanitize_display(dirty)
    assert "\x1b" not in clean
    assert clean.startswith("ok")
