"""票据 02 显示文档单元检查: 原位流式更新、终结校准、条目归属与排版.

覆盖票据验收 P02 (长文本原位更新、校准无重复/丢失)、P03 的投影部分
(同批多轮、迟到增量、length、错误、取消的条目归属) 与约束 ⑤⑥
(每行 visible_width <= width; 外部文本先显示清理)。
事件直接喂 Transcript, 不经 session (组合层行为在 test_conversation_view.py)。
"""

from __future__ import annotations

import pytest

from cicada.core.messages import AssistantMessage
from cicada.core.ports import ModelMetrics
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.rendering import sanitize_display
from cicada.ui.transcript import (
    LENGTH_NOTE,
    AssistantEntry,
    CancelledEntry,
    ErrorEntry,
    SystemEntry,
    Transcript,
    UserEntry,
)


def _long_text(lines: int = 20, per_line: int = 240) -> str:
    """超过旧 12 行/4096 bytes 阈值的确定性长文本。"""
    return "\n".join(f"第{i}行: {'流式回答内容。' * (per_line // 7)}" for i in range(lines))


def test_delta_accumulates_into_single_entry():
    t = Transcript()
    t.stream_assistant_delta("run-1", 0, "第一段。")
    t.stream_assistant_delta("run-1", 0, "第二段,")
    t.stream_assistant_delta("run-1", 0, "第三段。")
    entries = t.entries()
    assert len(entries) == 1
    assert isinstance(entries[0], AssistantEntry)
    assert entries[0].text == "第一段。第二段,第三段。"
    assert entries[0].finalized is False
    # 渲染为无前缀正文 (Pi 样式: 前置空行 + Markdown 排版, 样式剥离后文本不变)
    rendered = [strip_terminal_sequences(line).strip() for line in t.render(80)]
    assert [x for x in rendered if x] == ["第一段。第二段,第三段。"]


def test_completed_calibrates_same_entry_without_duplicate():
    t = Transcript()
    t.stream_assistant_delta("run-1", 0, "部分预览")
    t.complete_assistant("run-1", 0, AssistantMessage(text="完整校准正文", stop_reason="stop"))
    entries = t.entries()
    assert len(entries) == 1  # 完成不二次追加
    assert entries[0].text == "完整校准正文"
    assert entries[0].finalized is True
    assert entries[0].stop_reason == "stop"


def test_completed_without_prior_delta_creates_entry():
    t = Transcript()
    t.complete_assistant("run-1", 0, AssistantMessage(text="无增量回答", stop_reason="stop"))
    assert len(t.entries()) == 1
    assert t.entries()[0].text == "无增量回答"


def test_late_delta_does_not_revive_completed_entry():
    t = Transcript()
    t.complete_assistant("run-1", 0, AssistantMessage(text="已完成", stop_reason="stop"))
    t.stream_assistant_delta("run-1", 0, "迟到的增量")
    entries = t.entries()
    assert len(entries) == 1
    assert entries[0].text == "已完成"  # 迟到增量被丢弃


def test_same_batch_multi_turn_belonging():
    """同批多轮: 上一轮完成后, 下一轮增量进入新条目, 互不混淆。"""
    t = Transcript()
    t.stream_assistant_delta("run-1", 0, "轮0增量")
    t.complete_assistant("run-1", 0, AssistantMessage(text="轮0完成", stop_reason="tool_use"))
    t.stream_assistant_delta("run-1", 1, "轮1增量")
    t.complete_assistant("run-1", 1, AssistantMessage(text="轮1完成", stop_reason="stop"))
    assistant_entries = [e for e in t.entries() if isinstance(e, AssistantEntry)]
    assert [(e.run_id, e.turn_index, e.text) for e in assistant_entries] == [
        ("run-1", 0, "轮0完成"),
        ("run-1", 1, "轮1完成"),
    ]


def test_long_answer_stays_single_entry_and_keeps_full_text():
    """P02: 长于旧 12 行/4096 bytes 阈值的回答原位更新同一条目, 正文不裁剪。"""
    t = Transcript()
    full = _long_text(lines=20, per_line=240)  # > 12 行且远超 4096 bytes
    assert len(full.encode("utf-8")) > 4096
    half = len(full) // 2
    t.stream_assistant_delta("run-1", 0, full[:half])
    t.stream_assistant_delta("run-1", 0, full[half:])
    t.complete_assistant("run-1", 0, AssistantMessage(text=full, stop_reason="stop"))
    entries = t.entries()
    assert len(entries) == 1
    assert entries[0].text == full  # 保存已收到的完整正文
    # 80 列下渲染远超 12 行且全文可回放 (无 12 行/64 行窗口裁剪):
    # Markdown 重排会移动空白与缩进 (Pi outputPad 1 列), 剥样式并
    # 归一空白后逐字符等于原文 (无内容丢失)
    lines = t.render(80)
    assert len(lines) > 12
    replay = "".join(strip_terminal_sequences(line) for line in lines)
    assert "".join(replay.split()) == "".join(full.split())


def test_length_stop_reason_annotated_on_entry():
    t = Transcript()
    t.complete_assistant("run-1", 0, AssistantMessage(text="被截断的回答", stop_reason="length"))
    lines = [strip_terminal_sequences(line) for line in t.render(80)]
    assert [x.strip() for x in lines if x.strip()][0] == "被截断的回答"
    assert lines[-1].strip() == LENGTH_NOTE
    # 非 length 完成不带标注
    t.complete_assistant("run-1", 1, AssistantMessage(text="正常", stop_reason="stop"))
    rendered = [strip_terminal_sequences(line) for line in t.render(80)]
    assert rendered.count(LENGTH_NOTE) == 1


def test_error_assistant_entry_without_text_renders_nothing():
    t = Transcript()
    t.complete_assistant(
        "run-1", 0, AssistantMessage(text="", stop_reason="error", error="provider exploded")
    )
    assert t.render(80) == []  # 正文事实由请求级错误条目表达, 不打空行


def test_streaming_placeholder_before_first_delta():
    entry = AssistantEntry("run-1", 0)
    # 条目统一自带 1 行前导空行 (规格 A): 占位期间与完成时间距一致
    assert entry.render(80) == ["", "…"]


def test_entries_have_uniform_one_blank_separator():
    """规格 A: 任意相邻条目之间恰好 1 行透明空行 (assistant→user 不再紧贴)。"""
    t = Transcript()
    t.complete_assistant("run-1", 0, AssistantMessage(text="回答", stop_reason="stop"))
    t.append_user("追问")
    t.record_request_result("error", "炸了")
    t.record_request_result("aborted", None)
    lines = [strip_terminal_sequences(line) for line in t.render(80)]
    text_rows = [i for i, line in enumerate(lines) if line.strip()]
    for left, right in zip(text_rows, text_rows[1:]):
        assert right - left >= 2, f"条目间缺透明空行: {lines[left]!r} -> {lines[right]!r}"


def test_document_start_has_no_leading_blank():
    """规格 A: 文档起始处的前导空行被丢弃, 首屏顶部不留白行。"""
    t = Transcript()
    t.append_user("首个消息")
    assert t.render(80)[0] != ""
    t2 = Transcript()
    t2.complete_assistant("run-1", 0, AssistantMessage(text="回答", stop_reason="stop"))
    assert t2.render(80)[0] != ""


def test_record_request_result_error_and_cancel_and_normal():
    t = Transcript()
    t.record_request_result("stop", None)
    assert t.entries() == []  # 正常完成安静, 无内部 phase 标签条目
    t.record_request_result("error", "provider exploded")
    t.record_request_result("aborted", None)
    entries = t.entries()
    assert isinstance(entries[0], ErrorEntry)
    assert isinstance(entries[1], CancelledEntry)
    lines = t.render(80)
    assert any("provider exploded" in line for line in lines)
    assert any("已取消" in line for line in lines)
    assert not any("finalizing" in line or "运行结束" in line for line in lines)


def test_record_request_result_error_without_text_is_truthful():
    t = Transcript()
    t.record_request_result("error", None)
    lines = t.render(80)
    assert any("错误" in line for line in lines)


def test_metrics_accumulate_and_none_is_not_zero():
    t = Transcript()
    assert t.metrics_summary() == "计量: 不可得"
    t.add_metrics(ModelMetrics(input_tokens=100, output_tokens=40))
    t.add_metrics(ModelMetrics(input_tokens=30, output_tokens=None))
    # 累计口径: 输出累计 40 是已收到的部分, 标注部分缺失; 不当作 0+40 也不虚构补齐
    assert t.metrics_summary() == "计量: 累计输入 130 · 累计输出 40（部分调用不可得）"
    t.add_metrics(None)  # 整条计量缺失: 两个字段的累计都成为部分值, 不清空也不虚构
    assert (
        t.metrics_summary()
        == "计量: 累计输入 130（部分调用不可得） · 累计输出 40（部分调用不可得）"
    )
    other = Transcript()
    other.add_metrics(None)
    assert other.metrics_summary() == "计量: 不可得"


@pytest.mark.usefixtures("color_env")
def test_user_entry_keeps_lines_and_prefixes_each_row():
    """04 主题化: 用户消息为 Pi 背景盒 (上下 1 行、左右 1 列), 原文换行保留。"""
    t = Transcript()
    t.append_user("第一行\n第二行")
    lines = t.render(80)
    plain = [strip_terminal_sequences(line).rstrip() for line in lines]
    assert plain == ["", " 第一行", " 第二行", ""]
    # 背景色作用于整行 (含填充)
    from cicada.ui.theme import USER_MESSAGE_BG, bg

    assert lines[0] == bg(USER_MESSAGE_BG, " " * 80)
    assert all(line.startswith("\x1b[48;2;52;53;65m") for line in lines)


def test_user_and_assistant_entries_in_order():
    t = Transcript()
    t.append_user("问题")
    t.stream_assistant_delta("run-1", 0, "回答")
    t.append_system("系统提示")
    assert [type(e) for e in t.entries()] == [UserEntry, AssistantEntry, SystemEntry]
    rendered = [strip_terminal_sequences(line).strip() for line in t.render(80)]
    assert [x for x in rendered if x] == ["问题", "回答", "系统提示"]


@pytest.mark.parametrize("width", [20, 80, 160])
@pytest.mark.parametrize(
    "text",
    [
        "中文长文本" * 50,
        "ascii words " * 60,
        "混合 mixed 中English文 混合" * 20,
        "\x1b[31m红色 ANSI\x1b[0m 注入" * 10,
        "\x1b]0;title\x07osc 注入" + "标题后正文" * 30,
    ],
)
def test_every_rendered_line_fits_width(width, text):
    """五种注入文本 × 窄/中/宽列宽行宽有界 (v1 四档列宽矩阵的中档 40
    与 80 同分支, 由 20/80 边界与 160 宽档代表)。"""
    t = Transcript()
    t.append_user(text)
    t.stream_assistant_delta("run-1", 0, text)
    t.append_system(text)
    t.append_error(text)
    for line in t.render(width):
        assert visible_width(line) <= width


def test_external_control_sequences_sanitized_not_executed():
    t = Transcript()
    t.stream_assistant_delta("run-1", 0, "\x1b[31m红\x1b[0m\x1b]0;title\x07正文\x07剩余")
    lines = t.render(80)
    joined = "\n".join(lines)
    assert "\x1b" not in joined
    assert sanitize_display("红正文剩余") in strip_terminal_sequences(joined) or "红" in joined


def test_newline_and_tab_survive_sanitization_but_tab_widens():
    t = Transcript()
    t.append_system("a\tb")
    lines = t.render(80)
    # tab 归一化为 3 空格由 renderer 完成; 文档层不丢内容
    body = [line for line in lines if line.strip()]
    assert len(body) == 1
    assert "a\tb" in body[0]
