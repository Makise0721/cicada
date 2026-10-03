"""票据 03 聚焦检查: 工具原位卡片与输出展开 (P04/P05/相关 P11)。

事件驱动场景用固定 fake 事件序列: 内核事件的真实形状 (core/events.py),
经 AgentEventReceived 包装 — 与 application.ApplicationSession._on_core_event
的投递形态一致 (application.py:315)。会话组合场景用真实 ApplicationSession
+ FakeModel + 真实 glob 工具, 构造口径与 tests/test_application.py 相同;
不接真实模型、不启动 Ollama、不执行检查模式。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from cicada.application import (
    AgentEventReceived,
    ApplicationConfig,
    open_application,
)
from cicada.core.events import (
    AssistantCompleted,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.core.messages import AssistantMessage, ToolCall, ToolResult
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.tool_view import (
    PREVIEW_LINES,
    TOOL_BG_ERROR,
    TOOL_BG_PENDING,
    TOOL_BG_SUCCESS,
    ToolDocument,
)

RUN = "run-7"

# fake 事件场景常数 (本文件的固定序列事实, 不是模型行为):
# 会话场景剧本 = 1 次用户请求, 2 个模型轮次 (工具轮 + 回答轮), 1 次工具调用
SCENARIO_MODEL_CALLS = 2
SCENARIO_TOOL_CALLS = 1


def feed(doc: ToolDocument, *events: object) -> None:
    """按应用投递形态喂事件 (AgentEventReceived 包装)。"""
    for event in events:
        doc.apply(AgentEventReceived(event))


def plain_lines(doc: ToolDocument, width: int) -> list[str]:
    return [strip_terminal_sequences(line) for line in doc.render_lines(width)]


def started_card(doc: ToolDocument, *, call: str = "t1", name: str = "read", run: str = RUN, turn: int = 0) -> None:
    feed(doc, RunStarted(run), TurnStarted(run, turn), ToolStarted(run, call, name))


# --- P04: 开始→终态原位更新 ------------------------------------------------------------------


def test_started_then_completed_updates_same_entry():
    doc = ToolDocument()
    started_card(doc)
    assert len(doc.entries) == 1
    entry = doc.entries[0]
    assert (entry.run_id, entry.turn_index, entry.call_id) == (RUN, 0, "t1")
    assert entry.status == "running"

    running_lines = doc.render_lines(80)
    assert any(TOOL_BG_PENDING in line for line in running_lines)
    assert not any(TOOL_BG_SUCCESS in line for line in running_lines)

    feed(doc, ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="hello")))
    # 同一条目: 数量与身份不变, 状态与输出原位更新
    assert len(doc.entries) == 1
    done = doc.entries[0]
    assert (done.run_id, done.turn_index, done.call_id) == (RUN, 0, "t1")
    assert done.status == "success" and done.output_text == "hello"

    done_lines = doc.render_lines(80)
    assert any(TOOL_BG_SUCCESS in line for line in done_lines)
    assert not any(TOOL_BG_PENDING in line for line in done_lines)
    assert any("hello" in line for line in plain_lines(doc, 80))


def test_title_shows_args_summary_from_assistant_completed():
    doc = ToolDocument()
    message = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="t1", name="read", arguments_json=json.dumps({"path": "src/app.py"})),),
    )
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        AssistantCompleted(RUN, message),
        ToolStarted(RUN, "t1", "read"),
    )
    assert any("src/app.py" in line for line in plain_lines(doc, 80))


def test_reused_call_id_across_turns_not_merged():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "glob"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="glob", content="first")),
        TurnStarted(RUN, 1),
        ToolStarted(RUN, "t1", "glob"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="glob", content="second")),
    )
    entries = doc.entries
    assert len(entries) == 2
    assert [(entry.turn_index, entry.output_text) for entry in entries] == [(0, "first"), (1, "second")]
    lines = plain_lines(doc, 80)
    assert sum("first" in line for line in lines) == 1
    assert sum("second" in line for line in lines) == 1


def test_missing_start_reused_call_id_creates_second_card():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "edit"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="edit", content="first")),
        TurnStarted(RUN, 1),
        # 第二轮缺少 ToolStarted 的闭合结果: 建新终态卡, 不并入第一轮
        ToolCompleted(RUN, ToolResult(call_id="t1", name="edit", content="second")),
    )
    entries = doc.entries
    assert len(entries) == 2
    assert (entries[0].turn_index, entries[0].output_text) == (0, "first")
    assert (entries[1].turn_index, entries[1].output_text) == (1, "second")
    assert entries[1].status == "success" and not entries[1].cancelled


@pytest.mark.parametrize(
    "content",
    ["cancelled before execution", "cancelled during execution"],
)
def test_missing_start_cancelled_result_annotated(content: str):
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolCompleted(
            RUN,
            ToolResult(call_id="t2", name="powershell", content=content, is_error=True),
        ),
    )
    entry = doc.entries[0]
    assert entry.status == "error" and entry.cancelled and entry.is_error
    rendered = doc.render_lines(80)
    assert any(TOOL_BG_ERROR in line for line in rendered)
    assert not any(TOOL_BG_PENDING in line for line in rendered)
    assert any(content in line for line in plain_lines(doc, 80))


def test_error_result_without_cancel_keeps_error_background_only():
    doc = ToolDocument()
    started_card(doc, call="t1", name="edit")
    feed(
        doc,
        ToolCompleted(
            RUN, ToolResult(call_id="t1", name="edit", content="tool raised: boom", is_error=True)
        ),
    )
    entry = doc.entries[0]
    assert entry.status == "error" and not entry.cancelled
    assert any("tool raised: boom" in line for line in plain_lines(doc, 80))


def test_stale_events_after_terminal_recorded_not_fatal():
    doc = ToolDocument()
    started_card(doc)
    feed(doc, ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="ok")))
    feed(doc, ToolStarted(RUN, "t1", "read"))  # 迟到开始: 不能恢复已完成条目
    feed(doc, ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="dup")))
    assert len(doc.entries) == 1
    assert doc.entries[0].output_text == "ok"
    assert doc.ui_failures, "有界失败应记录 ui_failures 而非抛异常"


def test_cards_keep_arrival_order_and_in_place_update_does_not_reorder():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "read"),
        ToolStarted(RUN, "t2", "glob"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="r1")),
        TurnStarted(RUN, 1),
        ToolStarted(RUN, "t3", "edit"),
        ToolCompleted(RUN, ToolResult(call_id="t2", name="glob", content="g1")),
    )
    assert [entry.call_id for entry in doc.entries] == ["t1", "t2", "t3"]
    lines = plain_lines(doc, 80)
    positions = [
        next(index for index, line in enumerate(lines) if name in line)
        for name in ("read", "glob", "edit")
    ]
    assert positions == sorted(positions), "条目按到达序稳定排列"


# --- P05: 默认 10 行预览与 Ctrl+O 展开/收起只改展示 --------------------------------------------


def test_preview_default_10_lines_and_expand_toggle_display_only():
    doc = ToolDocument()
    body = [f"line-{index}" for index in range(1, 26)]  # 25 行输出
    body.append("[truncated=true reason=lines next_offset=26]")  # read 工具截断脚注形态
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "read"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="\n".join(body))),
    )
    entries_before = doc.entries

    collapsed = doc.render_lines(200)
    # 布局: 空行 + 上背景行 + 标题 + 前 10 行 + 折叠提示 + 下背景行
    assert len(collapsed) == 1 + 1 + 1 + PREVIEW_LINES + 1 + 1
    collapsed_plain = [strip_terminal_sequences(line) for line in collapsed]
    assert not any("line-11" in line for line in collapsed_plain)
    # 工具自身截断脚注在未展示的行里: 收起时不伪造出现
    assert not any("[truncated=true" in line for line in collapsed_plain)
    hint_rows = [line for line in collapsed_plain if "还有" in line and "Ctrl+O" in line]
    assert len(hint_rows) == 1 and "16" in hint_rows[0]  # 26 行收到, 显示 10 行

    doc.toggle_expand()
    assert doc.expanded
    expanded = doc.render_lines(200)
    # 展开: 已收到的 26 行全部显示, 不显示任何未收到的字节
    assert len(expanded) == 1 + 1 + 1 + len(body) + 1
    expanded_plain = [strip_terminal_sequences(line) for line in expanded]
    assert any("line-25" in line for line in expanded_plain)
    assert any("[truncated=true reason=lines next_offset=26]" in line for line in expanded_plain)
    assert not any("line-26" in line for line in expanded_plain)

    doc.toggle_expand()
    assert not doc.expanded
    assert doc.render_lines(200) == collapsed
    # 切换只改展示: 条目快照逐字段不变
    assert doc.entries == entries_before


def test_external_ansi_in_output_is_sanitized():
    doc = ToolDocument()
    started_card(doc)
    feed(
        doc,
        ToolCompleted(
            RUN, ToolResult(call_id="t1", name="read", content="\x1b[31mred\x1b[0m text")
        ),
    )
    rendered = doc.render_lines(80)
    assert not any("\x1b[31m" in line for line in rendered)
    assert any("red text" in line for line in plain_lines(doc, 80))


# --- 相关 P11: 可见列宽不破版 ------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 2, 3, 8, 40, 80, 120, 160])
def test_every_rendered_line_within_width(width: int):
    doc = ToolDocument()
    content = "\n".join(
        [
            "短行",
            "中" * 30,
            "x" * 200,
            "",
            "\x1b[31m外部 ANSI 应被清理\x1b[0m 中文 English mix",
            "🎉 emoji 行",
        ]
    )
    message = AssistantMessage(
        text="",
        tool_calls=(
            ToolCall(id="t1", name="powershell", arguments_json=json.dumps({"command": "Get-Content 很长的中文参数" * 5})),
        ),
    )
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        AssistantCompleted(RUN, message),
        ToolStarted(RUN, "t1", "powershell"),
    )
    assert all(visible_width(line) <= width for line in doc.render_lines(width))
    feed(
        doc,
        ToolCompleted(RUN, ToolResult(call_id="t1", name="powershell", content=content)),
    )
    assert all(visible_width(line) <= width for line in doc.render_lines(width))
    doc.toggle_expand()
    assert all(visible_width(line) <= width for line in doc.render_lines(width))


# --- 共享 fake 事件场景: 真实 ApplicationSession 驱动 ---------------------------------------------


def _prompt_info(root: Path) -> PromptInfo:
    return PromptInfo(
        version="tool-view-test",
        workspace=root,
        instructions_path=None,
        instructions_sha256=None,
        prompt_sha256=hashlib.sha256(b"tool-view-test").hexdigest(),
    )


def _script(*entries: dict) -> list:
    from cicada.__main__ import parse_script

    return parse_script(json.dumps(list(entries)))


async def test_session_cards_update_and_toggle_keeps_request_and_tool_counts(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    # 12 个模块文件: 让 glob 输出超过 10 行, 收起/展开呈现可观察差异
    for index in range(12):
        (root / f"mod_{index:02d}.py").write_text(f"# mod {index}\n", encoding="utf-8")
    # glob 依赖 GitInventory: 与 tests/test_application.py 的 _repo 同口径建仓
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "baseline"],
        check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"},
    )
    config = ApplicationConfig(
        workspace=root,
        system_prompt="tool view test",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(
            FakeModel(
                _script(
                    {
                        "tool_calls": [
                            {"id": "c1", "name": "glob", "arguments": {"pattern": "**/*.py"}}
                        ],
                        "tool_use": True,
                    },
                    {"text": "工具完成后的回答", "stop": True},
                )
            )
        ),
        max_turns=6,
        model_label="fake",
    )
    session = await open_application(config)
    doc = ToolDocument()
    updates: list = []

    def on_update(update) -> None:
        updates.append(update)
        doc.apply(update)

    session.subscribe(on_update)
    try:
        await session.execute("查找 py 文件")
    finally:
        await session.aclose()

    entries = doc.entries
    assert len(entries) == 1
    assert (entries[0].run_id, entries[0].turn_index, entries[0].call_id) == ("run-1", 0, "c1")
    assert entries[0].status == "success" and entries[0].name == "glob"
    # 参数摘要来自真实 AssistantCompleted 的 arguments_json
    lines = plain_lines(doc, 80)
    assert any("**/*.py" in line for line in lines)
    assert any(TOOL_BG_SUCCESS in line for line in doc.render_lines(80))

    # fake 场景常数: 一次请求、一次工具调用; 展开/收起前后不得变化
    model = session._app.agent._model
    assert isinstance(model, FakeModel)
    collapsed = doc.render_lines(80)
    updates_before = len(updates)
    doc.toggle_expand()
    expanded = doc.render_lines(80)
    doc.toggle_expand()
    assert doc.render_lines(80) == collapsed
    assert expanded != collapsed
    assert len(model.requests) == SCENARIO_MODEL_CALLS
    tool_started = [
        update.event
        for update in updates
        if isinstance(update, AgentEventReceived) and isinstance(update.event, ToolStarted)
    ]
    assert len(tool_started) == SCENARIO_TOOL_CALLS
    assert len(updates) == updates_before, "展开/收起与渲染期间不得产生新应用事件"
