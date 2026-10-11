"""紧凑工具显示聚焦检查 (tui-compact-interaction 票据 01; 契约 C1/C2)。

事件驱动场景用固定 fake 事件序列: 内核事件的真实形状 (core/events.py),
经 AgentEventReceived 包装 — 与 application.ApplicationSession._on_core_event
的投递形态一致 (application.py:315)。会话组合场景用真实 ApplicationSession
+ FakeModel + 真实 glob 工具, 构造口径与 tests/test_application.py 相同;
不接真实模型、不启动本地模型服务、不执行检查模式。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("color_env")

from cicada.application import (
    AgentEventReceived,
    ApplicationConfig,
    open_application,
)
from cicada.core.events import (
    AssistantCompleted,
    RunFinished,
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
    INITIAL_OUTPUT_LINES,
    MORE_OUTPUT_STEP,
    ToolDocument,
)

RUN = "run-7"

# fake 事件场景常数 (本文件的固定序列事实, 不是模型行为):
# 会话场景剧本 = 1 次用户请求, 2 个模型轮次 (工具轮 + 回答轮), 1 次工具调用
SCENARIO_MODEL_CALLS = 2
SCENARIO_TOOL_CALLS = 1

# 规格配色 (独立字面量来源: tui-aesthetics spec.md §B)
ACCENT_RUNNING = "\x1b[38;2;129;162;190m"  # #81a2be
ACCENT_ERROR = "\x1b[38;2;204;102;102m"    # #cc6666
OUTPUT_FG = "\x1b[38;2;168;168;168m"       # #a8a8a8


def feed(doc: ToolDocument, *events: object) -> None:
    """按应用投递形态喂事件 (AgentEventReceived 包装)。"""
    for event in events:
        doc.apply(AgentEventReceived(event))


def plain_lines(doc: ToolDocument, width: int) -> list[str]:
    return [strip_terminal_sequences(line) for line in doc.render_lines(width)]


def started_card(doc: ToolDocument, *, call: str = "t1", name: str = "read", run: str = RUN, turn: int = 0) -> None:
    feed(doc, RunStarted(run), TurnStarted(run, turn), ToolStarted(run, call, name))


def live_key(call: str = "t1", run: str = RUN, turn: int = 0) -> tuple[str, str, int, str]:
    return ("live", run, turn, call)


def completed_card(doc: ToolDocument, *, call: str = "t1", name: str = "read",
                   content: str = "ok", is_error: bool = False,
                   details: dict | None = None) -> None:
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, call, name),
        ToolCompleted(RUN, ToolResult(call_id=call, name=name, content=content,
                                      is_error=is_error, details=details)),
    )


# --- C1: 开始→终态原位更新与配对 ------------------------------------------------------------------


def test_started_then_completed_updates_same_entry():
    doc = ToolDocument()
    started_card(doc)
    assert len(doc.entries) == 1
    entry = doc.entries[0]
    assert (entry.run_id, entry.turn_index, entry.call_id) == (RUN, 0, "t1")
    assert entry.status == "running"

    running_plain = plain_lines(doc, 80)
    assert len(running_plain) == 2  # 前导空行 + 单工具标题行 (收起)
    assert any("· 进行中" in line for line in running_plain)

    feed(doc, ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="hello")))
    # 同一条目: 数量与身份不变, 状态与输出原位更新
    assert len(doc.entries) == 1
    done = doc.entries[0]
    assert (done.run_id, done.turn_index, done.call_id) == (RUN, 0, "t1")
    assert done.status == "success" and done.output_text == "hello"

    collapsed = plain_lines(doc, 80)
    assert not any("hello" in line for line in collapsed)  # 收起不显示输出
    doc.toggle_tool(live_key())
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
    # 两轮各成一组 (组键含 turn_index); 展开两组后输出各出现一次
    assert len(doc.groups) == 2
    doc.toggle_tool(live_key(turn=0))
    doc.toggle_tool(live_key(turn=1))
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


def test_missing_start_cancelled_result_annotated():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolCompleted(
            RUN,
            ToolResult(call_id="t2", name="powershell", content="cancelled during execution", is_error=True),
        ),
    )
    entry = doc.entries[0]
    assert entry.status == "error" and entry.cancelled and entry.is_error
    collapsed = plain_lines(doc, 80)
    assert any("· 已取消" in line for line in collapsed)  # 收起可辨认 (C2)
    doc.toggle_tool(live_key(call="t2"))
    assert any("cancelled during execution" in line for line in plain_lines(doc, 80))


def test_error_result_shows_failure_suffix_without_cancel():
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
    assert any("· 失败" in line for line in plain_lines(doc, 80))
    doc.toggle_tool(live_key())
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
    doc.toggle_group(("live", RUN, 0))
    doc.toggle_group(("live", RUN, 1))
    lines = plain_lines(doc, 80)
    positions = [
        next(index for index, line in enumerate(lines) if verb in line)
        for verb in ("读取", "查找文件", "修改文件")
    ]
    assert positions == sorted(positions), "条目按到达序稳定排列"


# --- C1: 组登记与封存 (等待/未确认不伪造完成) --------------------------------------------------------


def test_assistant_completed_registers_whole_group_waiting():
    doc = ToolDocument()
    message = AssistantMessage(
        text="",
        tool_calls=(
            ToolCall(id="t1", name="read", arguments_json=json.dumps({"path": "a.py"})),
            ToolCall(id="t2", name="read", arguments_json=json.dumps({"path": "b.py"})),
        ),
    )
    feed(doc, RunStarted(RUN), TurnStarted(RUN, 0), AssistantCompleted(RUN, message))
    # 登记整组: 两个等待态子项, 组已存在 (不漏不重复)
    assert len(doc.groups) == 1
    assert doc.groups[0].call_ids == ("t1", "t2")
    assert [entry.status for entry in doc.entries] == ["waiting", "waiting"]
    header = next(line for line in plain_lines(doc, 80) if line.strip())
    assert "读取文件 · 2 次" in header and "2 进行中" in header  # 未终结计数 (C2)
    feed(doc, ToolCompleted(RUN, ToolResult(call_id="t2", name="read", content="b")))
    # ToolCompleted 只更新对应子项; t1 保持等待
    assert [entry.status for entry in doc.entries] == ["waiting", "success"]
    feed(doc, RunFinished(RUN, "stop"))
    # 请求封存: 无结果子项标未确认, 不伪造完成
    assert [entry.status for entry in doc.entries] == ["unconfirmed", "success"]
    header = next(line for line in plain_lines(doc, 80) if line.strip())
    assert "1 未确认" in header


def test_mutation_unknown_counted_as_unconfirmed():
    doc = ToolDocument()
    completed_card(
        doc, name="edit", content="ok",
        details={"mutation": {"state": "unknown"}},
    )
    entry = doc.entries[0]
    assert entry.status == "success"  # 结构化 unknown 不改终态事实
    assert any("· 未确认" in line for line in plain_lines(doc, 80))  # 收起按未确认归类


# --- C2: 紧凑状态表与有界详情 ------------------------------------------------------------------------


def test_collapsed_group_is_single_line_with_counts():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "read"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="r1")),
        ToolStarted(RUN, "t2", "edit"),
        ToolCompleted(RUN, ToolResult(call_id="t2", name="edit", content="boom", is_error=True)),
    )
    lines = plain_lines(doc, 80)
    body = [line for line in lines if line.strip()]
    assert len(body) == 1, "多工具组收起只占一物理行"
    assert "调用工具 · 2 次" in body[0] and "1 失败" in body[0]
    assert body[0].startswith("+")


def test_group_expand_lists_children_without_detail():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "read"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="row-1")),
        ToolStarted(RUN, "t2", "read"),
        ToolCompleted(RUN, ToolResult(call_id="t2", name="read", content="row-2")),
    )
    doc.toggle_group(("live", RUN, 0))
    lines = plain_lines(doc, 80)
    body = [line for line in lines if line.strip()]
    assert len(body) == 3  # 组头 + 两个子工具短行
    assert "读取文件 · 2 次" in body[0]
    assert not any("row-1" in line for line in lines)  # 打开组只露列表, 不自动显示正文
    doc.toggle_tool(live_key(call="t2"))
    lines = plain_lines(doc, 80)
    assert any("row-2" in line for line in lines) and not any("row-1" in line for line in lines)


def test_detail_budget_paging_and_reset():
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
    assert len(collapsed) == 2  # 空行 + 标题; 收起不 wrap 大结果
    collapsed_plain = [strip_terminal_sequences(line) for line in collapsed]
    assert not any("line-1" in line for line in collapsed_plain)

    key = live_key()
    doc.toggle_tool(key)
    opened = plain_lines(doc, 200)
    # 详情: 标题 + 至多 6 输出物理行 + 「更多」分页动作
    assert len(opened) == 1 + 1 + INITIAL_OUTPUT_LINES + 1
    assert any("line-6" in line for line in opened)
    assert not any("line-7" in line for line in opened)
    # 工具自身截断脚注在未展示的行里: 初始预览不伪造出现
    assert not any("[truncated=true" in line for line in opened)
    assert opened[-1].strip() == "更多"

    doc.tool_output_more(key)  # 同一展开过程累加预算 (+20)
    more = plain_lines(doc, 200)
    assert any("line-25" in line for line in more)  # 26 行全部披露 (6+20 >= 26)
    assert any("[truncated=true reason=lines next_offset=26]" in line for line in more)
    assert not any("line-26" in line for line in more)  # 不显示未收到的字节
    assert more[-1].strip() == "收起详情"

    doc.tool_output_reset(key)  # 恢复初次预览 (与刚打开时同一布局)
    assert plain_lines(doc, 200) == opened

    doc.toggle_tool(key)  # 收起
    assert doc.render_lines(200) == collapsed
    # 切换只改展示: 条目快照逐字段不变
    assert doc.entries == entries_before


def test_reopen_detail_resets_budget():
    doc = ToolDocument()
    completed_card(doc, content="\n".join(f"row-{i}" for i in range(30)))
    key = live_key()
    doc.toggle_tool(key)
    doc.tool_output_more(key)
    assert any("row-25" in line for line in plain_lines(doc, 80))
    doc.toggle_tool(key)  # 收起
    doc.toggle_tool(key)  # 再打开: 预算回初始 6 行 (C2)
    reopened = plain_lines(doc, 80)
    assert any("row-5" in line for line in reopened)
    assert not any("row-6" in line for line in reopened)


def test_params_limited_to_two_physical_rows():
    doc = ToolDocument()
    message = AssistantMessage(
        text="",
        tool_calls=(ToolCall(
            id="t1", name="powershell",
            arguments_json=json.dumps({"command": "Get-Content 很长的中文参数 " * 20}),
        ),),
    )
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        AssistantCompleted(RUN, message),
        ToolStarted(RUN, "t1", "powershell"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="powershell", content="first-output")),
    )
    doc.toggle_tool(live_key())
    lines = plain_lines(doc, 40)
    assert any("参数:" in line for line in lines)
    output_row = next(index for index, line in enumerate(lines) if "first-output" in line)
    assert output_row <= 4  # 空行+标题+至多 2 参数物理行之后即是输出


def test_ctrl_o_unifies_groups_and_resets_detail():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "read"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="\n".join(f"a{i}" for i in range(30)))),
        ToolStarted(RUN, "t2", "glob"),
        ToolCompleted(RUN, ToolResult(call_id="t2", name="glob", content="g1")),
        TurnStarted(RUN, 1),
        ToolStarted(RUN, "t3", "edit"),
        ToolCompleted(RUN, ToolResult(call_id="t3", name="edit", content="e1")),
    )
    collapsed = doc.render_lines(80)
    doc.toggle_group(("live", RUN, 0))
    doc.toggle_tool(live_key(call="t1"))
    doc.tool_output_more(live_key(call="t1"))
    assert doc.toggle_expand() is True  # Ctrl+O: 统一全部组/单工具为展开
    plain = plain_lines(doc, 80)
    # C2: 组打开只露列表, 子工具详情重置收起 (逐项覆盖被统一)
    assert any("读取" in line for line in plain) and any("查找文件" in line for line in plain)
    assert not any("a0" in line for line in plain)
    assert all(group.expanded for group in doc.groups)
    # 单工具组 (turn 1) 展开即详情, 最多初始预览
    assert any("e1" in line for line in plain)
    doc.toggle_tool(live_key(call="t1"))  # 再开详情: 预算已回初始 6 行
    assert any("a5" in line for line in plain_lines(doc, 80))
    assert not any("a6" in line for line in plain_lines(doc, 80))
    assert doc.toggle_expand() is False  # 再次统一: 全部收起
    assert doc.render_lines(80) == collapsed


def test_external_ansi_in_output_is_sanitized():
    doc = ToolDocument()
    started_card(doc)
    feed(
        doc,
        ToolCompleted(
            RUN, ToolResult(call_id="t1", name="read", content="\x1b[31mred\x1b[0m text")
        ),
    )
    doc.toggle_tool(live_key())
    rendered = doc.render_lines(80)
    assert not any("\x1b[31m" in line for line in rendered)
    assert any("red text" in line for line in plain_lines(doc, 80))


# --- 外观纪律 (tui-aesthetics 规格 §B / 紧凑交互 C2: 无大背景, 标记一列) -------------------------------


def test_collapsed_rows_have_no_card_background():
    doc = ToolDocument()
    completed_card(doc)
    completed_card(doc, call="t2", name="edit", content="boom", is_error=True)
    for line in doc.render_lines(80):
        assert "\x1b[48;" not in line, "不再为每个工具绘制大背景"


def test_marker_and_suffix_accents():
    doc = ToolDocument()
    completed_card(doc, call="t1", name="edit", content="boom", is_error=True)
    lines = doc.render_lines(80)
    title = lines[1]
    plain = strip_terminal_sequences(title)
    assert plain.startswith("+ ") and "· 失败" in plain
    assert ACCENT_ERROR in title  # 失败类别着色 (标记与后缀)

    pending = ToolDocument()
    started_card(pending, name="glob")
    running_title = pending.render_lines(80)[1]
    assert "· 进行中" in strip_terminal_sequences(running_title)
    assert ACCENT_RUNNING in running_title


def test_output_dimmed_and_paging_hint():
    doc = ToolDocument()
    completed_card(doc, content="\n".join(f"line-{i}" for i in range(20)))
    doc.toggle_tool(live_key())
    lines = doc.render_lines(80)
    output_rows = [line for line in lines if "line-0" in strip_terminal_sequences(line)]
    assert output_rows and all(OUTPUT_FG in line for line in output_rows)
    paging = strip_terminal_sequences(lines[-1])
    assert paging.strip() == "更多"


# --- 相关 P11: 可见列宽不破版 (含极窄降级) -------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 8, 40, 160])
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


def test_narrow_header_keeps_counts_as_symbols():
    doc = ToolDocument()
    feed(
        doc,
        RunStarted(RUN),
        TurnStarted(RUN, 0),
        ToolStarted(RUN, "t1", "read"),
        ToolCompleted(RUN, ToolResult(call_id="t1", name="read", content="r1")),
        ToolStarted(RUN, "t2", "edit"),
        ToolCompleted(RUN, ToolResult(call_id="t2", name="edit", content="boom", is_error=True)),
    )
    wide = next(line for line in plain_lines(doc, 80) if line.strip())
    assert "1 失败" in wide
    narrow = next(line for line in plain_lines(doc, 14) if line.strip())
    assert "!1" in narrow  # 极窄: 计数退化为符号形式, 不被标题挤掉 (C2)


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
    # 12 个模块文件: 让 glob 输出超过 6 行初始预览, 收起/展开呈现可观察差异
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
    # 参数摘要来自真实 AssistantCompleted 的 arguments_json; 单工具收起为一行标题
    lines = plain_lines(doc, 80)
    assert any("**/*.py" in line for line in lines)
    assert len([line for line in lines if line.strip()]) == 1

    # fake 场景常数: 一次请求、一次工具调用; 展开/收起前后不得变化
    model = session._app.agent._model
    assert isinstance(model, FakeModel)
    collapsed = doc.render_lines(80)
    updates_before = len(updates)
    doc.toggle_expand()  # Ctrl+O: 单工具直接开详情 (6 行初始预览 + 分页动作)
    expanded = doc.render_lines(80)
    expanded_plain = [strip_terminal_sequences(line) for line in expanded]
    assert any("mod_00.py" in line for line in expanded_plain)  # 初始 6 行预览
    assert not any("sample.py" in line for line in expanded_plain)  # 字母序第 13 行, 未披露
    assert any("更多" in line for line in expanded_plain)  # 13 个 py 文件超出 6 行预览
    key = ("live", "run-1", 0, "c1")
    doc.tool_output_more(key)  # 「更多」每次 +20 输出物理行 (长路径折行, 两次必全覆盖)
    doc.tool_output_more(key)
    assert any("sample.py" in line for line in plain_lines(doc, 80))
    doc.tool_output_reset(key)  # 「收起详情」恢复 6 行初始预览
    reset_plain = plain_lines(doc, 80)
    assert not any("sample.py" in line for line in reset_plain)
    assert any("更多" in line for line in reset_plain)
    doc.toggle_expand()  # 统一收起后渲染回到最初收起态
    assert doc.render_lines(80) == collapsed
    assert len(model.requests) == SCENARIO_MODEL_CALLS
    tool_started = [
        update.event
        for update in updates
        if isinstance(update, AgentEventReceived) and isinstance(update.event, ToolStarted)
    ]
    assert len(tool_started) == SCENARIO_TOOL_CALLS
    assert len(updates) == updates_before, "展开/收起与渲染期间不得产生新应用事件"
