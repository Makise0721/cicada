"""票据 03 坐标/键位最小回归: 底栏思考徽标点击与 Shift+Tab (契约 C5/C6).

只覆盖票据 Validation 点名而现有检查无法证明的缺口: 实际呈现徽标的帧命中
(完整保留才可点, 被裁掉无 hit)、点击与键盘共用同一同步准入入口、忙碌静默
不排队、草稿与规范历史不受影响。徽标文案降级 (窄窗缩写/低高度省略/plain)
的一次性核对见票据回执 preview; 原生键位/观感人工核对归 04。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.application import ReasoningEffortChanged, open_application
from cicada.ui.commands import SlashCommandMenu
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.terminal_io import FakeTerminal, PointerEvent
from tests.ui.vt_screen import VtScreen
from test_application import _repo
from test_conversation_view import _wait_until
from test_deepseek_application import config as ds_config, response as ds_response


def _transport(bodies):
    def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=ds_response(None, "答案"))
    return httpx.MockTransport(handle)


class EffortHarness:
    """生产组合 headless 驱动: 与 run_tui_session 同一接线 (03 全量注入)。"""

    def __init__(self, session, *, columns: int = 60, rows: int = 16) -> None:
        self.session = session
        self.terminal = FakeTerminal(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.terminal.feed_screen = self.vt.feed
        from cicada.ui.conversation_view import build

        self.bundle = build(session, on_exit=lambda: self.frontend.request_exit(),
                            on_update=lambda: self.frontend.request_render())
        self.view = self.bundle["view"]
        self.view.editor.focused = True
        self.frontend = MainScreenFrontend(
            self.terminal,
            document_lines=self.bundle["document_lines"],
            layout_document=self.bundle["layout_document"],
            on_display_action=self.bundle["on_display_action"],
            on_shift_tab=self.bundle["on_shift_tab"],
            footer_badge_spans=self.bundle["footer_badge_spans"],
            editor=self.bundle["editor"],
            on_submit=self.bundle["on_submit"],
            on_escape=self.bundle["on_escape"],
            on_ctrl_c=self.bundle["on_ctrl_c"],
            on_exit=self.bundle["on_exit"],
            footer_lines=self.bundle["footer_lines"],
            command_menu=SlashCommandMenu(),
            status_lines=self.bundle["status_lines"],
            model_menu=self.bundle["model_menu"],
            on_model_confirm=self.bundle["on_model_confirm"],
            modal=self.bundle["modal"],
        )
        self.runner: asyncio.Task | None = None

    async def start(self) -> None:
        self.runner = asyncio.ensure_future(self.frontend.run_async())
        await self.settle()

    async def settle(self, ticks: int = 3) -> None:
        await asyncio.sleep(0.05)
        for _ in range(ticks):
            await asyncio.sleep(0)

    async def stop(self) -> list[str]:
        self.frontend.request_exit()
        return await asyncio.wait_for(self.runner, timeout=5)

    async def wait_idle(self) -> bool:
        return await _wait_until(
            lambda: self.session.state == "idle"
            and not self.view._gate.model_selecting
            and self.view._effort_pending is None
        )

    def click(self, x: int, y: int) -> None:
        geo = self.terminal.geometry
        self.terminal.emit([
            PointerEvent("down", x, y, geo, button="left"),
            PointerEvent("up", x, y, geo),
        ])

    def shift_tab(self) -> None:
        self.terminal.emit([KeyPress(Keys.BackTab, "\t")])


async def _open_session(tmp_path, bodies):
    return await open_application(ds_config(_repo(tmp_path), _transport(bodies)))


@pytest.mark.asyncio
async def test_badge_click_cycles_once_model_name_and_draft_untouched(tmp_path):
    bodies = []
    session = await _open_session(tmp_path, bodies)
    h = EffortHarness(session)
    changes = []
    session.subscribe(lambda u: changes.append(u) if isinstance(u, ReasoningEffortChanged) else None)
    try:
        await h.start()
        brand_row = h.vt.find_row("cicada · [high]")
        assert brand_row is not None, h.vt.screen_text()
        row = brand_row  # 单行底栏: 档位徽标与模型同一行 (2026-10-19 裁决)
        h.view.editor.insert_text("草稿")
        history_before = session.history
        h.click(12, row)  # "[high]" 区间: 品牌 "cicada" 后两列分隔, 徽标起第 9 列
        assert await h.wait_idle()
        await h.settle()  # 节流渲染落屏后再核对屏幕事实
        assert session.reasoning_effort.current == "max"  # 循环 high → max
        assert len(changes) == 1
        assert session.history == history_before and bodies == []  # 零请求/零模型调用
        assert h.vt.find_row("[max]") is not None
        assert h.view.editor.get_text() == "草稿"  # 点击不触碰草稿
        # 模型名 (徽标右侧) 无命中: 无新事务
        h.click(30, row)
        await h.settle()
        assert session.reasoning_effort.current == "max" and len(changes) == 1
        assert h.view._gate.model_selecting is False
    finally:
        await h.stop()
        await session.aclose()


@pytest.mark.asyncio
async def test_truncated_badge_has_no_hit(tmp_path):
    """极窄 (24 列): 徽标整体省略, 原徽标位置点击无动作 (C6 被裁掉无 hit)."""
    bodies = []
    session = await _open_session(tmp_path, bodies)
    h = EffortHarness(session, columns=24, rows=16)
    changes = []
    session.subscribe(lambda u: changes.append(u) if isinstance(u, ReasoningEffortChanged) else None)
    try:
        await h.start()
        text = "\n".join(h.vt.screen_text())
        assert "思考 high" not in text and "R:high" not in text, text
        h.click(20, h.terminal.rows - 2)
        await h.settle()
        assert session.reasoning_effort.current == "high" and changes == []
    finally:
        await h.stop()
        await session.aclose()


@pytest.mark.asyncio
async def test_shift_tab_cycles_busy_silent_no_queue(tmp_path):
    bodies = []
    session = await _open_session(tmp_path, bodies)
    h = EffortHarness(session)
    changes = []
    session.subscribe(lambda u: changes.append(u) if isinstance(u, ReasoningEffortChanged) else None)
    try:
        await h.start()
        h.view.editor.insert_text("草稿")
        h.shift_tab()
        assert await h.wait_idle()
        await h.settle()
        assert session.reasoning_effort.current == "max"  # 循环 high → max
        assert h.view.editor.get_text() == "草稿"  # BackTab 不进草稿、不改焦点
        # busy 屏障: executing 中 Shift+Tab 静默, 不排队不暗中受理
        assert h.view.on_submit("请求") is True
        assert await _wait_until(lambda: session.state == "executing")
        h.shift_tab()
        h.shift_tab()
        assert h.view._gate.model_selecting is False
        assert h.view._effort_pending is None
        assert await _wait_until(lambda: session.state == "idle" and not h.view._gate.busy())
        assert session.reasoning_effort.current == "max" and len(changes) == 1
        # 空闲恢复后可继续循环 (max → low)
        h.shift_tab()
        assert await h.wait_idle()
        await h.settle()
        assert session.reasoning_effort.current == "low" and len(changes) == 2
        # 菜单拥有 BackTab: 模型菜单活跃时不透传主会话、菜单不收起 (C6/D8)
        h.view.model_menu.open(session.model_options, current_id=session.config.model_label)
        h.shift_tab()
        await h.settle()
        assert session.reasoning_effort.current == "low" and len(changes) == 2
        assert h.view.model_menu.is_open
        h.view.model_menu.dismiss()
    finally:
        await h.stop()
        await session.aclose()
