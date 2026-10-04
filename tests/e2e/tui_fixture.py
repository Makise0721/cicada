"""Pi 主屏阶段验收夹具 (05/06): 固定 fake fixture 经公开入口 (--tui 新前端) 驱动.

本模块只提供可复用夹具 (模型/剧本/工作区/终端/等待), 不含测试; 验收场景在
test_tui_stage_acceptance.py。确定性: fake 模型剧本 + 真实 Git 工作区 + 真实
PowerShell 检查命令, 无真实模型调用。显示断言经 FakeTerminal -> MainScreen ->
VtScreen (真实 VT 输出后的屏幕与滚动历史), 不只断言 render() 返回值。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.__main__ import parse_script
from cicada.core.cancel import CancelToken
from cicada.core.ports import StreamDone, StreamEvent, TextDelta
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.fake_model import fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.ui.main_screen import visible_width
from cicada.ui.terminal_io import FakeTerminal
from tests.ui.vt_screen import VtScreen

CHECK_COMMAND = "exit 0"

ENTER = KeyPress(Keys.ControlM, "\r")
CTRL_C = KeyPress(Keys.ControlC, "\x03")
CTRL_D = KeyPress(Keys.ControlD, "\x04")
CTRL_O = KeyPress(Keys.ControlO, "\x0f")


def text_keys(text: str) -> list[KeyPress]:
    """可打印文本按键序列 (含中文)。"""
    return [KeyPress(ch, ch) for ch in text]


class ChunkedFakeModel:
    """parse_script JSON 剧本 + 可控分块/延迟的 fake 模型 (与单元夹具同构)."""

    def __init__(self, script_json: str, *, chunks: int = 4, delay: float = 0.0) -> None:
        self._script = parse_script(script_json)
        self._chunks = chunks
        self._delay = delay
        self.requests: list = []

    async def stream(self, request, cancel: CancelToken):
        self.requests.append(request)
        entry: list[StreamEvent] = self._script.pop(0) if self._script else [StreamDone("stop")]
        for event in entry:
            if cancel.cancelled:
                yield StreamDone("aborted")
                return
            if isinstance(event, TextDelta) and event.text and self._chunks > 1:
                size = -(-len(event.text) // self._chunks)
                for start in range(0, len(event.text), size):
                    if cancel.cancelled:
                        yield StreamDone("aborted")
                        return
                    await asyncio.sleep(self._delay)
                    yield TextDelta(event.text[start : start + size])
            else:
                await asyncio.sleep(self._delay)
                yield event


def script_json(*entries: dict) -> str:
    return json.dumps(list(entries))


def prompt_info(root: Path) -> PromptInfo:
    return PromptInfo(
        version="stage-acceptance",
        workspace=root,
        instructions_path=None,
        instructions_sha256=None,
        prompt_sha256=hashlib.sha256(b"stage-acceptance").hexdigest(),
    )


def make_config(
    root: Path,
    script: str,
    *,
    chunks: int = 4,
    delay: float = 0.0,
    plan: VerificationPlan | None = None,
    max_turns: int = 6,
    model_label: str = "fake",
):
    """公开入口的 ApplicationConfig: 可控分块/延迟的 fake 模型 (与 CLI --tui 同口径)."""
    from cicada.application import ApplicationConfig

    return ApplicationConfig(
        workspace=root,
        system_prompt="stage acceptance system prompt",
        prompt_info=prompt_info(root),
        model_plugin=fake_model_plugin(ChunkedFakeModel(script, chunks=chunks, delay=delay)),
        plan=plan,
        max_turns=max_turns,
        model_label=model_label,
    )


def stage_config(root: Path, script: str, *, plan: VerificationPlan | None = None):
    """06 固定序列的 ApplicationConfig: 4 分块 + 80ms 延迟长流."""
    return make_config(root, script, chunks=4, delay=0.08, plan=plan, max_turns=10)


def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "baseline"], check=True)
    return root


def check_plan(root: Path) -> VerificationPlan:
    from cicada.plugins.coding.verification import generate_verification_run_id

    return VerificationPlan(
        verification_run_id=generate_verification_run_id(),
        root=Path(__import__("os").path.realpath(root)),
        checks=(CheckDefinition("check-1", CHECK_COMMAND, timeout_s=30.0),),
    )


# 固定 agentic 剧本 (06 固定交互序列):
# 首问 = powershell 长输出(>10 行, 供展开) + 编辑 + 检查 → 长流式回答 (超过旧
# 12 行/4096B 预览阈值) → 追问(仅文本) → 修改(仅编辑, 检查过期) → 汇报 →
# 重检(编辑+检查) → 汇报 → 慢回答(供取消)
def stage_script() -> str:
    powershell_edit_check = {"tool_calls": [
        {"id": "p1", "name": "powershell",
         "arguments": {"command": (
             "foreach ($i in 1..14) { Write-Output ('powershell-line-{0:D2}' -f $i) }"
         ), "timeout": 15}},
        {"id": "e1", "name": "edit", "arguments": {
            "path": "sample.py",
            "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
        }},
        {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
    ], "tool_use": True}
    long_answer = (
        "## 检查报告\n\n以下是本轮修改的完整说明, 内容超过旧 12 行/4096 bytes "
        "活动预览阈值, 用于验收同一 assistant 条目的原位流式更新与完整回放。\n\n"
        + "".join(f"第 {i} 行: 内容要点 {i}, 含 `code` 片段与中文。\n\n" for i in range(1, 21))
        + "结论: 第一轮修改完成并通过检查。"
    )
    edit_only = {"tool_calls": [
        {"id": "e2", "name": "edit", "arguments": {
            "path": "sample.py",
            "edits": [{"old_text": "VALUE = 2", "new_text": "VALUE = 3"}],
        }},
    ], "tool_use": True}
    recheck = {"tool_calls": [
        {"id": "e3", "name": "edit", "arguments": {
            "path": "sample.py",
            "edits": [{"old_text": "VALUE = 3", "new_text": "VALUE = 4"}],
        }},
        {"id": "c2", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
    ], "tool_use": True}
    return script_json(
        powershell_edit_check,
        {"text": long_answer, "stop": True},
        {"text": "这是追问的回答。", "stop": True},
        edit_only,
        {"text": "又改了一处但没有重检。", "stop": True},
        recheck,
        {"text": "重新检查通过，可以交付。", "stop": True},
        {"text": "这一轮回答得很慢很慢很慢很慢很慢很慢很慢很慢, 用于取消验收。", "stop": True},
    )


class StageTerminal:
    """FakeTerminal + VtScreen 镜像: 屏幕与滚动历史的可见文本视图。

    set_size 重建 VtScreen (缩放后完整重绘清屏回放, 历史由新尺寸重新积累);
    visible 断言一律合并滚动历史与当前屏幕 (文档可能滚出视口)。
    """

    def __init__(self, columns: int = 100, rows: int = 30) -> None:
        self.columns = columns
        self.rows = rows
        self.terminal = FakeTerminal(columns, rows)
        self._attach()

    def _attach(self) -> None:
        self.vt = VtScreen(self.columns, self.rows)
        self.terminal.feed_screen = self.vt.feed

    # --- 测试驱动接口 -------------------------------------------------------------

    def emit(self, keys: list[KeyPress]) -> None:
        self.terminal.emit(keys)

    def set_size(self, columns: int, rows: int) -> None:
        self.columns, self.rows = columns, rows
        self.terminal.set_size(columns, rows)
        self._attach()

    # --- 可见文本视图 ---------------------------------------------------------------

    def visible_lines(self) -> list[str]:
        return [*self.vt.history_text(), *self.vt.screen_text()]

    def text(self) -> str:
        return "\n".join(self.visible_lines())

    def flat(self) -> str:
        """全文本拼平: 长路径/hash 允许跨换行匹配 (折行不插字符)。"""
        return self.text().replace("\n", "")

    def contains(self, needle: str) -> bool:
        return needle in self.text()

    def count(self, needle: str) -> int:
        return "\n".join(self.visible_lines()).count(needle)

    def visible_widths(self) -> list[int]:
        return [visible_width(line) for line in self.visible_lines()]


async def wait_until(predicate, timeout: float = 20.0, step: float = 0.01) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return False


async def clear_draft(stage: StageTerminal, draft: str, attempts: int = 10) -> bool:
    """空闲 Ctrl+C 清草稿; reservation 释放窗口 (毫秒级) 内按键按设计记录取消意图
    (键位表: 忙碌/未起跑窗口 Ctrl+C 不清草稿), 重试对齐真实用户的重复按键。"""
    for _ in range(attempts):
        stage.emit([CTRL_C])
        if await wait_until(lambda: draft not in stage.text(), timeout=0.5):
            return True
    return False


def model_of(session) -> ChunkedFakeModel:
    model = session._app.agent._model
    if type(model).__name__ == "RunPolicy":
        model = model.inner
    assert isinstance(model, ChunkedFakeModel)
    return model
