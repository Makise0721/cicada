"""06 阶段验收夹具: 固定 fake fixture 经公开入口 (--tui 生产组合) 驱动完整交互序列.

本模块只提供可复用夹具 (模型/剧本/工作区/sink/等待), 不含测试; 验收场景在
test_tui_stage_acceptance.py。确定性: fake 模型剧本 + 真实 Git 工作区 + 真实
PowerShell 检查命令, 无真实模型调用。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from cicada.__main__ import parse_script
from cicada.core.cancel import CancelToken
from cicada.core.ports import StreamDone, StreamEvent, TextDelta
from cicada.plugins.coding.verification_contracts import CheckDefinition, VerificationPlan
from cicada.plugins.fake_model import fake_model_plugin
from cicada.prompting import PromptInfo
from cicada.ui.rendering import record_fragments

CHECK_COMMAND = "exit 0"


class ChunkedFakeModel:
    """parse_script JSON 剧本 + 可控分块/延迟的 fake 模型 (与 03/05 夹具同构)."""

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


def stage_config(root: Path, script: str, *, plan: VerificationPlan | None = None):
    """公开入口的 ApplicationConfig: fake 模型 + 可选检查计划 (与 CLI --tui 同口径)."""
    from cicada.application import ApplicationConfig

    return ApplicationConfig(
        workspace=root,
        system_prompt="stage acceptance system prompt",
        prompt_info=prompt_info(root),
        model_plugin=fake_model_plugin(ChunkedFakeModel(script, chunks=4, delay=0.08)),
        plan=plan,
        max_turns=10,
        model_label="fake",
    )


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


# 固定 agentic 剧本: 首问(编辑+检查) → 追问(仅文本) → 修改(仅编辑, 检查过期) → 重检(编辑+检查) → 慢回答(供取消)
def stage_script() -> str:
    edit_check = {"tool_calls": [
        {"id": "e1", "name": "edit", "arguments": {
            "path": "sample.py",
            "edits": [{"old_text": "VALUE = 1", "new_text": "VALUE = 2"}],
        }},
        {"id": "c1", "name": "check", "arguments": {"action": "run", "check_id": "check-1"}},
    ], "tool_use": True}
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
        edit_check,
        {"text": "第一轮修改完成并通过检查。", "stop": True},
        {"text": "这是追问的回答。", "stop": True},
        edit_only,
        {"text": "又改了一处但没有重检。", "stop": True},
        recheck,
        {"text": "重新检查通过，可以交付。", "stop": True},
        {"text": "这一轮回答得很慢很慢很慢很慢很慢很慢很慢很慢, 用于取消验收。", "stop": True},
    )


class StageSink:
    """捕获生产渲染输出: render 记录渲染纯文本, lines 收 print_line 行文本."""

    def __init__(self) -> None:
        self.rendered: list[str] = []
        self.lines: list[str] = []

    def print_line(self, text: str = "") -> None:
        self.lines.append(text)

    def print_record(self, record) -> None:
        fragments = record_fragments(record)
        self.rendered.append("".join(text for _, text in fragments))

    def rendered_count(self, needle: str) -> int:
        return sum(1 for text in self.rendered if needle in text)


async def wait_until(predicate, timeout: float = 20.0, step: float = 0.01) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return False


def done_count(sink: StageSink) -> int:
    return sink.rendered_count("运行结束")


def model_of(session) -> ChunkedFakeModel:
    model = session._app.agent._model
    if type(model).__name__ == "RunPolicy":
        model = model.inner
    assert isinstance(model, ChunkedFakeModel)
    return model
