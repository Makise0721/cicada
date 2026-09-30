"""非交互入口: uv run python -m cicada --workspace <dir> --script <script.json> "<prompt>".

真实模型适配器 (本地 Ollama qwen) 尚未接入; 无 --script 时明确报错并以退出码 2 结束.
剧本 JSON 为条目列表, 每条目是一轮模型响应:
  {"text": "...", "tool_calls": [{"id", "name", "arguments": {...}}],
   "stop": true | "error": "..." | "length": true}
终结缺省: 有 tool_calls 视为 tool_use, 否则视为 stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from cicada.boot import App, BootError, bootstrap
from cicada.core.events import (
    AssistantCompleted,
    Event,
    RunFinished,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.core.ports import StreamDone, StreamEvent, TextDelta, ToolCallEvent
from cicada.plugins.coding.process import process_plugin
from cicada.plugins.coding.tool_edit import edit_plugin
from cicada.plugins.coding.tool_powershell import powershell_plugin
from cicada.plugins.coding.tool_read import read_plugin
from cicada.plugins.coding.tool_write import write_plugin
from cicada.plugins.coding.workspace import workspace_plugin
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.runtime.plugin import PluginDefinition

TOOL_CAPABILITIES = ("tool.read", "tool.edit", "tool.write", "tool.powershell")


def parse_script(raw: str) -> list[list[StreamEvent]]:
    """把剧本 JSON 转换为 FakeModel 剧本; 非法输入一律抛 ValueError."""
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"script is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError("script must be a JSON list of entries")
    script: list[list[StreamEvent]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"script entry {index} must be an object")
        events: list[StreamEvent] = []
        text = entry.get("text")
        if text:
            events.append(TextDelta(str(text)))
        tool_calls = entry.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise ValueError(f"script entry {index}: tool_calls must be a list")
        for call in tool_calls:
            if not isinstance(call, dict) or "id" not in call or "name" not in call:
                raise ValueError(
                    f"script entry {index}: each tool call must be an object with id and name"
                )
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"script entry {index}: tool call arguments must be an object"
                )
            events.append(ToolCallEvent(str(call["id"]), str(call["name"]), json.dumps(arguments)))
        if "error" in entry:
            events.append(StreamDone("error", str(entry["error"])))
        elif entry.get("length"):
            events.append(StreamDone("length"))
        elif entry.get("stop") or not tool_calls:
            events.append(StreamDone("stop"))
        else:
            events.append(StreamDone("tool_use"))
        script.append(events)
    return script


def default_definitions(workspace: Path, model: FakeModel) -> list[PluginDefinition]:
    return [
        workspace_plugin(workspace),
        process_plugin(),
        read_plugin(),
        edit_plugin(),
        write_plugin(),
        powershell_plugin(),
        fake_model_plugin(model),
    ]


def print_event(event: Event) -> None:
    if isinstance(event, RunStarted):
        print(f"=== {event.run_id}")
    elif isinstance(event, TurnStarted):
        print(f"--- turn {event.turn_index}")
    elif isinstance(event, AssistantCompleted):
        if event.message.text:
            print(event.message.text)
    elif isinstance(event, ToolStarted):
        print(f">> {event.name} ({event.call_id})")
    elif isinstance(event, ToolCompleted):
        marker = " [error]" if event.result.is_error else ""
        print(f"<< {event.result.name} ({event.result.call_id}){marker}")
        if event.result.content:
            print(event.result.content)
    elif isinstance(event, RunFinished):
        suffix = f" ({event.error})" if event.error else ""
        print(f"=== finished: {event.stop_reason}{suffix}")


async def _run(args: argparse.Namespace) -> int:
    if args.script is None:
        print(
            "真实模型适配器 (本地 Ollama qwen) 尚未接入; 请用 --script 提供 fake model 剧本",
            file=sys.stderr,
        )
        return 2
    try:
        script = parse_script(Path(args.script).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"invalid script: {exc}", file=sys.stderr)
        return 2
    try:
        app: App = await bootstrap(
            default_definitions(Path(args.workspace), FakeModel(script)),
            tool_capabilities=TOOL_CAPABILITIES,
        )
    except BootError as exc:
        print(f"boot failed: {exc}", file=sys.stderr)
        return 2
    app.agent.subscribe(print_event)
    try:
        result = await app.agent.run(args.prompt)
    finally:
        await app.aclose()
    return 0 if result.stop_reason == "stop" else 1


def main() -> int:
    # 与 powershell 工具一致: 入口输出统一 UTF-8, 不随控制台代码页变化
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="cicada", description="Cicada coding agent (scripted-model harness)"
    )
    parser.add_argument("--workspace", default=".", help="工作区根目录 (默认当前目录)")
    parser.add_argument("--script", help="fake model 剧本 JSON 文件")
    parser.add_argument("prompt", help="交给 agent 的 prompt")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
