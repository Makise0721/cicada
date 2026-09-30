"""I3 确定性跨层验收: 真实 Agent → Ollama 适配器(MockTransport) → 真实 read/edit → 下一轮请求.

不判断模型智能, 直接核对 provider 实际收到的 payload: system 唯一首位、num_ctx 透传、
工具结果 content 带来源/有界 diff, 以及内核计量记录与运行摘要。零网络。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from cicada.boot import bootstrap
from cicada.core.events import AssistantCompleted, Event, ModelCallCompleted
from cicada.plugins.coding.process import process_plugin
from cicada.plugins.coding.tool_edit import edit_plugin
from cicada.plugins.coding.tool_powershell import powershell_plugin
from cicada.plugins.coding.tool_read import read_plugin
from cicada.plugins.coding.tool_write import write_plugin
from cicada.plugins.coding.workspace import workspace_plugin
from cicada.plugins.ollama import OllamaConfig, OllamaModel
from cicada.prompting import compose_system_prompt
from cicada.reporting import build_run_summary
from cicada.runtime.plugin import PluginContext, PluginDefinition

TOOL_CAPABILITIES = ("tool.read", "tool.edit", "tool.write", "tool.powershell")


def _ndjson(*entries: dict) -> bytes:
    return b"\n".join(json.dumps(entry).encode("utf-8") for entry in entries) + b"\n"


def _chunk(content="", tool_calls=None, **extra) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message, "done": False, **extra}


def _done(input_tokens: int, output_tokens: int, duration_ns: int) -> dict:
    return {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": input_tokens,
        "eval_count": output_tokens,
        "total_duration": duration_ns,
    }


def _mock_ollama_plugin(model: OllamaModel) -> PluginDefinition:
    """经真实 boot 组装提供 model 能力; 仅 HTTP 传输被 MockTransport 替换."""

    def setup(ctx: PluginContext) -> None:
        ctx.provide("model", model)

    return PluginDefinition(name="mock-ollama", setup=setup, provides=frozenset({"model"}))


async def test_p3_context_flows_end_to_end(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    prompt_text, _ = compose_system_prompt(tmp_path)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        turn = len(requests) - 1
        if turn == 0:
            body = _ndjson(
                _chunk(tool_calls=[{"id": "call_1", "function": {"name": "read", "arguments": {"path": "a.txt"}}}]),
                _done(100, 10, 500_000_000),
            )
        elif turn == 1:
            body = _ndjson(
                _chunk(tool_calls=[{"id": "call_2", "function": {"name": "edit", "arguments": {
                    "path": "a.txt", "edits": [{"old_text": "beta", "new_text": "BETA"}]}}}]),
                _done(200, 20, 600_000_000),
            )
        else:
            body = _ndjson(_chunk("改完了"), _done(300, 30, 700_000_000))
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OllamaModel(client, OllamaConfig(options={"num_ctx": 32768}))
    app = await bootstrap(
        [
            workspace_plugin(tmp_path),
            process_plugin(),
            read_plugin(),
            edit_plugin(),
            write_plugin(),
            powershell_plugin(),
            _mock_ollama_plugin(model),
        ],
        tool_capabilities=TOOL_CAPABILITIES,
        system_prompt=prompt_text,
    )
    events: list[Event] = []
    app.agent.subscribe(events.append)
    try:
        result = await app.agent.run("读取 a.txt 并把 beta 改成 BETA")
    finally:
        await app.aclose()
        await client.aclose()

    assert result.stop_reason == "stop"
    assert len(requests) == 3

    # system: 每轮请求恰好一份且在首位; 普通历史不被改写 (提示不进入 result.messages)
    for payload in requests:
        assert payload["options"] == {"num_ctx": 32768}
        system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert payload["messages"][0] == {"role": "system", "content": prompt_text}
        assert payload["tools"]
    assert all(prompt_text not in getattr(m, "text", "") for m in result.messages)
    assert all(prompt_text not in getattr(getattr(m, "result", None), "content", "") for m in result.messages)

    # read 结果 (第 2 轮请求的 tool 消息): content 带来源头与续读 footer
    tool_msg_1 = requests[1]["messages"][-1]
    assert tool_msg_1["role"] == "tool" and tool_msg_1["tool_call_id"] == "call_1"
    assert tool_msg_1["content"].startswith('[file="')
    assert "a.txt" in tool_msg_1["content"]
    assert "lines=1-2 total_lines=2" in tool_msg_1["content"]
    assert "[truncated=false next_offset=none]" in tool_msg_1["content"]

    # edit 结果 (第 3 轮请求): content 带 applied 行与有界 diff
    tool_msg_2 = requests[2]["messages"][-1]
    assert tool_msg_2["tool_call_id"] == "call_2"
    assert "edit applied" in tool_msg_2["content"]
    assert "-beta" in tool_msg_2["content"] and "+BETA" in tool_msg_2["content"]

    # 真实工具副作用落地
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\nBETA\n"

    # 内核计量: 每轮一条记录, 值来自 terminal 行, 时序在 AssistantCompleted 之后
    assert len(result.model_calls) == 3
    assert [c.stop_reason for c in result.model_calls] == ["tool_use", "tool_use", "stop"]
    assert [c.metrics.input_tokens for c in result.model_calls] == [100, 200, 300]
    assert [c.metrics.output_tokens for c in result.model_calls] == [10, 20, 30]
    assert all(c.elapsed_s >= 0 for c in result.model_calls)
    kinds = [type(e) for e in events]
    mcc_positions = [i for i, k in enumerate(kinds) if k is ModelCallCompleted]
    assert len(mcc_positions) == 3
    for pos in mcc_positions:
        assert kinds[pos - 1] is AssistantCompleted

    # 摘要: 已知求和 + 覆盖率, provider 与本地耗时分别标注
    summary = build_run_summary(result)
    assert "model_calls=3 tools=2 tool_errors=0" in summary
    assert "provider_time_s=1.80 provider_reported_calls=3/3" in summary
    assert "input_tokens_known=600 input_reported_calls=3/3" in summary
    assert "output_tokens_known=60 output_reported_calls=3/3" in summary
