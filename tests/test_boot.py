import pytest

from cicada.boot import BootError, bootstrap
from cicada.core.ports import StreamDone, TextDelta, ToolCallEvent
from cicada.core.messages import ToolResultMessage
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.fake_tools import EchoTool, FailTool, fake_tools_plugin
from cicada.runtime.plugin import PluginDefinition
from cicada.runtime.runtime import CapabilityError


async def test_bootstrap_runs_end_to_end():
    model = FakeModel(
        [
            [
                ToolCallEvent("c1", "echo", '{"text": "你好"}'),
                ToolCallEvent("c2", "fail", "{}"),
                StreamDone("tool_use"),
            ],
            [TextDelta("完成"), StreamDone("stop")],
        ]
    )
    echo = EchoTool()
    app = await bootstrap(
        [fake_model_plugin(model), fake_tools_plugin([echo, FailTool()])],
        tool_capabilities=("tools",),
    )
    result = await app.agent.run("测试")
    assert result.stop_reason == "stop"
    assert echo.invocations == [{"text": "你好"}]
    tool_msgs = [m for m in result.messages if isinstance(m, ToolResultMessage)]
    assert [m.result.call_id for m in tool_msgs] == ["c1", "c2"]
    assert not tool_msgs[0].result.is_error
    assert tool_msgs[1].result.is_error
    assert "fail tool always raises" in tool_msgs[1].result.content
    await app.aclose()
    with pytest.raises(CapabilityError):
        app.runtime.capability("model")


async def test_bootstrap_fails_when_plugin_fails():
    cleaned = []

    def good_setup(ctx):
        ctx.provide("model", FakeModel([]))
        ctx.defer(lambda: cleaned.append("good"))

    def bad_setup(ctx):
        raise RuntimeError("boom")

    with pytest.raises(BootError, match="bad"):
        await bootstrap(
            [
                PluginDefinition("good", good_setup, provides=frozenset({"model"})),
                PluginDefinition("bad", bad_setup, requires=frozenset({"model"})),
            ],
            tool_capabilities=("tools",),
        )
    assert cleaned == ["good"]


async def test_bootstrap_fails_when_required_capability_missing():
    with pytest.raises(BootError, match="tools"):
        await bootstrap(
            [fake_model_plugin(FakeModel([]))],
            tool_capabilities=("tools",),
        )


async def test_bootstrap_accepts_single_tool_per_capability():
    def read_setup(ctx):
        ctx.provide("tool.read", EchoTool())

    echo = EchoTool()
    app = await bootstrap(
        [
            fake_model_plugin(FakeModel([[StreamDone("stop")]])),
            PluginDefinition("tool-read", read_setup, provides=frozenset({"tool.read"})),
        ],
        tool_capabilities=("tool.read",),
    )
    result = await app.agent.run("x")
    assert result.stop_reason == "stop"
    await app.aclose()


async def test_bootstrap_default_system_prompt_keeps_existing_behavior():
    model = FakeModel([[TextDelta("ok"), StreamDone("stop")]])
    app = await bootstrap(
        [fake_model_plugin(model), fake_tools_plugin([EchoTool()])],
        tool_capabilities=("tools",),
    )
    assert app.agent._system_prompt == ""
    result = await app.agent.run("x")
    assert result.stop_reason == "stop"
    await app.aclose()


async def test_bootstrap_passes_system_prompt_to_agent():
    model = FakeModel([[TextDelta("ok"), StreamDone("stop")]])
    app = await bootstrap(
        [fake_model_plugin(model), fake_tools_plugin([EchoTool()])],
        tool_capabilities=("tools",),
        system_prompt="be terse",
    )
    # K1 已落地: 请求逐轮携带 system_prompt; 这里一并核对组装行为与请求透传
    assert app.agent._system_prompt == "be terse"
    result = await app.agent.run("x")
    assert result.stop_reason == "stop"
    assert model.requests[0].system_prompt == "be terse"
    await app.aclose()


@pytest.mark.parametrize("bad_prompt", [123, None, ["be terse"], b"bytes"])
async def test_bootstrap_rejects_non_str_system_prompt_before_runtime_start(bad_prompt):
    setup_calls = []

    def setup(ctx):
        setup_calls.append("ran")
        ctx.provide("model", FakeModel([]))

    with pytest.raises(TypeError, match="system_prompt"):
        await bootstrap(
            [PluginDefinition("fake-model", setup, provides=frozenset({"model"}))],
            tool_capabilities=("tools",),
            system_prompt=bad_prompt,
        )
    assert setup_calls == []
