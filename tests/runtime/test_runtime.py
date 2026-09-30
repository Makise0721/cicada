import asyncio

import pytest

from cicada.runtime.plugin import DeclarationError, InstanceState, PluginDefinition
from cicada.runtime.runtime import CapabilityError, PluginRuntime, RegistrationError


def make_runtime(*definitions):
    runtime = PluginRuntime()
    for definition in definitions:
        runtime.register(definition)
    return runtime


async def test_start_activates_in_topo_order_and_require_returns_value():
    events = []

    def provider_setup(ctx):
        events.append("provider")
        ctx.provide("model", "fake-model-object")

    def consumer_setup(ctx):
        events.append("consumer")
        assert ctx.require("model") == "fake-model-object"

    runtime = make_runtime(
        PluginDefinition("consumer", consumer_setup, requires=frozenset({"model"})),
        PluginDefinition("provider", provider_setup, provides=frozenset({"model"})),
    )
    report = await runtime.start()
    assert events == ["provider", "consumer"]
    assert report.activated == ("provider", "consumer")
    assert report.failed == ()
    assert runtime.capability("model") == "fake-model-object"


async def test_setup_failure_rolls_back_activated_instances():
    cleaned = []

    def good_setup(ctx):
        ctx.provide("model", object())
        ctx.defer(lambda: cleaned.append("good"))

    def bad_setup(ctx):
        ctx.defer(lambda: cleaned.append("bad-partial"))
        raise RuntimeError("setup boom")

    runtime = make_runtime(
        PluginDefinition("good", good_setup, provides=frozenset({"model"})),
        PluginDefinition("bad", bad_setup, requires=frozenset({"model"})),
    )
    report = await runtime.start()
    assert report.failed == ("bad",)
    assert report.activated == ()
    assert cleaned == ["bad-partial", "good"]
    assert runtime.instance("good").state is InstanceState.STOPPED
    assert runtime.instance("bad").state is InstanceState.FAILED
    assert isinstance(runtime.instance("bad").error, RuntimeError)
    with pytest.raises(CapabilityError):
        runtime.capability("model")


async def test_declared_provides_must_actually_be_provided():
    def lazy_setup(ctx):
        pass  # 声明了 model 但未提供

    runtime = make_runtime(PluginDefinition("lazy", lazy_setup, provides=frozenset({"model"})))
    report = await runtime.start()
    assert report.failed == ("lazy",)
    assert isinstance(runtime.instance("lazy").error, DeclarationError)


async def test_stop_is_reverse_topo_and_consumer_finishes_before_provider():
    events = []

    def provider_setup(ctx):
        ctx.provide("model", object())

        async def provider_cleanup():
            events.append("provider-cleanup-start")
            await asyncio.sleep(0)
            events.append("provider-cleanup-end")

        ctx.defer(provider_cleanup)

    def consumer_setup(ctx):
        ctx.require("model")

        async def consumer_cleanup():
            events.append("consumer-cleanup-start")
            await asyncio.sleep(0)
            events.append("consumer-cleanup-end")

        ctx.defer(consumer_cleanup)

    runtime = make_runtime(
        PluginDefinition("provider", provider_setup, provides=frozenset({"model"})),
        PluginDefinition("consumer", consumer_setup, requires=frozenset({"model"})),
    )
    await runtime.start()
    await runtime.stop()
    assert events == [
        "consumer-cleanup-start",
        "consumer-cleanup-end",
        "provider-cleanup-start",
        "provider-cleanup-end",
    ]


async def test_stop_aggregates_cleanup_errors_but_stops_all():
    cleaned = []

    def a_setup(ctx):
        ctx.provide("x", 1)
        ctx.defer(lambda: cleaned.append("a"))

    def b_setup(ctx):
        ctx.require("x")

        def boom():
            raise ValueError("cleanup boom")

        ctx.defer(boom)

    runtime = make_runtime(
        PluginDefinition("a", a_setup, provides=frozenset({"x"})),
        PluginDefinition("b", b_setup, requires=frozenset({"x"})),
    )
    await runtime.start()
    with pytest.raises(ExceptionGroup):
        await runtime.stop()
    assert cleaned == ["a"]
    assert runtime.instance("a").state is InstanceState.STOPPED
    assert runtime.instance("b").state is InstanceState.FAILED


async def test_capability_after_stop_raises():
    runtime = make_runtime(
        PluginDefinition("p", lambda ctx: ctx.provide("m", 1), provides=frozenset({"m"}))
    )
    await runtime.start()
    await runtime.stop()
    with pytest.raises(CapabilityError):
        runtime.capability("m")


def test_duplicate_plugin_name_rejected():
    runtime = make_runtime(PluginDefinition("p", lambda ctx: None))
    with pytest.raises(RegistrationError):
        runtime.register(PluginDefinition("p", lambda ctx: None))


async def test_provide_and_require_outside_declaration_raise():
    def setup(ctx):
        with pytest.raises(DeclarationError):
            ctx.provide("nope", 1)
        with pytest.raises(DeclarationError):
            ctx.require("model")

    runtime = make_runtime(PluginDefinition("p", setup))
    report = await runtime.start()
    assert report.failed == ()
    assert runtime.instance("p").state is InstanceState.ACTIVE


async def test_double_provide_same_capability_raises():
    def setup(ctx):
        ctx.provide("m", 1)
        with pytest.raises(DeclarationError):
            ctx.provide("m", 2)

    runtime = make_runtime(PluginDefinition("p", setup, provides=frozenset({"m"})))
    report = await runtime.start()
    assert report.failed == ()
    assert runtime.capability("m") == 1
