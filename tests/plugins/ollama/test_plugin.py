from cicada.boot import bootstrap
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.ollama import OllamaConfig, OllamaModel, ollama_plugin


def test_plugin_definition_shape():
    definition = ollama_plugin()
    assert definition.name == "ollama-model"
    assert definition.provides == frozenset({"model"})
    assert definition.requires == frozenset()


async def test_boot_provides_model_capability():
    app = await bootstrap([ollama_plugin(OllamaConfig())], tool_capabilities=())
    try:
        model = app.runtime.capability("model")
        assert isinstance(model, OllamaModel)
        assert model.config.model == "qwen3.5:9b"
        assert app.report.activated == ("ollama-model",)
    finally:
        await app.aclose()


async def test_runtime_stop_closes_client():
    app = await bootstrap([ollama_plugin(OllamaConfig())], tool_capabilities=())
    model = app.runtime.capability("model")
    assert not model.client.is_closed
    await app.aclose()
    assert model.client.is_closed


async def test_assembly_swappable_with_fake_model():
    """与 fake_model 在组装层可互换: 同一 bootstrap(..., model_capability="model") 路径."""
    fake_app = await bootstrap([fake_model_plugin(FakeModel([]))], tool_capabilities=())
    try:
        assert fake_app.report.activated == ("fake-model",)
    finally:
        await fake_app.aclose()
    ollama_app = await bootstrap([ollama_plugin()], tool_capabilities=())
    try:
        assert isinstance(ollama_app.runtime.capability("model"), OllamaModel)
    finally:
        await ollama_app.aclose()
