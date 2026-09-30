import pytest

from cicada.runtime.graph import ResolutionError, resolve
from cicada.runtime.plugin import PluginDefinition


def define(name, provides=(), requires=()):
    return PluginDefinition(
        name=name,
        setup=lambda ctx: None,
        provides=frozenset(provides),
        requires=frozenset(requires),
    )


def test_topo_order_providers_first():
    consumer = define("consumer", requires=("model",))
    provider = define("provider", provides=("model",))
    order = resolve((consumer, provider))
    assert [d.name for d in order] == ["provider", "consumer"]


def test_missing_capability():
    with pytest.raises(ResolutionError) as exc_info:
        resolve((define("consumer", requires=("model",)),))
    problem = exc_info.value.problems[0]
    assert problem.kind == "missing"
    assert problem.capability == "model"
    assert problem.plugins == ("consumer",)


def test_duplicate_provider():
    with pytest.raises(ResolutionError) as exc_info:
        resolve((define("a", provides=("model",)), define("b", provides=("model",))))
    problem = exc_info.value.problems[0]
    assert problem.kind == "duplicate_provider"
    assert problem.capability == "model"
    assert set(problem.plugins) == {"a", "b"}


def test_cycle():
    a = define("a", provides=("x",), requires=("y",))
    b = define("b", provides=("y",), requires=("x",))
    with pytest.raises(ResolutionError) as exc_info:
        resolve((a, b))
    problem = exc_info.value.problems[0]
    assert problem.kind == "cycle"
    assert set(problem.plugins) == {"a", "b"}


def test_all_problems_reported_together():
    with pytest.raises(ResolutionError) as exc_info:
        resolve((define("c1", requires=("m1",)), define("c2", requires=("m2",))))
    assert len(exc_info.value.problems) == 2


def test_self_provided_requirement_is_cycle():
    with pytest.raises(ResolutionError) as exc_info:
        resolve((define("a", provides=("x",), requires=("x",)),))
    assert exc_info.value.problems[0].kind == "cycle"
