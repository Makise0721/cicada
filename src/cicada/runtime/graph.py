"""依赖图解析: 启动时静态校验, 拓扑排序, 全部问题一次性聚合报告."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cicada.runtime.plugin import PluginDefinition


@dataclass(frozen=True)
class Problem:
    kind: Literal["missing", "cycle", "duplicate_provider"]
    capability: str | None
    plugins: tuple[str, ...]
    detail: str


class ResolutionError(RuntimeError):
    def __init__(self, problems: tuple[Problem, ...]) -> None:
        self.problems = problems
        summary = "; ".join(p.detail for p in problems)
        super().__init__(f"plugin resolution failed: {summary}")


def resolve(definitions: tuple[PluginDefinition, ...]) -> tuple[PluginDefinition, ...]:
    """返回拓扑序 (provider 在 consumer 前); 任何问题聚合后一次抛出."""
    problems: list[Problem] = []

    providers: dict[str, str] = {}
    for definition in definitions:
        for capability in sorted(definition.provides):
            existing = providers.get(capability)
            if existing is not None:
                problems.append(
                    Problem(
                        "duplicate_provider",
                        capability,
                        (existing, definition.name),
                        f"capability {capability!r} provided by both {existing!r} and {definition.name!r}",
                    )
                )
            else:
                providers[capability] = definition.name

    # consumer -> 其依赖的 provider 名集合
    dependencies: dict[str, set[str]] = {d.name: set() for d in definitions}
    for definition in definitions:
        for capability in sorted(definition.requires):
            provider = providers.get(capability)
            if provider is None:
                problems.append(
                    Problem(
                        "missing",
                        capability,
                        (definition.name,),
                        f"plugin {definition.name!r} requires missing capability {capability!r}",
                    )
                )
            elif provider == definition.name:
                problems.append(
                    Problem(
                        "cycle",
                        capability,
                        (definition.name,),
                        f"plugin {definition.name!r} requires capability {capability!r} it provides itself",
                    )
                )
            else:
                dependencies[definition.name].add(provider)

    if problems:
        raise ResolutionError(tuple(problems))

    registration_order = [d.name for d in definitions]
    order: list[str] = []
    remaining = dependencies
    while remaining:
        ready = sorted(
            (name for name, deps in remaining.items() if not deps),
            key=registration_order.index,
        )
        if not ready:
            cycle = tuple(sorted(remaining))
            raise ResolutionError(
                (Problem("cycle", None, cycle, f"dependency cycle among: {', '.join(cycle)}"),)
            )
        for name in ready:
            order.append(name)
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)

    by_name = {d.name: d for d in definitions}
    return tuple(by_name[name] for name in order)
