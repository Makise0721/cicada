"""层间导入边界: 静态检查, 替代运行期才发现的越界依赖."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cicada"

FORBIDDEN = {
    "core": {"cicada.runtime", "cicada.plugins", "cicada.boot"},
    "runtime": {"cicada.core", "cicada.plugins", "cicada.boot"},
    "plugins": {"cicada.boot"},
}


def module_imports(source: str, package: tuple[str, ...]) -> set[str]:
    """解析 source 的全部导入并归一为绝对模块名.

    package 为 source 所在包 (如 ("cicada", "core")). 相对导入按 level 归一到
    绝对名; `from cicada import x` / `from . import x` 展开别名, 防止绕过前缀匹配.
    """
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    modules.add(node.module)
                if node.module == "cicada":
                    modules.update(f"cicada.{alias.name}" for alias in node.names)
            else:
                base = package[: len(package) - (node.level - 1)]
                target = (*base, *(node.module.split(".") if node.module else ()))
                if node.module:
                    modules.add(".".join(target))
                else:
                    modules.update(".".join((*target, alias.name)) for alias in node.names)
    return modules


def imports_of(path: Path) -> set[str]:
    parts = path.relative_to(SRC).with_suffix("").parts  # e.g. ("core", "agent")
    package = ("cicada", *parts[:-1])
    return module_imports(path.read_text(encoding="utf-8"), package)


def layer_violations() -> list[str]:
    offenders = []
    for layer, banned in FORBIDDEN.items():
        for file in sorted((SRC / layer).rglob("*.py")):
            for module in imports_of(file):
                root = ".".join(module.split(".")[:2])
                if root in banned:
                    offenders.append(f"{file.name} imports {module}")
    return offenders


def test_import_resolution_catches_relative_and_package_aliases():
    # 相对导入不得绕过边界
    assert "cicada.runtime.runtime" in module_imports(
        "from ..runtime.runtime import X", ("cicada", "core")
    )
    assert "cicada.runtime" in module_imports("from .. import runtime", ("cicada", "core"))
    assert "cicada.boot" in module_imports("from ..boot import bootstrap", ("cicada", "runtime"))
    # 包别名导入等价于导入该层
    assert "cicada.boot" in module_imports("from cicada import boot", ("cicada", "plugins"))
    assert "cicada.plugins" in module_imports("from cicada import plugins", ("cicada", "runtime"))
    # 绝对导入与 import 形式
    assert "cicada.runtime" in module_imports("import cicada.runtime as rt", ("cicada", "core"))
    assert "cicada.runtime.plugin" in module_imports(
        "from cicada.runtime.plugin import PluginDefinition", ("cicada", "core")
    )
    # 合法导入不受影响
    assert "cicada.core.cancel" in module_imports(
        "from cicada.core.cancel import CancelToken", ("cicada", "plugins")
    )
    assert "cicada.runtime.plugin" in module_imports(
        "from cicada.runtime.plugin import PluginDefinition", ("cicada", "plugins")
    )


def test_import_boundaries():
    assert layer_violations() == []
