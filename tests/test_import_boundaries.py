"""层间导入边界: 静态检查, 替代运行期才发现的越界依赖."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cicada"

FORBIDDEN = {
    "core": {"cicada.runtime", "cicada.plugins", "cicada.boot"},
    "runtime": {"cicada.core", "cicada.plugins", "cicada.boot"},
    "plugins": {"cicada.boot"},
}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_import_boundaries():
    offenders = []
    for layer, banned in FORBIDDEN.items():
        for file in sorted((SRC / layer).rglob("*.py")):
            for module in imports_of(file):
                root = ".".join(module.split(".")[:2])
                if root in banned:
                    offenders.append(f"{file.name} imports {module}")
    assert offenders == []
