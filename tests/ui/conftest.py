"""tests/ui 共享夹具入口: 使 tests/ 根的 helper 模块在聚焦运行时可导入.

pytest 无 __init__.py 的 prepend 导入只把被收集文件所在目录插入 sys.path;
单独运行 tests/ui 时 tests/ 不在路径上, test_application 等 helper 不可达。
此处把 tests/ 追加到 sys.path (append 不遮蔽既有项), 全量运行时同名模块
已经加载, sys.modules 命中不产生第二份实例。
"""

import sys
from pathlib import Path
import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.append(str(TESTS_ROOT))


@pytest.fixture
def color_env(monkeypatch):
    """有色外观断言显式选择有色环境，不依赖 pytest 宿主环境。"""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
