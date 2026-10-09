"""UI 会话的显示策略；渲染作用域不会修改环境或跨任务共享可变状态。"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DisplayStyle:
    plain: bool

    @classmethod
    def capture(cls, environ: Mapping[str, str] | None = None) -> "DisplayStyle":
        env = os.environ if environ is None else environ
        return cls(plain="NO_COLOR" in env or env.get("TERM") == "dumb")

    def sgr(self, parameters: str) -> str:
        """只供可信组件生成样式；renderer 控制序列不经过此入口。"""
        return "" if self.plain else f"\x1b[{parameters}m"

    @contextmanager
    def use(self) -> Iterator[None]:
        token = _render_style.set(self)
        try:
            yield
        finally:
            _render_style.reset(token)


_render_style: ContextVar[DisplayStyle | None] = ContextVar("cicada_render_style", default=None)


def current_style() -> DisplayStyle:
    """组合层冻结会话策略；独立纯渲染调用按调用时环境捕获。"""
    style = _render_style.get()
    return style if style is not None else DisplayStyle.capture()
