"""独立思考块 (票据 01): 完成／恢复记录的折叠显示与排版.

职责 (规格 D2 / 契约 C2):
- ReasoningBlock 是 AssistantEntry 内部对 immutable message.reasoning 的显示
  引用, 不是规范消息: 缺失记录 (None) 完全省略块; 完整空/完整正文/未完成
  各有清晰状态。流式预览态 (~) 属票据 02, 本模块已按四级短码规则预留。
- 展开状态只属于当前显示文档: Transcript 持有唯一 bool (初始 false), 块经
  只读 getter 获取, 区别于工具/Skill 的 expand_getter; /clear 保留显示文档
  因此保持展开状态。
- 渲染先 sanitize_display 再换行, 保持原始段落, 不解释 Markdown, 不把正文
  与答案拼接; 渲染缓存含宽度/样式/展开/内容版本, 不复用失效颜色或旧正文。
- 标题按 14/6/2/1 列四级降级, 状态字符 = 规范完整(含空)、! 未完成、~ 进行中
  预览; 完整空与非空在极窄标题中合并为完整状态。完整性由记录决定, 不靠
  防御截断碰巧选择; 收起始终只占标题行。
"""

from __future__ import annotations

from typing import Callable

from cicada.core.messages import ReasoningContent
from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.main_screen import visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display

_BODY_INDENT = "  "
_EXPAND_HINT = " · Ctrl+T 展开"
_COLLAPSE_HINT = " · Ctrl+T 收起"


class ReasoningBlock:
    """一条助手轮次的思考块: 记录四态 + 标题降级 + 净化排版。

    expand_getter 返回当前显示文档的思考展开 bool; 未注入时恒为收起。
    """

    def __init__(self, *, expand_getter: Callable[[], bool] | None = None) -> None:
        self._expand_getter = expand_getter
        self._text: str | None = None  # None = 无记录; str 含完整空串
        self._complete = False
        self._version = 0  # 记录替换即递增, 进渲染缓存键
        self._render_cache: tuple[int, DisplayStyle, bool, int, list[str]] | None = None

    @property
    def has_record(self) -> bool:
        """是否有思考记录 (缺失记录省略整块)。"""
        return self._text is not None

    def set_record(self, reasoning: ReasoningContent | None) -> None:
        """完成/恢复的权威记录替换; None 撤掉已有显示 (含后继预览撤除)。"""
        self._text = reasoning.text if reasoning is not None else None
        self._complete = reasoning.complete if reasoning is not None else False
        self._version += 1

    def render(self, width: int) -> list[str]:
        """按当前宽度排版; 缺失记录返回空, 收起只有标题行, 每行列宽 <= width。"""
        if not self.has_record:
            return []
        expanded = True if self._expand_getter is None else bool(self._expand_getter())
        style = current_style()
        cached = self._render_cache
        if cached is not None and cached[:4] == (width, style, expanded, self._version):
            return list(cached[4])
        lines = self._render_lines(width, expanded)
        self._render_cache = (width, style, expanded, self._version, list(lines))
        return lines

    def _render_lines(self, width: int, expanded: bool) -> list[str]:
        title = theme.fg(theme.MUTED_GRAY, self._title(width, expanded))
        if not expanded:
            return [title]  # 收起只占标题行 (规格 D2)
        cleaned = sanitize_display(self._text or "")
        if not cleaned.strip():
            return [title]  # 空正文/纯空白: 事实在标题状态里, 不补空壳
        body: list[str] = []
        content_width = max(1, width - visible_width(_BODY_INDENT))
        for logical in cleaned.split("\n"):
            for chunk in wrap_text_ansi(logical, content_width):
                body.append(theme.fg(theme.MUTED_GRAY, _BODY_INDENT + chunk))
        return [title, *body]

    def _title(self, width: int, expanded: bool) -> str:
        """四级标题降级 (D2): >=14 完整文字, 6-13 "思考 X", 2-5 "RX", 1 列 "X"。"""
        code = "=" if self._complete else "!"
        if width < 6:
            return f"R{code}" if width >= 2 else code
        if width < 14:
            return f"思考 {code}"
        status = "思考 未完成" if not self._complete else (
            "思考 已完成" if self._text else self._empty_status(width)
        )
        title = status
        marked = ("-" if expanded else "+") + " " + status
        if visible_width(marked) <= width:
            title = marked  # 折叠标记仅在放得下时加入
        hint = _COLLAPSE_HINT if expanded else _EXPAND_HINT
        if visible_width(title + hint) <= width:
            title += hint
        return title

    @staticmethod
    def _empty_status(width: int) -> str:
        """完整空记录状态: 放得下用全称, 否则缩为"思考 已完成 空" (>=14 列入口)。"""
        full = "思考 已完成 · 内容为空"
        return full if visible_width(full) <= width else "思考 已完成 空"


__all__ = ["ReasoningBlock"]
