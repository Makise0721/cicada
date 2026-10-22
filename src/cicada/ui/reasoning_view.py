"""思考正文的显示投影：默认灰色斜体直出，独立于答案和工具卡。

流中预览保留有界前缀，完成后由规范记录替换；未完成和截断仍明确标注。
展开正文不带标题或缩进，逐项收起时保留状态标题。所有文本先净化再排版。
"""

from __future__ import annotations

from typing import Callable

from cicada.core.messages import ReasoningContent
from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display

_PREVIEW_MAX_BYTES = 8192
_PREVIEW_NOTE = "预览已截断；完成后显示完整记录"
_PREVIEW_NOTE_SEALED = "预览已截断"


class ReasoningBlock:
    """一条助手轮次的思考块: 记录四态 + 流中预览 + 标题降级 + 净化排版。

    expand_getter 返回当前显示文档的思考展开默认值; 未注入时恒为展开
    (与工具/Skill 正文同一约定, 避免无展开手段的调用者看不到内容),
    生产组合 (transcript) 恒注入, 默认展开。
    逐项展开覆盖 (_override) 优先于文档默认值；收起时用状态标题提供展开入口。
    点击/文档焦点沿用逐项切换，不影响工具卡。
    """

    def __init__(self, *, expand_getter: Callable[[], bool] | None = None) -> None:
        self._expand_getter = expand_getter
        self._override: bool | None = None  # 逐项展开覆盖 (点击/键盘动作)
        self._text: str | None = None  # None = 无记录无预览; str 含完整空串/空预览
        self._complete = False
        self._preview_active = False  # ~ 态: 流中预览未获完成记录
        self._preview_bytes = 0
        self._preview_capped = False
        self._version = 0  # 记录替换即递增, 进渲染缓存键
        self._render_cache: tuple[int, DisplayStyle, bool, bool, int, list[str]] | None = None

    @property
    def has_record(self) -> bool:
        """是否有思考记录或预览 (缺失记录省略整块)。"""
        return self._text is not None

    def is_expanded(self) -> bool:
        """有效展开状态: 逐项覆盖优先, 否则文档默认值 (未注入恒展开)。"""
        if self._override is not None:
            return self._override
        return True if self._expand_getter is None else bool(self._expand_getter())

    def toggle(self) -> None:
        """逐项开关 (票据 01): 只覆盖本块; 04 用户裁决移除 Ctrl+T 统一切换,
        覆盖只由下一次逐项点击改回。"""
        self._override = not self.is_expanded()
        self._version += 1

    def set_record(self, reasoning: ReasoningContent | None) -> None:
        """完成/恢复的权威记录替换 (含预览原位替换); None 撤掉已有显示。"""
        self._text = reasoning.text if reasoning is not None else None
        self._complete = reasoning.complete if reasoning is not None else False
        self._preview_active = False
        self._preview_bytes = 0
        self._preview_capped = False
        self._version += 1

    def append_preview(self, text: str) -> None:
        """流中思考增量进入有界预览 (票据 02); 已封存/已有记录时忽略迟到增量。

        只编码当前增量并按剩余预算截取 UTF-8 前缀 (decode 忽略残缺尾字节,
        按码点截断), 不重扫此前预览; 超限后不继续累计, 也不再保留原文。
        """
        if self._text is not None and not self._preview_active:
            return
        if not self._preview_active:
            self._preview_active = True
            self._text = ""  # 空增量也建立条目: 区分未返回字段与已收到空字段
        if not self._preview_capped:
            remaining = _PREVIEW_MAX_BYTES - self._preview_bytes
            # 每个码点至少占一个字节；多取一个即可判定超限，编码成本有界。
            raw = text[:remaining + 1].encode("utf-8")
            if len(raw) > remaining:
                taken = raw[:remaining].decode("utf-8", "ignore")
                self._text += taken
                self._preview_bytes += len(taken.encode("utf-8"))
                self._preview_capped = True
            else:
                self._text += text
                self._preview_bytes += len(raw)
        self._version += 1

    def seal_preview(self) -> None:
        """无完成记录的防御性终止: 封存为未完成预览 (幂等), 不生成规范记录。"""
        if self._preview_active:
            self._preview_active = False
            self._complete = False
            self._version += 1

    def render(self, width: int, *, selected: bool = False) -> list[str]:
        """按当前宽度排版; 缺失记录返回空, 收起只有标题行, 每行列宽 <= width。"""
        if not self.has_record:
            return []
        expanded = self.is_expanded()
        style = current_style()
        cached = self._render_cache
        if cached is not None and cached[:5] == (width, style, expanded, selected, self._version):
            return list(cached[5])
        lines = self._render_lines(width, expanded, selected)
        self._render_cache = (width, style, expanded, selected, self._version, list(lines))
        return lines

    def _render_lines(self, width: int, expanded: bool, selected: bool) -> list[str]:
        title = theme.fg(theme.MUTED_GRAY, self._title(width, expanded, selected))
        if not expanded:
            return [title]  # 收起只占标题行 (规格 D2)
        cleaned = sanitize_display(self._text or "")
        if not cleaned.strip():
            return [] if self._preview_active or self._complete else [title]
        body: list[str] = []
        for logical in cleaned.split("\n"):
            for chunk in wrap_text_ansi(logical, width):
                body.append(self._body_line(truncate_to_width(chunk, width)))
        if self._preview_capped:
            note = _PREVIEW_NOTE if self._preview_active else _PREVIEW_NOTE_SEALED
            for chunk in wrap_text_ansi(note, width):
                body.append(self._body_line(truncate_to_width(chunk, width)))
        if not self._preview_active and not self._complete:
            body.append(title)
        return body

    @staticmethod
    def _body_line(text: str) -> str:
        style = current_style()
        return theme.fg_open(theme.MUTED_GRAY) + style.sgr("3") + text + style.sgr("0")

    def _title(self, width: int, expanded: bool, selected: bool) -> str:
        """四级标题降级 (D2): >=14 完整文字, 6-13 "思考 X", 2-5 "RX", 1 列 "X"。
        折叠标记一列: 选中 > / 展开 - / 收起 + (C2, plain 策略同样可辨认)。"""
        code = "~" if self._preview_active else ("=" if self._complete else "!")
        if width < 6:
            return f"R{code}" if width >= 2 else code
        if width < 14:
            return f"思考 {code}"
        status = (
            "思考 预览中" if self._preview_active
            else "思考 未完成" if not self._complete
            else "思考 已完成" if self._text else self._empty_status(width)
        )
        title = status
        marked = (">" if selected else ("-" if expanded else "+")) + " " + status
        if visible_width(marked) <= width:
            title = marked  # 折叠标记仅在放得下时加入
        return title

    @staticmethod
    def _empty_status(width: int) -> str:
        """完整空记录状态: 放得下用全称, 否则缩为"思考 已完成 空" (>=14 列入口)。"""
        full = "思考 已完成 · 内容为空"
        return full if visible_width(full) <= width else "思考 已完成 空"


__all__ = ["ReasoningBlock"]
