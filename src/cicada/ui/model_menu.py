"""/model 模型选择菜单 (DeepSeek 阶段票据 04; 契约 C7)。

只管理展示与光标状态: 打开时默认选中当前模型, 上下键循环, Enter 由前端
交回组合层确认 (UI 屏障与应用双向核对归 conversation_view), Esc 收起;
旧目录的普通编辑先收起菜单再按原键处理；带修订的跨供应商目录在菜单内搜索，
搜索状态不占用聊天草稿。菜单是本地操作: 不创建
UserMessage/请求意图, 不消耗模型轮。所有输出仍由主屏 renderer 写入。
"""

from __future__ import annotations

from cicada.models import ModelOption
from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.display_style import current_style
from cicada.ui.rendering import sanitize_display

MODEL_MENU_HINT = "选择模型  ↑/↓ 移动  Enter 确认  Esc 收起"
_CURRENT_MARKER = " （当前）"


class ModelMenu:
    """/model 打开的模型目录; 收起或空目录时 render 返回空。"""

    def __init__(self) -> None:
        self._options: tuple[ModelOption, ...] = ()
        self._current_id: str | None = None
        self._index = 0
        self._open = False
        self._all_options: tuple[ModelOption, ...] = ()
        self.search_enabled = False
        self.query = ''
        self.catalog_revision: str | None = None

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def selected(self) -> ModelOption | None:
        if not self._open or not self._options:
            return None
        return self._options[self._index]

    def open(self, options: tuple[ModelOption, ...], *, current_id: str | None,
             catalog_revision: str | None = None, search_enabled: bool = False) -> None:
        """打开目录并默认选中当前模型 (契约 C7); 空目录保持收起。"""
        if not options:
            self._options = ()
            self._current_id = None
            self._open = False
            return
        self._options = options
        self._all_options = options
        self.catalog_revision = catalog_revision
        self.search_enabled = search_enabled
        self.query = ''
        self._current_id = current_id
        ids = [option.id for option in options]
        self._index = ids.index(current_id) if current_id in ids else 0
        self._open = True

    def filter(self, query: str) -> None:
        """目录搜索只修改菜单状态，不占用编辑器草稿。"""
        if not self.search_enabled:
            return
        self.query = sanitize_display(query)[:256]
        folded = self.query.casefold()
        self._options = tuple(item for item in self._all_options
                              if folded in item.label.casefold() or folded in item.id.casefold())
        self._index = 0

    def move(self, delta: int) -> bool:
        """上下键循环选择; 收起或空目录返回 False (键位交还编辑器)。"""
        if not self._open or not self._options:
            return False
        self._index = (self._index + delta) % len(self._options)
        return True

    def dismiss(self) -> bool:
        """收起菜单; 返回是否实际收起 (供前端决定是否拦截 Esc)。"""
        if not self._open:
            return False
        self._open = False
        return True

    def render(self, width: int, max_rows: int) -> list[str]:
        if not self._open or max_rows <= 0:
            return []
        hint = ('Enter 切换 · Ctrl+D 默认 · Ctrl+O 关闭思考 · Ctrl+L 配置 · Esc 返回'
                if self.search_enabled else MODEL_MENU_HINT)
        lines = [theme.fg(theme.FOOTER_DIM, truncate_to_width(hint, width))]
        if self.search_enabled and max_rows > 1:
            lines.append(theme.fg(theme.FOOTER_DIM, truncate_to_width('搜索：' + self.query, width)))
        if not self._options:
            if len(lines) < max_rows:
                lines.append(theme.fg(theme.MUTED_GRAY, truncate_to_width('没有匹配模型', width)))
            return lines
        available = max_rows - len(lines)
        if available <= 0:
            return lines[:max_rows]
        show_count = not self.search_enabled and len(self._options) > available and available > 1
        count = min(len(self._options), available - int(show_count))
        start = max(0, min(self._index - count // 2, len(self._options) - count))
        for index in range(start, start + count):
            lines.append(self._row(self._options[index], index == self._index, width))
        if show_count:
            lines.append(theme.fg(theme.FOOTER_DIM, truncate_to_width(
                f"  ({self._index + 1}/{len(self._options)})", width
            )))
        return lines

    def _row(self, option: ModelOption, selected: bool, width: int) -> str:
        prefix = truncate_to_width('→ ' if selected else '  ', max(0, width))
        marker = _CURRENT_MARKER if option.id == self._current_id else ""
        if option.is_default:
            marker += ' （默认）'
        marker = truncate_to_width(marker, max(0, width - visible_width(prefix)))
        label_text = option.label + (' · ' + (option.reason or '不可用') if not option.available else '')
        label = truncate_to_width(sanitize_display(label_text), max(0, width - visible_width(prefix) - visible_width(marker)))
        if selected:
            # 选中行整行背景高亮 (tui-aesthetics 规格 §D): 行内只用只开序列,
            # 末尾一次复位 (内层复位会清掉背景)
            used = visible_width(prefix) + visible_width(label) + visible_width(marker)
            pad = " " * max(0, width - used)
            return (
                theme.bg_open(theme.MENU_SEL_BG)
                + theme.fg_open(theme.EDITOR_BORDER) + prefix
                + theme.fg_open(theme.MD_LINK) + label
                + (theme.fg_open(theme.MUTED_GRAY) + marker if marker else "")
                + pad + current_style().sgr("0")
            )
        return prefix + label + (theme.fg(theme.MUTED_GRAY, marker) if marker else "")


__all__ = ["MODEL_MENU_HINT", "ModelMenu"]
