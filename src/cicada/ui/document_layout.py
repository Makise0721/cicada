"""同源文档布局 DTO 与自有文档视口 (tui-compact-interaction 票据 01; 契约 C3).

职责:
- DisplayAction/HitSpan/EntryAnchor/DocumentLayout 是同一次排版的只读产物:
  文字、点击区域与锚点来自同一个布局结果 (D5), 不另算点击坐标。动作只允许
  显式显示语义 (组/工具/思考/Skill toggle、工具 more/reset、jump_latest、
  effort_cycle 预留给 03); 禁止任意命令字符串、提交或外部打开。
- 条目身份: 工具组/工具/思考用 (scope, run_id, turn_index[, call_id]) (C1);
  文档条目 (用户/系统/Skill 等) 只有顺序身份, 用 transcript 插入序号
  ("entry"/"skill", display_id) — 文档条目只追加不重排, 序号即稳定身份。
- EntryLayout 是单个 transcript 条目的布局片段 (行/命中/锚点, 行号相对条目
  首行); Transcript 负责拼接成 DocumentLayout 并校正行号。
- DocumentViewport 是前端持有的纯状态机: 跟随底部/回看滚动、展开与 resize
  锚定、F6 文档焦点的动作选择。不读终端、不写屏; 帧合成输入是 DocumentLayout
  与文档区高度, 输出是呈现窗口行 (至多 height 行)。
"""

from __future__ import annotations

from dataclasses import dataclass

# DisplayAction.kind 允许值 (C3): 其余种类 (命令/提交/外部打开) 禁止。
ACTION_TOGGLE = "toggle"
ACTION_MORE = "more"
ACTION_RESET = "reset"
ACTION_JUMP_LATEST = "jump_latest"
ACTION_EFFORT_CYCLE = "effort_cycle"  # 03 思考强度标记预留

# 滚轮: 每个标准 notch 移动的物理行数 (C3)
WHEEL_LINES_PER_NOTCH = 3

# 目标身份首元素
TARGET_GROUP = "group"      # ("group", scope, run_id, turn_index)
TARGET_TOOL = "tool"        # ("tool", scope, run_id, turn_index, call_id)
TARGET_REASONING = "reasoning"  # ("reasoning", run_id, turn_index)
TARGET_SKILL = "skill"      # ("skill", display_id)
TARGET_ENTRY = "entry"      # ("entry", display_id) — 无动作条目的锚点身份
TARGET_LATEST = "latest"    # ("latest",) — 「有更新 · 最新」
TARGET_ASSISTANT = "assistant"  # ("assistant", run_id, turn_index)


@dataclass(frozen=True)
class DisplayAction:
    """明确的显示动作: kind ∈ ACTION_*, target_id 为稳定条目身份元组。"""

    kind: str
    target_id: tuple


@dataclass(frozen=True)
class HitSpan:
    """点击区域: 零基物理行与半开列区间 [start_column, end_column)。

    列按 visible_width 规则 (ANSI 零宽、CJK 双列); 正文不生成动作, 只有实际
    呈现的标题/动作有命中 (D4/D5)。
    """

    row: int
    start_column: int
    end_column: int
    action: DisplayAction


@dataclass(frozen=True)
class EntryAnchor:
    """物理行 -> 稳定条目身份及条目内显示行 (resize/展开锚定依据, C3)。"""

    item_id: tuple
    item_row: int


@dataclass(frozen=True)
class DocumentLayout:
    """整份显示文档的一次排版结果: lines/hits/anchors 同源。"""

    lines: tuple[str, ...]
    hits: tuple[HitSpan, ...]
    anchors: tuple[EntryAnchor | None, ...]  # 与 lines 等长; 空白/间隔行为 None

    def __post_init__(self) -> None:
        if len(self.anchors) != len(self.lines):
            raise ValueError("anchors 必须与 lines 等长")


EMPTY_LAYOUT = DocumentLayout((), (), ())


@dataclass(frozen=True)
class EntryLayout:
    """单条 transcript 条目的布局片段; row 相对条目首行。"""

    lines: tuple[str, ...]
    hits: tuple[HitSpan, ...]
    anchors: tuple[EntryAnchor | None, ...]

    def __post_init__(self) -> None:
        if len(self.anchors) != len(self.lines):
            raise ValueError("anchors 必须与 lines 等长")


def wrap_plain_entry(lines: list[str], item_id: tuple) -> EntryLayout:
    """无动作条目的默认包装: 每行锚到条目身份, 无命中区域。"""
    return EntryLayout(
        tuple(lines),
        (),
        tuple(EntryAnchor(item_id, index) for index in range(len(lines))),
    )


def owner_action(action: DisplayAction) -> DisplayAction | None:
    """动作被收起隐藏时的归属标题 (C2: 隐藏的动作回到所属标题)。

    more/reset 归所属工具标题; 工具标题归所属组标题; 其余无上层。
    """
    if action.kind in (ACTION_MORE, ACTION_RESET) and action.target_id[:1] == (TARGET_TOOL,):
        return DisplayAction(ACTION_TOGGLE, action.target_id)
    if action.target_id[:1] == (TARGET_TOOL,) and len(action.target_id) >= 5:
        return DisplayAction(ACTION_TOGGLE, (TARGET_GROUP, *action.target_id[1:4]))
    return None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


class DocumentViewport:
    """自有文档视口 (前端持有): 滚动/跟随/锚定与文档焦点动作选择的纯状态机。

    每帧 frame(layout, height, indicator) 产出呈现窗口 (至多 height 行, 含
    「有更新 · 最新」指示行); 键位方法只依据最近一次呈现帧的布局快照, 保证
    Enter 不执行屏幕外旧目标。不读终端尺寸、不写屏。
    """

    def __init__(self) -> None:
        self.scroll_top = 0
        self.follow = True
        self.focus = False  # F6 文档焦点
        self.selection: DisplayAction | None = None
        self.indicator_visible = False
        self._anchor: EntryAnchor | None = None
        self._pending_action: tuple[DisplayAction, int] | None = None  # 动作 + 期望屏幕行
        self._scroll_dirty = False  # 显式滚动指令后的下一帧只做 clamp, 不被锚点覆盖
        self._last_layout: DocumentLayout = EMPTY_LAYOUT
        self._last_height = 0       # 上一帧文档区高度 (含指示行)
        self._last_content_rows = 0  # 上一帧实际文档内容行 (不含指示行)

    # --- 帧合成 -----------------------------------------------------------------

    def frame(self, layout: DocumentLayout, height: int, indicator: str | None = None) -> list[str]:
        """按当前滚动状态切出呈现窗口; height<=0 或空文档时无命中/滚动。"""
        if height <= 0 or not layout.lines:
            self.indicator_visible = False
            self._last_layout = layout if layout.lines else EMPTY_LAYOUT
            self._last_height = max(0, height)
            self._last_content_rows = 0
            return []
        total = len(layout.lines)
        max_scroll = max(0, total - height)

        pending = self._pending_action
        self._pending_action = None
        anchored_by_action = False
        if pending is not None:
            # 展开/收起锚定 (C3): 被操作标题保持在原屏幕行, 空间不够时 clamp
            action, screen_row = pending
            row = self._find_action_row(layout, action)
            if row is not None:
                self.follow = False
                self.scroll_top = _clamp(row - screen_row, 0, max_scroll)
                anchored_by_action = True
        if self.follow:
            self.scroll_top = max_scroll
        else:
            if (
                not anchored_by_action
                and not self._scroll_dirty
                and layout is not self._last_layout
                and self._anchor is not None
            ):
                # resize/全局展开等重排: 按稳定条目身份重找锚点
                resolved = self._resolve_anchor(layout, self._anchor)
                if resolved is not None:
                    self.scroll_top = resolved
            self.scroll_top = _clamp(self.scroll_top, 0, max_scroll)
        self._scroll_dirty = False

        # 「有更新 · 最新」: 脱离跟随且窗口下方仍有内容时占窗口末行, 不伪造数量
        below = total - (self.scroll_top + height)
        self.indicator_visible = bool(indicator) and not self.follow and below > 0
        content_rows = height - (1 if self.indicator_visible else 0)
        window = list(layout.lines[self.scroll_top : self.scroll_top + content_rows])
        if self.indicator_visible and indicator is not None:
            window.append(indicator)

        if self.focus:
            self._repair_selection(layout)
        self._anchor = self._anchor_at(layout, self.scroll_top)
        self._last_layout = layout
        self._last_height = height
        self._last_content_rows = content_rows
        return window

    # --- 文档焦点 (F6) ------------------------------------------------------------

    def enter_focus(self) -> bool:
        """F6 进入文档焦点: 文档高度 0 或无文档时不进入 (C3)。返回是否进入。"""
        layout = self._last_layout
        if self._last_height <= 0 or not layout.lines:
            return False
        self.focus = True
        actions = self._ordered_actions(layout)
        if self.selection is not None and self.selection not in actions:
            self.selection = None  # 上次动作已失效: 选可见区最近动作
        if self.selection is None and actions:
            self.selection = self._nearest_visible_action(actions)
        if self.selection is not None:
            self._scroll_action_into_view(layout, self.selection)
        return True

    def exit_focus(self) -> None:
        self.focus = False

    def move_selection(self, delta: int) -> None:
        """Up/Down: 选择上/下一条可操作标题或分页动作, 必要时滚动到可见。"""
        layout = self._last_layout
        actions = self._ordered_actions(layout)
        if not actions:
            self.selection = None
            return
        if self.selection is None or self.selection not in actions:
            self.selection = self._nearest_visible_action(actions)
        else:
            index = actions.index(self.selection)
            self.selection = actions[_clamp(index + delta, 0, len(actions) - 1)]
        if self.selection is not None:
            self._scroll_action_into_view(layout, self.selection)

    def page(self, direction: int) -> None:
        """PageUp/PageDown: 移动 max(1, 文档高度−1) 行; 焦点留在可见区最近动作。"""
        step = max(1, self._last_height - 1)
        self._scroll_by(direction * step)
        self._selection_to_visible()

    def go_home(self) -> None:
        """Home: 到文档头 (脱离跟随)。"""
        self.follow = False
        self.scroll_top = 0
        self._scroll_dirty = True
        self._selection_to_visible()

    def go_end(self) -> None:
        """End: 到最新并恢复跟随。"""
        layout = self._last_layout
        max_scroll = max(0, len(layout.lines) - self._last_height)
        self.follow = True
        self.scroll_top = max_scroll
        self._selection_to_visible()

    def jump_to_latest(self) -> None:
        """「最新」动作: 恢复跟随 (与 End 同语义)。"""
        self.go_end()

    def wheel_scroll(self, notches: int) -> None:
        """滚轮 (02, C3): 每 notch 移动 WHEEL_LINES_PER_NOTCH 物理行。

        notches 正 = 向文档头回看, 负 = 向尾部; 到达底部自动恢复跟随。
        """
        if notches:
            self._scroll_by(notches * WHEEL_LINES_PER_NOTCH)

    def frame_hits(self, width: int) -> tuple[HitSpan, ...]:
        """最近一次 frame() 呈现窗口的帧行命中 (02, D5 同源)。

        布局命中平移到帧内行 (scroll_top 为偏移), 只含实际呈现的标题/动作;
        「有更新 · 最新」指示行整体可点。空窗口无命中。列沿用布局的
        visible_width 规则 (CJK 双列、ANSI 零宽), 前端不再另算坐标。
        """
        if self._last_content_rows <= 0 or not self._last_layout.lines:
            return ()
        top = self.scroll_top
        bottom = top + self._last_content_rows
        hits = tuple(
            HitSpan(hit.row - top, hit.start_column, hit.end_column, hit.action)
            for hit in self._last_layout.hits
            if top <= hit.row < bottom
        )
        if self.indicator_visible:
            hits += (
                HitSpan(
                    self._last_content_rows, 0, width,
                    DisplayAction(ACTION_JUMP_LATEST, (TARGET_LATEST,)),
                ),
            )
        return hits

    def execute_selected(self) -> DisplayAction | None:
        """Enter: 执行选中的显示动作; 屏幕外旧目标不执行 (返回 None)。

        动作入展开锚定队列 (下一帧保持其标题在原屏幕行); jump_latest 由视口
        内部消费, 不返回给组合层。
        """
        if self.selection is None:
            return None
        if self.selection.kind == ACTION_JUMP_LATEST:
            # 合成动作无布局命中行: 指示行可见即表达可达, 由视口内部消费
            if self.indicator_visible:
                self.jump_to_latest()
            return None
        layout = self._last_layout
        row = self._find_action_row(layout, self.selection)
        if row is None:
            return None
        if not (self.scroll_top <= row < self.scroll_top + self._last_content_rows):
            return None  # 选中动作不在可见内容区: Enter 静默
        self._pending_action = (self.selection, row - self.scroll_top)
        return self.selection

    # --- 内部: 滚动与选择 -------------------------------------------------------------

    def _scroll_by(self, delta: int) -> None:
        layout = self._last_layout
        max_scroll = max(0, len(layout.lines) - self._last_height)
        self.follow = False
        self.scroll_top = _clamp(self.scroll_top + delta, 0, max_scroll)
        self._scroll_dirty = True
        if self.scroll_top >= max_scroll:
            self.follow = True  # 到达底部恢复跟随 (C3)

    def _ordered_actions(self, layout: DocumentLayout) -> list[DisplayAction]:
        """全部动作按 (行, 列) 排序去重; 指示行可见时 jump_latest 排在最后。"""
        seen: list[DisplayAction] = []
        for hit in sorted(layout.hits, key=lambda h: (h.row, h.start_column)):
            if not seen or seen[-1] != hit.action:
                if hit.action not in seen:
                    seen.append(hit.action)
        if self.indicator_visible:
            seen.append(DisplayAction(ACTION_JUMP_LATEST, (TARGET_LATEST,)))
        return seen

    def _find_action_row(self, layout: DocumentLayout, action: DisplayAction) -> int | None:
        for hit in layout.hits:
            if hit.action == action:
                return hit.row
        return None

    def _visible_bounds(self) -> tuple[int, int]:
        return self.scroll_top, self.scroll_top + self._last_content_rows

    def _nearest_visible_action(self, actions: list[DisplayAction]) -> DisplayAction | None:
        """可见区内最接近窗口顶部的动作; 无可见动作时 None。"""
        layout = self._last_layout
        top, bottom = self._visible_bounds()
        best: tuple[int, DisplayAction] | None = None
        for action in actions:
            row = self._find_action_row(layout, action)
            if row is None:
                continue  # jump_latest 等合成动作无布局行, 可见性由指示行表达
            if top <= row < bottom:
                distance = row - top
                if best is None or distance < best[0]:
                    best = (distance, action)
        if best is not None:
            return best[1]
        if self.indicator_visible and actions:
            return actions[-1]  # 仅指示动作可见
        return None

    def _scroll_action_into_view(self, layout: DocumentLayout, action: DisplayAction) -> None:
        row = self._find_action_row(layout, action)
        if row is None:
            return  # 合成动作 (jump_latest): 无需滚动
        top, bottom = self._visible_bounds()
        if row < top:
            self.follow = False
            self.scroll_top = row
        elif row >= bottom:
            self.follow = False
            self.scroll_top = row - self._last_content_rows + 1
        else:
            return
        max_scroll = max(0, len(layout.lines) - self._last_height)
        self.scroll_top = _clamp(self.scroll_top, 0, max_scroll)
        self._scroll_dirty = True
        if self.scroll_top >= max_scroll:
            self.follow = True

    def _selection_to_visible(self) -> None:
        """滚动后焦点留在可见区最近动作; 无动作时清空选择 (Enter 静默)。"""
        actions = self._ordered_actions(self._last_layout)
        if not actions:
            self.selection = None
            return
        self.selection = self._nearest_visible_action(actions)

    def _repair_selection(self, layout: DocumentLayout) -> None:
        """帧级选择维护: 动作被收起/消失时回到所属标题, 再退化可见区最近。"""
        if self.selection is None:
            return
        if self._find_action_row(layout, self.selection) is not None:
            return
        if self.selection.kind == ACTION_JUMP_LATEST:
            if not self.indicator_visible:
                self.selection = None
            return
        owner = owner_action(self.selection)
        while owner is not None:
            if self._find_action_row(layout, owner) is not None:
                self.selection = owner
                return
            owner = owner_action(owner)
        actions = self._ordered_actions(layout)
        self.selection = self._nearest_visible_action(actions) if actions else None

    # --- 内部: 锚定 -----------------------------------------------------------------

    def _anchor_at(self, layout: DocumentLayout, row: int) -> EntryAnchor | None:
        """窗口顶行的锚点; 间隔/空白行向就近行借 (下优先后上, 限 4 行)。"""
        total = len(layout.lines)
        if total == 0:
            return None
        row = _clamp(row, 0, total - 1)
        for candidate in range(row, min(row + 4, total)):
            anchor = layout.anchors[candidate]
            if anchor is not None:
                return anchor
        for candidate in range(row - 1, max(row - 4, -1), -1):
            anchor = layout.anchors[candidate]
            if anchor is not None:
                return anchor
        return None

    def _resolve_anchor(self, layout: DocumentLayout, anchor: EntryAnchor) -> int | None:
        """在新布局中重找锚点行: 同条目同显示行 -> 同条目最近行 (收起落标题)
        -> 最近仍存在的前驱 -> None (调用方保留 clamp 后位置/文档边界)。"""
        same_item: list[tuple[int, int]] = []  # (item_row 距离, 物理行)
        for row, candidate in enumerate(layout.anchors):
            if candidate is None:
                continue
            if candidate == anchor:
                return row
            if candidate.item_id == anchor.item_id:
                same_item.append((abs(candidate.item_row - anchor.item_row), row))
        if same_item:
            same_item.sort(key=lambda item: (item[0], item[1]))
            return same_item[0][1]
        # 条目不存在: 旧布局中向文档头方向找最近仍存在的前驱条目
        old = self._last_layout
        old_row = self._find_item_row(old, anchor.item_id)
        if old_row is not None:
            new_items = {a.item_id for a in layout.anchors if a is not None}
            for candidate_row in range(old_row - 1, -1, -1):
                candidate = old.anchors[candidate_row]
                if candidate is not None and candidate.item_id in new_items:
                    resolved = self._resolve_anchor_item_first_row(layout, candidate.item_id)
                    if resolved is not None:
                        return resolved
        return None

    @staticmethod
    def _find_item_row(layout: DocumentLayout, item_id: tuple) -> int | None:
        for row, anchor in enumerate(layout.anchors):
            if anchor is not None and anchor.item_id == item_id:
                return row
        return None

    @staticmethod
    def _resolve_anchor_item_first_row(layout: DocumentLayout, item_id: tuple) -> int | None:
        best: tuple[int, int] | None = None
        for row, anchor in enumerate(layout.anchors):
            if anchor is not None and anchor.item_id == item_id:
                if best is None or anchor.item_row < best[0]:
                    best = (anchor.item_row, row)
        return best[1] if best is not None else None


__all__ = [
    "ACTION_EFFORT_CYCLE",
    "ACTION_JUMP_LATEST",
    "ACTION_MORE",
    "ACTION_RESET",
    "ACTION_TOGGLE",
    "DisplayAction",
    "DocumentLayout",
    "DocumentViewport",
    "EMPTY_LAYOUT",
    "EntryAnchor",
    "EntryLayout",
    "HitSpan",
    "TARGET_ASSISTANT",
    "TARGET_ENTRY",
    "TARGET_GROUP",
    "TARGET_LATEST",
    "TARGET_REASONING",
    "TARGET_SKILL",
    "TARGET_TOOL",
    "WHEEL_LINES_PER_NOTCH",
    "owner_action",
    "wrap_plain_entry",
]
