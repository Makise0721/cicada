"""紧凑工具显示 (tui-compact-interaction 票据 01; 契约 C1/C2, ADR 0011).

同一含工具模型轮次聚合成一个只读工具组 (D1): 组键 (display_scope, run_id,
turn_index), 工具键再加 call_id; AssistantCompleted 取得 tool_calls 时登记整组
(等待态), ToolStarted/ToolCompleted 只更新对应子项, RunFinished/请求终结把
无结果子项封存为未确认 (不伪造完成)。恢复经 add_completed_card 以终态入组,
组身份沿用 restored 命名空间。

默认收起: 多工具组一行概括 (动作 · 次数 · 异常/活动计数), 单工具一行标题;
打开组只列子工具短行, 再开单条才显示详情——参数至多 2 物理行 + 输出至多
6 物理行 + 短分页动作 (更多每次 +20 输出行, 收起详情恢复 6 行)。渲染只为
已披露内容换行, 收起不全量 wrap 大结果; 换行/净化只作用于显示副本, 规范
参数与结果原文不变 (C1)。

展开状态按显示身份独立保存; 新建/恢复默认收起, /clear 不动本文档。Ctrl+O
统一设置所有现有组/单工具为同一开合状态并作为后续默认值 (组打开只露列表,
子工具详情重置收起, 单工具最多初始预览); 逐项 toggle 只覆盖当前条目, 下一次
全局快捷键重新统一。组收起保留子项展开标志, 再打开时子项输出预算回到初始值。

同源布局 (C3): layout_group 产出 (行, 命中, 锚点) 三元组, 命中只落在实际呈现
的标题/分页动作上, 正文不生成动作; 渲染缓存键含宽度/版本/样式/选中动作。

外观 (tui-aesthetics 纪律): 无卡片背景与上下 padding, 组间至多一个空行;
行内样式只用只开 SGR 或 theme.fg 包裹, 折叠标记取列宽 1 字符 (+/-/选中 >),
plain 策略下无转义序列。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypeAlias

from cicada.core.events import (
    AssistantCompleted,
    RunFinished,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
)
from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.checks import artifact_line as _artifact_line_from_details
from cicada.ui.document_layout import (
    ACTION_MORE,
    ACTION_RESET,
    ACTION_TOGGLE,
    TARGET_GROUP,
    TARGET_TOOL,
    DisplayAction,
    EntryAnchor,
    EntryLayout,
    HitSpan,
)
from cicada.ui.main_screen import (
    slice_by_column,
    truncate_to_width,
    visible_width,
    word_wrap_line,
)
from cicada.ui.rendering import sanitize_display

# 详情预算 (契约 C2): 初次输出物理行 / 更多步进 / 参数物理行
INITIAL_OUTPUT_LINES = 6
MORE_OUTPUT_STEP = 20
PARAMS_MAX_LINES = 2
ARTIFACT_MAX_LINES = 2

# 标题/参数文本的字符上限 (摘要本身有界, 展示行再按列宽裁剪)
_TARGET_MAX_CHARS = 120
_PARAMS_MAX_CHARS = 240
_PARAM_VALUE_MAX_CHARS = 80

_PREFERRED_ARG_KEYS = (
    "path",
    "file_path",
    "command",
    "pattern",
    "action",
    "check_id",
    "query",
    "url",
)

_UI_FAILURES_MAX = 100

# 内核取消路径固定哨兵前缀 (core/agent.py _dispatch / _finish_pending_calls)
_CANCELLED_PREFIX = "cancelled"

CardStatus: TypeAlias = Literal["waiting", "running", "success", "error", "unconfirmed"]

# 工具类别文案: (单条动作动词, 组概括标签); 未知工具不捏造动作, 用安全工具名
_TOOL_LABELS: dict[str, tuple[str, str]] = {
    "read": ("读取", "读取文件"),
    "powershell": ("运行命令", "运行命令"),
    "glob": ("查找文件", "查找文件"),
    "grep": ("搜索内容", "搜索内容"),
    "edit": ("修改文件", "修改文件"),
    "write": ("写入文件", "写入文件"),
    "check": ("运行检查", "运行检查"),
}
_MIXED_GROUP_LABEL = "调用工具"

# 收起态异常/活动事实 (C2): (类别, 计数文案, 极窄符号); 极窄优先 ? ! c ~
_CATEGORY_ORDER = ("unconfirmed", "failed", "cancelled", "active")
_CATEGORY_WORD = {"unconfirmed": "未确认", "failed": "失败", "cancelled": "取消", "active": "进行中"}
_CATEGORY_SYMBOL = {"unconfirmed": "?", "failed": "!", "cancelled": "c", "active": "~"}

_CATEGORY_ACCENT = {
    "active": theme.TOOL_ACCENT_RUNNING,
    "ok": theme.MUTED_GRAY,
    "unconfirmed": theme.TOOL_ACCENT_ERROR,
    "failed": theme.TOOL_ACCENT_ERROR,
    "cancelled": theme.TOOL_ACCENT_ERROR,
}


def _clean_text(text: str) -> str:
    """单行显示副本: 显示清理 + 换行折叠为空格 (标题/摘要用)。"""
    return sanitize_display(str(text)).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _clean_output(content: str) -> str:
    """输出区显示副本: 规范换行后清理; 已收到的字节除控制序列外不增删。"""
    normalized = str(content).replace("\r\n", "\n").replace("\r", "")
    return sanitize_display(normalized)


def _ellipsize_middle(text: str, width: int) -> str:
    """可辨认省略: 保留首尾 (路径 basename 不丢); 过窄退化尾部省略。"""
    if visible_width(text) <= width:
        return text
    if width <= 1:
        return truncate_to_width(text, width, "")
    head_w = (width - 1) // 2
    tail_w = width - 1 - head_w
    head = slice_by_column(text, 0, head_w, strict=True)
    tail = slice_by_column(text, visible_width(text) - tail_w, tail_w, strict=True)
    return head + "…" + tail


def _scope_of(run_id: str) -> str:
    """显示命名空间 (C1): 恢复投影的 run_id 以 restored: 前缀隔离。"""
    return "restored" if run_id.startswith("restored:") else "live"


@dataclass(frozen=True)
class ToolCardView:
    """工具条目只读快照 (测试与组合方观察用); output_text 为已收到的完整输出。"""

    run_id: str
    turn_index: int
    call_id: str
    name: str
    status: CardStatus
    is_error: bool
    cancelled: bool
    output_text: str


@dataclass(frozen=True)
class ToolGroupView:
    """工具组只读快照: 身份 + 调用序 + 有效展开状态。"""

    scope: str
    run_id: str
    turn_index: int
    call_ids: tuple[str, ...]
    expanded: bool


class _ToolCard:
    """单条工具调用的内部可变状态; 真实终态 (success/error) 只写一次, 不回退。"""

    __slots__ = (
        "run_id",
        "turn_index",
        "call_id",
        "name",
        "args_target",
        "args_params",
        "status",
        "is_error",
        "cancelled",
        "content",
        "artifact",
        "mutation_state",
        "expanded",
        "budget",
    )

    def __init__(
        self, *, run_id: str, turn_index: int, call_id: str, name: str,
        args_target: str, args_params: str,
    ) -> None:
        self.run_id = run_id
        self.turn_index = turn_index
        self.call_id = call_id
        self.name = name
        self.args_target = args_target  # 标题目标 (单行, 有界)
        self.args_params = args_params  # 详情参数文本 (单行, 有界)
        self.status: CardStatus = "waiting"
        self.is_error = False
        self.cancelled = False
        self.content = ""
        self.artifact: str | None = None
        self.mutation_state: str | None = None  # 结构化 details["mutation"]["state"]
        self.expanded: bool | None = None  # None = 收起 (子项默认不开详情)
        self.budget = INITIAL_OUTPUT_LINES  # 输出物理行预算 (打开/组重开时重置)

    @property
    def terminal(self) -> bool:
        """已收到真实终态结果; 未确认 (封存无结果) 仍可被迟到真实结果更新。"""
        return self.status in ("success", "error")

    def start(self, name: str) -> None:
        self.name = name
        if self.status == "waiting":
            self.status = "running"

    def complete(self, content: str, is_error: bool, artifact: str | None, mutation_state: str | None) -> None:
        self.content = _clean_output(content)
        self.is_error = bool(is_error)
        self.status = "error" if is_error else "success"
        # 取消哨兵只用于标注: 内容原文照旧显示
        self.cancelled = self.is_error and self.content.lstrip().startswith(_CANCELLED_PREFIX)
        self.artifact = artifact
        self.mutation_state = mutation_state

    def seal_unconfirmed(self) -> None:
        """运行封存仍无结果: 等待/进行中 -> 未确认 (不伪造完成)。"""
        if self.status in ("waiting", "running"):
            self.status = "unconfirmed"


class _ToolGroup:
    """同一模型轮次的工具组 (D1): 保留原始调用顺序与配对。"""

    __slots__ = ("scope", "run_id", "turn_index", "cards", "expanded")

    def __init__(self, scope: str, run_id: str, turn_index: int) -> None:
        self.scope = scope
        self.run_id = run_id
        self.turn_index = turn_index
        self.cards: list[_ToolCard] = []
        self.expanded: bool | None = None  # None = 跟随文档默认值

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.scope, self.run_id, self.turn_index)


def _card_category(card: _ToolCard) -> str:
    """收起态关键风险分类 (C2): 未确认含结构化 mutation unknown (不认文字推断)。"""
    if card.status in ("waiting", "running"):
        return "active"
    if card.status == "unconfirmed":
        return "unconfirmed"
    if card.cancelled:
        return "cancelled"
    if card.mutation_state == "unknown":
        return "unconfirmed"
    if card.status == "error":
        return "failed"
    return "ok"


class ToolDocument:
    """紧凑工具显示文档片段: 应用事件 -> 工具组 -> 同源布局片段。

    apply() 接受 ApplicationSession 通知的 ApplicationUpdate (取其 .event
    字段) 或直接的内核 Event; 其余更新类型静默忽略。组按创建序稳定排列,
    只追加不重排; 逐项展开状态与输出预算只改展示, 不触请求或工具。
    """

    def __init__(self) -> None:
        self._groups: list[_ToolGroup] = []
        self._group_by_key: dict[tuple[str, str, int], _ToolGroup] = {}
        self._cards: list[_ToolCard] = []  # 到达序 (跨组)
        self._by_key: dict[tuple[str, int, str], _ToolCard] = {}
        self._current_turn: dict[str, int] = {}
        self._args_by_call: dict[tuple[str, str], str] = {}
        self._default_expanded = False
        self._version = 0
        self._group_cache: dict[int, tuple[int, int, DisplayStyle, DisplayAction | None, EntryLayout]] = {}
        self.ui_failures: list[str] = []

    # --- 事件入口 ---------------------------------------------------------------

    def apply(self, update: object) -> None:
        """消费一条应用更新; 工具与运行事件更新组, 其余忽略。"""
        event = getattr(update, "event", update)
        if isinstance(event, TurnStarted):
            self._current_turn[event.run_id] = event.turn_index
        elif isinstance(event, AssistantCompleted):
            for call in event.message.tool_calls:
                self._args_by_call[(event.run_id, call.id)] = call.arguments_json
            if event.message.tool_calls:
                self._declare_group(event.run_id, event.message)
        elif isinstance(event, ToolStarted):
            self._on_tool_started(event)
        elif isinstance(event, ToolCompleted):
            self._on_tool_completed(event)
        elif isinstance(event, RunFinished):
            self.seal_run(event.run_id)
        # 其余事件类型不是工具显示的事实来源, 静默忽略。

    def _declare_group(self, run_id: str, message) -> None:
        """AssistantCompleted 取得 tool_calls 时登记整组 (C1): 等待态子项按
        声明顺序建立, 已有子项 (防御性早到事件) 原位保留。"""
        turn = self._current_turn.get(run_id, 0)
        group = self._ensure_group("live", run_id, turn)
        for call in message.tool_calls:
            key = (run_id, turn, call.id)
            if key in self._by_key:
                continue
            target, params = self._summarize_args(call.arguments_json)
            self._append_card(
                group,
                _ToolCard(
                    run_id=run_id, turn_index=turn, call_id=call.id,
                    name=_clean_text(call.name), args_target=target, args_params=params,
                ),
            )

    def _on_tool_started(self, event: ToolStarted) -> None:
        turn = self._current_turn.get(event.run_id, 0)
        key = (event.run_id, turn, event.call_id)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing.terminal:
                # 迟到开始事件不能恢复已完成条目 (规格 §原位文本和工具更新)
                self._record_failure(f"stale ToolStarted for terminal card {key}; ignored")
                return
            existing.start(_clean_text(event.name))  # 重复开始: 幂等刷新
            self._touch()
            return
        group = self._ensure_group("live", event.run_id, turn)
        target, params = self._summarize_args(self._args_by_call.get((event.run_id, event.call_id)))
        self._append_card(
            group,
            _ToolCard(
                run_id=event.run_id, turn_index=turn, call_id=event.call_id,
                name=_clean_text(event.name), args_target=target, args_params=params,
            ),
        )
        self._by_key[key].start(_clean_text(event.name))

    def _on_tool_completed(self, event: ToolCompleted) -> None:
        result = event.result
        run_id = event.run_id
        artifact = _artifact_line_from_details(result.details)
        mutation_state = _mutation_state_of(result.details)
        # 1) 同 run 同 call 的未终结条目: 原位更新 (含等待/进行中/未确认)
        for card in reversed(self._cards):
            if card.run_id == run_id and card.call_id == result.call_id and not card.terminal:
                card.complete(result.content, result.is_error, artifact, mutation_state)
                self._touch()
                return
        # 2) 缺少 ToolStarted 的闭合结果: 以当前轮建终态卡 (跨轮复用 call_id 不合并)
        turn = self._current_turn.get(run_id, 0)
        key = (run_id, turn, result.call_id)
        if key in self._by_key:
            self._record_failure(f"duplicate ToolCompleted for terminal card {key}; ignored")
            return
        group = self._ensure_group("live", run_id, turn)
        target, params = self._summarize_args(self._args_by_call.get((run_id, result.call_id)))
        card = _ToolCard(
            run_id=run_id, turn_index=turn, call_id=result.call_id,
            name=_clean_text(result.name), args_target=target, args_params=params,
        )
        card.complete(result.content, result.is_error, artifact, mutation_state)
        self._append_card(group, card)

    def seal_run(self, run_id: str) -> None:
        """运行终结封存: 无结果子项标注未确认 (C1); 幂等。"""
        for group in self._groups:
            if group.run_id != run_id:
                continue
            for card in group.cards:
                if card.status in ("waiting", "running"):
                    card.seal_unconfirmed()
                    self._touch()

    def add_completed_card(
        self, *, run_id: str, turn_index: int, call_id: str, name: str,
        arguments_json: str | None, content: str, is_error: bool,
        artifact: str | None = None, details: dict | None = None,
        unconfirmed: bool = False,
    ) -> tuple[int, bool]:
        """恢复投影入口 (票据 05/01): 以终态挂一张卡, 返回 (组下标, 是否新建组)。

        run_id 由调用方命名空间化 (restored:<request_id>); 同键重复是调用方缺陷,
        记 ui_failures 并返回现有组。unconfirmed=True 用于模型已声明但存档无对应
        结果的中断轮调用: 以未确认如实显示, 不伪造完成也不伪造输出。
        """
        scope = _scope_of(run_id)
        key = (run_id, turn_index, call_id)
        existing = self._by_key.get(key)
        if existing is not None:
            self._record_failure(f"duplicate restored card {key}; ignored")
            group = self._group_by_key[(scope, run_id, turn_index)]
            return self._groups.index(group), False
        group, created = self._ensure_group_tracked(scope, run_id, turn_index)
        target, params = self._summarize_args(arguments_json)
        card = _ToolCard(
            run_id=run_id, turn_index=turn_index, call_id=call_id,
            name=_clean_text(name), args_target=target, args_params=params,
        )
        if unconfirmed:
            card.status = "unconfirmed"
            card.is_error = is_error
        else:
            card.complete(content, is_error, artifact, _mutation_state_of(details))
        self._append_card(group, card)
        return self._groups.index(group), created

    # --- 只读视图 ---------------------------------------------------------------

    @property
    def expanded(self) -> bool:
        """全局默认展开标志 (Ctrl+O 统一状态的默认值; 逐项覆盖不改变它)。"""
        return self._default_expanded

    @property
    def entries(self) -> tuple[ToolCardView, ...]:
        """当前条目快照 (按到达序); 修改快照不影响文档。"""
        return tuple(
            ToolCardView(
                run_id=card.run_id,
                turn_index=card.turn_index,
                call_id=card.call_id,
                name=card.name,
                status=card.status,
                is_error=card.is_error,
                cancelled=card.cancelled,
                output_text=card.content,
            )
            for card in self._cards
        )

    @property
    def groups(self) -> tuple[ToolGroupView, ...]:
        """当前组快照 (按创建序); 组合层按数量差挂载新组条目。"""
        return tuple(
            ToolGroupView(
                scope=group.scope,
                run_id=group.run_id,
                turn_index=group.turn_index,
                call_ids=tuple(card.call_id for card in group.cards),
                expanded=self._group_expanded(group),
            )
            for group in self._groups
        )

    @property
    def version(self) -> int:
        """单调内容版本 (条目/输出/展开/预算变化即递增); 渲染缓存键用。"""
        return self._version

    # --- 展开状态 -----------------------------------------------------------------

    def toggle_expand(self) -> bool:
        """Ctrl+O (C2): 统一所有现有组/单工具为同一开合状态并作为后续默认值。

        组打开只露列表 (子项详情重置收起), 单工具最多初始预览; 逐项覆盖被
        重新统一。返回新的默认标志 (组合层同步 Skill 正文用)。
        """
        self._default_expanded = not self._default_expanded
        for group in self._groups:
            group.expanded = None
            for card in group.cards:
                card.expanded = None
                card.budget = INITIAL_OUTPUT_LINES
        self._touch()
        return self._default_expanded

    def toggle_group(self, key: tuple[str, str, int]) -> None:
        """逐项开关组 (点击/键盘动作): 只覆盖当前组; 再打开时子项预算回初始值。"""
        group = self._group_by_key.get(tuple(key))
        if group is None:
            self._record_failure(f"toggle_group unknown key {key}; ignored")
            return
        opening = not self._group_expanded(group)
        group.expanded = opening
        if opening:
            for card in group.cards:
                card.budget = INITIAL_OUTPUT_LINES
        self._touch()

    def toggle_tool(self, key: tuple[str, str, int, str]) -> None:
        """逐项开关工具详情; 单工具组直接开关详情 (无中间组头层)。再次打开从
        初始输出行预算开始 (C2)。"""
        card = self._card_by_key(tuple(key))
        if card is None:
            self._record_failure(f"toggle_tool unknown key {key}; ignored")
            return
        group = self._group_by_key[(card_scope(card), card.run_id, card.turn_index)]
        if len(group.cards) == 1:
            self.toggle_group(group.key)
            return
        if not self._group_expanded(group):
            # 组收起时子项不可达 (防御: 动作只来自已呈现命中)
            self._record_failure(f"toggle_tool on collapsed group {group.key}; ignored")
            return
        opening = not self._card_expanded(card)
        card.expanded = opening
        if opening:
            card.budget = INITIAL_OUTPUT_LINES
        self._touch()

    def tool_output_more(self, key: tuple[str, str, int, str]) -> None:
        """「更多」: 同一展开过程累加输出预算 (每次 +20 物理行)。"""
        card = self._card_by_key(tuple(key))
        if card is None:
            self._record_failure(f"tool_output_more unknown key {key}; ignored")
            return
        card.budget += MORE_OUTPUT_STEP
        self._touch()

    def tool_output_reset(self, key: tuple[str, str, int, str]) -> None:
        """「收起详情」: 恢复初次预览 (6 输出行), 详情保持打开。"""
        card = self._card_by_key(tuple(key))
        if card is None:
            self._record_failure(f"tool_output_reset unknown key {key}; ignored")
            return
        card.budget = INITIAL_OUTPUT_LINES
        self._touch()

    # --- 展示 -------------------------------------------------------------------

    def layout_group(
        self, index: int, width: int, selected: DisplayAction | None = None
    ) -> EntryLayout:
        """单组同源布局 (票据 01 组合): 行/命中/锚点同一构造产物。

        越界是组合层缺陷: 有界处理 (记 ui_failures 返回空布局), 不抛异常杀渲染。
        """
        width = max(1, int(width))
        if not 0 <= index < len(self._groups):
            self._record_failure(f"layout_group index out of range: {index}")
            return EntryLayout((), (), ())
        cached = self._group_cache.get(index)
        style = current_style()
        if cached is not None and cached[:4] == (width, self._version, style, selected):
            return cached[4]
        layout = self._layout_group(self._groups[index], width, selected)
        self._group_cache[index] = (width, self._version, style, selected, layout)
        return layout

    def render_group_lines(self, index: int, width: int) -> list[str]:
        """兼容包装: 同一布局的 lines (不另算点击坐标, C3)。"""
        return list(self.layout_group(index, width).lines)

    def render_lines(self, width: int) -> list[str]:
        """按当前列宽产出全部组的显示行 (测试/诊断用)。"""
        width = max(1, int(width))
        lines: list[str] = []
        for index in range(len(self._groups)):
            lines.extend(self.layout_group(index, width).lines)
        return lines

    def _layout_group(
        self, group: _ToolGroup, width: int, selected: DisplayAction | None
    ) -> EntryLayout:
        """组布局 (C2 状态表): 前导空行 + 标题/子项短行 + 有界详情。"""
        lines: list[str] = [""]
        anchors: list[EntryAnchor | None] = [None]
        hits: list[HitSpan] = []
        group_key = group.key
        group_action = DisplayAction(ACTION_TOGGLE, (TARGET_GROUP, *group_key))
        expanded = self._group_expanded(group)

        if len(group.cards) == 1:
            # 单工具: 省略冗余组头, 标题直接开关详情
            card = group.cards[0]
            tool_key = (card_scope(card), card.run_id, card.turn_index, card.call_id)
            action = DisplayAction(ACTION_TOGGLE, (TARGET_TOOL, *tool_key))
            row = len(lines)
            lines.append(self._tool_title_line(card, width, "", expanded, selected == action))
            anchors.append(EntryAnchor((TARGET_TOOL, *tool_key), 0))
            hits.append(HitSpan(row, 0, width, action))
            if expanded:
                self._append_detail(lines, hits, anchors, card, width, "  ", tool_key, selected)
            return EntryLayout(tuple(lines), tuple(hits), tuple(anchors))

        row = len(lines)
        lines.append(self._group_header_line(group, width, expanded, selected == group_action))
        anchors.append(EntryAnchor((TARGET_GROUP, *group_key), 0))
        hits.append(HitSpan(row, 0, width, group_action))
        if expanded:
            for card in group.cards:
                tool_key = (card_scope(card), card.run_id, card.turn_index, card.call_id)
                action = DisplayAction(ACTION_TOGGLE, (TARGET_TOOL, *tool_key))
                row = len(lines)
                lines.append(
                    self._tool_title_line(card, width, "  ", self._card_expanded(card), selected == action)
                )
                anchors.append(EntryAnchor((TARGET_TOOL, *tool_key), 0))
                hits.append(HitSpan(row, 0, width, action))
                if self._card_expanded(card):
                    self._append_detail(lines, hits, anchors, card, width, "    ", tool_key, selected)
        return EntryLayout(tuple(lines), tuple(hits), tuple(anchors))

    def _group_header_line(
        self, group: _ToolGroup, width: int, expanded: bool, selected: bool
    ) -> str:
        """组标题: 标记 + 动作概括 · 次数 + 异常/活动计数; 计数不被标题挤掉。"""
        marker = self._marker(group_worst_category(group), expanded, selected)
        label = f"{_group_label(group)} · {len(group.cards)} 次"
        counts = _category_counts(group)
        full_tail = "".join(f" · {counts[c]} {_CATEGORY_WORD[c]}" for c in _CATEGORY_ORDER if counts.get(c))
        compact_tail = "".join(f" {_CATEGORY_SYMBOL[c]}{counts[c]}" for c in _CATEGORY_ORDER if counts.get(c))
        full = f"{marker} {label}{full_tail}"
        if visible_width(full) <= width:
            return self._style_tail(full, full_tail, counts)
        compact = f"{marker} {label}{compact_tail}"
        if visible_width(compact) <= width:
            return self._style_tail(compact, compact_tail, counts)
        marker_plain = ">" if selected else ("-" if expanded else "+")
        avail = width - visible_width(marker_plain) - 1 - visible_width(compact_tail)
        if avail >= 1 and compact_tail:
            return self._style_tail(
                f"{marker_plain} {_ellipsize_middle(label, avail)}{compact_tail}", compact_tail, counts
            )
        return truncate_to_width(f"{marker_plain}{compact_tail}" if compact_tail else f"{marker_plain} {label}", width)

    def _style_tail(self, line: str, tail: str, counts: dict[str, int]) -> str:
        """异常/活动计数染色 (标记已含样式); 无计数原样。"""
        if not tail:
            return line
        worst = next((c for c in _CATEGORY_ORDER if counts.get(c)), "ok")
        return line[: len(line) - len(tail)] + theme.fg(_CATEGORY_ACCENT[worst], tail)

    def _marker(self, category: str, expanded: bool, selected: bool) -> str:
        """折叠标记: 选中 > / 展开 - / 收起 + (一列); 异常与活动按类别着色。"""
        glyph = ">" if selected else ("-" if expanded else "+")
        return theme.fg(_CATEGORY_ACCENT[category], glyph)

    def _tool_title_line(
        self, card: _ToolCard, width: int, indent: str, expanded: bool, selected: bool
    ) -> str:
        """工具短行/单工具标题: 标记 + 动作 + 目标 + 状态后缀; 一行物理行。"""
        indent = indent if visible_width(indent) < width else ""
        category = _card_category(card)
        verb = _TOOL_LABELS.get(card.name, (card.name, card.name))[0]
        body = f"{verb} {card.args_target}" if card.args_target else verb
        suffix, flag = _status_suffix(card, category)
        marker = self._marker(category, expanded, selected)
        prefix = f"{indent}{marker} "
        full = prefix + body + suffix
        if visible_width(full) <= width:
            return self._style_suffix(full, suffix, category)
        marker_plain = ">" if selected else ("-" if expanded else "+")
        plain_prefix = f"{indent}{marker_plain} "
        tail = f" {flag}" if flag else ""
        avail = width - visible_width(plain_prefix) - visible_width(tail)
        if avail >= 2:
            return self._style_suffix(plain_prefix + _ellipsize_middle(body, avail) + tail, tail, category)
        return truncate_to_width(f"{indent}{marker_plain}{flag}", width)

    def _style_suffix(self, line: str, suffix: str, category: str) -> str:
        if not suffix:
            return line
        return line[: len(line) - len(suffix)] + theme.fg(_CATEGORY_ACCENT[category], suffix)

    def _append_detail(
        self,
        lines: list[str],
        hits: list[HitSpan],
        anchors: list[EntryAnchor | None],
        card: _ToolCard,
        width: int,
        indent: str,
        tool_key: tuple[str, str, int, str],
        selected: DisplayAction | None,
    ) -> None:
        """工具详情 (C2): 参数至多 2 物理行 + 输出至多预算物理行 + 短分页动作。

        只为已披露内容换行: 输出逐逻辑行 wrap 到预算+1 即停, 收起/短预览不全量
        wrap 大结果; 分页不丢字不重复 (wrap 前缀)。锚点行为条目内显示行。
        """
        indent = indent if visible_width(indent) < width else ""
        cw = max(1, width - visible_width(indent))
        item_id = (TARGET_TOOL, *tool_key)
        item_row = 1

        def emit(line: str) -> None:
            nonlocal item_row
            lines.append(line)
            anchors.append(EntryAnchor(item_id, item_row))
            item_row += 1

        hint = theme.TOOL_HINT_FG
        if card.args_params:
            for chunk in _bounded_chunks(f"参数: {card.args_params}", cw, PARAMS_MAX_LINES):
                emit(theme.fg(hint, indent + chunk))
        if card.artifact:
            for chunk in _bounded_chunks(card.artifact, cw, ARTIFACT_MAX_LINES):
                emit(theme.fg(hint, indent + chunk))
        budget = max(INITIAL_OUTPUT_LINES, card.budget)
        shown = 0
        has_more = False
        if card.terminal and card.content:
            for logical in card.content.split("\n"):
                for chunk in word_wrap_line(logical, cw):
                    if shown >= budget:
                        has_more = True
                        break
                    # 极窄防御: 单图素超宽 (1 列 CJK) 截断, 不破版 ( reasoning_view 同纪律)
                    emit(theme.fg(theme.TOOL_OUTPUT_FG, indent + truncate_to_width(chunk.text, cw)))
                    shown += 1
                if has_more:
                    break
        if has_more or card.budget > INITIAL_OUTPUT_LINES:
            row = len(lines)
            line = self._paging_line(card, indent, cw, tool_key, selected, has_more, row, hits)
            emit(line)

    def _paging_line(
        self,
        card: _ToolCard,
        indent: str,
        cw: int,
        tool_key: tuple[str, str, int, str],
        selected: DisplayAction | None,
        has_more: bool,
        row: int,
        hits: list[HitSpan],
    ) -> str:
        """短分页动作行: 「更多」(还有未披露输出) 与「收起详情」(预算超初始)。

        两个动作同一行各持命中区间; 选中动作用有色加粗 (plain 用 [方括号])。
        """
        segments: list[tuple[str, DisplayAction]] = []
        if has_more:
            segments.append(("更多", DisplayAction(ACTION_MORE, (TARGET_TOOL, *tool_key))))
        if card.budget > INITIAL_OUTPUT_LINES:
            segments.append(("收起详情", DisplayAction(ACTION_RESET, (TARGET_TOOL, *tool_key))))
        style = current_style()
        limit = visible_width(indent) + cw
        line = indent
        col = visible_width(indent)
        pending_hits: list[HitSpan] = []
        for index, (label, action) in enumerate(segments):
            if index:
                line += theme.fg(theme.TOOL_HINT_FG, " · ")
                col += 3
            start = col
            if action == selected:
                if style.plain:
                    line += f"[{label}]"
                else:
                    line += style.sgr("1") + theme.fg_open(theme.TOOL_ACCENT_RUNNING) + label + style.sgr("0")
                col += visible_width(label) + (2 if style.plain else 0)
            else:
                line += theme.fg(theme.TOOL_HINT_FG, label)
                col += visible_width(label)
            pending_hits.append(HitSpan(row, start, col, action))
        if visible_width(line) > limit:
            # 极窄防御: 动作标签超宽时截断, 命中区域同步放弃 (不产生不可达动作)
            return truncate_to_width(line, limit)
        hits.extend(pending_hits)
        return line

    # --- 内部 -------------------------------------------------------------------

    def _group_expanded(self, group: _ToolGroup) -> bool:
        return group.expanded if group.expanded is not None else self._default_expanded

    @staticmethod
    def _card_expanded(card: _ToolCard) -> bool:
        return bool(card.expanded)

    def _ensure_group(self, scope: str, run_id: str, turn_index: int) -> _ToolGroup:
        return self._ensure_group_tracked(scope, run_id, turn_index)[0]

    def _ensure_group_tracked(self, scope: str, run_id: str, turn_index: int) -> tuple[_ToolGroup, bool]:
        """取或建组; 返回 (组, 是否新建) — 新建组由组合层挂载显示条目。"""
        key = (scope, run_id, turn_index)
        group = self._group_by_key.get(key)
        if group is None:
            group = _ToolGroup(scope, run_id, turn_index)
            self._group_by_key[key] = group
            self._groups.append(group)
            self._touch()
            return group, True
        return group, False

    def _append_card(self, group: _ToolGroup, card: _ToolCard) -> None:
        group.cards.append(card)
        self._cards.append(card)
        self._by_key[(card.run_id, card.turn_index, card.call_id)] = card
        self._touch()

    def _card_by_key(self, key: tuple[str, str, int, str]) -> _ToolCard | None:
        scope, run_id, turn_index, call_id = key
        return self._by_key.get((run_id, turn_index, call_id))

    def _summarize_args(self, raw: str | None) -> tuple[str, str]:
        """(标题目标, 详情参数) 单行有界文本; 全部来自真实事件文本。"""
        if raw is None:
            return "", ""
        text = raw.strip()
        if not text or text == "{}":
            return "", ""
        try:
            obj = json.loads(text)
        except ValueError:
            return "", self._cap(_clean_text(text), _PARAMS_MAX_CHARS)
        if isinstance(obj, dict) and obj:
            target = ""
            for key in _PREFERRED_ARG_KEYS:
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    target = _clean_text(value)
                    break
                if value is not None and not isinstance(value, (dict, list)):
                    target = _clean_text(str(value))
                    break
            params = self._cap(
                _clean_text(" · ".join(f"{key}: {self._param_value(value)}" for key, value in obj.items())),
                _PARAMS_MAX_CHARS,
            )
            return self._cap(target, _TARGET_MAX_CHARS), params
        if isinstance(obj, list):
            return "", self._cap(_clean_text(f"（{len(obj)} 项）"), _PARAMS_MAX_CHARS)
        return "", self._cap(_clean_text(str(obj)), _PARAMS_MAX_CHARS)

    @staticmethod
    def _param_value(value: object) -> str:
        """参数值摘要: 标量有界原文, 容器只报项数 (不默认输出 raw JSON, D2)。"""
        if isinstance(value, str):
            text = value if len(value) <= _PARAM_VALUE_MAX_CHARS else value[:_PARAM_VALUE_MAX_CHARS] + "…"
            return text
        if isinstance(value, (dict, list)):
            return f"（{len(value)} 项）"
        return str(value)

    @staticmethod
    def _cap(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[:limit] + "…"

    def _touch(self) -> None:
        self._version += 1

    def _record_failure(self, message: str) -> None:
        if len(self.ui_failures) < _UI_FAILURES_MAX:
            self.ui_failures.append(message)
        self._touch()


def card_scope(card: _ToolCard) -> str:
    return _scope_of(card.run_id)


def _mutation_state_of(details: dict | None) -> str | None:
    """结构化文件修改事实 (C2): 只读 details["mutation"]["state"], 不认文字推断。"""
    if not isinstance(details, dict):
        return None
    mutation = details.get("mutation")
    if not isinstance(mutation, dict):
        return None
    state = mutation.get("state")
    return state if isinstance(state, str) else None


def _status_suffix(card: _ToolCard, category: str) -> tuple[str, str]:
    """(完整后缀, 极窄符号); 正常完成无后缀。"""
    if category == "active":
        return (" · 等待", "~") if card.status == "waiting" else (" · 进行中", "~")
    if category == "failed":
        return " · 失败", "!"
    if category == "cancelled":
        return " · 已取消", "c"
    if category == "unconfirmed":
        return " · 未确认", "?"
    return "", ""


def _group_label(group: _ToolGroup) -> str:
    """组概括标签 (C2): 纯单类用准确动作, 混合为「调用工具」; N 是调用次数。"""
    names = {card.name for card in group.cards}
    if len(names) == 1:
        name = next(iter(names))
        return _TOOL_LABELS.get(name, (name, name))[1]
    return _MIXED_GROUP_LABEL


def _category_counts(group: _ToolGroup) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in group.cards:
        category = _card_category(card)
        if category == "ok":
            continue
        counts[category] = counts.get(category, 0) + 1
    return counts


def group_worst_category(group: _ToolGroup) -> str:
    for category in _CATEGORY_ORDER:
        if any(_card_category(card) == category for card in group.cards):
            return category
    return "ok"


def _bounded_chunks(text: str, width: int, max_lines: int) -> list[str]:
    """单行文本按宽度换行并限物理行数; 超限末行尾部省略 (参数/工件区用)。

    极窄防御: 单图素超过宽度时 (如 1 列里的 CJK) 截断到宽度, 与
    ReasoningBlock 正文同一纪律。
    """
    bounded = max(1, width)
    chunks = [truncate_to_width(chunk.text, bounded) for chunk in word_wrap_line(text, bounded)]
    if len(chunks) <= max_lines:
        return chunks
    kept = chunks[:max_lines]
    kept[-1] = truncate_to_width(kept[-1], bounded)  # 省略号计入宽度预算
    return kept


class ToolGroupEntry:
    """单组 transcript 条目适配 (票据 01 组合): 按创建序委托 ToolDocument 布局。

    每个工具组在创建时经 transcript.add_entry 挂载一个本适配条目, 保持文档
    顺序与消息交错一致; 渲染按 (宽度, 文档版本, 样式, 选中动作) 缓存。
    """

    def __init__(self, document: "ToolDocument", index: int) -> None:
        self._document = document
        self._index = index

    def render(self, width: int) -> list[str]:
        """兼容包装 (C3): 同一布局的 lines。"""
        return self._document.render_group_lines(self._index, width)

    def layout(
        self, width: int, selected: DisplayAction | None = None, display_id: int = 0
    ) -> EntryLayout:
        return self._document.layout_group(self._index, width, selected)


__all__ = [
    "ARTIFACT_MAX_LINES",
    "INITIAL_OUTPUT_LINES",
    "MORE_OUTPUT_STEP",
    "PARAMS_MAX_LINES",
    "CardStatus",
    "ToolDocument",
    "ToolCardView",
    "ToolGroupEntry",
    "ToolGroupView",
]
