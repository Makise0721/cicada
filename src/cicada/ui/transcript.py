"""显示文档 (票据 02): 真实会话消息条目的原位流式状态与排版.

职责 (规格 §原位文本和工具更新 / 设计 §显示文档与事件):
- 保存用户/助手/系统/错误/取消条目, 顺序与身份稳定; assistant 条目以
  (run_id, turn_index) 定位, AssistantTextDelta 累积、AssistantCompleted 校准
  同一条目, 完成后不二次追加, 迟到增量不复活已完成条目。
- 不沿用旧 12 行/4096 bytes 活动预览与 64 行正文裁剪: 保存已收到的完整
  正文, 渲染按当前宽度全量换行, 不擅自删掉显示历史。
- 外部文本先显示清理 (sanitize_display) 再排版; 每一行产出保证
  visible_width <= width, 超宽即本模块缺陷 (renderer 只做防御截断)。
- 文档不是规范历史, 不参与模型请求; 工具条目 (票据 03) 可经 add_entry
  挂载本文档, 条目协议只有 render(width) -> list[str] 一个方法。
- Skill 显式请求 (local-skills-v1 票据 03): append_user 用中立解码器识别
  封装, 成功时挂 SkillRequestEntry (UserEntry 显示实现, text 保留完整
  原文本); 展开状态经构造时注入的只读 getter 读全局展开标志 (Ctrl+O 与
  工具输出共享), 未注入时完整显示正文。解码失败保持原 UserEntry。
- 思考块 (thinking-budget-visibility 票据 01): 助手条目内答案前独立呈现
  message.reasoning (ui/reasoning_view), 缺失省略; 展开状态是本文档唯一
  bool, 初始收起, 区别于工具/Skill 的 expand_getter (04 用户裁决移除
  Ctrl+T 统一切换, 逐项点击为唯一入口)。
- 思考流中预览 (票据 02): stream_reasoning_delta 把只读思考增量送入同一条目
  的有界预览 (空增量也建立条目); finish_reasoning_previews 按 run_id 幂等封存
  未完成预览并登记终结运行, 迟到增量不复活。
- ModelCallCompleted 计量在此累计: None 字段显示"不可得", 不当作 0;
  展示出口 (底栏) 由票据 04 组装, 本模块只提供口径文本。
"""

from __future__ import annotations

from typing import Callable, Protocol

from cicada.core.messages import AssistantMessage
from cicada.core.ports import ModelMetrics
from cicada.skills import SkillInvocation, decode_skill_request
from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.document_layout import (
    ACTION_TOGGLE,
    TARGET_ASSISTANT,
    TARGET_ENTRY,
    TARGET_REASONING,
    TARGET_SKILL,
    DisplayAction,
    EntryAnchor,
    EntryLayout,
    DocumentLayout,
    HitSpan,
    wrap_plain_entry,
)
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.reasoning_view import ReasoningBlock
from cicada.ui.rendering import sanitize_display

ERROR_PREFIX = "错误: "
CANCELLED_TEXT = "已取消: 本次请求在完成前被中止。"
LENGTH_NOTE = "（回答因达到输出长度上限被截断）"
STREAM_PLACEHOLDER = "…"


class CacheMetricsTotal:
    """调用快照的只读展示累计；单边值不参与成对份额。"""

    def __init__(self) -> None:
        self.calls = 0
        self.paired_calls = 0
        self.hit: int | None = None
        self.miss: int | None = None
        self.paired_hit = 0
        self.paired_tokens = 0

    def add(self, metrics: ModelMetrics | None) -> None:
        self.calls += 1
        if metrics is None:
            return
        hit, miss = metrics.cache_hit_input_tokens, metrics.cache_miss_input_tokens
        if hit is not None:
            self.hit = (self.hit or 0) + hit
        if miss is not None:
            self.miss = (self.miss or 0) + miss
        if hit is not None and miss is not None:
            self.paired_calls += 1
            self.paired_hit += hit
            self.paired_tokens += hit + miss

    def summary(self, label: str) -> str:
        if not self.calls:
            return f"{label}: 尚无调用"
        hit = "不可得" if self.hit is None else str(self.hit)
        miss = "不可得" if self.miss is None else str(self.miss)
        share = f"{self.paired_hit / self.paired_tokens:.1%}" if self.paired_tokens else "不可得"
        partial = "（部分不可得）" if self.paired_calls < self.calls else ""
        return (
            f"{label}: 已知命中 {hit} · 已知未命中 {miss}"
            f" · 成对样本份额 {share} · 覆盖 {self.paired_calls}/{self.calls} 次{partial}"
        )


class TranscriptEntry(Protocol):
    """显示文档条目协议 (票据 03 的工具条目同样实现)。"""

    def render(self, width: int) -> list[str]: ...


def _wrap_lines(text: str, width: int) -> list[str]:
    """清理后的多行文本按宽度换行; 每行可见列宽 <= width。"""
    lines: list[str] = []
    for logical in text.split("\n"):
        lines.extend(wrap_text_ansi(logical, max(1, width)))
    return lines


def _wrap_with_prefix(prefix: str, text: str, width: int) -> list[str]:
    """带前缀的换行: 前缀占列, 正文按剩余宽度换行。"""
    content_width = max(1, width - visible_width(prefix))
    lines: list[str] = []
    for logical in text.split("\n"):
        for chunk in wrap_text_ansi(logical, content_width):
            lines.append(prefix + chunk)
    return lines if lines else [prefix.rstrip()]


def _fit(lines: list[str], width: int) -> list[str]:
    """逐行防御: 任何超宽行截断到 width (正常换行后不应触发)。"""
    result: list[str] = []
    for line in lines:
        if visible_width(line) > width:
            line = truncate_to_width(line, width)
        result.append(line)
    return result


class UserEntry:
    """用户消息条目: Pi 样式背景盒 (上下 1 行、左右 1 列), 原文换行保留。

    与所有条目一致自带 1 行前导空行 (间距与条目类型无关)。
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        return ["", *theme.user_message_lines(self.text, width)]


class SkillRequestEntry(UserEntry):
    """Skill 显式请求条目 (local-skills-v1 票据 03): 用户任务 + 独立 Skill 文档部分.

    只在 append_user 解码成功时使用; text 保留完整封装原文本 (UserEntry 契约),
    显示派生自记录数据, 不读当前 Skill 文件 (恢复共用同一条目)。默认收起正文,
    展开状态经只读 getter 取全局展开默认 (与工具输出共用 Ctrl+O); 未注入
    getter 时完整显示正文, 避免无展开手段的调用者看不到内容。
    票据 01 (契约 C1/C2): 逐项展开覆盖优先于全局默认; 标题行带折叠标记并
    持 toggle 命中 (目标 ("skill", display_id)), 正文不生成动作。
    """

    def __init__(
        self, text: str, invocation: SkillInvocation,
        expand_getter: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(text)
        self.invocation = invocation
        self._expand_getter = expand_getter
        self._override: bool | None = None  # 逐项展开覆盖 (票据 01)
        self._render_cache: tuple[int, bool, bool, DisplayStyle, list[str], tuple[int, int]] | None = None

    def is_expanded(self) -> bool:
        if self._override is not None:
            return self._override
        return True if self._expand_getter is None else bool(self._expand_getter())

    def toggle(self) -> None:
        """逐项开关 (票据 01): 只覆盖本条目, 下一次 Ctrl+O 重新统一。"""
        self._override = not self.is_expanded()

    def set_expanded(self, expanded: bool) -> None:
        """Ctrl+O 统一 (票据 01): 现有 Skill 与后续默认值同为该开合状态。"""
        self._override = expanded

    def render(self, width: int) -> list[str]:
        return list(self.layout(width).lines)

    def layout(
        self, width: int, selected: DisplayAction | None = None, display_id: int = 0
    ) -> EntryLayout:
        expanded = self.is_expanded()
        action = DisplayAction(ACTION_TOGGLE, (TARGET_SKILL, display_id))
        is_selected = selected == action
        cached = self._render_cache
        style = current_style()
        if cached is not None and cached[:4] == (width, expanded, is_selected, style):
            lines, (header_start, header_rows) = list(cached[4]), cached[5]
        else:
            lines, header_start, header_rows = self._render_lines(width, expanded, is_selected)
            self._render_cache = (width, expanded, is_selected, style, list(lines), (header_start, header_rows))
        hits = tuple(
            HitSpan(row, 0, width, action)
            for row in range(header_start, min(header_start + header_rows, len(lines)))
        )
        anchors = tuple(
            None if row == 0 else EntryAnchor((TARGET_SKILL, display_id), row)
            for row in range(len(lines))
        )
        return EntryLayout(tuple(lines), hits, anchors)

    def _render_lines(self, width: int, expanded: bool, selected: bool) -> tuple[list[str], int, int]:
        """返回 (显示行, 标题起始行, 标题行数); 标题行是同源命中的唯一区域。"""
        invocation = self.invocation
        # 用户任务部分: 与普通用户消息同一背景盒样式
        lines = ["", *theme.user_message_lines(invocation.task, width)]
        header_start = len(lines)
        # 标题整体显示清理后才换行 (wrap_text_ansi 契约要求调用方先剥 ANSI):
        # source_path 是记录文本, 存档/粘贴可携带任意控制序列 (审查处置 F2)
        marker = ">" if selected else ("-" if expanded else "+")
        header = sanitize_display(
            f"{marker} Skill 文档 {invocation.name}（{invocation.source_path}）"
            + (f" · sha256:{invocation.body_sha256[:12]}" if expanded else "")
        )
        header_chunks = wrap_text_ansi(header, max(1, width))
        for chunk in header_chunks:
            lines.append(theme.fg(theme.MD_CODE, chunk))
        if not expanded:
            for chunk in wrap_text_ansi("正文已收起 · Ctrl+O 展开完整文档", max(1, width)):
                lines.append(theme.fg(theme.MUTED_GRAY, chunk))
            return lines, header_start, len(header_chunks)
        # 正文来自记录: 规范换行后显示清理, 与工具输出同一纪律 (不执行外部序列)
        body = sanitize_display(
            invocation.body.replace("\r\n", "\n").replace("\r", "\n")
        )
        lines.extend(_wrap_with_prefix("  ", body, width))
        return lines, header_start, len(header_chunks)


class AssistantEntry:
    """助手消息条目: 以 (run_id, turn_index) 定位, 流式累积, 完成校准。

    渲染为离屏 Markdown (Pi 样式, 前置空行, 左右 1 列); 缓存按
    (正文, 宽度) 复用, 流式增量未变化时不重复排版。
    思考块 (票据 01): 答案前独立呈现, complete/restore 以 message.reasoning
    权威替换 (None 撤掉); 顺序为思考、答案、终结标注。
    """

    def __init__(
        self, run_id: str, turn_index: int,
        *, reasoning_expand_getter: Callable[[], bool] | None = None,
    ) -> None:
        self.run_id = run_id
        self.turn_index = turn_index
        self.text = ""
        self.finalized = False
        self.stop_reason: str | None = None
        self.error: str | None = None
        self.reasoning = ReasoningBlock(expand_getter=reasoning_expand_getter)
        self._render_cache: tuple[str, int, DisplayStyle, list[str]] | None = None

    def render(self, width: int) -> list[str]:
        return list(self.layout(width).lines)

    def layout(
        self, width: int, selected: DisplayAction | None = None, display_id: int = 0
    ) -> EntryLayout:
        """同源布局 (票据 01): 思考标题行持 toggle 命中, 锚点区分思考/答案。"""
        reasoning_action = DisplayAction(ACTION_TOGGLE, (TARGET_REASONING, self.run_id, self.turn_index))
        reasoning_lines = self.reasoning.render(width, selected=selected == reasoning_action)
        cleaned = sanitize_display(self.text)
        if cleaned:
            cached = self._render_cache
            style = current_style()
            if cached is not None and cached[:3] == (cleaned, width, style):
                lines = list(cached[3])
            else:
                lines = theme.assistant_markdown_lines(cleaned, width)
                self._render_cache = (cleaned, width, style, list(lines))
        elif not self.finalized:
            lines = [STREAM_PLACEHOLDER]  # 已建条目尚未收到任何增量
        elif reasoning_lines:
            lines = []  # 完成无正文但有思考: 空行 + 思考块
        else:
            return EntryLayout((), (), ())  # 完成但无正文: 事实由错误/取消条目表达
        # 统一前导空行; 思考块在答案前 (票据 01: 思考、答案、终结标注)
        result = ["", *reasoning_lines, *lines]
        assistant_rows = len(lines)
        if self.finalized and self.stop_reason == "length":
            result.append(theme.fg(theme.MUTED_GRAY, LENGTH_NOTE))
            assistant_rows += 1
        anchors: list[EntryAnchor | None] = [None]
        anchors += [
            EntryAnchor((TARGET_REASONING, self.run_id, self.turn_index), row)
            for row in range(len(reasoning_lines))
        ]
        anchors += [
            EntryAnchor((TARGET_ASSISTANT, self.run_id, self.turn_index), row)
            for row in range(assistant_rows)
        ]
        hits = (HitSpan(1, 0, width, reasoning_action),) if reasoning_lines else ()
        return EntryLayout(tuple(result), hits, tuple(anchors))


class SystemEntry:
    """系统条目: 命令输出等程序事实, 无角色前缀。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        prefix = " " if width > 4 else ""
        return ["", *_wrap_with_prefix(prefix, sanitize_display(self.text), width)]


class InformationEntry(SystemEntry):
    """命令信息块: 事实正文与标题/分组装饰分开保存。"""

    def __init__(self, title: str, sections: list[tuple[str, list[str]]]) -> None:
        super().__init__("\n".join(row for _, rows in sections for row in rows))
        self.title = title
        self.sections = [(heading, list(rows)) for heading, rows in sections]

    def render(self, width: int) -> list[str]:
        return theme.information_lines(self.title, self.sections, width)


class ErrorEntry:
    """错误条目: 如实显示错误事实 (模型错误/请求被拒/异常)。"""

    def __init__(self, text: str) -> None:
        self.text = text

    def render(self, width: int) -> list[str]:
        return ["", *_wrap_with_prefix(ERROR_PREFIX, sanitize_display(self.text), width)]


class CancelledEntry:
    """取消条目: 本次请求被用户中止的事实。"""

    def render(self, width: int) -> list[str]:
        return ["", *_wrap_lines(CANCELLED_TEXT, width)]


class Transcript:
    """显示文档: 条目有序集合 + 助手条目定位索引 + 计量累计。

    expand_getter (local-skills-v1 票据 03): 只读展开标志 getter, 生产组合注入
    工具文档的全局展开状态 (Ctrl+O 同时影响工具输出与 Skill 正文); 未注入时
    SkillRequestEntry 完整显示正文。
    思考展开 (票据 01): 本文档唯一 bool, 初始 false (文档默认收起; 04 用户
    裁决移除 Ctrl+T 统一切换后不再变化, 逐项覆盖由点击维护); /clear 保留
    显示文档因此保留该状态, 新建/恢复文档自然重置。
    """

    def __init__(self, *, expand_getter: Callable[[], bool] | None = None) -> None:
        self._entries: list[TranscriptEntry] = []
        self._entry_ids: list[int] = []  # 与 _entries 平行的显示序号 (只追加, 稳定身份)
        self._next_display_id = 0
        self._assistant_index: dict[tuple[str, int], AssistantEntry] = {}
        self._expand_getter = expand_getter
        self._reasoning_expanded = False
        # 票据 02: 已终结运行的 run_id 集合 (与显示文档同生命周期);
        # 终结运行的迟到思考增量被忽略, 不新建终结运行的新预览
        self._finished_runs: set[str] = set()
        # 计量累计: None = 尚无任何数值 (显示不可得); unknown = 有调用但该字段缺失
        self._input_total: int | None = None
        self._output_total: int | None = None
        self._input_unknown = False
        self._output_unknown = False
        self._cache_total = CacheMetricsTotal()
        self._metric_calls: set[tuple[str, int]] = set()

    # --- 条目管理 ----------------------------------------------------------------

    def _add(self, entry: TranscriptEntry) -> None:
        self._entries.append(entry)
        self._entry_ids.append(self._next_display_id)
        self._next_display_id += 1

    def entries(self) -> list[TranscriptEntry]:
        """当前条目快照 (顺序即文档顺序), 供测试与诊断。"""
        return list(self._entries)

    def add_entry(self, entry: TranscriptEntry) -> None:
        """追加任意条目 (票据 03 的工具条目经此挂载, 保持文档顺序稳定)。"""
        self._add(entry)

    def append_user(self, text: str) -> UserEntry:
        """用户条目: 中立解码器识别 Skill 显式请求封装 (local-skills-v1 票据 03)。

        解码成功挂 SkillRequestEntry (UserEntry 显示实现), 解码失败 (含全部
        普通文本) 保持原 UserEntry; 返回契约与完整原 text 不变。
        """
        invocation = decode_skill_request(text)
        entry: UserEntry
        if invocation is not None:
            entry = SkillRequestEntry(text, invocation, expand_getter=self._expand_getter)
        else:
            entry = UserEntry(text)
        self._add(entry)
        return entry

    def append_system(self, text: str) -> SystemEntry:
        entry = SystemEntry(text)
        self._add(entry)
        return entry

    def append_information(self, title: str, sections: list[tuple[str, list[str]]]) -> None:
        self._add(InformationEntry(title, sections))

    def append_error(self, text: str) -> ErrorEntry:
        entry = ErrorEntry(text)
        self._add(entry)
        return entry

    def append_cancelled(self) -> CancelledEntry:
        entry = CancelledEntry()
        self._add(entry)
        return entry

    # --- 助手条目原位更新 -----------------------------------------------------------

    def stream_assistant_delta(self, run_id: str, turn_index: int, text: str) -> None:
        """AssistantTextDelta: 累积到 (run_id, turn_index) 条目; 已完成条目丢弃迟到增量。"""
        if not text:
            return
        entry = self._assistant_entry(run_id, turn_index)
        if entry.finalized:
            return  # 迟到增量不复活已完成条目
        entry.text += text

    def stream_reasoning_delta(self, run_id: str, turn_index: int, text: str) -> None:
        """AssistantReasoningDelta (票据 02): 有界预览进入 (run_id, turn_index) 条目。

        空字符串同样建立条目 (区分未返回字段与已收到空字段); 已完成条目与
        已终结运行的迟到增量忽略, 不复活条目、不新建终结运行的新预览。
        """
        if run_id in self._finished_runs:
            return
        entry = self._assistant_entry(run_id, turn_index)
        if entry.finalized:
            return  # 完成后迟到增量: 权威记录已原位替换, 不复活预览
        entry.reasoning.append_preview(text)

    def finish_reasoning_previews(self, run_id: str) -> None:
        """按 run_id 封存尚未完成的思考预览 (票据 02, 幂等)。

        无完成记录的防御性终止只封存为未完成预览, 不生成规范思考或
        complete=true; 已有权威记录的条目不受影响。之后该运行的迟到思考
        增量被忽略 (终结运行集合与显示文档同生命周期)。
        """
        self._finished_runs.add(run_id)
        for entry in self._entries:
            if isinstance(entry, AssistantEntry) and entry.run_id == run_id:
                entry.reasoning.seal_preview()

    def complete_assistant(self, run_id: str, turn_index: int, message: AssistantMessage) -> None:
        """AssistantCompleted: 校准同一条目的完整正文与终结事实, 不二次追加。

        message 本身不带 turn_index, 由调用方按内核 TurnStarted 跟踪的当前轮传入。
        思考记录 (票据 01): message.reasoning 权威替换条目思考块, None 撤掉。
        """
        entry = self._assistant_entry(run_id, turn_index)
        entry.text = message.text  # 完成事件是权威: 校准 (非追加) 同一条目
        entry.finalized = True
        entry.stop_reason = message.stop_reason
        entry.error = message.error
        entry.reasoning.set_record(message.reasoning)

    def restore_assistant(self, key_run_id: str, turn_index: int, message: AssistantMessage) -> None:
        """恢复投影 (票据 05): 以终态直建助手条目; key_run_id 必须命名空间化,
        避免与本进程新 run_id (如 run-1) 碰撞 (A13)。

        思考 (票据 01): 原始记录直接进显示引用, 命名空间与归属不变。
        """
        entry = self._assistant_entry(key_run_id, turn_index)
        entry.text = message.text
        entry.finalized = True
        entry.stop_reason = message.stop_reason
        entry.error = message.error
        entry.reasoning.set_record(message.reasoning)

    def toggle_reasoning_item(self, run_id: str, turn_index: int) -> bool:
        """逐项开关指定思考块 (紧凑交互票据 01); 条目缺失或无记录返回 False。"""
        entry = self._assistant_index.get((run_id, turn_index))
        if entry is None or not entry.reasoning.has_record:
            return False
        entry.reasoning.toggle()
        return True

    def set_skills_expanded(self, expanded: bool) -> None:
        """Ctrl+O 统一 (紧凑交互票据 01): 现有 Skill 条目与该标志同步。"""
        for entry in self._entries:
            if isinstance(entry, SkillRequestEntry):
                entry.set_expanded(expanded)

    def toggle_skill(self, display_id: int) -> bool:
        """逐项开关指定 Skill 条目 (紧凑交互票据 01); 条目缺失返回 False。"""
        for entry, entry_id in zip(self._entries, self._entry_ids):
            if entry_id == display_id and isinstance(entry, SkillRequestEntry):
                entry.toggle()
                return True
        return False

    def _assistant_entry(self, run_id: str, turn_index: int) -> AssistantEntry:
        key = (run_id, turn_index)
        entry = self._assistant_index.get(key)
        if entry is None:
            entry = AssistantEntry(
                run_id, turn_index,
                reasoning_expand_getter=lambda: self._reasoning_expanded,
            )
            self._assistant_index[key] = entry
            self._add(entry)
        return entry

    # --- 请求终结与计量 ---------------------------------------------------------------

    def record_request_result(self, stop_reason: str, error: str | None) -> None:
        """请求级终结事实: 错误/取消如实成条; 正常完成不产生任何条目 (安静界面)。

        length 不单独立条: 截断事实标注在对应 assistant 条目上。
        """
        if stop_reason == "aborted":
            self.append_cancelled()
        elif stop_reason == "error":
            self.append_error(error or "模型请求以错误结束，但未提供错误信息。")

    def add_metrics(self, metrics: ModelMetrics | None, *, call_key: tuple[str, int] | None = None) -> None:
        """累计一次模型调用的计量; None 字段记不可得, 不当作 0。"""
        if call_key is not None:
            if call_key in self._metric_calls:
                return
            self._metric_calls.add(call_key)
        self._cache_total.add(metrics)
        if metrics is None:
            self._input_unknown = True
            self._output_unknown = True
            return
        if metrics.input_tokens is None:
            self._input_unknown = True
        else:
            self._input_total = (self._input_total or 0) + metrics.input_tokens
        if metrics.output_tokens is None:
            self._output_unknown = True
        else:
            self._output_total = (self._output_total or 0) + metrics.output_tokens

    def metrics_summary(self) -> str:
        """计量口径文本: 跨调用累计; 无任何数值时为不可得, 不虚构 0。

        部分调用的字段缺失时如实标注, 已收到的累计不作废也不夸大为完整值。
        """
        if self._input_total is None and self._output_total is None:
            return "计量: 不可得"

        def fmt(label: str, total: int | None, unknown: bool) -> str:
            if total is None:
                return f"{label} 不可得"
            if unknown:
                return f"{label} {total}（部分调用不可得）"
            return f"{label} {total}"

        return f"计量: 累计{fmt('输入', self._input_total, self._input_unknown)} · 累计{fmt('输出', self._output_total, self._output_unknown)}"

    def metrics_fact(self) -> tuple[int | None, bool, int | None, bool]:
        """底栏用计量事实: (输入累计, 输入含缺失, 输出累计, 输出含缺失)。"""
        return (self._input_total, self._input_unknown, self._output_total, self._output_unknown)

    def cache_summary(self) -> str:
        """仅本进程普通调用，跨模型累计；不导入恢复历史或摘要计量。"""
        return self._cache_total.summary("缓存（本进程普通调用，跨模型累计）")

    # --- 渲染 ----------------------------------------------------------------------

    def layout(self, width: int, selected: DisplayAction | None = None) -> DocumentLayout:
        """整份文档同源布局 (紧凑交互票据 01, 契约 C3): 文字/命中/锚点同一次排版。

        有 layout 方法的条目 (AssistantEntry/SkillRequestEntry/ToolGroupEntry)
        直接产出片段, 其余条目 wrap_plain_entry 只持锚点身份; 首部前导空行
        丢弃, 命中行同步下移。
        """
        lines: list[str] = []
        hits: list[HitSpan] = []
        anchors: list[EntryAnchor | None] = []
        for entry, display_id in zip(self._entries, self._entry_ids):
            layout_method = getattr(entry, "layout", None)
            if callable(layout_method):
                piece = layout_method(width, selected=selected, display_id=display_id)
            else:
                piece = wrap_plain_entry(entry.render(width), (TARGET_ENTRY, display_id))
            offset = len(lines)
            lines.extend(piece.lines)
            hits.extend(
                HitSpan(hit.row + offset, hit.start_column, hit.end_column, hit.action)
                for hit in piece.hits
            )
            anchors.extend(piece.anchors)
        dropped = 0
        while dropped < len(lines) and lines[dropped] == "":
            dropped += 1
        if dropped:
            del lines[:dropped]
            del anchors[:dropped]
            hits = [
                HitSpan(hit.row - dropped, hit.start_column, hit.end_column, hit.action)
                for hit in hits
                if hit.row >= dropped
            ]
        return DocumentLayout(tuple(_fit(lines, width)), tuple(hits), tuple(anchors))

    def render(self, width: int) -> list[str]:
        """整份文档按当前宽度排版; 每行可见列宽 <= width (同源布局兼容出口)。"""
        return list(self.layout(width).lines)


__all__ = [
    "CANCELLED_TEXT",
    "ERROR_PREFIX",
    "LENGTH_NOTE",
    "STREAM_PLACEHOLDER",
    "AssistantEntry",
    "CancelledEntry",
    "ErrorEntry",
    "InformationEntry",
    "SkillRequestEntry",
    "SystemEntry",
    "Transcript",
    "TranscriptEntry",
    "UserEntry",
]
