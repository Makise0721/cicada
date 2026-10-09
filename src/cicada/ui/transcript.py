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
  bool (toggle_reasoning, Ctrl+T), 初始收起, 区别于工具/Skill 的 expand_getter。
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
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.reasoning_view import ReasoningBlock
from cicada.ui.rendering import sanitize_display

ERROR_PREFIX = "错误: "
CANCELLED_TEXT = "已取消: 本次请求在完成前被中止。"
LENGTH_NOTE = "（回答因达到输出长度上限被截断）"
STREAM_PLACEHOLDER = "…"


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
    展开状态经只读 getter 取全局展开标志 (与工具输出共用 Ctrl+O); 未注入
    getter 时完整显示正文, 避免无展开手段的调用者看不到内容。
    """

    def __init__(
        self, text: str, invocation: SkillInvocation,
        expand_getter: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(text)
        self.invocation = invocation
        self._expand_getter = expand_getter
        self._render_cache: tuple[int, bool, DisplayStyle, list[str]] | None = None

    def render(self, width: int) -> list[str]:
        expanded = True if self._expand_getter is None else bool(self._expand_getter())
        cached = self._render_cache
        style = current_style()
        if cached is not None and cached[:3] == (width, expanded, style):
            return list(cached[3])
        lines = self._render_lines(width, expanded)
        self._render_cache = (width, expanded, style, list(lines))
        return lines

    def _render_lines(self, width: int, expanded: bool) -> list[str]:
        invocation = self.invocation
        # 用户任务部分: 与普通用户消息同一背景盒样式
        lines = ["", *theme.user_message_lines(invocation.task, width)]
        # 标题整体显示清理后才换行 (wrap_text_ansi 契约要求调用方先剥 ANSI):
        # source_path 是记录文本, 存档/粘贴可携带任意控制序列 (审查处置 F2)
        header = sanitize_display(
            f"Skill 文档 {invocation.name}（{invocation.source_path}）"
            + (f" · sha256:{invocation.body_sha256[:12]}" if expanded else "")
        )
        for chunk in wrap_text_ansi(header, max(1, width)):
            lines.append(theme.fg(theme.MD_CODE, chunk))
        if not expanded:
            for chunk in wrap_text_ansi("正文已收起 · Ctrl+O 展开完整文档", max(1, width)):
                lines.append(theme.fg(theme.MUTED_GRAY, chunk))
            return lines
        # 正文来自记录: 规范换行后显示清理, 与工具输出同一纪律 (不执行外部序列)
        body = sanitize_display(
            invocation.body.replace("\r\n", "\n").replace("\r", "\n")
        )
        lines.extend(_wrap_with_prefix("  ", body, width))
        return lines


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
        reasoning_lines = self.reasoning.render(width)  # 思考块自带缓存与样式
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
            return []  # 完成但无正文 (如错误路径): 事实由错误/取消条目表达
        # 统一前导空行; 思考块在答案前 (票据 01: 思考、答案、终结标注)
        result = ["", *reasoning_lines, *lines]
        if self.finalized and self.stop_reason == "length":
            result.append(theme.fg(theme.MUTED_GRAY, LENGTH_NOTE))
        return result


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
    思考展开 (票据 01): 本文档唯一 bool, 初始 false, Ctrl+T 切换全部现有与
    后续思考块; /clear 保留显示文档因此保留该状态, 新建/恢复文档自然重置。
    """

    def __init__(self, *, expand_getter: Callable[[], bool] | None = None) -> None:
        self._entries: list[TranscriptEntry] = []
        self._assistant_index: dict[tuple[str, int], AssistantEntry] = {}
        self._expand_getter = expand_getter
        self._reasoning_expanded = False
        # 计量累计: None = 尚无任何数值 (显示不可得); unknown = 有调用但该字段缺失
        self._input_total: int | None = None
        self._output_total: int | None = None
        self._input_unknown = False
        self._output_unknown = False

    # --- 条目管理 ----------------------------------------------------------------

    def entries(self) -> list[TranscriptEntry]:
        """当前条目快照 (顺序即文档顺序), 供测试与诊断。"""
        return list(self._entries)

    def add_entry(self, entry: TranscriptEntry) -> None:
        """追加任意条目 (票据 03 的工具条目经此挂载, 保持文档顺序稳定)。"""
        self._entries.append(entry)

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
        self._entries.append(entry)
        return entry

    def append_system(self, text: str) -> SystemEntry:
        entry = SystemEntry(text)
        self._entries.append(entry)
        return entry

    def append_information(self, title: str, sections: list[tuple[str, list[str]]]) -> None:
        self._entries.append(InformationEntry(title, sections))

    def append_error(self, text: str) -> ErrorEntry:
        entry = ErrorEntry(text)
        self._entries.append(entry)
        return entry

    def append_cancelled(self) -> CancelledEntry:
        entry = CancelledEntry()
        self._entries.append(entry)
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

    def toggle_reasoning(self) -> None:
        """Ctrl+T (票据 01): 切换本文档全部思考块展开状态; 只改显示。"""
        self._reasoning_expanded = not self._reasoning_expanded

    def _assistant_entry(self, run_id: str, turn_index: int) -> AssistantEntry:
        key = (run_id, turn_index)
        entry = self._assistant_index.get(key)
        if entry is None:
            entry = AssistantEntry(
                run_id, turn_index,
                reasoning_expand_getter=lambda: self._reasoning_expanded,
            )
            self._assistant_index[key] = entry
            self._entries.append(entry)
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

    def add_metrics(self, metrics: ModelMetrics | None) -> None:
        """累计一次模型调用的计量; None 字段记不可得, 不当作 0。"""
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

    # --- 渲染 ----------------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        """整份文档按当前宽度排版; 每行可见列宽 <= width。

        条目自带 1 行前导空行; 文档起始处的前导空行丢弃 (首屏顶部不留白)。
        """
        lines: list[str] = []
        for entry in self._entries:
            lines.extend(entry.render(width))
        while lines and lines[0] == "":
            lines.pop(0)
        return _fit(lines, width)


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
