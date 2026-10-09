"""启动欢迎区 (tui-startup-welcome 票据 01): 一次捕获的启动快照与操作入口.

职责 (规格 D1–D6 / 契约 C1–C5):
- WelcomeFacts 是 ConversationView 初始化时一次捕获的不可变安全值 (版本/工作区/
  启动模型/思考/指令文件名/Skills 计数/恢复判定); 不持有 ApplicationConfig、
  LaunchIdentity 或凭据对象。事实只取自已发布的会话只读出口, 不读插件闭包,
  不调用 login_view。
- WelcomeEntry 是 TranscriptEntry 实现, 每个 ConversationView 只追加一次。
  新会话七行 (标题/工作区/启动模型与思考/指令/Skills/命令入口/键位), 恢复三行;
  宽窗 (>=88 列) 新会话左侧并排 28×6 蝉标志 (有色真彩格 / plain ASCII 轮廓,
  资产为 welcome_assets 受控静态数据), 恢复短版无标志; 24–47 列降为四行
  (恢复两行), 1–23 列仅 Cicada 与 /help, 宽度 <= 0 返回空行集。
- 动态值捕获时先 sanitize_display、换行/Tab 折成空格、按字段上限裁剪; 渲染再按
  当前可见列宽截断。plain 策略 (捕获时 NO_COLOR 存在或 TERM=dumb) 输出零 ESC;
  有色模式只对截断后的文本套样式, 每行末尾一次复位, 不污染相邻条目。
- 纯展示: 不发模型请求、不写历史/持久记录、无订阅/异步任务; render 无副作用,
  resize/模型切换/登录配置/clear/compact 都不重建快照 (启动值归属不变)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Mapping

from cicada.ui import theme
from cicada.ui.main_screen import truncate_to_width
from cicada.ui.rendering import sanitize_display
from cicada.ui.welcome_assets import LOGO_ASCII_28, LOGO_CELLS_28, LOGO_PALETTE

if TYPE_CHECKING:
    from cicada.application import ApplicationSession
    from cicada.persistence.contracts import LaunchIdentity

_UNKNOWN = "不可得"
_EFFORT_LABELS = ("low", "high", "max")

# 字段展示上限 (规格 D6, Unicode 字符数; 不影响原始事实)
_MAX_VERSION_CHARS = 64
_MAX_MODEL_CHARS = 128
_MAX_INSTRUCTIONS_CHARS = 128
_MAX_WORKSPACE_CHARS = 256

_COMMANDS_LINE = "/help 帮助 · /model 模型 · /login 配置"
_KEYS_LINE = "Enter 发送 · Shift+Enter 换行"
_RESUME_COMMANDS_LINE = "/help 帮助 · /status 状态"
_HELP_LINE = "/help 帮助"

# 宽窗并排排版 (规格 D3 / 契约 C3): 左缩进 1 + 标志 28 + 间隔 3 + 文字 + 右留白 1
_WIDE_MIN_WIDTH = 88
_LOGO_WIDTH = 28
_LOGO_GAP = 3
_LOGO_BLANK = " " * _LOGO_WIDTH


def _color_logo_row(cells: tuple[tuple[str, str | None, str | None], ...]) -> str:
    """同色连续段共享只开 SGR; 无着色段前先复位, 行末一次复位 (颜色不越界)。"""
    parts: list[str] = []
    current: tuple[str | None, str | None] = (None, None)
    index = 0
    while index < len(cells):
        color = (cells[index][1], cells[index][2])
        end = index
        while end < len(cells) and (cells[end][1], cells[end][2]) == color:
            end += 1
        if color != current:
            if current != (None, None):
                parts.append("\x1b[0m")
            fg, bg = color
            if fg is not None:
                parts.append(theme.fg_open(LOGO_PALETTE[fg]))
            if bg is not None:
                parts.append(theme.bg_open(LOGO_PALETTE[bg]))
            current = color
        parts.append("".join(cell[0] for cell in cells[index:end]))
        index = end
    if current != (None, None):
        parts.append("\x1b[0m")
    return "".join(parts)


# 静态数据一次组装, render 只投影不再整形; ASCII 行尾空格补齐到 28 列
_COLORED_LOGO_ROWS: tuple[str, ...] = tuple(_color_logo_row(row) for row in LOGO_CELLS_28)
_PLAIN_LOGO_ROWS: tuple[str, ...] = tuple(line.ljust(_LOGO_WIDTH) for line in LOGO_ASCII_28)


@dataclass(frozen=True)
class WelcomeFacts:
    """一次捕获的启动快照安全值; skills_count None = 未启用 (与 0 个区分)。"""

    version: str | None  # None = 版本不可得
    workspace: str
    model_label: str  # 空串 = 不可得
    thinking_label: str
    instructions_name: str | None  # None = 未指定
    skills_count: int | None
    resumed: bool


def _clean_field(value: str, limit: int) -> str:
    """显示清理 + 换行/Tab 折成空格 + 字段字符上限; 不成为终端控制序列。"""
    cleaned = sanitize_display(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return cleaned[:limit]


def _thinking_label(identity: LaunchIdentity | None) -> str:
    """思考显示决策表 (规格 D2): 未知保持不可得, 不升级为推定默认值。"""
    if identity is None or identity.think is None:
        return _UNKNOWN
    if not identity.think:
        return "off"
    if identity.provider == "deepseek":
        effort = identity.options.get("reasoning_effort")
        return effort if isinstance(effort, str) and effort in _EFFORT_LABELS else _UNKNOWN
    return "开启"


def capture_welcome_facts(session: ApplicationSession) -> WelcomeFacts:
    """从会话公开只读出口捕获一次启动事实; 无 I/O 副作用、无模型请求、无凭据访问。"""
    try:
        version: str | None = importlib_metadata.version("cicada")
    except importlib_metadata.PackageNotFoundError:
        version = None
    if version is not None:
        version = _clean_field(version, _MAX_VERSION_CHARS)
    config = session.config
    identity = config.persistence.identity if config.persistence is not None else None
    instructions_path = config.prompt_info.instructions_path
    skills = session.skills
    persistence = session.persistence
    # 恢复判定 (规格 D4): 优先 persistence.mode; 无持久化时以恢复事实兜底
    resumed = (
        persistence.mode == "resume"
        if persistence is not None
        else bool(session.restored_requests or session.restored_interrupted)
    )
    return WelcomeFacts(
        version=version,
        workspace=_clean_field(
            theme.format_workspace(str(config.workspace)), _MAX_WORKSPACE_CHARS
        ),
        model_label=_clean_field(config.model_label, _MAX_MODEL_CHARS),
        thinking_label=_thinking_label(identity),
        instructions_name=(
            None
            if instructions_path is None
            else _clean_field(instructions_path.name, _MAX_INSTRUCTIONS_CHARS)
        ),
        skills_count=None if skills is None else len(skills.infos),
        resumed=resumed,
    )


def build_welcome_entry(
    session: ApplicationSession, *, environ: Mapping[str, str] | None = None
) -> WelcomeEntry:
    """捕获事实与局部无色策略 (规格 D5), 构造本视图唯一的欢迎条目。"""
    env = os.environ if environ is None else environ
    plain = "NO_COLOR" in env or env.get("TERM") == "dumb"
    return WelcomeEntry(capture_welcome_facts(session), plain=plain)


class WelcomeEntry:
    """显示文档首条: 启动快照文字块。渲染无副作用, 快照不刷新 (契约 C4)。"""

    def __init__(self, facts: WelcomeFacts, *, plain: bool) -> None:
        self.facts = facts
        self._plain = plain

    def render(self, width: int) -> list[str]:
        """按当前宽度投影; 每行可见列宽 <= width, 宽度 <= 0 返回空行集。"""
        if width <= 0:
            return []
        if width >= _WIDE_MIN_WIDTH and not self.facts.resumed:
            return self._wide_lines(width)
        lines = ["", *(
            self._style(truncate_to_width(text, width), title=index == 0)
            for index, text in enumerate(self._text_lines(width))
        )]
        return lines

    def _wide_lines(self, width: int) -> list[str]:
        """宽窗新会话: 左缩进 1 + 28×6 标志 + 间隔 3 + 七行摘要; 第七行左侧空白。"""
        text_budget = width - (1 + _LOGO_WIDTH + _LOGO_GAP + 1)
        logo = _PLAIN_LOGO_ROWS if self._plain else _COLORED_LOGO_ROWS
        lines = [""]
        for index, text in enumerate(self._text_lines(width)):
            left = logo[index] if index < len(logo) else _LOGO_BLANK
            styled = self._style(truncate_to_width(text, text_budget), title=index == 0)
            lines.append(" " + left + "   " + styled)
        return lines

    # --- 文案 (规格 D2/D4, 行序固定) --------------------------------------------------

    def _text_lines(self, width: int) -> list[str]:
        if width < 24:
            return ["Cicada", "/help"]
        if width < 48:
            if self.facts.resumed:
                return ["Cicada · 恢复会话", _HELP_LINE]
            return [
                self._title(),
                f"启动模型: {self._model()}",
                f"指令: {self._instructions()} · Skills: {self._skills_brief()}",
                _HELP_LINE,
            ]
        if self.facts.resumed:
            return [
                self._title(),
                f"启动模型: {self._model()} · 思考 {self.facts.thinking_label}",
                _RESUME_COMMANDS_LINE,
            ]
        return [
            self._title(),
            f"工作区: {self.facts.workspace}",
            f"启动模型: {self._model()} · 思考 {self.facts.thinking_label}",
            f"指令: {self._instructions()}",
            self._skills_line(),
            _COMMANDS_LINE,
            _KEYS_LINE,
        ]

    def _title(self) -> str:
        version = self.facts.version if self.facts.version is not None else "版本不可得"
        kind = "恢复会话" if self.facts.resumed else "新会话"
        return f"Cicada {version} · {kind} · 启动快照"

    def _model(self) -> str:
        return self.facts.model_label or _UNKNOWN

    def _instructions(self) -> str:
        return self.facts.instructions_name if self.facts.instructions_name is not None else "未指定"

    def _skills_brief(self) -> str:
        return "未启用" if self.facts.skills_count is None else f"{self.facts.skills_count} 个"

    def _skills_line(self) -> str:
        if self.facts.skills_count is None:
            return "Skills: 未启用"
        if self.facts.skills_count == 0:
            return "Skills: 0 个"
        return f"Skills: {self.facts.skills_count} 个 · /skill 查看目录"

    # --- 样式 (规格 D5): plain 零 ESC; 有色只套截断后文本, 行末一次复位 ------------------

    def _style(self, line: str, *, title: bool) -> str:
        if self._plain or not line:
            return line
        if title:
            return f"\x1b[1m{line}\x1b[0m"
        if line.startswith("/help") or line.startswith("Enter"):
            return theme.fg(theme.MD_CODE, line)
        return line


__all__ = ["WelcomeEntry", "WelcomeFacts", "build_welcome_entry", "capture_welcome_facts"]
