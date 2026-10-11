"""启动欢迎区 (tui-startup-welcome; 票据 04 起为 r2 Pi 风格, 无图标).

职责 (规格 D1–D6 + r2 修订 / 契约 C1–C5):
- WelcomeFacts 是 ConversationView 初始化时一次捕获的不可变安全值 (版本/工作区/
  启动模型/思考/指令文件名/Skills 计数/恢复判定); 不持有 ApplicationConfig、
  LaunchIdentity 或凭据对象。事实只取自已发布的会话只读出口, 不读插件闭包,
  不调用 login_view。
- WelcomeEntry 是 TranscriptEntry 实现, 每个 ConversationView 只追加一次。
  新会话完整块 (>=48 列, r2 Pi 风格): 加粗标题 + 低亮键位行 + 低亮命令入口行 +
  空行 + [工作区]/[启动模型]/[指令]/[Skills] 四个段组 (段标题默认前景, 内容缩进
  2 列低亮); 恢复会话三行短版 (不加段组)。24–47 列降为四行 (恢复两行),
  1–23 列仅 Cicada 与 /help, 宽度 <= 0 返回空行集。r2 起无图标资产,
  所有宽度共用同一文字布局。
- 动态值捕获时先 sanitize_display、换行/Tab 折成空格、按字段上限裁剪; 渲染再按
  当前可见列宽截断。plain 策略 (捕获时 NO_COLOR 存在或 TERM=dumb) 输出零 ESC;
  有色模式只对截断后的文本套样式, 每行末尾一次复位, 不污染相邻条目。
- 纯展示: 不发模型请求、不写历史/持久记录、无订阅/异步任务; render 无副作用,
  resize/模型切换/登录配置/clear/compact 都不重建快照 (启动值归属不变)。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Mapping

from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle
from cicada.ui.main_screen import truncate_to_width
from cicada.ui.rendering import sanitize_display

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

# r2 文案 (Pi 风格; 行序固定, 规格 r2 修订); Ctrl+O 表述为"工具/Skill"
# (thinking-budget-visibility 契约 C3; 04 用户裁决移除 Ctrl+T 条目),
# 不改变原发送和取消按键
_KEYS_LINE = (
    "Enter 发送 · Shift+Enter 换行 · Esc/Ctrl+C 中断 · Ctrl+D 退出 · / 命令"
    " · Ctrl+O 工具/Skill"
)
_COMMANDS_LINE = "/help 全部命令 · /model 模型 · /login 配置 · /status 状态"
_RESUME_COMMANDS_LINE = "/help 帮助 · /status 状态"
_HELP_LINE = "/help 帮助"


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
    session: ApplicationSession, *, environ: Mapping[str, str] | None = None,
    display_style: DisplayStyle | None = None,
) -> WelcomeEntry:
    """捕获事实与局部无色策略 (规格 D5), 构造本视图唯一的欢迎条目。"""
    style = display_style if display_style is not None else DisplayStyle.capture(environ)
    return WelcomeEntry(capture_welcome_facts(session), plain=style.plain)


class WelcomeEntry:
    """显示文档首条: 启动快照文字块。渲染无副作用, 快照不刷新 (契约 C4)。"""

    def __init__(self, facts: WelcomeFacts, *, plain: bool) -> None:
        self.facts = facts
        self._plain = plain

    def render(self, width: int) -> list[str]:
        """按当前宽度投影; 每行可见列宽 <= width, 宽度 <= 0 返回空行集。"""
        if width <= 0:
            return []
        return [
            "",
            *(
                self._style(role, truncate_to_width(text, width))
                for role, text in self._lines(width)
            ),
        ]

    # --- 文案 (规格 D2/D4 + r2 行序, (角色, 文本) 对) ----------------------------------

    def _lines(self, width: int) -> list[tuple[str, str]]:
        if width < 24:
            return [("title", "Cicada"), ("dim", "/help")]
        if width < 48:
            if self.facts.resumed:
                return [("title", "Cicada · 恢复会话"), ("dim", _HELP_LINE)]
            return [
                ("title", self._title()),
                ("dim", f"启动模型: {self._model()}"),
                ("dim", f"指令: {self._instructions()} · Skills: {self._skills_brief()}"),
                ("dim", _HELP_LINE),
            ]
        if self.facts.resumed:
            return [
                ("title", self._title()),
                ("dim", f"启动模型: {self._model()} · 思考 {self.facts.thinking_label}"),
                ("dim", _RESUME_COMMANDS_LINE),
            ]
        return [
            ("title", self._title()),
            ("dim", _KEYS_LINE),
            ("dim", _COMMANDS_LINE),
            ("dim", ""),
            ("section", "[工作区]"),
            ("dim", f"  {self.facts.workspace}"),
            ("section", "[启动模型]"),
            ("dim", f"  {self._model()} · 思考 {self.facts.thinking_label}"),
            ("section", "[指令]"),
            ("dim", f"  {self._instructions()}"),
            ("section", "[Skills]"),
            ("dim", f"  {self._skills_content()}"),
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

    def _skills_content(self) -> str:
        if self.facts.skills_count is None:
            return "未启用"
        if self.facts.skills_count == 0:
            return "0 个"
        return f"{self.facts.skills_count} 个 · /skill 查看目录"

    # --- 样式 (规格 r2): plain 零 ESC; 有色只套截断后文本, 行末一次复位 -----------------

    def _style(self, role: str, line: str) -> str:
        if self._plain or not line:
            return line
        if role == "title":
            return f"\x1b[1m{line}\x1b[0m"
        if role == "dim":
            with DisplayStyle(plain=False).use():
                return theme.fg(theme.MUTED_GRAY, line)
        return line  # section 段标题默认前景


__all__ = ["WelcomeEntry", "WelcomeFacts", "build_welcome_entry", "capture_welcome_facts"]
