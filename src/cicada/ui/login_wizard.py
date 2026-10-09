"""共享向导只编辑安全视图；启动宿主独占 Terminal 并拥有保存任务。"""

import asyncio
from dataclasses import replace
from typing import Callable

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.plugins.deepseek.protocol import REASONING_EFFORTS
from cicada.ui.rendering import sanitize_display
from cicada.ui.terminal_io import Terminal
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.user_config import (
    LoginResult, LoginSubmission, LoginView, SecretValue, UserConfigError,
    UserConfigStore, UserDefaults,
)


class LoginWizard:
    """无终端、应用或文件控制权；主屏宿主可复用 handle/render/submission。"""

    def __init__(self, view: LoginView, *, save_defaults: UserDefaults | None = None,
                 explicit: dict[str, object] | None = None, title: str | None = None,
                 display_style: DisplayStyle | None = None):
        self.display_style = display_style if display_style is not None else current_style()
        self.view = view
        self.title = title
        self.defaults = save_defaults or UserDefaults(
            view.provider, view.model, view.think, view.reasoning_effort,
            view.num_ctx, view.max_output_tokens)
        self.explicit = dict(explicit or {})
        self.step = 'provider'
        self.selection = 0
        self.credential_action = 'keep' if view.credential_present else 'replace'
        self._key = ''
        self._credential: SecretValue | None = None
        self.error: str | None = None
        self.submitting = False
        self.cancelled = False
        self.target = view.actions[0]
        self._viewport: tuple[int, int | None] | None = None
        self._confirm_blocked = False

    @property
    def _thinking(self) -> tuple[str, ...]:
        return ('off', *REASONING_EFFORTS)

    def clear_secret(self) -> None:
        self._key = ''
        self._credential = None

    def handle(self, key: KeyPress) -> str | None:
        """返回 cancel/submit 或 None；提交占位同步设置，批次多 Enter 不重复。"""
        if self.submitting or self.cancelled:
            return None
        if key.key in (Keys.Escape, Keys.ControlC):
            self.cancelled = True
            self.clear_secret()
            return 'cancel'
        if key.key == Keys.ControlI:  # 明确的返回键，密码 Backspace 保持删除语义。
            previous = {'credential': 'provider', 'thinking': 'credential', 'confirm':
                        'credential' if self.view.credentials_only else 'thinking'}
            self.step = previous.get(self.step, self.step)
            self.selection = 0
            self.error = None
            return None
        if key.key in (Keys.Up, Keys.Down):
            count = (2 if self.step == 'credential' and self.view.credential_present
                     else len(self._thinking) if self.step == 'thinking'
                     else len(self.view.actions) if self.step == 'confirm' else 1)
            self.selection = (self.selection + (1 if key.key == Keys.Down else -1)) % count
            return None
        if self.step == 'credential':
            self.credential_action = ('keep' if self.view.credential_present and self.selection == 0
                                      else 'replace')
            if self.credential_action == 'replace':
                if key.key in (Keys.ControlH, Keys.Backspace):
                    self._key = self._key[:-1]
                elif key.key == Keys.BracketedPaste or not isinstance(key.key, Keys):
                    candidate = key.data if key.key == Keys.BracketedPaste else self._key + key.data
                    try:
                        # 非法 paste 整体拒绝；现有输入不被覆盖，也不会落入聊天。
                        if '\n' in candidate or '\r' in candidate:
                            raise UserConfigError('credential', 'invalid-key')
                        SecretValue(candidate)
                    except UserConfigError:
                        self.error = '凭据须单行、无内部空白或控制字符，且不超过 4096 UTF-8 bytes'
                    else:
                        self._key = candidate.strip()
                        self.error = None
        if key.key not in (Keys.Enter, Keys.ControlM):
            return None
        if self.step == 'provider':
            self.step = 'credential'
            self.selection = 0
        elif self.step == 'credential':
            if self.credential_action == 'replace':
                try:
                    self._credential = SecretValue(self._key)
                except UserConfigError:
                    self.error = '请输入有效的单行凭据'
                    return None
            else:
                self.clear_secret()
            self._key = ''
            self.step = 'confirm' if self.view.credentials_only else 'thinking'
            self.selection = (0 if not self.defaults.think else
                              self._thinking.index(self.defaults.reasoning_effort))
            if self.view.credentials_only:
                self.selection = 0
        elif self.step == 'thinking':
            value = self._thinking[self.selection]
            self.defaults = replace(self.defaults, think=value != 'off',
                reasoning_effort=self.defaults.reasoning_effort if value == 'off' else value)
            self.step = 'confirm'
            self.selection = 0
        elif self.step == 'confirm':
            if self._viewport is not None:
                # 同批按键也先按最近宿主尺寸计算，不能绕过未显示完整的确认信息。
                self.render_lines(*self._viewport)
                if self._confirm_blocked:
                    return None
            self.target = self.view.actions[self.selection]
            self.submitting = True
            return 'submit'
        self.error = None
        return None

    def submission(self) -> LoginSubmission:
        if not self.submitting:
            raise RuntimeError('login submission is not reserved')
        return LoginSubmission(self.defaults, self.credential_action, self.view.revision,
                               self.target, self._credential)

    def render_lines(self, columns: int, rows: int | None = None) -> list[str]:
        with self.display_style.use():
            return self._render_lines(columns, rows)

    def _render_lines(self, columns: int, rows: int | None = None) -> list[str]:
        """Pi 式对话框结构，沿用主屏主题；低窗口优先保留输入和选中动作。"""
        width = max(1, columns)
        self._viewport = (columns, rows)
        self._confirm_blocked = self.step == 'confirm' and not self.submitting and width < 18
        pad = '  ' if width >= 24 else ' ' if width > 4 else ''
        content_width = max(1, width - len(pad) * 2)
        items: list[tuple[int, str]] = []
        required: set[int] = set()

        def row(text: str, *, color: str = theme.ASSISTANT_TEXT, priority: int = 2,
                wrap: bool = False, mandatory: bool = False) -> None:
            clean = sanitize_display(text)
            chunks = wrap_text_ansi(clean, content_width) if wrap else [truncate_to_width(clean, content_width)]
            chunks = [truncate_to_width(chunk, content_width) for chunk in chunks]
            if mandatory:
                required.update(range(len(items), len(items) + len(chunks)))
            items.extend((priority, pad + theme.fg(color, chunk)) for chunk in chunks)

        def space() -> None:
            items.append((5, ''))

        def choice(index: int, label: str, description: str) -> None:
            selected = index == self.selection
            marker = ('→ ' if selected else '  ') if content_width >= 3 else ''
            label = truncate_to_width(sanitize_display(label), max(1, content_width - len(marker)))
            remaining = content_width - len(marker) - visible_width(label) - 3
            detail = ('   ' + sanitize_display(description)
                      if description and visible_width(description) <= remaining else '')
            body = (pad + theme.fg_open(theme.EDITOR_BORDER) + marker
                    + theme.fg_open(theme.MD_LINK if selected else theme.ASSISTANT_TEXT) + label
                    + theme.fg_open(theme.MUTED_GRAY) + detail)
            if selected:
                body = theme.bg_open(theme.MENU_SEL_BG) + body
                body += ' ' * max(0, width - visible_width(body))
                if self.step == 'confirm':
                    required.add(len(items))
            items.append((-3 if selected else 1, body + self.display_style.sgr('0')))

        steps = ('provider', 'credential', 'confirm') if self.view.credentials_only else (
            'provider', 'credential', 'thinking', 'confirm')
        names = {'provider': '提供商', 'credential': '凭据', 'thinking': '思考', 'confirm': '确认'}
        number = steps.index(self.step) + 1
        title = self.title or ('Cicada · 补齐凭据' if self.view.credentials_only else 'Cicada · 首次配置')
        compact = rows is not None and rows < 14
        caption = (f'{names[self.step]} · {number}/{len(steps)}' if compact else title)
        heading = pad + theme.fg(theme.EDITOR_BORDER, self.display_style.sgr('1') + truncate_to_width(caption, content_width))
        border = theme.fg(theme.EDITOR_BORDER, '─' * width)
        progress = pad + '   '.join(theme.fg(
            theme.EDITOR_BORDER if step == self.step else theme.MUTED_GRAY,
            ('› ' if step == self.step else '') + names[step]) for step in steps)
        header = [heading] if compact else [border, heading, progress, '']

        if self.submitting:
            row('● 正在可靠保存，请等待…', color=theme.EDITOR_BORDER, priority=-3, wrap=True)
        elif self.step == 'provider':
            choice(0, 'DeepSeek', 'API key')
        elif self.step == 'credential':
            if self.view.credential_present:
                labels = {'environment': '环境变量', 'file': '凭据文件', 'login': '本次登录'}
                row('凭据已配置 · 来源：' + labels.get(self.view.credential_source, '已配置'),
                    color=theme.MUTED_GRAY, priority=1)
                space()
                choice(0, '保留凭据', '')
                choice(1, '替换凭据', '')
            if not self.view.credential_present or self.selection == 1:
                space()
                masked = '*' * min(len(self._key), 32)
                row('API key: ' + (masked or '输入或粘贴单行凭据'),
                    color=theme.ASSISTANT_TEXT if masked else theme.MUTED_GRAY, priority=-3)
        elif self.step == 'thinking':
            row(self.view.model, color=theme.MUTED_GRAY)
            space()
            for index, value in enumerate(self._thinking):
                choice(index, '关闭思考' if value == 'off' else value, '')
        else:
            d = self.defaults
            row('模型：' + self.view.model, priority=1, wrap=True, mandatory=True)
            row(('本次思考（存档只读）：' + (self.view.reasoning_effort if self.view.think else 'off')
                 if self.view.credentials_only else
                 '思考：' + (d.reasoning_effort if d.think else 'off')),
                priority=1, wrap=True, mandatory=True)
            space()
            row('保存位置：' + str(self.view.save_path), color=theme.MUTED_GRAY,
                priority=3, wrap=True, mandatory=True)
            if self.view.credentials_only:
                row('仅补凭据；存档设置不变', color=theme.MD_LINK,
                    priority=-1, wrap=True, mandatory=True)
            else:
                actual = replace(d, **{k: v for k, v in self.explicit.items() if v is not None})
                if actual != d:
                    row('本次 CLI 覆盖：' + actual.model + ' · 思考=' +
                        (actual.reasoning_effort if actual.think else 'off') +
                        f' · 窗口/输出={actual.num_ctx}/{actual.max_output_tokens}',
                        color=theme.MD_LINK, priority=0, wrap=True, mandatory=True)
            if self.view.credential_source == 'environment':
                row('下次启动优先环境凭据',
                    color=theme.MD_LINK, priority=0, wrap=True, mandatory=True)
            space()
            for index, target in enumerate(self.view.actions):
                choice(index, '保存并应用' if target == 'save-and-apply' else
                       '仅保存' if self.title else '确认保存',
                       '后继请求生效' if target == 'save-and-apply' else
                       '下次新会话生效' if self.title else '继续启动')
            if self.title and not self.view.credentials_only:
                scope = ('后继请求生效'
                         if self.view.actions[self.selection] == 'save-and-apply' else
                         '下次新会话生效')
                selected_label = '保存并应用' if self.view.actions[self.selection] == 'save-and-apply' else '仅保存'
                if content_width < 2 + visible_width(selected_label) + 3 + visible_width(scope):
                    row(scope, color=theme.MUTED_GRAY, priority=-1, wrap=True, mandatory=True)
        if self.error:
            row(self.error, color=theme.DIFF_REMOVED, priority=-2, wrap=True)

        if rows is not None and rows <= 2:
            self._confirm_blocked = self.step == 'confirm'
            active = sorted(range(len(items)), key=lambda i: (items[i][0], i))[:max(1, rows)]
            return [items[i][1] for i in sorted(active)]

        hint = ('请等待保存完成' if self.submitting else
                '↑↓ 选择 · Enter 确认 · Tab 返回 · Esc 取消')
        if width < 52:
            hint = '保存中…' if self.submitting else '↑↓ 选择 · Enter 确认 · Esc 取消'
        footer = [pad + theme.fg(theme.MUTED_GRAY, truncate_to_width(hint, content_width))]
        if not compact:
            footer = ['', *footer, border]
        if rows is not None:
            budget = max(0, rows - len(header) - len(footer))
            self._confirm_blocked = self._confirm_blocked or (
                self.step == 'confirm' and not self.submitting and len(required) > budget)
            if self._confirm_blocked:
                row('窗口太小，请放大后确认', color=theme.DIFF_REMOVED, priority=-4, wrap=True)
                retained = set(sorted(range(len(items)), key=lambda i: (items[i][0], i))[:budget])
            else:
                retained = set(required)
                extra = sorted((i for i in range(len(items)) if i not in retained),
                               key=lambda i: (items[i][0], i))[:max(0, budget - len(retained))]
                retained.update(extra)
            items = [item for i, item in enumerate(items) if i in retained]
        lines = [*header, *(text for _, text in items), *footer]
        # 纯文本先限宽，套 SGR 后不再截断（会剥掉颜色）；极窄窗的进度条单独限宽。
        if not compact:
            lines[2] = pad + theme.fg(theme.MUTED_GRAY, truncate_to_width(
                ' · '.join(names[step] for step in steps), content_width)) if visible_width(progress) > width else progress
        return lines[:max(1, rows)] if rows is not None else lines


class LoginModal:
    """主屏的可选模态槽；复用向导，无文件或 Terminal 控制权。"""

    def __init__(self, on_cancel: Callable[[], None],
                 on_submit: Callable[[LoginSubmission], None], *,
                 display_style: DisplayStyle | None = None):
        self.display_style = display_style if display_style is not None else current_style()
        self.wizard: LoginWizard | None = None
        self._on_cancel = on_cancel
        self._on_submit = on_submit

    @property
    def is_open(self) -> bool:
        return self.wizard is not None

    def open(self, view: LoginView) -> None:
        self.wizard = LoginWizard(view, title='Cicada · 登录配置', display_style=self.display_style)

    def dismiss(self) -> None:
        if self.wizard is not None:
            self.wizard.clear_secret()
        self.wizard = None

    def handle(self, key: KeyPress) -> None:
        if self.wizard is None:
            return
        action = self.wizard.handle(key)
        if action == 'cancel':
            self.dismiss()
            self._on_cancel()
        elif action == 'submit':
            submission = self.wizard.submission()
            self.wizard.clear_secret()
            self._on_submit(submission)

    def render(self, columns: int, rows: int) -> list[str]:
        return self.wizard.render_lines(columns, max(1, rows)) if self.wizard else []


async def run_startup_login(
    terminal: Terminal, store: UserConfigStore, view: LoginView, *,
    save_defaults: UserDefaults | None = None, explicit: dict[str, object] | None = None,
) -> tuple[LoginResult | None, list[str]]:
    """应用 open 前唯一 writer；结束并恢复后才可进入主屏。"""
    wizard = LoginWizard(view, save_defaults=save_defaults, explicit=explicit)
    done = asyncio.Event()
    save_task: asyncio.Task | None = None
    failed = False
    errors: list[str] = []

    def render() -> None:
        nonlocal failed
        try:
            lines = wizard.render_lines(terminal.columns, max(1, terminal.rows - 1))
            terminal.write('\x1b[?25l\x1b[2J\x1b[H' + '\r\n'.join(lines))
        except Exception:
            failed = True
            errors.append('login terminal rendering failed')
            done.set()

    def on_input(batch) -> None:
        nonlocal save_task
        if failed or done.is_set():
            return
        for key in batch:
            action = wizard.handle(key)
            if action == 'cancel':
                done.set()
                return
            if action == 'submit':
                submission = wizard.submission()
                wizard.clear_secret()
                save_task = asyncio.create_task(store.save(submission))
                save_task.add_done_callback(lambda _: done.set())
                break
            render()
            if failed:
                break
        render()

    result = None
    try:
        terminal.start(on_input, render)
        render()
        try:
            while not done.is_set():
                try:
                    await asyncio.wait_for(done.wait(), timeout=0.05)
                except TimeoutError:
                    pass
                if getattr(terminal, 'input_failed', False):
                    failed = True
                    errors.append('login terminal input failed')
                    done.set()
        except asyncio.CancelledError:
            if save_task is None:
                raise
            save_task.cancel()
        if save_task is not None:
            if failed:
                save_task.cancel()
            while True:
                try:
                    result = await asyncio.shield(save_task)
                    break
                except asyncio.CancelledError:
                    if save_task.cancelled():
                        result = LoginResult(False, False, 'not-requested', 'cancelled')
                        break
                    save_task.cancel()
        if failed:
            result = LoginResult(bool(result and result.defaults_saved), False,
                                 'not-requested', 'terminal-failed')
    except Exception:
        result = LoginResult(False, False, 'not-requested', 'terminal-failed')
        errors.append('login terminal failed')
    finally:
        wizard.clear_secret()
        try:
            terminal.write('\x1b[2J\x1b[H\x1b[?25h')
        except Exception:
            errors.append('login terminal cursor restore failed')
        try:
            if terminal.stop():
                # 适配器的底层异常可能包含输入内容，只返回固定恢复失败事实。
                errors.append('login terminal restore errors')
        except Exception:
            errors.append('login terminal mode restore failed')
    return result, errors
