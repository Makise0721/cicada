"""共享向导只编辑安全视图；启动宿主独占 Terminal 并拥有保存任务。"""

import asyncio
from dataclasses import replace
from typing import Callable

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.plugins.deepseek.protocol import REASONING_EFFORTS
from cicada.ui.rendering import sanitize_display
from cicada.ui.terminal_io import Terminal
from cicada.ui.main_screen import truncate_to_width
from cicada.user_config import (
    LoginResult, LoginSubmission, LoginView, SecretValue, UserConfigError,
    UserConfigStore, UserDefaults,
)


class LoginWizard:
    """无终端、应用或文件控制权；主屏宿主可复用 handle/render/submission。"""

    def __init__(self, view: LoginView, *, save_defaults: UserDefaults | None = None,
                 explicit: dict[str, object] | None = None, title: str | None = None):
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

    def render_lines(self, columns: int) -> list[str]:
        lines = [self.title or ('Cicada · 补齐凭据' if self.view.credentials_only else 'Cicada · 首次配置'),
                 '↑/↓ 选择 · Enter 继续/保存 · Tab 返回 · Esc/Ctrl+C 取消', '']
        if self.submitting:
            lines.append('正在可靠保存，请等待…')
        elif self.step == 'provider':
            lines.append('> DeepSeek')
        elif self.step == 'credential':
            if self.view.credential_present:
                labels = {'environment': '环境变量', 'file': '凭据文件', 'login': '本次登录'}
                lines.append('凭据已配置 · 来源：' + labels.get(self.view.credential_source, '已配置'))
                for index, text in enumerate(('保留凭据', '替换凭据')):
                    lines.append(('> ' if index == self.selection else '  ') + text)
            if not self.view.credential_present or self.selection == 1:
                lines.append('API key: ' + '*' * min(len(self._key), 32))
        elif self.step == 'thinking':
            lines.append('本次内置模型：' + self.view.model)
            for index, value in enumerate(self._thinking):
                lines.append(('> ' if index == self.selection else '  ') +
                             ('关闭思考' if value == 'off' else value))
        else:
            d = self.defaults
            lines.extend(('提供商：DeepSeek', '本次模型：' + self.view.model,
                ('本次思考（存档只读）：' + (self.view.reasoning_effort if self.view.think else 'off')
                 if self.view.credentials_only else
                 '保存思考：' + (d.reasoning_effort if d.think else 'off')),
                '保存位置：' + str(self.view.save_path), self.view.effect))
            if self.view.credentials_only:
                lines.append('仅补凭据；保留已有用户默认值与存档设置')
            else:
                lines.append(f'保留窗口/输出上限：{d.num_ctx}/{d.max_output_tokens}')
                actual = replace(d, **{k: v for k, v in self.explicit.items() if v is not None})
                if actual != d:
                    lines.append('本次 CLI 覆盖：' + actual.model + ' · 思考=' +
                                 (actual.reasoning_effort if actual.think else 'off') +
                                 f' · 窗口/输出={actual.num_ctx}/{actual.max_output_tokens}')
            if self.view.credential_source == 'environment':
                lines.append('后续启动仍优先环境变量；保留不会复制环境凭据')
            for index, target in enumerate(self.view.actions):
                lines.append(('> ' if index == self.selection else '  ') +
                             ('保存并应用' if target == 'save-and-apply' else '确认保存'))
        if self.error:
            lines.append(self.error)
        return [truncate_to_width(sanitize_display(line), max(1, columns)) for line in lines]


class LoginModal:
    """主屏的可选模态槽；复用向导，无文件或 Terminal 控制权。"""

    def __init__(self, on_cancel: Callable[[], None],
                 on_submit: Callable[[LoginSubmission], None]):
        self.wizard: LoginWizard | None = None
        self._on_cancel = on_cancel
        self._on_submit = on_submit

    @property
    def is_open(self) -> bool:
        return self.wizard is not None

    def open(self, view: LoginView) -> None:
        self.wizard = LoginWizard(view, title='Cicada · 登录配置')

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
        lines = self.wizard.render_lines(columns) if self.wizard else []
        return lines[-max(1, rows):]


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
            lines = wizard.render_lines(terminal.columns)
            # 小窗口保留当前操作行；resize 重新按实际尺寸绘制。
            height = max(1, terminal.rows - 1)
            if len(lines) > height:
                lines = lines[-height:]
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
