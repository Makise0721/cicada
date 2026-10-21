"""多供应商配置的纯显示/编辑状态；秘密只在提交时交给应用。"""

from dataclasses import fields, replace
import unicodedata

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display
from cicada.user_config import (
    LoginSubmission, LoginView, ModelProfile, ProviderProfile, SecretValue,
    UserConfigError, UserDefaults,
)


class ProviderWizard:
    """主屏和首次启动共用；不读取文件、环境变量或网络。"""

    def __init__(self, view: LoginView, *, save_defaults: UserDefaults | None = None,
                 explicit: dict[str, object] | None = None, title: str | None = None,
                 display_style: DisplayStyle | None = None,
                 initial_set_default: bool | None = None):
        self.view = view
        self.title = title or 'Cicada · 配置模型'
        self.display_style = display_style or current_style()
        self.explicit = dict(explicit or {})
        self.step = 'provider'
        self.selection = next((i for i, item in enumerate(view.providers) if item.id == view.provider), 0)
        self.error: str | None = None
        self.submitting = False
        self.cancelled = False
        self.target = view.actions[0]
        self.set_default = ((title is None if initial_set_default is None else initial_set_default)
                            and not view.credentials_only)
        self.defaults = save_defaults or UserDefaults(
            view.provider, view.model, view.think, view.reasoning_effort,
            view.num_ctx, view.max_output_tokens)
        self._credential: SecretValue | None = None
        self.credential_action = 'keep'
        self._text = ''
        self._edit_field = ''
        self._return_step = 'settings'
        self._viewport: tuple[int, int | None] | None = None
        self._confirm_blocked = False
        self._provider: dict[str, object] = {}
        self._models: list[ModelProfile] = []
        self._model: dict[str, object] = {}
        self._old_model_key: str | None = None
        self._selected_model_key = ''
        self._initial_provider_id: str | None = None
        if view.credentials_only:
            existing = next((item for item in view.providers if item.id == view.provider), view.providers[0])
            self._load_provider(existing)
            self.step = 'credential'

    def clear_secret(self) -> None:
        self._credential = None
        if self._edit_field == 'api_key':
            self._text = ''

    def _load_provider(self, provider: ProviderProfile) -> None:
        self.clear_secret()
        self.credential_action = 'keep'
        self._initial_provider_id = provider.id
        self._provider = {key: getattr(provider, key) for key in
                          ('id', 'name', 'protocol', 'base_url', 'default_model', 'auth', 'api_key_env')}
        self._models = list(provider.models)
        self._original_auth = provider.auth, provider.api_key_env
        key = self.view.model if provider.id == self.view.provider else provider.default_model
        self._selected_model_key = key if key in {item.key for item in self._models} else provider.default_model

    def _template(self, kind: str) -> None:
        self.clear_secret()
        self.credential_action = 'keep'
        self._initial_provider_id = None
        if kind == 'deepseek':
            self._provider = dict(id='deepseek', name='DeepSeek', protocol='deepseek-chat',
                                  base_url='https://api.deepseek.com', default_model='deepseek-flash',
                                  auth='bearer', api_key_env='DEEPSEEK_API_KEY')
            self._models = [ModelProfile(key=model, model_id=model, name=model,
                                        max_output_tokens=32768, think=True)
                            for model in ('deepseek-flash', 'deepseek-v4-pro')]
        else:
            self._provider = dict(id='openai' if kind == 'openai' else 'custom',
                                  name='OpenAI' if kind == 'openai' else '自定义供应商',
                                  protocol='openai-chat-completions',
                                  base_url='https://api.openai.com/v1' if kind == 'openai' else '',
                                  default_model='coder', auth='bearer',
                                  api_key_env='OPENAI_API_KEY' if kind == 'openai' else None)
            self._models = []
        base_id = str(self._provider['id'])
        existing = {item.id for item in self.view.providers}
        number = 2
        while self._provider['id'] in existing:
            self._provider['id'] = base_id + '-' + str(number)
            number += 1
        if number > 2:
            self._provider['name'] = str(self._provider['name']) + f' ({number - 1})'
        self._selected_model_key = str(self._provider['default_model'])
        self._original_auth = self._provider['auth'], self._provider['api_key_env']

    def _choices(self) -> list[tuple[str, str]]:
        if self.step == 'provider':
            return [(item.id, item.name) for item in self.view.providers] + [
                ('+deepseek', '添加 DeepSeek 连接'), ('+openai', '添加 OpenAI Chat 连接'),
                ('+custom', '添加自定义连接')]
        if self.step == 'settings':
            rows = [('model', '模型：' + (self._selected_model_key or '尚未配置')),
                    ('connection', '连接：' + str(self._provider['name'])),
                    ('credential', '凭据：' + self._credential_label()),
                    ('default', '设为下次启动默认：' + ('是' if self.set_default else '否')),
                    ('confirm', '继续保存')]
            return rows
        if self.step == 'connection':
            return [(key, label + '：' + str(self._provider[key] or '')) for key, label in (
                ('id', '实例 ID'), ('name', '名称'), ('protocol', '协议'), ('base_url', 'API 根地址'))] + [('back', '返回')]
        if self.step == 'models':
            return [(item.key, item.name or item.model_id) for item in self._models] + [('add', '添加模型'), ('back', '返回')]
        if self.step == 'model_settings':
            keys = [('key', '本地标识'), ('model_id', '远端模型 ID'), ('name', '显示名称'),
                    ('context_window', '上下文窗口'), ('max_output_tokens', '输出上限'), ('tools', '工具调用')]
            if self._provider['protocol'] == 'deepseek-chat':
                keys += [('think', '思考'), ('reasoning_effort', '思考强度')]
            else:
                keys += [('stream_usage', '流式 usage'), ('output_token_field', '输出限制字段')]
            return [(key, label + '：' + str(self._model[key])) for key, label in keys] + [('back', '保存模型并返回')]
        if self.step == 'credential':
            rows = [('keep', '保留当前凭据来源'), ('api_key', '输入新的 API Key（文件保存）'),
                    ('api_key_env', '使用环境变量'), ('none', '无认证（仅本机接口）')]
            return rows[:3] if self.view.credentials_only else rows
        if self.step == 'confirm':
            return [(action, '保存并应用' if action == 'save-and-apply' else '保存配置')
                    for action in self.view.actions]
        return []

    def _credential_label(self) -> str:
        if self._provider.get('auth') == 'none':
            return '无认证'
        if self.credential_action == 'replace':
            return '新 API Key（已隐藏）'
        if self._provider.get('api_key_env'):
            return '环境变量 ' + str(self._provider['api_key_env'])
        return '保留文件凭据（保存时核对）'

    def _edit(self, field: str, value: object, return_step: str) -> None:
        self._edit_field = field
        self._text = '' if field == 'api_key' else str(value or '')
        self._return_step = return_step
        self.step = 'edit'
        self.selection = 0

    def _model_start(self, key: str | None) -> None:
        selected = next((item for item in self._models if item.key == key), None)
        self._old_model_key = selected.key if selected is not None else None
        self._model = {item.name: getattr(selected, item.name) for item in fields(ModelProfile)} if selected else dict(
            key='coder' if not self._models else 'model' + str(len(self._models) + 1), model_id='', name='',
            context_window=131072, max_output_tokens=8192, tools=True, think=False,
            reasoning_effort='high', stream_usage=True, output_token_field='max_tokens')
        self.step, self.selection = 'model_settings', 0

    def _save_model(self) -> None:
        model = ModelProfile(**self._model)
        if model.key != self._old_model_key and any(item.key == model.key for item in self._models):
            raise UserConfigError('config', 'duplicate-model-key')
        self._models = [model if item.key == self._old_model_key else item for item in self._models]
        if self._old_model_key is None:
            self._models.append(model)
        if not self._provider['default_model'] or self._provider['default_model'] == self._old_model_key:
            self._provider['default_model'] = model.key
        if self._provider['default_model'] not in {item.key for item in self._models}:
            self._provider['default_model'] = model.key
        self._selected_model_key = model.key

    def _build(self) -> ProviderProfile:
        if (self._provider['id'] != self._initial_provider_id
                and any(item.id == self._provider['id'] for item in self.view.providers)):
            raise UserConfigError('config', 'duplicate-provider-id')
        provider = ProviderProfile(**self._provider, models=tuple(self._models))
        selected = next(item for item in provider.models if item.key == self._selected_model_key)
        if not selected.tools:
            raise UserConfigError('config', 'tools-required')
        self.defaults = UserDefaults(provider.id, selected.key, selected.think, selected.reasoning_effort,
                                     selected.context_window, selected.max_output_tokens)
        return provider

    def handle(self, key: KeyPress) -> str | None:
        if self.submitting or self.cancelled:
            return None
        if key.key in (Keys.Escape, Keys.ControlC):
            self.cancelled = True
            self.clear_secret()
            self._text = ''
            return 'cancel'
        if key.key == Keys.ControlI:
            if self.step == 'edit' and self._edit_field == 'api_key':
                self._text = ''
            self.step = (self._return_step if self.step == 'edit' else 'credential'
                         if self.view.credentials_only else 'provider' if self.step == 'settings'
                         else 'settings' if self._provider else 'provider')
            self.selection = 0
            self.error = None
            return None
        if self.step == 'edit':
            if key.key in (Keys.ControlH, Keys.Backspace):
                self._text = self._text[:-1]
            elif key.key == Keys.ControlU:
                self._text = ''
            elif key.key == Keys.BracketedPaste or not isinstance(key.key, Keys):
                value = key.data if key.key == Keys.BracketedPaste else self._text + key.data
                limit = 4096 if self._edit_field == 'api_key' else 2048
                if (len(value.encode('utf-8', errors='replace')) > limit
                        or any(unicodedata.category(char).startswith('C') for char in value)):
                    self.error = '请输入单行内容，不支持控制字符或超长输入'
                    return None
                self._text = value
                self.error = None
            if key.key not in (Keys.ControlM, Keys.Enter):
                return None
            try:
                self._accept_text()
            except (ValueError, StopIteration):
                self.error = '输入无效，请检查格式与限制'
            return None
        choices = self._choices()
        if key.key in (Keys.Up, Keys.Down):
            self.selection = (self.selection + (1 if key.key == Keys.Down else -1)) % max(1, len(choices))
            return None
        if key.key not in (Keys.ControlM, Keys.Enter) or not choices:
            return None
        action = choices[self.selection][0]
        self.error = None
        try:
            return self._accept_choice(action)
        except UserConfigError as exc:
            self.error = '配置无效：' + exc.error_code
        except (ValueError, StopIteration):
            self.error = '请先配置有效的连接与模型'
        return None

    def _accept_text(self) -> None:
        field, value = self._edit_field, self._text.strip()
        if field == 'api_key':
            self._credential = SecretValue(value)
            self.credential_action = 'replace'
            self._provider.update(auth='bearer', api_key_env=None)
            self._text = ''
            self.step, self.selection = ('confirm', 0) if self.view.credentials_only else ('settings', 2)
        elif field == 'api_key_env':
            import re
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', value) is None:
                raise ValueError
            self.clear_secret()
            self.credential_action = 'keep'
            self._provider.update(auth='bearer', api_key_env=value)
            self.step, self.selection = ('confirm', 0) if self.view.credentials_only else ('settings', 2)
        else:
            target = self._model if self._return_step == 'model_settings' else self._provider
            target[field] = int(value) if field in ('context_window', 'max_output_tokens') else value
            self.step, self.selection = self._return_step, 0

    def _accept_choice(self, action: str) -> str | None:
        if self.step == 'provider':
            if action.startswith('+'):
                self._template(action[1:])
            else:
                self._load_provider(next(item for item in self.view.providers if item.id == action))
            self.step, self.selection = 'settings', 0
        elif self.step == 'settings':
            if action == 'default':
                self.set_default = not self.set_default
            elif action == 'model':
                self.step, self.selection = 'models', 0
            elif action == 'confirm':
                self._build()
                self.step, self.selection = 'confirm', 0
            else:
                self.step, self.selection = action, 0
        elif self.step == 'connection':
            if action == 'back':
                self.step, self.selection = 'settings', 0
            elif action == 'protocol':
                values = ('deepseek-chat', 'openai-chat-completions')
                self._provider['protocol'] = values[1 - values.index(self._provider['protocol'])]
                if self._provider['protocol'] == 'openai-chat-completions':
                    self._models = [replace(item, think=False) for item in self._models]
            else:
                self._edit(action, self._provider[action], 'connection')
        elif self.step == 'models':
            if action == 'back':
                self.step, self.selection = 'settings', 0
            else:
                self._model_start(None if action == 'add' else action)
        elif self.step == 'model_settings':
            if action == 'back':
                self._save_model()
                self.step, self.selection = 'settings', 0
            elif action in ('tools', 'think', 'stream_usage'):
                self._model[action] = not self._model[action]
            elif action in ('reasoning_effort', 'output_token_field'):
                values = ('low', 'high', 'max') if action == 'reasoning_effort' else ('max_tokens', 'max_completion_tokens')
                self._model[action] = values[(values.index(self._model[action]) + 1) % len(values)]
            else:
                self._edit(action, self._model[action], 'model_settings')
        elif self.step == 'credential':
            if action == 'keep':
                self.clear_secret()
                self.credential_action = 'keep'
                self._provider.update(auth=self._original_auth[0], api_key_env=self._original_auth[1])
                self.step, self.selection = ('confirm' if self.view.credentials_only else 'settings'), 0
            elif action == 'none':
                self.clear_secret()
                self.credential_action = 'keep'
                self._provider.update(auth='none', api_key_env=None)
                self.step, self.selection = 'settings', 0
            else:
                self._edit(action, self._provider.get(action), 'credential')
        elif self.step == 'confirm':
            self._build()
            if self._viewport is not None:
                self.render_lines(*self._viewport)
            if not self._confirm_blocked:
                self.target = action
                self.submitting = True
                return 'submit'
        return None

    def submission(self) -> LoginSubmission:
        if not self.submitting:
            raise RuntimeError('login submission is not reserved')
        provider = self._build()
        return LoginSubmission(self.defaults, self.credential_action, self.view.revision, self.target,
                               self._credential, provider, self.set_default)

    def render_lines(self, columns: int, rows: int | None = None) -> list[str]:
        self._viewport = columns, rows
        width = max(1, columns)
        pad = '  ' if width >= 24 else ''
        content = max(1, width - len(pad) * 2)
        with self.display_style.use():
            lines = [theme.fg(theme.EDITOR_BORDER, '─' * width),
                     pad + theme.fg(theme.EDITOR_BORDER, truncate_to_width(self.title, content))]
            body: list[str] = []

            def row(text: str, color=theme.ASSISTANT_TEXT):
                for part in wrap_text_ansi(sanitize_display(text), content):
                    body.append(pad + theme.fg(color, truncate_to_width(part, content)))

            if self.submitting:
                row('正在可靠保存，请等待…')
            elif self.step == 'edit':
                names = {'api_key': 'API key', 'api_key_env': '环境变量名', 'id': '实例 ID',
                         'base_url': 'API 根地址', 'model_id': '远端模型 ID'}
                row(names.get(self._edit_field, self._edit_field))
                text = '*' * min(32, len(self._text)) if self._edit_field == 'api_key' else self._text
                row(text or '输入或粘贴单行内容', theme.MD_LINK)
                row('Enter 保存此项 · Ctrl+U 清空 · Tab 返回', theme.MUTED_GRAY)
            else:
                if self.step == 'confirm':
                    row('供应商：' + str(self._provider['name']))
                    row('协议：' + str(self._provider['protocol']))
                    row('地址：' + str(self._provider['base_url']))
                    model = next((item for item in self._models if item.key == self._selected_model_key), None)
                    if model:
                        row('模型：' + model.key + ' → ' + model.model_id)
                        row(f'窗口/输出：{model.context_window}/{model.max_output_tokens}')
                        row('思考：' + (model.reasoning_effort if model.think else 'off'))
                    row('凭据：' + self._credential_label())
                    row('下次启动默认：' + ('改为此模型' if self.set_default else '保持原设置'))
                    row('保存位置：' + str(self.view.save_path))
                    if self.explicit:
                        row('本次启动仍按显式 CLI 参数覆盖', theme.MUTED_GRAY)
                    row('仅补凭据，存档配置保持' if self.view.credentials_only else self.view.effect,
                        theme.MUTED_GRAY)
                choices = self._choices()
                required_count = len(body)
                available = max(1, (rows or 30) - len(lines) - 2)
                visible = min(len(choices), max(1, available - (required_count if self.step == 'confirm' else 0)))
                start = max(0, min(self.selection - visible // 2, len(choices) - visible))
                for index in range(start, start + visible):
                    selected = index == self.selection
                    label = ('→ ' if selected else '  ') + choices[index][1]
                    label = truncate_to_width(sanitize_display(label), content)
                    if selected:
                        body.append(theme.bg_open(theme.MENU_SEL_BG) + pad + theme.fg_open(theme.MD_LINK)
                                    + label + ' ' * max(0, width - len(pad) - visible_width(label))
                                    + self.display_style.sgr('0'))
                    else:
                        body.append(pad + theme.fg(theme.ASSISTANT_TEXT, label))
            if self.error:
                row(self.error, theme.DIFF_REMOVED)
            self._confirm_blocked = (self.step == 'confirm' and not self.submitting
                                     and (width < 18 or (rows is not None and len(lines) + len(body) + 2 > rows)))
            if self._confirm_blocked:
                row('窗口太小，请放大后确认', theme.DIFF_REMOVED)
            footer = pad + theme.fg(theme.MUTED_GRAY, truncate_to_width(
                '↑↓ 选择 · Enter 确认 · Tab 返回 · Esc 取消', content))
            if rows is not None:
                budget = max(0, rows - len(lines) - 1)
                if len(body) > budget:
                    body = body[:max(0, budget - 1)] + body[-1:] if budget else []
            result = [*lines, *body, footer]
            return result[:max(1, rows)] if rows is not None else result
