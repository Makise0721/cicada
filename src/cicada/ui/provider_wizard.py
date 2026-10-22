"""多供应商配置的纯显示/编辑状态；秘密只在提交时交给应用。"""

from dataclasses import fields, replace
import re
import unicodedata

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.ui import theme
from cicada.ui.display_style import DisplayStyle, current_style
from cicada.ui.main_screen import truncate_to_width, visible_width, wrap_text_ansi
from cicada.ui.rendering import sanitize_display
from cicada.provider_presets import is_opencode_go_endpoint, opencode_go_profile
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
        self.selection = 0
        self.query = ''
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
        self._custom_form = False
        self._quick_connect = False
        self._model_settings_return = 'settings'
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
        self._quick_connect = (provider.protocol == 'openai-chat-completions'
                               and is_opencode_go_endpoint(provider.base_url))
        self._custom_form = not (
            (provider.id == 'deepseek' and provider.protocol == 'deepseek-chat')
            or (provider.id == 'openai' and provider.protocol == 'openai-chat-completions'
                and provider.base_url.rstrip('/') == 'https://api.openai.com/v1')
            or self._quick_connect)
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
        self._custom_form = kind == 'custom'
        self._quick_connect = kind == 'opencode-go'
        if kind == 'deepseek':
            self._provider = dict(id='deepseek', name='DeepSeek', protocol='deepseek-chat',
                                  base_url='https://api.deepseek.com', default_model='deepseek-flash',
                                  auth='bearer', api_key_env='DEEPSEEK_API_KEY')
            self._models = [ModelProfile(key=model, model_id=model, name=model,
                                        max_output_tokens=32768, think=True)
                            for model in ('deepseek-flash', 'deepseek-v4-pro')]
        elif kind == 'opencode-go':
            preset = opencode_go_profile()
            self._provider = {key: getattr(preset, key) for key in
                              ('id', 'name', 'protocol', 'base_url', 'default_model', 'auth', 'api_key_env')}
            self._models = list(preset.models)
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
            choices = [('+custom', '自定义供应商')]
            for item in self.view.providers:
                status = ('已配置' if self.view.credential_present else '未配置') if item.id == self.view.provider else '已添加'
                choices.append((item.id, item.name + ' · ' + status))
            for kind, name, protocol in (
                ('deepseek', 'DeepSeek', 'deepseek-chat'),
                ('openai', 'OpenAI Chat', 'openai-chat-completions'),
            ):
                exists = any(item.protocol == protocol and (
                    kind == 'deepseek' or item.base_url.rstrip('/') == 'https://api.openai.com/v1')
                    for item in self.view.providers)
                choices.append(('+' + kind, name + ('（新连接）' if exists else '') + ' · 未配置'))
            exists = any(is_opencode_go_endpoint(item.base_url) for item in self.view.providers)
            choices.append(('+opencode-go', 'OpenCode Go' + ('（新连接）' if exists else '') + ' · 未配置'))
            query = self.query.casefold()
            return [(key, label) for key, label in choices if query in (key + ' ' + label).casefold()]
        if self.step == 'custom':
            credential = ('无需认证' if self._provider['auth'] == 'none' else
                          '已输入' if self.credential_action == 'replace' else
                          '保留现有凭据' if self._provider.get('api_key_env') or (
                              self._initial_provider_id == self.view.provider and self.view.credential_present)
                          else '输入 API 密钥')
            rows = [(key, label + '：' + str(value)) for key, label, value in (
                ('id', 'Provider ID', self._provider['id']),
                ('name', '显示名称', self._provider['name']),
                ('base_url', 'API 地址', self._provider['base_url'] or '输入 API 地址'),
                ('protocol', 'API 协议', self._protocol_label()),
                ('api_key', 'API 密钥', credential))]
            rows += [('model:' + item.key, item.name or item.model_id) for item in self._models]
            rows += [('add_model', '+ 添加模型'), ('advanced', '高级设置')]
            creating = self._initial_provider_id is None
            rows += [(action, ('创建并应用' if creating else '保存并应用') if action == 'save-and-apply'
                      else ('创建供应商' if creating else '保存配置')) for action in self.view.actions]
            return rows + [('cancel', '取消')]
        if self.step == 'protocol':
            return [('openai-chat-completions', 'OpenAI Chat Completions'),
                    ('deepseek-chat', 'DeepSeek 官方 API')]
        if self.step == 'custom_advanced':
            return [('credential', '凭据来源'),
                    ('default', '设为下次启动默认：' + ('是' if self.set_default else '否'))] + [
                        ('model:' + item.key, (item.name or item.model_id) + ' · 模型参数')
                        for item in self._models] + [('back', '返回')]
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
            from cicada.user_config import supports_thinking

            if supports_thinking(self._provider['protocol'], self._provider['base_url'], self._model['model_id']):
                keys += [('think', '思考'), ('reasoning_effort', '思考强度')]
            if self._provider['protocol'] != 'deepseek-chat':
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

    def _settings_step(self) -> str:
        return 'custom' if self._custom_form else 'settings'

    def _protocol_label(self) -> str:
        return ('OpenAI Chat Completions' if self._provider['protocol'] == 'openai-chat-completions'
                else 'DeepSeek 官方 API')

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

    def _model_start(self, key: str | None, *, return_step: str = 'settings') -> None:
        selected = next((item for item in self._models if item.key == key), None)
        self._old_model_key = selected.key if selected is not None else None
        self._model_settings_return = return_step
        self._model = {item.name: getattr(selected, item.name) for item in fields(ModelProfile)} if selected else dict(
            key='coder' if not self._models else 'model' + str(len(self._models) + 1), model_id='', name='',
            context_window=131072, max_output_tokens=8192, tools=True, think=False,
            reasoning_effort='high', stream_usage=True, output_token_field='max_tokens')
        self.step, self.selection = 'model_settings', 0

    def _new_model_key(self, model_id: str) -> str:
        base = re.sub(r'[^A-Za-z0-9._-]+', '-', model_id).strip('-._')[:64] or 'model'
        used = {item.key for item in self._models}
        key, number = base, 2
        while key in used:
            suffix = '-' + str(number)
            key = base[:64 - len(suffix)] + suffix
            number += 1
        return key

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
        if self._custom_form:
            if not self._provider['base_url']:
                raise UserConfigError('config', 'missing-api-address')
            if not self._models:
                raise UserConfigError('config', 'missing-model')
            if (self._initial_provider_id is None and self._provider['auth'] == 'bearer'
                    and self.credential_action == 'keep' and not self._provider['api_key_env']):
                raise UserConfigError('config', 'missing-credential')
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
                         if self.view.credentials_only else 'provider' if self.step in ('settings', 'custom')
                         else self._settings_step() if self._provider else 'provider')
            self.selection = 0
            if self.step == 'provider':
                self.query = ''
            self.error = None
            return None
        if self.step == 'provider':
            if key.key in (Keys.ControlH, Keys.Backspace):
                self.query = self.query[:-1]
            elif key.key == Keys.ControlU:
                self.query = ''
            elif key.key == Keys.BracketedPaste or not isinstance(key.key, Keys):
                value = key.data if key.key == Keys.BracketedPaste else self.query + key.data
                if len(value) <= 256 and not any(unicodedata.category(char).startswith('C') for char in value):
                    self.query = value
                else:
                    return None
            if key.key in (Keys.ControlH, Keys.Backspace, Keys.ControlU, Keys.BracketedPaste) or not isinstance(key.key, Keys):
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
                return self._accept_text()
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
            self.error = {'missing-api-address': '请填写 API 地址', 'missing-model': '请添加模型',
                          'missing-credential': '请填写 API 密钥'}.get(exc.error_code, '配置无效：' + exc.error_code)
        except (ValueError, StopIteration):
            self.error = '请先配置有效的连接与模型'
        return None

    def _accept_text(self) -> str | None:
        field, value = self._edit_field, self._text.strip()
        if field == 'api_key':
            self._credential = SecretValue(value)
            self.credential_action = 'replace'
            self._provider.update(auth='bearer', api_key_env=None)
            self._text = ''
            if self._quick_connect:
                self._build()
                self.target = ('save-and-apply' if 'save-and-apply' in self.view.actions
                               and not self.view.credentials_only else 'defaults-only')
                self.submitting = True
                return 'submit'
            self.step, self.selection = (('confirm', 0) if self.view.credentials_only else
                                        (self._settings_step(), 4 if self._custom_form else 2))
        elif field == 'api_key_env':
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', value) is None:
                raise ValueError
            self.clear_secret()
            self.credential_action = 'keep'
            self._provider.update(auth='bearer', api_key_env=value)
            self.step, self.selection = (('confirm', 0) if self.view.credentials_only else
                                        (self._settings_step(), 4 if self._custom_form else 2))
        elif field == 'model_id' and self._return_step == 'custom':
            if any(item.model_id == value and item.key != self._old_model_key for item in self._models):
                raise UserConfigError('config', 'duplicate-model')
            self._model['model_id'] = value
            if self._old_model_key is None:
                self._model['key'] = self._new_model_key(value)
            self._save_model()
            self.step = 'custom'
            self.selection = next(i for i, (key, _) in enumerate(self._choices())
                                  if key == 'model:' + self._selected_model_key)
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
            self.step, self.selection = self._settings_step(), 0
            if not self._custom_form and self._provider['auth'] != 'none':
                self._edit('api_key', '', 'settings')
        elif self.step == 'custom':
            if action == 'cancel':
                self.cancelled = True
                self.clear_secret()
                self._text = ''
                return 'cancel'
            if action in self.view.actions:
                self._build()
                if self._viewport is not None:
                    self.render_lines(*self._viewport)
                if not self._confirm_blocked:
                    self.target, self.submitting = action, True
                    return 'submit'
            elif action == 'protocol':
                self.step, self.selection = 'protocol', 0
            elif action == 'advanced':
                self.step, self.selection = 'custom_advanced', 0
            elif action == 'add_model' or action.startswith('model:'):
                key = None if action == 'add_model' else action.removeprefix('model:')
                self._model_start(key)
                self._edit('model_id', self._model['model_id'], 'custom')
            else:
                self._edit(action, self._provider.get(action), 'custom')
        elif self.step == 'protocol':
            self._provider['protocol'] = action
            if action == 'openai-chat-completions':
                self._models = [replace(item, think=False) for item in self._models]
            self.step, self.selection = 'custom', 3
        elif self.step == 'custom_advanced':
            if action == 'default':
                self.set_default = not self.set_default
            elif action == 'back':
                self.step, self.selection = 'custom', 0
            elif action.startswith('model:'):
                self._model_start(action.removeprefix('model:'), return_step='custom_advanced')
            else:
                self.step, self.selection = 'credential', 0
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
                self.step, self.selection = self._model_settings_return, 0
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
                self.step, self.selection = ('confirm' if self.view.credentials_only else self._settings_step()), 0
            elif action == 'none':
                self.clear_secret()
                self.credential_action = 'keep'
                self._provider.update(auth='none', api_key_env=None)
                self.step, self.selection = self._settings_step(), 0
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
        if self.step == 'custom':
            return self._render_custom(columns, rows)
        width = max(1, columns)
        pad = '  ' if width >= 24 and self.step != 'provider' else ''
        content = max(1, width - len(pad))
        with self.display_style.use():
            name = str(self._provider.get('name', '供应商'))
            editing_key = self.step == 'edit' and self._edit_field == 'api_key'
            title = ('选择要配置的供应商：' if self.step == 'provider' else
                     '登录 ' + name if editing_key else '配置 ' + name)
            heading = self.display_style.sgr('1') + theme.fg(theme.MD_CODE, truncate_to_width(title, content))
            lines = [pad + heading, '']
            body: list[str] = []

            def row(text: str, color=theme.ASSISTANT_TEXT):
                for part in wrap_text_ansi(sanitize_display(text), content):
                    body.append(pad + theme.fg(color, truncate_to_width(part, content)))

            if self.submitting:
                row('正在保存…', theme.MUTED_GRAY)
            elif self.step == 'edit':
                names = {'api_key_env': '环境变量名', 'id': 'Provider ID', 'name': '显示名称',
                         'base_url': 'API 地址', 'model_id': '远端模型 ID',
                         'key': '本地标识', 'context_window': '上下文窗口',
                         'max_output_tokens': '输出上限'}
                row('请输入 ' + name + ' API Key' if editing_key else names.get(self._edit_field, self._edit_field))
                value = '*' * min(32, len(self._text)) if editing_key else self._text
                prompt_width = max(0, content - 2)
                prompt = truncate_to_width(value, max(0, prompt_width - 1)) + ('█' if prompt_width else '')
                row('> ' + prompt)
                row('(Esc/Ctrl+C 取消，Enter 确认，Tab 返回设置)', theme.MUTED_GRAY)
            else:
                if self.step == 'provider':
                    prompt = '> ' + truncate_to_width(self.query, max(0, content - 3))
                    lines.extend([theme.fg(theme.ASSISTANT_TEXT, truncate_to_width(prompt + '█', width)), ''])
                if self.step == 'confirm':
                    row('供应商：' + name)
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
                reserve = 3 if self.step == 'provider' else 4
                available = max(1, (rows or 30) - len(lines) - len(body) - reserve)
                visible = min(len(choices), 8, available)
                start = max(0, min(self.selection - visible // 2, len(choices) - visible))
                for index in range(start, start + visible):
                    selected = index == self.selection
                    label, separator, status = choices[index][1].partition(' · ') if self.step == 'provider' else (choices[index][1], '', '')
                    prefix = '→ ' if selected else '  '
                    label = truncate_to_width(sanitize_display(prefix + label), content)
                    suffix = truncate_to_width(separator + status, max(0, content - visible_width(label)))
                    body.append(pad + theme.fg(theme.MD_CODE if selected else theme.ASSISTANT_TEXT, label)
                                + theme.fg(theme.MUTED_GRAY, suffix))
                if self.step == 'provider':
                    if choices:
                        row(f'  ({self.selection + 1}/{len(choices)})', theme.MUTED_GRAY)
                    else:
                        row('没有匹配的供应商', theme.MUTED_GRAY)
            if self.error:
                row(self.error, theme.DIFF_REMOVED)
            footer = ['', theme.fg(theme.MD_LINK, '─' * width)]
            if self.step not in ('provider', 'edit') and not self.submitting:
                footer.insert(0, pad + theme.fg(theme.MUTED_GRAY, truncate_to_width(
                    '↑↓ 选择 · Enter 确认 · Tab 返回 · Esc 取消', content)))
            self._confirm_blocked = (self.step == 'confirm' and not self.submitting
                                     and (width < 18 or (rows is not None and len(lines) + len(body) + len(footer) > rows)))
            if self._confirm_blocked:
                row('窗口太小，请放大后确认', theme.DIFF_REMOVED)
            if rows is not None:
                budget = max(0, rows - len(lines) - len(footer))
                if len(body) > budget:
                    body = body[:max(0, budget - 1)] + body[-1:] if budget else []
            result = [*lines, *body, *footer]
            return result[:max(1, rows)] if rows is not None else result

    def _render_custom(self, columns: int, rows: int | None) -> list[str]:
        width = max(1, columns)
        pad = '  ' if width >= 24 else ''
        content = max(1, width - len(pad))
        with self.display_style.use():
            header = [pad + theme.fg(theme.MD_CODE, truncate_to_width('自定义模型 API', content)), '']
            body: list[str] = []
            selected_line = 0

            def row(text: str, index: int | None = None):
                nonlocal selected_line
                selected = index == self.selection
                if selected:
                    selected_line = len(body)
                prefix = '→ ' if selected else '  '
                color = theme.MD_CODE if selected else theme.ASSISTANT_TEXT
                for part in wrap_text_ansi(sanitize_display(prefix + text), content):
                    body.append(pad + theme.fg(color, truncate_to_width(part, content)))

            if self.submitting:
                row('正在保存…')
            else:
                choices = self._choices()
                for index, (_, label) in enumerate(choices[:5]):
                    row(label, index)
                body.extend(['', pad + theme.fg(theme.MUTED_GRAY, truncate_to_width('模型目录', content))])
                models = [(i, label) for i, (key, label) in enumerate(choices) if key.startswith('model:')]
                if models:
                    current = next((i for i, (index, _) in enumerate(models) if index == self.selection), 0)
                    count = min(3, len(models))
                    start = max(0, min(current - count // 2, len(models) - count))
                    for index, label in models[start:start + count]:
                        row(label, index)
                else:
                    body.append(pad + theme.fg(theme.MUTED_GRAY, truncate_to_width('  尚未添加模型', content)))
                for index, (key, label) in enumerate(choices[5:], 5):
                    if not key.startswith('model:'):
                        if key in ('advanced', self.view.actions[0]):
                            body.append('')
                        row(label, index)
            if self.error:
                body.append(pad + theme.fg(theme.DIFF_REMOVED, truncate_to_width(self.error, content)))
            footer = [pad + theme.fg(theme.MUTED_GRAY, truncate_to_width(
                '↑↓ 选择 · Enter 编辑或确认 · Esc 取消', content)),
                theme.fg(theme.MD_LINK, '─' * width)]
            self._confirm_blocked = (width < 24 or (rows is not None and len(header) + len(body) + len(footer) > rows))
            if self._confirm_blocked and not self.submitting:
                footer.insert(0, theme.fg(theme.DIFF_REMOVED, truncate_to_width('请放大窗口后保存', width)))
            if rows is not None:
                budget = max(0, rows - len(header) - len(footer))
                start = max(0, min(selected_line - budget // 2, len(body) - budget))
                body = body[start:start + budget]
            lines = [*header, *body, *footer]
            return lines[:max(1, rows)] if rows is not None else lines
