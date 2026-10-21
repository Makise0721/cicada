"""用户 TOML/凭据 JSON 一致读取与受拥有的原子发布。"""

import asyncio
import json
import msvcrt
import os
import tempfile
import threading
import time
import tomllib
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

from cicada.user_config import (
    LoginResult, LoginSubmission, LoginView, SecretValue, UserConfigError,
    UserConfigSnapshot, UserDefaults, ModelProfile, ProviderProfile, _profile_key,
)

FILE_LIMIT = 64 * 1024
DEFAULT_FIELDS = frozenset(('provider', 'model', 'think', 'reasoning_effort',
                            'num_ctx', 'max_output_tokens'))
WRITE_LOCK_TIMEOUT_S = 2.0


def _config_bytes(defaults: UserDefaults) -> bytes:
    lines = ['version = 1', '[defaults]']
    for name in sorted(DEFAULT_FIELDS):
        value = getattr(defaults, name)
        # JSON 的已校验目录字符串也是合法的 TOML basic string。
        lines.append(f'{name} = {json.dumps(value, ensure_ascii=False)}')
    if defaults.credential_ref is not None:
        lines.extend(('', '[credentials.deepseek]', f'ref = "{defaults.credential_ref}"'))
    raw = ('\n'.join(lines) + '\n').encode('utf-8')
    _defaults(raw)  # 与读取使用同一白名单及严格类型边界。
    return raw


def _create_default_config(root: Path) -> None:
    """只在目标仍缺失时发布完整默认配置，不覆盖并发创建的文件。"""
    temporary = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.config.toml-', suffix='.tmp', dir=root)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(_config_bytes(UserDefaults()))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, root / 'config.toml')
        except FileExistsError:
            pass
    except OSError:
        raise UserConfigError('config', 'write-failed') from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                raise UserConfigError('config', 'write-failed') from None


def _atomic_replace(path: Path, raw: bytes, cancelled: threading.Event) -> None:
    """worker 独占临时文件；replace 之前最后检查尚未接受的取消。"""
    if len(raw) > FILE_LIMIT:
        raise UserConfigError('save', 'file-too-large')
    temporary = None
    replaced = False
    try:
        fd, name = tempfile.mkstemp(prefix=f'.{path.name}-', suffix='.tmp', dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if cancelled.is_set():
            raise UserConfigError('save', 'cancelled')
        os.replace(temporary, path)
        replaced = True
    finally:
        if temporary is not None and not replaced:
            temporary.unlink(missing_ok=True)


def _object(value, allowed, kind, field_name=None):
    if type(value) is not dict:
        raise UserConfigError(kind, 'invalid-type', field_name)
    if set(value) - set(allowed):
        # 未知字段名也可能是秘密；不回显任意用户输入。
        raise UserConfigError(kind, 'unknown-field', field_name)
    return value


def _version(data, kind):
    if type(data.get('version')) is not int or data['version'] != 1:
        raise UserConfigError(kind, 'unsupported-version', 'version')


def _ref(value, kind):
    try:
        ref = UUID(value) if type(value) is str else None
    except ValueError:
        ref = None
    if ref is None or ref.version != 4 or str(ref) != value:
        raise UserConfigError(kind, 'invalid-ref', 'ref')
    return value


def _read(path: Path, kind: str) -> bytes | None:
    try:
        with path.open('rb') as stream:
            raw = stream.read(FILE_LIMIT + 1)
    except FileNotFoundError:
        return None
    except OSError:
        raise UserConfigError(kind, 'read-failed') from None
    if len(raw) > FILE_LIMIT:
        raise UserConfigError(kind, 'file-too-large')
    return raw


def _legacy_defaults(raw: bytes | None) -> tuple[UserDefaults, frozenset[str]]:
    if raw is None:
        return UserDefaults(), frozenset()
    try:
        data = tomllib.loads(raw.decode('utf-8'))
    except (UnicodeError, tomllib.TOMLDecodeError, RecursionError):
        raise UserConfigError('config', 'invalid-format') from None
    _object(data, ('version', 'defaults', 'credentials'), 'config')
    _version(data, 'config')
    values = _object(data.get('defaults', {}), DEFAULT_FIELDS, 'config', 'defaults')
    for name, value in values.items():
        expected = bool if name == 'think' else int if name in ('num_ctx', 'max_output_tokens') else str
        if type(value) is not expected:
            raise UserConfigError('config', 'invalid-type', name)
    if values.get('provider', 'deepseek') != 'deepseek':
        raise UserConfigError('config', 'unsupported-value', 'provider')
    credentials = _object(data.get('credentials', {}), ('deepseek',), 'config', 'credentials')
    provider = _object(credentials.get('deepseek', {}), ('ref',), 'config', 'credentials.deepseek')
    ref = _ref(provider['ref'], 'config') if 'ref' in provider else None
    defaults = UserDefaults(**values, credential_ref=ref)
    try:
        _legacy_provider(defaults)
    except ValueError:
        raise UserConfigError('config', 'invalid-model-settings', 'defaults') from None
    return defaults, frozenset(values)


def _legacy_provider(defaults: UserDefaults) -> ProviderProfile:
    models = tuple(ModelProfile(key=model, model_id=model, name=model,
                                context_window=defaults.num_ctx,
                                max_output_tokens=defaults.max_output_tokens,
                                think=defaults.think, reasoning_effort=defaults.reasoning_effort)
                   for model in ('deepseek-flash', 'deepseek-v4-pro'))
    return ProviderProfile('deepseek', 'DeepSeek', 'deepseek-chat', 'https://api.deepseek.com',
                           models, defaults.model, api_key_env='DEEPSEEK_API_KEY',
                           credential_ref=defaults.credential_ref)


def _model_defaults(provider: ProviderProfile, key: str) -> UserDefaults:
    model = next((model for model in provider.models if model.key == key), None)
    if model is None:
        raise UserConfigError('config', 'unknown-model')
    return UserDefaults(provider.id, model.key, model.think, model.reasoning_effort,
                        model.context_window, model.max_output_tokens, provider.credential_ref)


def _configuration(raw: bytes | None) -> tuple[UserDefaults, frozenset[str], tuple[ProviderProfile, ...], int]:
    if raw is None:
        defaults = UserDefaults()
        return defaults, frozenset(), (_legacy_provider(defaults),), 1
    try:
        data = tomllib.loads(raw.decode('utf-8'))
    except (UnicodeError, tomllib.TOMLDecodeError, RecursionError):
        raise UserConfigError('config', 'invalid-format') from None
    if type(data.get('version')) is int and data['version'] == 1:
        defaults, fields = _legacy_defaults(raw)
        return defaults, fields, (_legacy_provider(defaults),), 1
    _object(data, ('version', 'defaults', 'providers'), 'config')
    if type(data.get('version')) is not int or data['version'] != 2:
        raise UserConfigError('config', 'unsupported-version', 'version')
    values = _object(data.get('defaults'), ('provider', 'model'), 'config', 'defaults')
    records = data.get('providers')
    if type(records) is not dict or not 1 <= len(records) <= 64:
        raise UserConfigError('config', 'invalid-provider-settings')
    providers = []
    provider_fields = ('name', 'protocol', 'base_url', 'models', 'default_model', 'auth',
                       'api_key_env', 'credential_ref')
    model_fields = ('id', 'name', 'context_window', 'max_output_tokens', 'tools', 'think',
                    'reasoning_effort', 'stream_usage', 'output_token_field')
    for provider_id, record in records.items():
        _object(record, provider_fields, 'config', 'providers')
        model_records = record.get('models')
        if type(model_records) is not dict or not 1 <= len(model_records) <= 128:
            raise UserConfigError('config', 'invalid-model-settings')
        models = []
        for model_key, model in model_records.items():
            _object(model, model_fields, 'config', 'models')
            if not {'id', 'context_window', 'max_output_tokens', 'tools'} <= set(model):
                raise UserConfigError('config', 'missing-model-settings')
            model_values = dict(model)
            model_id = model_values.pop('id')
            models.append(ModelProfile(model_key, model_id, **model_values))
        if not {'name', 'protocol', 'base_url', 'default_model'} <= set(record):
            raise UserConfigError('config', 'missing-provider-settings')
        provider_values = {name: value for name, value in record.items() if name != 'models'}
        if 'credential_ref' in provider_values:
            _ref(provider_values['credential_ref'], 'config')
        providers.append(ProviderProfile(provider_id, models=tuple(models), **provider_values))
    selected = next((provider for provider in providers if provider.id == values.get('provider')), None)
    if selected is None:
        raise UserConfigError('config', 'unknown-provider')
    defaults = _model_defaults(selected, values.get('model'))
    return defaults, DEFAULT_FIELDS, tuple(providers), 2


def _defaults(raw: bytes | None) -> tuple[UserDefaults, frozenset[str]]:
    defaults, fields, _, _ = _configuration(raw)
    return defaults, fields


def _catalog_bytes(providers: tuple[ProviderProfile, ...], defaults: UserDefaults) -> bytes:
    def scalar(value):
        return json.dumps(value, ensure_ascii=False)
    lines = ['version = 2', '', '[defaults]', f'provider = {scalar(defaults.provider)}',
             f'model = {scalar(defaults.model)}']
    for provider in providers:
        lines.extend(('', f'[providers.{scalar(provider.id)}]'))
        for name in ('name', 'protocol', 'base_url', 'default_model', 'auth',
                     'api_key_env', 'credential_ref'):
            value = getattr(provider, name)
            if value is not None:
                lines.append(f'{name} = {scalar(value)}')
        for model in provider.models:
            lines.extend(('', f'[providers.{scalar(provider.id)}.models.{scalar(model.key)}]',
                          f'id = {scalar(model.model_id)}'))
            for name in ('name', 'context_window', 'max_output_tokens', 'tools', 'think',
                         'reasoning_effort', 'stream_usage', 'output_token_field'):
                lines.append(f'{name} = {scalar(getattr(model, name))}')
    raw = ('\n'.join(lines) + '\n').encode('utf-8')
    if len(raw) > FILE_LIMIT:
        raise UserConfigError('save', 'file-too-large')
    _configuration(raw)
    return raw


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise UserConfigError('credentials', 'duplicate-field')
        result[name] = value
    return result


def _entries(raw: bytes | None) -> dict:
    if raw is None:
        raise UserConfigError('credentials', 'missing-ref', 'ref')
    try:
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, UserConfigError):
            raise
        raise UserConfigError('credentials', 'invalid-format') from None
    _object(data, ('version', 'entries'), 'credentials')
    _version(data, 'credentials')
    entries = data.get('entries')
    if type(entries) is not dict:
        raise UserConfigError('credentials', 'invalid-type', 'entries')
    for entry_ref, entry in entries.items():
        _ref(entry_ref, 'credentials')
        _object(entry, ('provider', 'api_key'), 'credentials', 'entries')
        if not _profile_key(entry.get('provider')):
            raise UserConfigError('credentials', 'unsupported-value', 'provider')
    return entries


def _credential(raw: bytes | None, ref: str, provider_id: str = 'deepseek') -> SecretValue:
    entry = _entries(raw).get(ref)
    if entry is None:
        raise UserConfigError('credentials', 'missing-ref', 'ref')
    if entry.get('provider') != provider_id:
        raise UserConfigError('credentials', 'unsupported-value', 'provider')
    try:
        return SecretValue(entry.get('api_key'))
    except UserConfigError:
        raise UserConfigError('credentials', 'invalid-key', 'api_key') from None


class FileUserConfigStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        # 一份最近观察供02核对；原始字节只留在Adapter，不派生key hash。
        self._observation: tuple[str, bytes | None, bytes | None, str, str] | None = None
        self._save_task: asyncio.Task | None = None

    def ensure_default_config(self) -> bool:
        """启动时补齐缺失的配置；返回是否观察到缺失。load 保持只读。"""
        if _read(self.root / 'config.toml', 'config') is not None:
            return False
        _create_default_config(self.root)
        return True

    def read_profiles(self) -> tuple[ProviderProfile, ...]:
        """只读连接目录，不打开凭据文件，也不读取环境变量值。"""
        return _configuration(_read(self.root / 'config.toml', 'config'))[2]

    def load(self, environment_key: str | None = None, *, provider_id: str | None = None,
             model_key: str | None = None) -> UserConfigSnapshot:
        for _ in range(3):
            raw = _read(self.root / 'config.toml', 'config')
            defaults, fields, providers, version = _configuration(raw)
            selected_id = provider_id if provider_id is not None else defaults.provider
            provider = next((item for item in providers if item.id == selected_id), None)
            if provider is None:
                raise UserConfigError('config', 'unknown-provider')
            key = model_key if model_key is not None else (
                defaults.model if selected_id == defaults.provider else provider.default_model)
            if version == 2 or provider_id is not None or model_key is not None:
                defaults = _model_defaults(provider, key)
            environment = None
            value = None
            if provider.api_key_env is not None:
                if provider.id == 'deepseek' and provider.api_key_env == 'DEEPSEEK_API_KEY' and environment_key is not None:
                    value = (environment_key if version == 2 or provider_id is not None
                             or environment_key.strip() else None)
                elif version == 2 or provider_id is not None:
                    value = os.environ.get(provider.api_key_env)
            if value is not None:
                try:
                    environment = SecretValue(value)
                except UserConfigError:
                    raise UserConfigError('environment', 'invalid-key') from None
            credential_raw = None
            source = 'environment' if environment is not None else 'missing'
            secret, error = environment, None
            if environment is None and defaults.credential_ref is not None:
                try:
                    credential_raw = _read(self.root / 'credentials.json', 'credentials')
                    secret = _credential(credential_raw, defaults.credential_ref, provider.id)
                    source = 'file'
                except UserConfigError as exc:
                    error = exc
                if _read(self.root / 'config.toml', 'config') != raw:
                    continue
                if error is not None:
                    raise error
            revision = str(uuid4())
            self._observation = revision, raw, credential_raw, source, provider.id
            return UserConfigSnapshot(defaults, fields, secret, source, revision, providers)
        raise UserConfigError('config', 'configuration-changing')

    def read_view(self, environment_key: str | None = None, *,
                  provider_id: str | None = None) -> LoginView:
        snapshot = self.load(environment_key, provider_id=provider_id)
        d = snapshot.defaults
        provider = next(item for item in snapshot.providers if item.id == d.provider)
        return LoginView(d.provider, d.model, d.think, d.reasoning_effort, d.num_ctx,
                         d.max_output_tokens, snapshot.credential is not None or provider.auth == 'none',
                         snapshot.credential_source, self.root / 'config.toml',
                         '之后新建会话使用这些默认值', ('defaults-only',), snapshot.revision,
                         providers=tuple(replace(item, credential_ref=None) for item in snapshot.providers))

    async def save(self, submission: LoginSubmission) -> LoginResult:
        """第一次 await 前占位；取消请求传给 worker，等待实际终态并如实返回。"""
        if self._save_task is not None:
            return LoginResult(False, False, 'not-requested', 'operation-in-progress')
        observation = self._observation
        if observation is None or observation[0] != submission.revision:
            return LoginResult(False, False, 'not-requested', 'configuration-changed')
        cancelled = threading.Event()
        task = asyncio.create_task(asyncio.to_thread(self._save, submission, observation, cancelled))
        self._save_task = task
        try:
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled.set()
                    # 多次 cancel 也不能丢掉拥有关系，或把已经发布的文件报告为未保存。
        finally:
            self._save_task = None

    def _save_profiles(self, submission, raw, observation, cancelled) -> None:
        current_defaults, _, profiles, _ = _configuration(raw)
        providers = {provider.id: provider for provider in profiles}
        proposed = submission.provider_profile
        if proposed is None:
            proposed = providers.get(submission.defaults.provider)
            if proposed is None:
                raise UserConfigError('save', 'unknown-provider')
            selected = next((model for model in proposed.models if model.key == submission.defaults.model), None)
            if selected is None:
                raise UserConfigError('save', 'unknown-model')
            selected = replace(selected, think=submission.defaults.think,
                               reasoning_effort=submission.defaults.reasoning_effort,
                               context_window=submission.defaults.num_ctx,
                               max_output_tokens=submission.defaults.max_output_tokens)
            proposed = replace(proposed, models=tuple(selected if item.key == selected.key else item
                                                     for item in proposed.models))
        if (submission.defaults.provider != proposed.id
                or submission.defaults.model not in {model.key for model in proposed.models}):
            raise UserConfigError('save', 'invalid-model-settings')
        existing = providers.get(proposed.id)
        ref = existing.credential_ref if existing is not None else None
        if proposed.auth == 'none':
            if submission.credential_action == 'replace':
                raise UserConfigError('save', 'unexpected-credential')
            ref = None
        credentials_path = self.root / 'credentials.json'
        credential_raw = None
        if observation[3] == 'file':
            credential_raw = _read(credentials_path, 'credentials')
            if credential_raw != observation[2]:
                raise UserConfigError('save', 'configuration-changed')
        credentials = None
        if submission.credential_action == 'replace':
            if credential_raw is None:
                credential_raw = _read(credentials_path, 'credentials')
            entries = {} if credential_raw is None else _entries(credential_raw)
            if ref is not None:
                _credential(credential_raw, ref, proposed.id)
            ref = str(uuid4())
            entries[ref] = {'provider': proposed.id, 'api_key': submission.credential.reveal()}
            credentials = json.dumps({'version': 1, 'entries': entries}, ensure_ascii=False).encode('utf-8')
            if len(credentials) > FILE_LIMIT:
                raise UserConfigError('save', 'file-too-large')
        elif (proposed.auth == 'bearer' and ref is None and proposed.api_key_env is None):
            raise UserConfigError('save', 'missing-credential')
        proposed = replace(proposed, credential_ref=ref)
        providers[proposed.id] = proposed
        defaults = (_model_defaults(proposed, submission.defaults.model) if submission.set_default
                    else current_defaults)
        target_raw = _catalog_bytes(tuple(providers.values()), defaults)
        if credentials is not None:
            _atomic_replace(credentials_path, credentials, cancelled)
        _atomic_replace(self.root / 'config.toml', target_raw, cancelled)

    def _save(self, submission, observation, cancelled) -> LoginResult:
        lock_fd = None
        locked = False
        published = False
        error_code = None
        try:
            if cancelled.is_set():
                raise UserConfigError('save', 'cancelled')
            self.root.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(self.root / 'config.lock', os.O_CREAT | os.O_RDWR, 0o600)
            deadline = time.monotonic() + WRITE_LOCK_TIMEOUT_S
            while True:
                if cancelled.is_set():
                    raise UserConfigError('save', 'cancelled')
                try:
                    os.lseek(lock_fd, 0, os.SEEK_SET)
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise UserConfigError('save', 'lock-timeout') from None
                    time.sleep(0.01)
            config_path = self.root / 'config.toml'
            credentials_path = self.root / 'credentials.json'
            raw = _read(config_path, 'config')
            if raw != observation[1]:
                raise UserConfigError('save', 'configuration-changed')
            if submission.provider_profile is not None or _configuration(raw)[3] == 2:
                self._save_profiles(submission, raw, observation, cancelled)
                published = True
            else:
                old, _ = _defaults(raw)
                credential_raw = None
                if observation[3] == 'file':
                    credential_raw = _read(credentials_path, 'credentials')
                    if credential_raw != observation[2]:
                        raise UserConfigError('save', 'configuration-changed')
                ref = old.credential_ref
                credentials = None
                if submission.credential_action == 'replace':
                    entries = {}
                    if ref is not None:
                        if observation[3] != 'file':
                            credential_raw = _read(credentials_path, 'credentials')
                        previous = _credential(credential_raw, ref)
                        entries[ref] = {'provider': 'deepseek', 'api_key': previous.reveal()}
                    ref = str(uuid4())
                    entries[ref] = {'provider': 'deepseek', 'api_key': submission.credential.reveal()}
                    credentials = json.dumps({'version': 1, 'entries': entries}).encode('utf-8')
                elif observation[3] == 'missing':
                    raise UserConfigError('save', 'missing-credential')
                target = replace(submission.defaults, credential_ref=ref)
                target_raw = _config_bytes(target)
                if credentials is not None:
                    _atomic_replace(credentials_path, credentials, cancelled)
                _atomic_replace(config_path, target_raw, cancelled)
                published = True  # 唯一 defaults_saved 接受点；之后不回滚。
        except UserConfigError as exc:
            error_code = exc.error_code
        except OSError:
            error_code = 'write-failed'
        finally:
            if lock_fd is not None:
                try:
                    if locked:
                        os.lseek(lock_fd, 0, os.SEEK_SET)
                        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    error_code = 'lock-release-failed'
                finally:
                    try:
                        os.close(lock_fd)
                    except OSError:
                        error_code = 'lock-release-failed'
        return LoginResult(published, False, 'not-requested', error_code)
