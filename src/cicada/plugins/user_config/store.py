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

from cicada.plugins.deepseek.protocol import DeepSeekConfig
from cicada.user_config import (
    LoginResult, LoginSubmission, LoginView, SecretValue, UserConfigError,
    UserConfigSnapshot, UserDefaults,
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


def _atomic_replace(path: Path, raw: bytes, cancelled: threading.Event) -> None:
    """worker 独占临时文件；replace 之前最后检查尚未接受的取消。"""
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


def _defaults(raw: bytes | None) -> tuple[UserDefaults, frozenset[str]]:
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
        DeepSeekConfig(model=defaults.model, think=defaults.think,
                       reasoning_effort=defaults.reasoning_effort, num_ctx=defaults.num_ctx,
                       max_tokens=defaults.max_output_tokens)
    except ValueError:
        raise UserConfigError('config', 'invalid-model-settings', 'defaults') from None
    return defaults, frozenset(values)


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise UserConfigError('credentials', 'duplicate-field')
        result[name] = value
    return result


def _credential(raw: bytes | None, ref: str) -> SecretValue:
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
    selected = None
    for entry_ref, entry in entries.items():
        _ref(entry_ref, 'credentials')
        _object(entry, ('provider', 'api_key'), 'credentials', 'entries')
        if entry.get('provider') != 'deepseek':
            raise UserConfigError('credentials', 'unsupported-value', 'provider')
        try:
            secret = SecretValue(entry.get('api_key'))
        except UserConfigError:
            raise UserConfigError('credentials', 'invalid-key', 'api_key') from None
        if entry_ref == ref:
            selected = secret
    if selected is None:
        raise UserConfigError('credentials', 'missing-ref', 'ref')
    return selected


class FileUserConfigStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        # 一份最近观察供02核对；原始字节只留在Adapter，不派生key hash。
        self._observation: tuple[str, bytes | None, bytes | None, str] | None = None
        self._save_task: asyncio.Task | None = None

    def load(self, environment_key: str | None = None) -> UserConfigSnapshot:
        environment = None
        if environment_key is not None and environment_key.strip():
            try:
                environment = SecretValue(environment_key)
            except UserConfigError:
                raise UserConfigError('environment', 'invalid-key') from None
        for _ in range(3):
            raw = _read(self.root / 'config.toml', 'config')
            defaults, fields = _defaults(raw)
            credential_raw = None
            source = 'environment' if environment is not None else 'missing'
            secret, error = environment, None
            if environment is None and defaults.credential_ref is not None:
                try:
                    credential_raw = _read(self.root / 'credentials.json', 'credentials')
                    secret = _credential(credential_raw, defaults.credential_ref)
                    source = 'file'
                except UserConfigError as exc:
                    error = exc
                if _read(self.root / 'config.toml', 'config') != raw:
                    continue
                if error is not None:
                    raise error
            revision = str(uuid4())
            self._observation = revision, raw, credential_raw, source
            return UserConfigSnapshot(defaults, fields, secret, source, revision)
        raise UserConfigError('config', 'configuration-changing')

    def read_view(self, environment_key: str | None = None) -> LoginView:
        snapshot = self.load(environment_key)
        d = snapshot.defaults
        return LoginView(d.provider, d.model, d.think, d.reasoning_effort, d.num_ctx,
                         d.max_output_tokens, snapshot.credential is not None,
                         snapshot.credential_source, self.root / 'config.toml',
                         '之后新建会话使用这些默认值', ('defaults-only',), snapshot.revision)

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
