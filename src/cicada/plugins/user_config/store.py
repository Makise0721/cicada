"""用户 TOML/凭据 JSON 读取；只读路径不创建目录或锁文件。"""

import json
import tomllib
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
        self._observation: tuple[str, bytes | None, bytes | None] | None = None

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
            self._observation = revision, raw, credential_raw
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
        """01只冻结签名；02实现受拥有的可靠写入。"""
        raise NotImplementedError('user configuration writing is not implemented')
