"""中立的用户默认配置契约；秘密不属于历史或持久会话 DTO。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
import unicodedata
import ipaddress
import re
from urllib.parse import urlsplit

ConfigSource = Literal['cli', 'user', 'archive', 'default']
CredentialSource = Literal['environment', 'file', 'login', 'missing']
LoginTarget = Literal['defaults-only', 'save-and-apply']


class UserConfigError(ValueError):
    """固定分类错误，不包含文件输入、秘密、引用或底层异常文本。"""

    def __init__(self, file_kind: str, error_code: str, field_name: str | None = None):
        self.file_kind = file_kind
        self.error_code = error_code
        self.field_name = field_name
        super().__init__(f'user configuration {file_kind}: {error_code}' +
                         (f' ({field_name})' if field_name else ''))


class SecretValue:
    """短期秘密引用；无通用序列化入口，只有显式 reveal 可访问值。"""

    __slots__ = ('_value',)

    def __init__(self, value: str):
        if not isinstance(value, str):
            raise UserConfigError('credential', 'invalid-key')
        value = value.strip()
        if (not value or len(value.encode('utf-8', errors='replace')) > 4096
                or any(c.isspace() or unicodedata.category(c).startswith('C') for c in value)):
            raise UserConfigError('credential', 'invalid-key')
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return 'SecretValue(<hidden>)'

    __str__ = __repr__


@dataclass(frozen=True)
class UserDefaults:
    provider: str = 'deepseek'
    model: str = 'deepseek-flash'
    think: bool = True
    reasoning_effort: str = 'high'
    num_ctx: int = 131072
    max_output_tokens: int = 32768
    credential_ref: str | None = field(default=None, repr=False)


def _profile_text(value: object, *, limit: int = 256, empty: bool = False) -> bool:
    return (isinstance(value, str) and (empty or bool(value)) and len(value) <= limit
            and value == value.strip()
            and not any(unicodedata.category(c).startswith('C') for c in value))


def _profile_key(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', value) is not None


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_id: str
    name: str = ''
    context_window: int = 131072
    max_output_tokens: int = 8192
    tools: bool = True
    think: bool = False
    reasoning_effort: str = 'high'
    stream_usage: bool = True
    output_token_field: str = 'max_tokens'

    def __post_init__(self) -> None:
        if (not _profile_key(self.key) or not _profile_text(self.model_id)
                or any(c.isspace() for c in self.model_id)
                or not _profile_text(self.name, limit=128, empty=True)
                or type(self.context_window) is not int or not 2048 <= self.context_window <= 16777216
                or type(self.max_output_tokens) is not int or not 0 < self.max_output_tokens <= self.context_window
                or any(type(value) is not bool for value in (self.tools, self.think, self.stream_usage))
                or self.reasoning_effort not in ('low', 'high', 'max')
                or self.output_token_field not in ('max_tokens', 'max_completion_tokens')):
            raise UserConfigError('config', 'invalid-model-settings')


@dataclass(frozen=True)
class ProviderProfile:
    id: str
    name: str
    protocol: str
    base_url: str
    models: tuple[ModelProfile, ...]
    default_model: str
    auth: str = 'bearer'
    api_key_env: str | None = None
    credential_ref: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (not _profile_key(self.id) or not _profile_text(self.name, limit=128)
                or self.protocol not in ('deepseek-chat', 'openai-chat-completions')
                or not isinstance(self.models, tuple) or not 1 <= len(self.models) <= 128
                or any(not isinstance(model, ModelProfile) for model in self.models)
                or len({model.key for model in self.models}) != len(self.models)
                or self.default_model not in {model.key for model in self.models}
                or self.auth not in ('bearer', 'none')
                or (self.api_key_env is not None and (not isinstance(self.api_key_env, str)
                    or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', self.api_key_env) is None))
                or not _profile_text(self.base_url, limit=2048)
                or any(c.isspace() for c in self.base_url)):
            raise UserConfigError('config', 'invalid-provider-settings')
        try:
            endpoint = urlsplit(self.base_url)
            hostname = endpoint.hostname
            loopback = hostname == 'localhost'
            if hostname and not loopback:
                try:
                    loopback = ipaddress.ip_address(hostname).is_loopback
                except ValueError:
                    pass
            if (endpoint.scheme not in ('https', 'http') or not hostname
                    or endpoint.username is not None or endpoint.password is not None
                    or endpoint.query or endpoint.fragment
                    or (endpoint.scheme == 'http' and not loopback)
                    or (self.auth == 'none' and not loopback)):
                raise ValueError
            endpoint.port
        except ValueError:
            raise UserConfigError('config', 'invalid-provider-settings') from None
        if self.protocol == 'deepseek-chat':
            if (self.base_url.rstrip('/') != 'https://api.deepseek.com' or self.auth != 'bearer'
                    or any(model.model_id not in ('deepseek-flash', 'deepseek-v4-pro')
                           or model.context_window > 1048576 or model.max_output_tokens > 393216
                           for model in self.models)):
                raise UserConfigError('config', 'invalid-provider-settings')
        elif any(model.think for model in self.models):
            raise UserConfigError('config', 'unsupported-thinking')
        if self.auth == 'none' and (self.api_key_env is not None or self.credential_ref is not None):
            raise UserConfigError('config', 'invalid-provider-settings')


@dataclass(frozen=True)
class UserConfigSnapshot:
    """启动私有读取结果；fields 区分用户明确填写与内置缺省值。"""

    defaults: UserDefaults
    fields: frozenset[str]
    credential: SecretValue | None = field(repr=False)
    credential_source: CredentialSource
    revision: str = field(repr=False)
    providers: tuple[ProviderProfile, ...] = ()


@dataclass(frozen=True)
class ResolvedLaunchConfig:
    defaults: UserDefaults
    sources: tuple[tuple[str, ConfigSource], ...]
    credential: SecretValue | None = field(repr=False)
    credential_source: CredentialSource


@dataclass(frozen=True)
class LoginView:
    provider: str
    model: str
    think: bool
    reasoning_effort: str
    num_ctx: int
    max_output_tokens: int
    credential_present: bool
    credential_source: CredentialSource
    save_path: Path
    effect: str
    actions: tuple[LoginTarget, ...]
    revision: str = field(repr=False)
    credentials_only: bool = False
    providers: tuple[ProviderProfile, ...] = ()


@dataclass(frozen=True)
class LoginSubmission:
    defaults: UserDefaults
    credential_action: Literal['keep', 'replace']
    revision: str = field(repr=False)
    target: LoginTarget = 'defaults-only'
    credential: SecretValue | None = field(default=None, repr=False)
    provider_profile: ProviderProfile | None = None
    set_default: bool = True

    def __post_init__(self) -> None:
        if self.target not in ('defaults-only', 'save-and-apply'):
            raise ValueError('invalid login target')
        if not ((self.credential_action == 'keep' and self.credential is None) or
                (self.credential_action == 'replace' and isinstance(self.credential, SecretValue))):
            raise ValueError('invalid login credential action')
        if self.provider_profile is not None and not isinstance(self.provider_profile, ProviderProfile):
            raise ValueError('invalid provider profile')
        if type(self.set_default) is not bool:
            raise ValueError('invalid default selection')


@dataclass(frozen=True)
class LoginResult:
    defaults_saved: bool
    session_applied: bool
    application_outcome: Literal['applied', 'unchanged', 'not-requested', 'blocked', 'failed', 'closing']
    error_code: str | None = None


class UserConfigStore(Protocol):
    """文件 Adapter 负责观察版本及受拥有的可靠保存。"""

    def load(self, environment_key: str | None = None, *, provider_id: str | None = None,
             model_key: str | None = None) -> UserConfigSnapshot: ...

    def read_view(self, environment_key: str | None = None, *,
                  provider_id: str | None = None) -> LoginView: ...

    def read_profiles(self) -> tuple[ProviderProfile, ...]: ...

    async def save(self, submission: LoginSubmission) -> LoginResult: ...


def resolve_launch_config(
    snapshot: UserConfigSnapshot, *, explicit: dict[str, object],
    archive: UserDefaults | None = None,
) -> ResolvedLaunchConfig:
    """普通恢复只采用存档；新建/采用当前配置逐字段解析。"""
    values = {}
    sources = []
    for name in ('provider', 'model', 'think', 'reasoning_effort', 'num_ctx', 'max_output_tokens'):
        if explicit.get(name) is not None:
            values[name], source = explicit[name], 'cli'
        elif archive is not None:
            values[name], source = getattr(archive, name), 'archive'
        else:
            values[name] = getattr(snapshot.defaults, name)
            source = 'user' if name in snapshot.fields else 'default'
        sources.append((name, source))
    return ResolvedLaunchConfig(UserDefaults(**values), tuple(sources),
                                snapshot.credential, snapshot.credential_source)
