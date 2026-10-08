"""中立的用户默认配置契约；秘密不属于历史或持久会话 DTO。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
import unicodedata

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


@dataclass(frozen=True)
class UserConfigSnapshot:
    """启动私有读取结果；fields 区分用户明确填写与内置缺省值。"""

    defaults: UserDefaults
    fields: frozenset[str]
    credential: SecretValue | None = field(repr=False)
    credential_source: CredentialSource
    revision: str = field(repr=False)


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


@dataclass(frozen=True)
class LoginSubmission:
    defaults: UserDefaults
    credential_action: Literal['keep', 'replace']
    revision: str = field(repr=False)
    target: LoginTarget = 'defaults-only'
    credential: SecretValue | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.target not in ('defaults-only', 'save-and-apply'):
            raise ValueError('invalid login target')
        if not ((self.credential_action == 'keep' and self.credential is None) or
                (self.credential_action == 'replace' and isinstance(self.credential, SecretValue))):
            raise ValueError('invalid login credential action')


@dataclass(frozen=True)
class LoginResult:
    defaults_saved: bool
    session_applied: bool
    application_outcome: Literal['applied', 'unchanged', 'not-requested', 'blocked', 'failed', 'closing']
    error_code: str | None = None


class UserConfigStore(Protocol):
    """文件 Adapter 负责观察版本，save 的可靠写入由后继票实现。"""

    def load(self, environment_key: str | None = None) -> UserConfigSnapshot: ...

    def read_view(self, environment_key: str | None = None) -> LoginView: ...

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
