"""中立模型身份; 不含凭据, 不依赖插件或持久化实现."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Callable
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit
import re

from cicada.budget import BudgetConfig, RequestTokenCounter
from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelPort, ModelRequest, StreamEvent
from cicada.user_config import SecretValue


@dataclass(frozen=True)
class SafeModelIdentity:
    """调用配置的不可变身份; options 仅允许已知的非秘密 scalar pairs."""

    provider: str
    model: str
    base_url: str | None
    think: bool | None
    options: tuple[tuple[str, str | int | bool | None], ...] = ()
    max_request_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"SafeModelIdentity.{name} must be a nonempty str")
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise ValueError("SafeModelIdentity.base_url must be str or None")
        if self.base_url is not None:
            endpoint = urlsplit(self.base_url)
            if (endpoint.scheme not in ("http", "https") or not endpoint.hostname
                    or endpoint.username is not None or endpoint.password is not None
                    or endpoint.query or endpoint.fragment):
                raise ValueError("SafeModelIdentity.base_url must be a credential-free endpoint")
        if self.think is not None and not isinstance(self.think, bool):
            raise ValueError("SafeModelIdentity.think must be bool or None")
        if self.max_request_bytes is not None and (
            type(self.max_request_bytes) is not int or self.max_request_bytes <= 0
        ):
            raise ValueError("SafeModelIdentity.max_request_bytes must be positive int or None")
        if not isinstance(self.options, tuple):
            raise ValueError("SafeModelIdentity.options must be immutable pairs")
        seen: set[str] = set()
        for pair in self.options:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("SafeModelIdentity.options must be immutable pairs")
            key, value = pair
            if key not in (
                "num_ctx", "num_predict", "max_tokens", "reasoning_effort", "protocol",
                "model_key", "profile_version", "stream_usage", "output_token_field",
            ) or key in seen:
                raise ValueError("SafeModelIdentity.options has unknown or duplicate key")
            seen.add(key)
            if key == "reasoning_effort":
                if value not in ("low", "high", "max"):
                    raise ValueError("SafeModelIdentity.reasoning_effort is invalid")
            elif key == "protocol":
                if value not in ("deepseek-chat", "openai-chat-completions"):
                    raise ValueError("SafeModelIdentity.protocol is invalid")
            elif key == "model_key":
                if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
                    raise ValueError("SafeModelIdentity.model_key is invalid")
            elif key == "profile_version":
                if type(value) is not int or value != 1:
                    raise ValueError("SafeModelIdentity.profile_version is invalid")
            elif key == "stream_usage":
                if type(value) is not bool:
                    raise ValueError("SafeModelIdentity.stream_usage must be bool")
            elif key == "output_token_field":
                if value not in ("max_tokens", "max_completion_tokens"):
                    raise ValueError("SafeModelIdentity.output_token_field is invalid")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"SafeModelIdentity.{key} must be positive int")


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    available: bool = True
    reason: str | None = None
    is_default: bool = False


@dataclass(frozen=True)
class ModelBinding:
    model: ModelPort
    summary_model: ModelPort
    identity: SafeModelIdentity
    summary_identity: SafeModelIdentity
    budget: BudgetConfig
    summary_budget: BudgetConfig
    validate_request: Callable[[ModelRequest], None]
    token_counter: RequestTokenCounter | None = None
    summary_token_counter: RequestTokenCounter | None = None
    option_id: str | None = None
    label: str | None = None
    catalog_revision: str | None = None


@dataclass(frozen=True)
class ReasoningEffortState:
    """当前会话思考强度的已生效状态 (C5); current=None 表示思考关闭."""

    levels: tuple[str, ...]          # 支持档位; 空元组 = 无该能力
    current: str | None              # 实际生效档位; None = 关闭
    enabled: bool                    # False = provider 不支持强度切换


@runtime_checkable
class ReasoningEffortPort(Protocol):
    """可选思考强度能力 (03, C5): 从最新 binding 的普通模型取得, 不保留过期 provider.

    prepare 纯本地: 复用模型/端点/客户端/凭据与其他普通参数, 不发网络、
    不读写用户配置; 摘要模型始终禁用思考。
    """

    @property
    def reasoning_efforts(self) -> tuple[str, ...]: ...

    def prepare_reasoning_effort(self, effort: str) -> ModelBinding: ...


@runtime_checkable
class ModelSelectionPort(Protocol):
    @property
    def initial_binding(self) -> ModelBinding: ...

    @property
    def options(self) -> tuple[ModelOption, ...]: ...

    def prepare_model(self, option_id: str) -> ModelBinding: ...


@dataclass(frozen=True)
class CacheRequestObservation:
    """最近一次实际发送相对前次的本地形状观察 (缓存 03, 契约 C7).

    sequence 按本实例实际 HTTP 发送尝试递增 (本地拒绝与取消前零 send 不计);
    changes 是冻结变化码; preserved_messages 为完全相同的前导映射消息数
    (不含 leading system 与派生尾部), 首次可观察发送为 None。
    不携带正文、hash、凭据或回执。
    """

    sequence: int
    changes: tuple[str, ...]
    preserved_messages: int | None


@runtime_checkable
class CacheDiagnosticsPort(Protocol):
    """可选只读发送观察能力 (缓存 03, 契约 C7); 不扩大 ModelPort 必备方法.

    变化码冻结为 first_request、system_changed、tools_changed、
    messages_extended、messages_replaced、request_context_changed、unchanged;
    观察由 provider 在实际发送启动的同步接缝更新, 不持久、不跨进程恢复。
    """

    @property
    def cache_observation(self) -> CacheRequestObservation | None: ...


@dataclass(frozen=True)
class ConfigurationRequest:
    """登录应用的准备请求: 目标安全身份 + 短期凭据动作; 秘密不进入身份或历史."""

    identity: SafeModelIdentity
    credential_action: Literal["keep", "replace"]
    credential: SecretValue | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SafeModelIdentity):
            raise ValueError("ConfigurationRequest.identity must be a SafeModelIdentity")
        if not ((self.credential_action == "keep" and self.credential is None) or
                (self.credential_action == "replace" and isinstance(self.credential, SecretValue))):
            raise ValueError("invalid configuration credential action")


@runtime_checkable
class ModelConfigurationPort(Protocol):
    """独立可选配置能力; 准备纯本地, 复用既有 client, 不改旧对象, 不发网络."""

    def prepare_configuration(self, request: ConfigurationRequest) -> ModelBinding: ...


class ModelRoute:
    """应用拥有的稳定出口; 每次开始流时捕获绑定, 发布仅替换引用."""

    def __init__(self, binding: ModelBinding, *, summary: bool = False) -> None:
        self.binding = binding
        self.summary = summary

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        binding = self.binding
        model = binding.summary_model if self.summary else binding.model
        async for event in model.stream(request, cancel):
            yield event
