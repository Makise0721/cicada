"""中立模型身份; 不含凭据, 不依赖插件或持久化实现."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


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
            if key not in ("num_ctx", "num_predict", "max_tokens", "reasoning_effort") or key in seen:
                raise ValueError("SafeModelIdentity.options has unknown or duplicate key")
            seen.add(key)
            if key == "reasoning_effort":
                if value not in ("low", "high", "max"):
                    raise ValueError("SafeModelIdentity.reasoning_effort is invalid")
            elif type(value) is not int or value <= 0:
                raise ValueError(f"SafeModelIdentity.{key} must be positive int")
