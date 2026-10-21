"""Chat 兼容子集的本地配置、历史资格及确定性请求映射。"""

from dataclasses import dataclass
import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit

from cicada.core.messages import ToolResultMessage, UserMessage, validate_history_structure
from cicada.core.ports import ModelRequest
from cicada.models import SafeModelIdentity

REQUEST_LIMIT = 2 * 1024 * 1024
_LOCAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def is_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    model_key: str
    model: str
    base_url: str
    num_ctx: int
    max_tokens: int
    stream_usage: bool = True
    output_token_field: str = "max_tokens"
    max_request_bytes: int = REQUEST_LIMIT

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or _LOCAL_ID.fullmatch(x) is None
               for x in (self.provider, self.model_key)):
            raise ValueError("invalid Chat provider or model key")
        if (not isinstance(self.model, str) or not self.model or len(self.model) > 256
                or any(c.isspace() or unicodedata.category(c).startswith("C") for c in self.model)):
            raise ValueError("invalid Chat model ID")
        if not isinstance(self.base_url, str) or len(self.base_url) > 2048:
            raise ValueError("invalid Chat endpoint")
        try:
            endpoint = urlsplit(self.base_url)
            _ = endpoint.port  # 同时核对无效端口；不在错误中回显输入。
        except ValueError:
            raise ValueError("invalid Chat endpoint") from None
        if (endpoint.scheme not in ("http", "https") or not endpoint.hostname
                or endpoint.username is not None or endpoint.password is not None
                or endpoint.query or endpoint.fragment or "?" in self.base_url or "#" in self.base_url
                or (endpoint.scheme == "http" and not is_loopback(endpoint.hostname))
                or any(c.isspace() or unicodedata.category(c).startswith("C") for c in self.base_url)):
            raise ValueError("Chat endpoint must be credential-free HTTPS or loopback HTTP")
        if type(self.stream_usage) is not bool or self.output_token_field not in (
                "max_tokens", "max_completion_tokens"):
            raise ValueError("invalid Chat compatibility settings")
        for name, limit in (("num_ctx", 16_777_216), ("max_tokens", 16_777_216),
                            ("max_request_bytes", REQUEST_LIMIT)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= limit:
                raise ValueError("invalid Chat request limits")
        if self.num_ctx < 2048 or self.max_tokens > self.num_ctx:
            raise ValueError("invalid Chat context/output limits")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def identity(self) -> SafeModelIdentity:
        return SafeModelIdentity(
            self.provider, self.model, self.base_url, False,
            (("num_ctx", self.num_ctx), ("max_tokens", self.max_tokens),
             ("protocol", "openai-chat-completions"), ("model_key", self.model_key),
             ("profile_version", 1), ("stream_usage", self.stream_usage),
             ("output_token_field", self.output_token_field)), self.max_request_bytes,
        )

    def validate_request(self, request: ModelRequest) -> None:
        validate_history_structure(request.messages)
        if request.request_context and not request.tools:
            raise ValueError("Chat requires tools to carry a nonempty request context")


def _schema_copy(value):
    if isinstance(value, dict):
        return {key: _schema_copy(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_schema_copy(item) for item in value]
    return value


def build_request(request: ModelRequest, config: ChatConfig) -> dict:
    config.validate_request(request)
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for message in request.messages:
        if isinstance(message, UserMessage):
            mapped = {"role": "user", "content": message.text}
        elif isinstance(message, ToolResultMessage):
            mapped = {"role": "tool", "content": message.result.content,
                      "tool_call_id": message.result.call_id}
        else:
            mapped = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                mapped["tool_calls"] = [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": call.arguments_json}}
                    for call in message.tool_calls
                ]
        messages.append(mapped)
    if request.request_context:
        messages.append({"role": "user", "content": request.request_context})
    body = {"model": config.model, "messages": messages, "stream": True,
            config.output_token_field: config.max_tokens}
    if config.stream_usage:
        body["stream_options"] = {"include_usage": True}
    if request.tools:
        body["tools"] = [
            {"type": "function", "function": {"name": tool.name,
                "description": tool.description, "parameters": _schema_copy(tool.parameters)}}
            for tool in request.tools
        ]
    return body
