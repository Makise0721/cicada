"""包内固定 Flash/Pro 资源; 构造/配置准备不读词表、不联网."""

import hashlib
import json
from importlib import resources

from cicada.budget import LocalTokenCount
from cicada.core.ports import ModelRequest
from ..protocol import DeepSeekConfig, build_request, request_body_bytes


# manifest 固定了词表/参考模板/adapter/依赖及许可证身份。
MANIFEST_SHA256 = "75bc6e53c9f8666f36c541fb57bd9a2d94b53355b8e0fe008e2b7782edbba132"
PROFILE = "deepseek-flash-v4.1:manifest-sha256:" + MANIFEST_SHA256
PRO_MANIFEST_SHA256 = "e4a06ecba470788ec377407b04f18179ad743445fd552384e6b48e56c512ded8"
PRO_PROFILE = "deepseek-pro-v4-0813:manifest-sha256:" + PRO_MANIFEST_SHA256


def render_body(body: dict) -> str:
    """只在私有发送映射上承载 tools; 整串含 BOS 与生成头."""
    if body["model"] == "deepseek-flash":
        from ._flash_template import _encode_messages_text as encode
    elif body["model"] == "deepseek-v4-pro":
        from ._pro_template import encode_messages as encode
    else:
        raise NotImplementedError("unsupported profile")

    messages = body["messages"]
    if any(not isinstance(m.get("content"), (str, type(None)))
           or not isinstance(m.get("reasoning_content", ""), str) for m in messages):
        raise NotImplementedError("text-only input")
    if body.get("tools"):
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = body["tools"]
    return encode(
        messages, thinking_mode="thinking" if body["thinking"]["type"] == "enabled" else "chat",
        reasoning_effort=body.get("reasoning_effort"),
    )


class _TokenCounter:
    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._load_reason: str | None = None
        self._memo: tuple[bytes, LocalTokenCount] | None = None

    def _load(self) -> str | None:
        if self._tokenizer is not None or self._load_reason is not None:
            return self._load_reason
        try:
            import tokenizers
        except ImportError:
            self._load_reason = "dependency-unavailable"
            return self._load_reason
        if tokenizers.__version__ != "0.23.3":
            self._load_reason = "dependency-unavailable"
            return self._load_reason
        try:
            root = resources.files(__package__)
            manifest_bytes = root.joinpath(self.asset_dir + "/manifest.json").read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != self.manifest_sha256:
                raise ValueError("manifest identity")
            manifest = json.loads(manifest_bytes)
            for name, identity in manifest["assets"].items():
                data = root.joinpath(self.asset_dir + "/" + name).read_bytes()
                if len(data) != identity["size"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
                    raise ValueError("asset identity")
                if name == "tokenizer.json":
                    tokenizer_bytes = data
            adapter = root.joinpath(self.adapter).read_bytes()
            if hashlib.sha256(adapter).hexdigest() != manifest["adapter_sha256"]:
                raise ValueError("adapter identity")
            tokenizer = tokenizers.Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizer = tokenizer
        except FileNotFoundError:
            self._load_reason = "asset-missing"
        except Exception:
            self._load_reason = "asset-invalid"
        return self._load_reason

    def count_request(self, request: ModelRequest) -> LocalTokenCount:
        # 资格错误必须裸 ValueError 交给应用拒绝, 不作为 unavailable 回退。
        body = build_request(request, self.config)
        if self.config.model != self.alias:
            return LocalTokenCount(None, None, "unsupported-profile")
        try:
            wire = request_body_bytes(body)
        except (TypeError, ValueError, UnicodeError):
            return LocalTokenCount(None, self.profile, "unsupported-input")
        if len(wire) > self.config.max_request_bytes:
            return LocalTokenCount(None, self.profile, "request-too-large")
        # 完整最终 body (含配置) + 固定 profile; 只存一个摘要及结果。
        key = hashlib.sha256(self.profile.encode() + wire).digest()
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1]
        reason = self._load()
        if reason is not None:
            result = LocalTokenCount(None, self.profile, reason)
        else:
            try:
                text = render_body(body)
            except (NotImplementedError, AssertionError, TypeError, ValueError):
                result = LocalTokenCount(None, self.profile, "unsupported-input")
            except Exception:
                result = LocalTokenCount(None, self.profile, "count-failed")
            else:
                try:
                    tokens = len(self._tokenizer.encode(text, add_special_tokens=False).ids)
                    result = LocalTokenCount(tokens, self.profile)
                except Exception:
                    result = LocalTokenCount(None, self.profile, "count-failed")
        self._memo = key, result
        return result


class FlashTokenCounter(_TokenCounter):
    alias = "deepseek-flash"
    asset_dir = "flash"
    adapter = "_flash_template.py"
    manifest_sha256 = MANIFEST_SHA256
    profile = PROFILE


class ProTokenCounter(_TokenCounter):
    alias = "deepseek-v4-pro"
    asset_dir = "pro"
    adapter = "_pro_template.py"
    manifest_sha256 = PRO_MANIFEST_SHA256
    profile = PRO_PROFILE
