"""拥有响应生命周期的 DeepSeek SSE ModelPort；零自动 HTTP 重试。"""

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import httpx

from cicada.budget import BudgetConfig
from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelRequest, StreamEvent
from cicada.costs import CostError, CostLedger
from cicada.models import (
    CacheRequestObservation, ConfigurationRequest, ModelBinding, ModelOption, SafeModelIdentity,
    SUMMARY_OUTPUT_TOKENS,
)
from cicada.plugins.deepseek.protocol import (
    BASE_URL, MODEL_IDS, REASONING_EFFORTS, REQUEST_LIMIT, DeepSeekConfig, build_request, request_body_bytes,
)
from cicada.plugins.deepseek.sse import Completion
from cicada.plugins.chat_stream import (
    close_response as _close_response, wait_operation as _wait_operation, stream_chat,
)

DEFAULT_TIMEOUT = httpx.Timeout(connect=10, read=60, write=30, pool=10)
CALL_DEADLINE_S = 300.0


@dataclass(frozen=True)
class _RequestShape:
    """上次实际发送的不可逆形状摘要 (契约 C7); 不保存正文副本或历史。

    identity 变化在本实现中表现为新普通路由实例的首调用, 不推测服务端身份。
    """

    identity: SafeModelIdentity
    system: str | None
    tools: str | None
    messages: tuple[str, ...]  # 核心映射消息摘要, 不含 leading system 与派生尾部
    tail: str | None


def _digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request_shape(request: ModelRequest, identity: SafeModelIdentity, body: dict) -> _RequestShape:
    """最终 provider 映射的形状摘要; 只覆盖实际发送内容 (契约 C7)."""
    messages = body["messages"]
    start = 1 if messages and messages[0].get("role") == "system" else 0
    end = len(messages) - 1 if request.request_context else len(messages)
    return _RequestShape(
        identity=identity,
        system=_digest(messages[0]) if start else None,
        tools=_digest(body["tools"]) if "tools" in body else None,
        messages=tuple(_digest(message) for message in messages[start:end]),
        tail=_digest(messages[-1]) if request.request_context else None,
    )


class DeepSeekModel:
    def __init__(self, client: httpx.AsyncClient, config: DeepSeekConfig, api_key: str, *,
                 cost_control: CostLedger | None = None,
                 identity_override: SafeModelIdentity | None = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip() or any(c in api_key for c in "\r\n"):
            raise ValueError("DEEPSEEK_API_KEY is missing or invalid")
        self.client = client
        self.config = config
        self._api_key = api_key
        self._cost_control = cost_control
        if identity_override is not None:
            options = dict(identity_override.options)
            if (identity_override.model != config.model or identity_override.base_url != BASE_URL
                    or identity_override.think != config.think
                    or identity_override.max_request_bytes != config.max_request_bytes
                    or options.get("num_ctx") != config.num_ctx
                    or options.get("max_tokens") != config.max_tokens
                    or options.get("reasoning_effort") != config.reasoning_effort
                    or options.get("protocol", "deepseek-chat") != "deepseek-chat"):
                raise ValueError("DeepSeek identity does not match its effective configuration")
        self._identity_override = identity_override
        # 缓存 03 (契约 C7): 每实例一份比较基线与最新观察; 换模型/强度/配置
        # 准备出的新实例从空开始, 不持久、不跨进程恢复。
        self._observation: CacheRequestObservation | None = None
        self._send_sequence = 0
        self._shape_baseline: _RequestShape | None = None
        from .tokenizer import FlashTokenCounter, ProTokenCounter
        counter_type = FlashTokenCounter if config.model == "deepseek-flash" else ProTokenCounter
        self._token_counter = counter_type(config)

    @property
    def identity(self) -> SafeModelIdentity:
        return self._identity_override or self.config.identity

    def _identity_for(self, config: DeepSeekConfig) -> SafeModelIdentity | None:
        if self._identity_override is None:
            return None
        options = dict(self._identity_override.options)
        options.update(num_ctx=config.num_ctx, max_tokens=config.max_tokens,
                       reasoning_effort=config.reasoning_effort)
        if config.model != self.config.model and "model_key" in options:
            options["model_key"] = config.model
        return replace(self._identity_override, model=config.model, think=config.think,
                       options=tuple(options.items()), max_request_bytes=config.max_request_bytes)

    @property
    def cache_observation(self) -> CacheRequestObservation | None:
        """最近一次实际发送相对前次的本地形状观察; 未观察或诊断失败为 None."""
        return self._observation

    @property
    def options(self) -> tuple[ModelOption, ...]:
        return tuple(ModelOption(model, model) for model in MODEL_IDS)

    def set_cost_control(self, ledger: CostLedger) -> None:
        """装配时绑定本进程唯一账本，后继配置不能换额度。"""
        if self._cost_control is not None and self._cost_control is not ledger:
            raise CostError('model already has a different cost allowance')
        self._cost_control = ledger

    @property
    def initial_binding(self) -> ModelBinding:
        summary_config = replace(self.config, think=False, max_tokens=min(SUMMARY_OUTPUT_TOKENS, self.config.max_tokens))
        summary = DeepSeekModel(self.client, summary_config, self._api_key, cost_control=self._cost_control,
                                identity_override=self._identity_for(summary_config))
        return ModelBinding(self, summary, self.identity, summary.identity,
                            BudgetConfig(self.config.num_ctx, self.config.max_tokens),
                            BudgetConfig(summary_config.num_ctx, summary_config.max_tokens),
                            self.config.validate_request, self._token_counter, summary._token_counter)

    def prepare_model(self, option_id: str) -> ModelBinding:
        config = replace(self.config, model=option_id)
        return DeepSeekModel(self.client, config, self._api_key, cost_control=self._cost_control,
                             identity_override=self._identity_for(config)).initial_binding

    @property
    def reasoning_efforts(self) -> tuple[str, ...]:
        return REASONING_EFFORTS

    def prepare_reasoning_effort(self, effort: str) -> ModelBinding:
        """思考强度准备 (03, C5): 纯本地, 复用 client/凭据与普通参数.

        普通模型显式 think=true 并设置档位; initial_binding 的摘要通道保持
        think=false (摘要思考禁用)。不合法档位按 ValueError 拒绝, 零网络。
        """
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"unknown reasoning effort: {effort!r}")
        config = replace(self.config, think=True, reasoning_effort=effort)
        return DeepSeekModel(self.client, config, self._api_key, cost_control=self._cost_control,
                             identity_override=self._identity_for(config)).initial_binding

    def prepare_configuration(self, request: ConfigurationRequest) -> ModelBinding:
        """登录应用准备: 复用既有 client, 零网络/文件 I/O, 旧对象保持不可变。"""
        identity = request.identity
        if identity.provider != self.identity.provider or type(identity.think) is not bool:
            raise ValueError("DeepSeek configuration requires a deepseek identity with explicit think")
        if identity.base_url not in (None, BASE_URL):
            raise ValueError("DeepSeek configuration requires the official endpoint")
        options = dict(identity.options)
        config = DeepSeekConfig(
            model=identity.model, think=identity.think,
            reasoning_effort=options.get("reasoning_effort", "high"),
            num_ctx=options.get("num_ctx", 131072),
            max_tokens=options.get("max_tokens", 32768),
            max_request_bytes=identity.max_request_bytes or REQUEST_LIMIT,
        )
        key = self._api_key if request.credential_action == "keep" else request.credential.reveal()
        options.update(dict(config.identity.options))
        return DeepSeekModel(self.client, config, key, cost_control=self._cost_control,
                             identity_override=replace(identity, base_url=BASE_URL,
                                 options=tuple(options.items()),
                                 max_request_bytes=config.max_request_bytes)).initial_binding

    def _record_observation(self, request: ModelRequest, body: dict) -> None:
        """缓存 03 (契约 C7): client.send 实际启动的同步接缝调用一次。

        sequence 按本实例实际发送尝试递增; 本地资格拒绝、字节守卫与取消前零
        send 不会到达本方法, 发送后失败仍保留观察。诊断异常不改变发送或终结:
        显示不可得并清空比较基线, 下次可观察发送为 first_request, 不假称前次。
        """
        self._send_sequence += 1
        try:
            shape = _request_shape(request, self.identity, body)
            baseline = self._shape_baseline
            if baseline is None or baseline.identity != shape.identity:
                observation = CacheRequestObservation(
                    self._send_sequence, ("first_request",), None)
            else:
                changes: list[str] = []
                if shape.system != baseline.system:
                    changes.append("system_changed")
                if shape.tools != baseline.tools:
                    changes.append("tools_changed")
                if shape.messages != baseline.messages:
                    prefix = len(baseline.messages)
                    if (len(shape.messages) > prefix
                            and shape.messages[:prefix] == baseline.messages):
                        changes.append("messages_extended")
                    else:
                        changes.append("messages_replaced")
                if shape.tail != baseline.tail:
                    changes.append("request_context_changed")
                if not changes:
                    changes = ["unchanged"]
                preserved = 0
                for sent, previous in zip(shape.messages, baseline.messages):
                    if sent != previous:
                        break
                    preserved += 1
                observation = CacheRequestObservation(
                    self._send_sequence, tuple(changes), preserved)
            self._shape_baseline = shape
            self._observation = observation
        except Exception:
            self._shape_baseline = None
            self._observation = None

    async def stream(self, request: ModelRequest, cancel: CancelToken) -> AsyncIterator[StreamEvent]:
        async for event in stream_chat(
            request, cancel, client=self.client, endpoint=BASE_URL + "/chat/completions",
            api_key=self._api_key, identity=self.identity,
            build_body=lambda value: build_request(value, self.config),
            completion_factory=lambda: Completion(think=self.config.think),
            max_request_bytes=self.config.max_request_bytes, max_tokens=self.config.max_tokens,
            timeout=DEFAULT_TIMEOUT, deadline_seconds=CALL_DEADLINE_S, label="deepseek",
            cost_control=self._cost_control, before_send=self._record_observation,
        ):
            yield event
