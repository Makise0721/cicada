"""标准 Chat 文本/工具流；网络生命周期复用同一受拥有实现。"""

from dataclasses import replace
from urllib.parse import urlsplit

import httpx

from cicada.budget import BudgetConfig
from cicada.core.cancel import CancelToken
from cicada.core.ports import ModelMetrics, ModelRequest, ReasoningDelta
from cicada.costs import CostError, CostLedger
from cicada.models import ConfigurationRequest, ModelBinding, SUMMARY_OUTPUT_TOKENS
from cicada.plugins.chat_stream import stream_chat
from cicada.plugins.deepseek.sse import Completion, ProtocolError
from cicada.plugins.openai_chat.protocol import ChatConfig, build_request, is_loopback
from cicada.user_config import SecretValue

DEFAULT_TIMEOUT = httpx.Timeout(connect=10, read=60, write=30, pool=10)
CALL_DEADLINE_S = 300.0


class ChatCompletion(Completion):
    def __init__(self):
        super().__init__(think=False)

    def consume(self, frame: str):
        events = list(super().consume(frame))
        if any(isinstance(event, ReasoningDelta) for event in events):
            raise ProtocolError("private reasoning is not supported by the Chat profile")
        if self.metrics is not None:
            self.metrics = ModelMetrics(self.metrics.input_tokens, self.metrics.output_tokens)
        yield from events

    def terminal(self):
        calls, reason, error = super().terminal()
        return calls, reason, ("chat response has an unsuccessful finish" if error else None)


class ChatModel:
    def __init__(self, client: httpx.AsyncClient, config: ChatConfig, api_key: str | None, *,
                 cost_control: CostLedger | None = None):
        self.client = client
        self.config = config
        if api_key is None:
            if not is_loopback(urlsplit(config.base_url).hostname):
                raise ValueError("unauthenticated Chat requires a loopback endpoint")
            self._api_key = None
        else:
            self._api_key = SecretValue(api_key).reveal()
        self._cost_control = cost_control

    def set_cost_control(self, ledger: CostLedger) -> None:
        if self._cost_control is not None and self._cost_control is not ledger:
            raise CostError("model already has a different cost allowance")
        self._cost_control = ledger

    @property
    def initial_binding(self) -> ModelBinding:
        summary_config = replace(self.config, max_tokens=min(SUMMARY_OUTPUT_TOKENS, self.config.max_tokens))
        summary = ChatModel(self.client, summary_config, self._api_key,
                            cost_control=self._cost_control)
        return ModelBinding(
            self, summary, self.config.identity, summary_config.identity,
            BudgetConfig(self.config.num_ctx, self.config.max_tokens),
            BudgetConfig(summary_config.num_ctx, summary_config.max_tokens),
            self.config.validate_request,
        )

    def prepare_configuration(self, request: ConfigurationRequest) -> ModelBinding:
        identity = request.identity
        options = dict(identity.options)
        if (identity.provider != self.config.provider or identity.think not in (False, None)
                or options.get("protocol") != "openai-chat-completions"):
            raise ValueError("Chat configuration requires the same provider and supported protocol")
        config = ChatConfig(
            identity.provider, options["model_key"], identity.model, identity.base_url,
            options["num_ctx"], options["max_tokens"], options.get("stream_usage", True),
            options.get("output_token_field", "max_tokens"),
            identity.max_request_bytes or self.config.max_request_bytes,
        )
        key = (self._api_key if request.credential_action == "keep"
               else request.credential.reveal())
        return ChatModel(self.client, config, key, cost_control=self._cost_control).initial_binding

    async def stream(self, request: ModelRequest, cancel: CancelToken):
        async for event in stream_chat(
            request, cancel, client=self.client, endpoint=self.config.endpoint,
            api_key=self._api_key, identity=self.config.identity,
            build_body=lambda value: build_request(value, self.config), completion_factory=ChatCompletion,
            max_request_bytes=self.config.max_request_bytes, max_tokens=self.config.max_tokens,
            timeout=DEFAULT_TIMEOUT, deadline_seconds=CALL_DEADLINE_S, label="chat",
            cost_control=self._cost_control,
        ):
            yield event
