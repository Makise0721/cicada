"""OpenCode Go 的固定兼容配置与 DeepSeek 思考协议。"""

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
import re
from typing import Callable
from uuid import uuid4

from cicada.budget import BudgetConfig
from cicada.models import ModelBinding, SUMMARY_OUTPUT_TOKENS
from cicada.core.ports import ModelMetrics
from cicada.plugins.chat_stream import stream_chat
from cicada.plugins.deepseek.protocol import DeepSeekConfig, REASONING_EFFORTS, build_request as build_thinking_request
from cicada.plugins.deepseek.sse import Completion
from cicada.plugins.openai_chat.model import ChatCompletion, ChatModel, DEFAULT_TIMEOUT, CALL_DEADLINE_S
from cicada.plugins.openai_chat.protocol import ChatConfig, build_request
from cicada.provider_presets import OPENCODE_GO_MODEL, OPENCODE_GO_CONTEXT_WINDOW, is_opencode_go_endpoint


@dataclass(frozen=True)
class GoConfig(ChatConfig):
    think: bool = False
    reasoning_effort: str = 'high'

    def __post_init__(self):
        super().__post_init__()
        # 旧Kimi配置仍可恢复；新建内置连接仅使用新的默认模型。
        context_limit = {OPENCODE_GO_MODEL: OPENCODE_GO_CONTEXT_WINDOW, 'kimi-k2.6': 262144}.get(self.model)
        if (not is_opencode_go_endpoint(self.base_url) or context_limit is None
                or self.num_ctx > context_limit or self.max_tokens > 32768
                or self.output_token_field != 'max_tokens'):
            raise ValueError('unsupported OpenCode Go model or limits')
        if (type(self.think) is not bool or self.reasoning_effort not in REASONING_EFFORTS
                or (self.think and self.model != OPENCODE_GO_MODEL)):
            raise ValueError('unsupported OpenCode Go thinking configuration')

    @property
    def thinking_config(self):
        return DeepSeekConfig(think=self.think, reasoning_effort=self.reasoning_effort,
            num_ctx=self.num_ctx, max_tokens=self.max_tokens, max_request_bytes=self.max_request_bytes)

    def validate_request(self, request):
        if self.think:
            self.thinking_config.validate_request(request)
        else:
            super().validate_request(request)

    @property
    def identity(self):
        identity = super().identity
        options = identity.options + (('compatibility_profile', 'opencode-go-v1'),)
        if self.think:
            options += (('reasoning_effort', self.reasoning_effort),)
        return replace(identity, think=self.think, options=options)


class GoCompletion(Completion):
    def consume(self, frame):
        events = list(super().consume(frame))
        if self.metrics is not None:
            self.metrics = ModelMetrics(self.metrics.input_tokens, self.metrics.output_tokens)
        yield from events


class GoModel(ChatModel):
    def __init__(self, client, config: GoConfig, api_key, *, cost_control=None,
                 session_id_provider: Callable[[], str] | None = None):
        super().__init__(client, config, api_key, cost_control=cost_control)
        fallback = str(uuid4())
        self._session_id_provider = session_id_provider or (lambda: fallback)

    @property
    def initial_binding(self):
        summary_config = replace(self.config, think=False, max_tokens=min(SUMMARY_OUTPUT_TOKENS, self.config.max_tokens))
        summary = GoModel(self.client, summary_config, self._api_key, cost_control=self._cost_control,
                          session_id_provider=self._session_id_provider)
        return ModelBinding(self, summary, self.config.identity, summary_config.identity,
            BudgetConfig(self.config.num_ctx, self.config.max_tokens),
            BudgetConfig(summary_config.num_ctx, summary_config.max_tokens), self.config.validate_request)

    def prepare_configuration(self, request):
        identity, options = request.identity, dict(request.identity.options)
        if identity.provider != self.config.provider or type(identity.think) is not bool:
            raise ValueError('invalid OpenCode Go identity')
        candidate = replace(self.config, model=identity.model, model_key=options.get('model_key'),
            base_url=identity.base_url, num_ctx=options.get('num_ctx'), max_tokens=options.get('max_tokens'),
            stream_usage=options.get('stream_usage', True),
            output_token_field=options.get('output_token_field', 'max_tokens'),
            think=identity.think, reasoning_effort=options.get('reasoning_effort', 'high'),
            max_request_bytes=identity.max_request_bytes)
        if candidate.identity != identity:
            raise ValueError('invalid OpenCode Go identity')
        key = self._api_key if request.credential_action == 'keep' else request.credential.reveal()
        return GoModel(self.client, candidate, key, cost_control=self._cost_control,
                       session_id_provider=self._session_id_provider).initial_binding

    @property
    def reasoning_efforts(self):
        return REASONING_EFFORTS if self.config.model == OPENCODE_GO_MODEL else ()

    def prepare_reasoning_effort(self, effort):
        if effort not in self.reasoning_efforts:
            raise ValueError('unsupported OpenCode Go reasoning effort')
        config = replace(self.config, think=True, reasoning_effort=effort)
        return GoModel(self.client, config, self._api_key, cost_control=self._cost_control,
                       session_id_provider=self._session_id_provider).initial_binding

    async def stream(self, request, cancel):
        def body(value):
            if self.config.think:
                mapped = build_thinking_request(value, self.config.thinking_config)
                mapped['model'] = self.config.model
                if not self.config.stream_usage:
                    mapped.pop('stream_options', None)
                return mapped
            mapped = build_request(value, self.config)
            mapped['thinking'] = {'type': 'disabled'}
            return mapped

        # 在受拥有的stream_chat准备段内解析；失败使用其固定错误及零发送语义。
        def checked_body(value):
            session_id = self._session_id_provider()
            if not isinstance(session_id, str) or re.fullmatch(r'[A-Za-z0-9._-]{1,128}', session_id) is None:
                raise ValueError('invalid conversation routing identity')
            headers['x-opencode-session'] = session_id
            return body(value)

        try:
            client_version = version('cicada')
        except PackageNotFoundError:
            client_version = '0+local'
        headers = {'User-Agent': 'cicada/' + client_version}
        async for event in stream_chat(request, cancel, client=self.client, endpoint=self.config.endpoint,
            api_key=self._api_key, identity=self.config.identity, build_body=checked_body,
            completion=GoCompletion(think=True) if self.config.think else ChatCompletion(),
            max_request_bytes=self.config.max_request_bytes,
            max_tokens=self.config.max_tokens, timeout=DEFAULT_TIMEOUT, deadline_seconds=CALL_DEADLINE_S,
            label='OpenCode Go', cost_control=self._cost_control, request_headers=headers):
            yield event
