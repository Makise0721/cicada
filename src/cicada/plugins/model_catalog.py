"""进程拥有的模型目录与纯本地绑定准备；目录不随当前模型替换。"""

from __future__ import annotations

import os
from http.cookiejar import DefaultCookiePolicy
from dataclasses import replace
from typing import Callable
from uuid import uuid4

import httpx

from cicada.models import ModelBinding, ModelOption, SafeModelIdentity
from cicada.provider_presets import is_opencode_go_endpoint
from cicada.runtime.plugin import PluginDefinition
from cicada.user_config import ModelProfile, ProviderProfile, SecretValue, UserConfigError, UserConfigSnapshot


class _NoCookies(DefaultCookiePolicy):
    """模型协议只用显式认证，连接复用不携带隐式Cookie会话。"""

    def set_ok(self, cookie, request):
        return False

    def return_ok(self, cookie, request):
        return False


def option_id(provider: str, model_key: str) -> str:
    return f'{provider}/{model_key}'


def profile_identity(provider: ProviderProfile, model: ModelProfile) -> SafeModelIdentity:
    """与 Adapter 共享的安全快照；旧官方身份保持旧编码。"""
    from cicada.plugins.deepseek.protocol import DeepSeekConfig

    if provider.protocol == 'deepseek-chat':
        identity = DeepSeekConfig(model=model.model_id, think=model.think,
            reasoning_effort=model.reasoning_effort, num_ctx=model.context_window,
            max_tokens=model.max_output_tokens).identity
        if provider.id == 'deepseek' and model.key == model.model_id:
            return identity
        return replace(identity, provider=provider.id, options=identity.options + (
            ('protocol', provider.protocol), ('model_key', model.key), ('profile_version', 1)))
    from cicada.plugins.openai_chat import ChatConfig

    config_type = ChatConfig
    settings = {}
    if is_opencode_go_endpoint(provider.base_url):
        from cicada.plugins.openai_chat.opencode_go import GoConfig
        config_type = GoConfig
        settings = dict(think=model.think, reasoning_effort=model.reasoning_effort)
    return config_type(provider=provider.id, model_key=model.key, model=model.model_id,
        base_url=provider.base_url, num_ctx=model.context_window,
        max_tokens=model.max_output_tokens, stream_usage=model.stream_usage,
        output_token_field=model.output_token_field, **settings).identity


class ModelCatalog:
    """一个 scope 的目录、连接池和准备入口；所有变更均为本地。"""

    is_catalog = True

    def __init__(self, profiles: tuple[ProviderProfile, ...], selected: str, *,
                 credential: Callable[[str, str], SecretValue | UserConfigSnapshot | None], defer: Callable,
                 default_id: str | None = None, initial_model=None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.profiles = profiles
        self.revision = str(uuid4())
        self.selected = selected
        self.default_id = default_id or selected
        self._credential = credential
        self._defer = defer
        self._initial_model = initial_model
        self._initial: ModelBinding | None = None
        self._transport = transport
        self._clients: dict[tuple[str, str, str], httpx.AsyncClient] = {}
        self._cost_control = None
        self._session_id_provider = lambda: self._transient_session_id
        self._transient_session_id = str(uuid4())
        if initial_model is not None and hasattr(initial_model, 'client'):
            provider, _ = self.lookup(selected)
            initial_model.client.cookies.clear()
            initial_model.client.cookies.jar.set_policy(_NoCookies())
            self._clients[(provider.id, provider.protocol, provider.base_url.rstrip('/'))] = initial_model.client

    def lookup(self, selected: str) -> tuple[ProviderProfile, ModelProfile]:
        for provider in self.profiles:
            for model in provider.models:
                if option_id(provider.id, model.key) == selected:
                    return provider, model
        raise UserConfigError('config', 'unknown-model')

    def set_session_id_provider(self, provider: Callable[[], str]) -> None:
        self._session_id_provider = provider

    @property
    def options(self) -> tuple[ModelOption, ...]:
        choices = []
        for provider in self.profiles:
            present = (provider.auth == 'none' or provider.credential_ref is not None
                       or bool(provider.api_key_env and os.environ.get(provider.api_key_env)))
            for model in provider.models:
                key = option_id(provider.id, model.key)
                reason = None if present and model.tools else ('缺少凭据' if not present else '不支持工具调用')
                choices.append(ModelOption(key, f'{provider.name} · {model.name or model.model_id}',
                    available=reason is None, reason=reason, is_default=key == self.default_id))
        return tuple(choices)

    def reload(self, profiles: tuple[ProviderProfile, ...], *, default_id: str | None = None) -> None:
        if profiles != self.profiles or (default_id is not None and default_id != self.default_id):
            self.profiles = profiles
            self.revision = str(uuid4())
        if default_id is not None:
            self.default_id = default_id

    def set_cost_control(self, ledger) -> None:
        if self._cost_control is not None and self._cost_control is not ledger:
            raise ValueError('model catalog already has a cost allowance')
        self._cost_control = ledger
        if self._initial_model is not None:
            self._initial_model.set_cost_control(ledger)

    def _decorate(self, binding: ModelBinding, provider: ProviderProfile, model: ModelProfile) -> ModelBinding:
        return replace(binding, option_id=option_id(provider.id, model.key),
            label=f'{provider.name} / {model.name or model.model_id}', catalog_revision=self.revision)

    @property
    def initial_binding(self) -> ModelBinding:
        if self._initial is None:
            provider, model = self.lookup(self.selected)
            if not model.tools:
                raise UserConfigError('config', 'tools-required')
            if self._initial_model is not None:
                self._initial = self._decorate(self._initial_model.initial_binding, provider, model)
            else:
                self._initial = self.prepare_model(self.selected)
        return self._initial

    def prepare_model(self, selected: str, *, credential: SecretValue | None = None,
                      think: bool | None = None) -> ModelBinding:
        provider, model = self.lookup(selected)
        if not model.tools:
            raise UserConfigError('config', 'tools-required')
        if think is not None:
            model = replace(model, think=think)
        resolved = credential if credential is not None else self._credential(provider.id, model.key)
        if isinstance(resolved, UserConfigSnapshot):
            observed = next((p for p in resolved.providers if p.id == provider.id), None)
            observed_model = next((m for m in observed.models if m.key == model.key), None) if observed else None
            if (observed is None or observed_model is None
                    or (observed.protocol, observed.base_url, observed.auth, observed_model.model_id)
                    != (provider.protocol, provider.base_url, provider.auth, model.model_id)):
                raise UserConfigError('config', 'configuration-changed')
            secret = resolved.credential
        else:
            secret = resolved
        if provider.auth != 'none' and secret is None:
            raise UserConfigError('credential', 'missing-credential')
        key = (provider.id, provider.protocol, provider.base_url.rstrip('/'))
        client = self._clients.get(key)
        if client is None:
            # 上限来自有界目录；旧修订创建的连接仍由同scope拥有，数量亦有硬上限。
            if len(self._clients) >= 64:
                raise UserConfigError('config', 'connection-limit')
            client = httpx.AsyncClient(transport=self._transport, trust_env=False, follow_redirects=False)
            client.cookies.jar.set_policy(_NoCookies())
            self._defer(client.aclose)
            self._clients[key] = client
        if provider.protocol == 'deepseek-chat':
            from cicada.plugins.deepseek.model import DeepSeekModel
            from cicada.plugins.deepseek.protocol import DeepSeekConfig

            config = DeepSeekConfig(model=model.model_id, think=model.think,
                reasoning_effort=model.reasoning_effort, num_ctx=model.context_window,
                max_tokens=model.max_output_tokens)
            adapter = DeepSeekModel(client, config, secret.reveal(),
                cost_control=self._cost_control, identity_override=profile_identity(provider, model))
        else:
            from cicada.plugins.openai_chat import ChatConfig, ChatModel

            config_type, model_type = ChatConfig, ChatModel
            extra = {}
            settings = {}
            if is_opencode_go_endpoint(provider.base_url):
                from cicada.plugins.openai_chat.opencode_go import GoConfig, GoModel
                config_type, model_type = GoConfig, GoModel
                settings = dict(think=model.think, reasoning_effort=model.reasoning_effort)
                extra['session_id_provider'] = lambda: self._session_id_provider()
            config = config_type(provider=provider.id, model_key=model.key, model=model.model_id,
                base_url=provider.base_url, num_ctx=model.context_window,
                max_tokens=model.max_output_tokens, stream_usage=model.stream_usage,
                output_token_field=model.output_token_field, **settings)
            adapter = model_type(client, config, secret.reveal() if secret else None,
                cost_control=self._cost_control, **extra)
        return self._decorate(adapter.initial_binding, provider, model)

    async def stream(self, request, cancel):
        async for event in self.initial_binding.model.stream(request, cancel):
            yield event


def model_catalog_plugin(profiles: tuple[ProviderProfile, ...], selected: str, *,
                         credential: Callable[[str, str], SecretValue | UserConfigSnapshot | None],
                         default_id: str | None = None, initial_plugin: PluginDefinition | None = None,
                         transport: httpx.AsyncBaseTransport | None = None) -> PluginDefinition:
    def setup(ctx):
        initial = None
        if initial_plugin is not None:
            class InitialContext:
                def defer(self, cleanup):
                    ctx.defer(cleanup)

                def provide(self, name, value):
                    nonlocal initial
                    if name != 'model' or initial is not None:
                        raise ValueError('invalid initial model capability')
                    initial = value

            initial_plugin.setup(InitialContext())
        ctx.provide('model', ModelCatalog(profiles, selected, credential=credential,
            defer=ctx.defer, default_id=default_id, initial_model=initial, transport=transport))

    return PluginDefinition(name='model-catalog', setup=setup,
        provides=frozenset({'model'}), requires=frozenset())
