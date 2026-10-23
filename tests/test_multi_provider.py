"""统一目录经过应用事务、真实SQLite及受控HTTP；全部凭据为本用例哨兵。"""

import json
from dataclasses import replace

import httpx
import pytest

from cicada.application import ModelSelectionError, open_application
from cicada.plugins.model_catalog import model_catalog_plugin, profile_identity
from cicada.plugins.session_store import read_session_identity
from cicada.plugins.user_config import FileUserConfigStore
from cicada.user_config import (
    LoginSubmission, ModelProfile, ProviderProfile, SecretValue, UserConfigSnapshot, UserDefaults,
)
from test_application import _repo
from test_deepseek_application import config, response


def profiles():
    model = ModelProfile('coder', 'org/same-model', context_window=32768, max_output_tokens=2048)
    return (
        ProviderProfile('personal', 'Personal', 'openai-chat-completions',
            'https://personal.example/v1', (model,), model.key, api_key_env='CICADA_TEST_PERSONAL'),
        ProviderProfile('work', 'Work', 'openai-chat-completions',
            'https://work.example/api/v1', (model,), model.key, api_key_env='CICADA_TEST_WORK'),
    )


def application_config(root, transport, providers, selected='personal/coder', session_id=None):
    provider = next(p for p in providers if selected.startswith(p.id + '/'))
    model = next(m for m in provider.models if selected == f'{provider.id}/{m.key}')
    safe = profile_identity(provider, model)
    base = config(root, transport, session_id)
    launch = replace(base.persistence.identity, provider=safe.provider, model=safe.model,
        base_url=safe.base_url, think=safe.think, options=dict(safe.options),
        max_request_bytes=safe.max_request_bytes)
    plugin = model_catalog_plugin(providers, selected,
        credential=lambda pid, key: SecretValue(f'offline-{pid}-key'), transport=transport)
    return replace(base, model_plugin=plugin, persistence=replace(base.persistence, identity=launch))


async def test_cross_provider_same_model_summary_and_resume(tmp_path):
    root = _repo(tmp_path)
    providers = profiles()
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append((str(request.url), request.headers['authorization'], body))
        return httpx.Response(200, content=response(None, 'offline answer'))

    transport = httpx.MockTransport(handle)
    session = await open_application(application_config(root, transport, providers))
    try:
        await session.execute('first')
        await session.execute('second')
        history = session.history
        before = session.persistence.epoch_id
        await session.select_model('work/coder', catalog_revision=session.model_catalog_revision)
        assert session.history == history and session.persistence.epoch_id != before
        assert session.current_model_option_id == 'work/coder'
        assert {o.id for o in session.model_options} == {'personal/coder', 'work/coder'}
        await session.compact_context()
        assert seen[-1][0] == 'https://work.example/api/v1/chat/completions'
        assert seen[-1][1] == 'Bearer offline-work-key'
        assert 'tools' not in seen[-1][2] and seen[-1][2]['max_tokens'] == 2048
        await session.execute('continue')
        sid = session.persistence.session_id
        assert all(item[2]['model'] == 'org/same-model' for item in seen)
    finally:
        await session.aclose()
    stored = read_session_identity(directory=root / '.cicada' / 'sessions', session_id=sid, workspace=root)
    restored_config = application_config(root, transport, providers, 'work/coder', sid)
    restored_config = replace(restored_config, persistence=replace(restored_config.persistence, identity=stored.launch))
    restored = await open_application(restored_config)
    try:
        await restored.execute('restored')
        assert seen[-1][1] == 'Bearer offline-work-key'
        await restored.select_model('personal/coder')
        await restored.execute('switch back')
        assert seen[-1][0] == 'https://personal.example/v1/chat/completions'
        assert seen[-1][1] == 'Bearer offline-personal-key'
    finally:
        await restored.aclose()


async def test_catalog_revision_and_incompatible_thinking_preserve_binding(tmp_path):
    root = _repo(tmp_path)
    deepseek_model = ModelProfile('deepseek-flash', 'deepseek-flash', think=True,
        context_window=32768, max_output_tokens=2048)
    deepseek = ProviderProfile('deepseek', 'DeepSeek', 'deepseek-chat', 'https://api.deepseek.com',
        (deepseek_model,), deepseek_model.key, api_key_env='CICADA_TEST_DEEPSEEK')
    providers = profiles() + (deepseek,)
    seen = []

    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        calls = (("read-1", "read", {"path": "a.txt"}),) if len(seen) == 1 else ()
        thinking = body.get('thinking', {}).get('type') == 'enabled'
        return httpx.Response(200, content=response('thought' if thinking else None, calls=calls))

    session = await open_application(application_config(root, httpx.MockTransport(handle), providers))
    try:
        await session.execute('read a.txt')
        history, epoch = session.history, session.persistence.epoch_id
        with pytest.raises(ModelSelectionError):
            await session.select_model('deepseek/deepseek-flash')
        assert session.history == history and session.persistence.epoch_id == epoch
        assert session.current_model_option_id == 'personal/coder'
        with pytest.raises(ModelSelectionError, match='目录'):
            await session.select_model('work/coder', catalog_revision='stale')
        await session.select_model('deepseek/deepseek-flash', think=False)
        await session.execute('continue without thinking')
        assert seen[-1]['thinking'] == {'type': 'disabled'}
        assert all('reasoning_content' not in message for message in seen[-1]['messages'])
        await session.select_model('work/coder')
        assert session.reasoning_effort.enabled is False
        assert session.current_model_option_id == 'work/coder'
    finally:
        await session.aclose()


def test_cli_selects_configured_instance_and_rejects_archive_endpoint_drift(monkeypatch, tmp_path):
    from cicada import __main__ as cli
    from test_cli import _runtime
    from test_user_config_startup import invoke

    monkeypatch.setenv('CICADA_TEST_PERSONAL', 'offline-personal-key')
    monkeypatch.setenv('CICADA_TEST_WORK', 'offline-work-key')
    user = tmp_path / 'user'
    user.mkdir()
    text = '''version = 2
[defaults]
provider = "personal"
model = "coder"
'''
    for provider in profiles():
        text += f'''
[providers.{provider.id}]
name = "{provider.name}"
protocol = "openai-chat-completions"
base_url = "{provider.base_url}"
default_model = "coder"
api_key_env = "{provider.api_key_env}"
[providers.{provider.id}.models.coder]
id = "org/same-model"
context_window = 32768
max_output_tokens = 2048
tools = true
'''
    path = user / 'config.toml'
    path.write_text(text, encoding='utf-8')
    seen = []

    def handle(request):
        seen.append((str(request.url), request.headers['authorization']))
        return httpx.Response(200, content=response(None, 'offline'))

    original = cli.model_catalog_plugin
    monkeypatch.setattr(cli, 'model_catalog_plugin', lambda *a, **kw: original(
        *a, **kw, transport=httpx.MockTransport(handle)))
    arguments = ['--workspace', str(tmp_path), '--provider', 'work', '--model', 'coder', '--save-session', 'first']
    code, out, err = invoke(monkeypatch, arguments, user)
    assert code == 0, err + out
    assert seen[-1] == ('https://work.example/api/v1/chat/completions', 'Bearer offline-work-key')
    sid = out.split('[session id=')[1].split(' ')[0]
    saved = read_session_identity(directory=_runtime(tmp_path) / 'sessions', session_id=sid, workspace=tmp_path)
    assert saved.launch.provider == 'work' and saved.launch.model == 'org/same-model'
    assert saved.launch.options['model_key'] == 'coder'
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--resume', sid, 'second'], user)
    assert code == 0, err + out
    path.write_text(text.replace('https://work.example/api/v1', 'https://changed.example/v1'), encoding='utf-8')
    before = len(seen)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--resume', sid, 'must reject'], user)
    assert code == 2 and len(seen) == before
    assert '--resume-with-current-config' in err


async def test_save_apply_rotates_key_without_resetting_current_default_or_cost(tmp_path, monkeypatch):
    from cicada.costs import CostLedger
    from cicada.plugins.user_config import FileUserConfigStore

    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    root = _repo(tmp_path)
    store = FileUserConfigStore(tmp_path / 'user')
    provider = replace(profiles()[0], api_key_env=None)
    defaults = UserDefaults(provider=provider.id, model='coder', think=False,
        num_ctx=32768, max_output_tokens=2048)
    view = store.read_view()
    result = await store.save(LoginSubmission(defaults, 'replace', view.revision,
        credential=SecretValue('offline-initial-key'), provider_profile=provider))
    assert result.defaults_saved
    seen = []
    transport = httpx.MockTransport(lambda req: (
        seen.append(req.headers['authorization']) or httpx.Response(200, content=response(None))))
    providers = store.read_profiles()
    prepared = application_config(root, transport, providers)
    plugin = model_catalog_plugin(providers, 'personal/coder',
        credential=lambda pid, key: store.load(provider_id=pid, model_key=key), transport=transport,
        default_id='personal/coder')
    ledger = CostLedger(root / '.cicada' / 'cost', max_calls=4)
    session = await open_application(replace(prepared, model_plugin=plugin, cost_control=ledger),
        login_store_factory=lambda: store)
    try:
        await session.execute('before rotation')
        previous = session.persistence.epoch_id
        view = session.login_view()
        selected = next(p for p in view.providers if p.id == provider.id)
        result = await session.configure_login(LoginSubmission(defaults, 'replace', view.revision,
            target='save-and-apply', credential=SecretValue('offline-rotated-key'),
            provider_profile=selected, set_default=False))
        assert result.defaults_saved and result.session_applied
        assert session.persistence.epoch_id != previous
        await session.execute('after rotation')
        assert seen == ['Bearer offline-initial-key', 'Bearer offline-rotated-key']
        assert session.cost.calls == 2
        assert store.read_default_reference() == ('personal', 'coder')
        assert session.current_model_option_id == 'personal/coder'
        snapshot = session._app.runtime.capability('model')
        assert snapshot.default_id == 'personal/coder'
        # 设置默认必须拒绝菜单打开后的外部修改，不能把旧连接写回配置。
        revision = session.model_catalog_revision
        path = store.root / 'config.toml'
        changed = path.read_text(encoding='utf-8').replace('https://personal.example/v1', 'https://new.example/v1')
        path.write_text(changed, encoding='utf-8')
        with pytest.raises(ModelSelectionError, match='目录'):
            await session.set_default_model('personal/coder', catalog_revision=revision)
        assert path.read_text(encoding='utf-8') == changed and not session.login_busy
        # 保存后的刷新失败也须向调用者和订阅者返回同一个“已保存、未应用”结果。
        view = session.login_view()
        selected = next(p for p in view.providers if p.id == provider.id)
        updates = []
        session.subscribe(updates.append)
        from cicada.user_config import UserConfigError
        def failed_refresh():
            raise UserConfigError('config', 'invalid-format')
        monkeypatch.setattr(store, 'read_profiles', failed_refresh)
        result = await session.configure_login(LoginSubmission(defaults, 'keep', view.revision,
            target='save-and-apply', provider_profile=selected, set_default=False))
        assert result.defaults_saved and not result.session_applied
        assert result.application_outcome == 'blocked' and result.error_code == 'configuration-changed'
        assert updates[-1].result == result and session.cost.calls == 2
    finally:
        await session.aclose()


async def test_preparation_refuses_new_credentials_for_changed_connection(tmp_path):
    root = _repo(tmp_path)
    providers = profiles()
    changed = replace(providers[1], base_url='https://changed.example/v1')
    snapshot = UserConfigSnapshot(UserDefaults(provider='work', model='coder'), frozenset(),
        SecretValue('offline-new-host-key'), 'file', 'offline-revision', (providers[0], changed))
    sent = []
    transport = httpx.MockTransport(lambda req: (
        sent.append(req) or httpx.Response(200, content=response(None))))
    initial = application_config(root, transport, providers)
    plugin = model_catalog_plugin(providers, 'personal/coder', transport=transport,
        credential=lambda pid, key: SecretValue('offline-personal-key') if pid == 'personal' else snapshot)
    session = await open_application(replace(initial, model_plugin=plugin))
    try:
        before = session.persistence.epoch_id
        with pytest.raises(ModelSelectionError):
            await session.select_model('work/coder')
        assert not sent and session.persistence.epoch_id == before
        assert session.current_model_option_id == 'personal/coder'
    finally:
        await session.aclose()


async def test_instances_at_same_endpoint_never_share_server_cookies(tmp_path):
    root = _repo(tmp_path)
    first, second = profiles()
    second = replace(second, base_url=first.base_url)
    providers = first, second
    seen = []

    def handle(request):
        seen.append((request.headers['authorization'], request.headers.get('cookie')))
        return httpx.Response(200, headers={'set-cookie': 'session=offline-cookie; Path=/'}, content=response(None))

    session = await open_application(application_config(root, httpx.MockTransport(handle), providers))
    try:
        await session.execute('one')
        await session.execute('same instance')
        await session.select_model('work/coder')
        await session.execute('another instance')
        assert [auth for auth, _ in seen] == ['Bearer offline-personal-key'] * 2 + ['Bearer offline-work-key']
        assert all(cookie is None for _, cookie in seen)
    finally:
        await session.aclose()


@pytest.mark.parametrize('previous_cancel,set_default', [(False, True), (True, True), (True, False)])
def test_first_run_can_save_custom_provider_and_use_it_immediately(monkeypatch, tmp_path, previous_cancel, set_default):
    import cicada.ui
    from cicada import __main__ as cli
    from test_cli import _CaptureStream
    from test_user_config_startup import invoke

    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)
    user = tmp_path / 'user'
    if previous_cancel:
        FileUserConfigStore(user).ensure_default_config()
    seen = []

    async def configure(store, snapshot, defaults, **kwargs):
        assert kwargs['first_run']
        view = store.read_view()
        provider = replace(profiles()[0], api_key_env=None)
        result = await store.save(LoginSubmission(
            UserDefaults(provider='personal', model='coder', think=False, num_ctx=32768, max_output_tokens=2048),
            'replace', view.revision, credential=SecretValue('offline-first-run-key'),
            provider_profile=provider, set_default=set_default))
        return replace(result, selected_provider='personal', selected_model='coder')

    async def tui(session, *, initial_prompt):
        assert session.current_model_option_id == 'personal/coder'
        result = await session.execute(initial_prompt)
        return result.exit_code

    def handle(request):
        seen.append((str(request.url), request.headers['authorization']))
        return httpx.Response(200, content=response(None))

    original = cli.model_catalog_plugin
    monkeypatch.setattr(cli, '_launch_first_run', configure)
    monkeypatch.setattr(cicada.ui, 'run_tui_session', tui)
    monkeypatch.setattr(cli, 'model_catalog_plugin', lambda *a, **kw: original(
        *a, **kw, transport=httpx.MockTransport(handle)))
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', 'first'], user)
    assert code == 0, err + out
    assert seen == [('https://personal.example/v1/chat/completions', 'Bearer offline-first-run-key')]
    assert FileUserConfigStore(user).read_default_reference() == (
        ('personal', 'coder') if set_default else (UserDefaults().provider, UserDefaults().model))


@pytest.mark.parametrize('credentials_only', [False, True])
async def test_startup_login_accepts_selected_nondefault_model(monkeypatch, tmp_path, credentials_only):
    from cicada import __main__ as cli
    from cicada.ui import login_wizard, terminal_io

    monkeypatch.delenv('CICADA_TEST_PERSONAL', raising=False)
    provider = profiles()[0]
    secondary = replace(provider.models[0], key='secondary', model_id='org/secondary')
    provider = replace(provider, models=provider.models + (secondary,))
    store = FileUserConfigStore(tmp_path / 'user')
    initial = store.load()
    await store.save(LoginSubmission(
        UserDefaults(provider=provider.id, model='coder', think=False), 'keep', initial.revision,
        provider_profile=provider, set_default=True))
    snapshot = store.load(provider_id=provider.id, model_key='secondary')

    async def save_selected(terminal, selected_store, view, **kwargs):
        assert view.model == 'secondary' and view.credentials_only == credentials_only
        result = await selected_store.save(LoginSubmission(snapshot.defaults, 'replace', view.revision,
            credential=SecretValue('offline-secondary-key'), provider_profile=replace(provider, api_key_env=None),
            set_default=False))
        return replace(result, selected_provider=provider.id, selected_model='secondary'), []

    monkeypatch.setattr(login_wizard, 'run_startup_login', save_selected)
    monkeypatch.setattr(terminal_io, 'PromptToolkitTerminal', terminal_io.FakeTerminal)
    result = await cli._launch_first_run(store, snapshot, snapshot.defaults,
        explicit={}, credentials_only=credentials_only)
    assert result.defaults_saved and result.selected_model == 'secondary'
    assert store.read_default_reference() == (provider.id, 'coder')


@pytest.mark.parametrize('reason', ['no price', 'call limit'])
async def test_cost_rejection_explains_reason_and_preserves_binding(tmp_path, reason):
    from decimal import Decimal
    from cicada.costs import CostLedger, CostProfile, ModelPrice

    root = _repo(tmp_path)
    providers = profiles()
    provider, model = providers[0], providers[0].models[0]
    profile = CostProfile('https://example.invalid/prices', '2026-10-21', 'a' * 64, (
        ModelPrice(model.model_id, Decimal('1'), Decimal('2'), provider.id, provider.protocol,
            provider.base_url, model.context_window),), schema_version=2)
    ledger = (CostLedger(root / '.cicada' / 'cost', limit=Decimal('10'), profile=profile)
              if reason == 'no price' else CostLedger(root / '.cicada' / 'cost', max_calls=1))
    sent = []
    def handle(request):
        sent.append(request)
        return httpx.Response(200, content=response(None))
    prepared = application_config(root, httpx.MockTransport(handle), providers)
    session = await open_application(replace(prepared, cost_control=ledger))
    try:
        if reason == 'call limit':
            await session.execute('consume allowance')
        before, history, epoch = len(sent), session.history, session.persistence.epoch_id
        with pytest.raises(ModelSelectionError, match=reason):
            await session.select_model('work/coder')
        assert len(sent) == before and session.history == history
        assert session.persistence.epoch_id == epoch and session.current_model_option_id == 'personal/coder'
    finally:
        await session.aclose()


def test_declared_tools_capability_blocks_initial_official_provider(monkeypatch, tmp_path):
    from cicada import __main__ as cli
    from test_user_config_startup import invoke

    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    store = FileUserConfigStore(tmp_path / 'user')
    snapshot = store.load()
    provider = replace(snapshot.providers[0], models=tuple(
        replace(model, tools=False) for model in snapshot.providers[0].models))
    async def save():
        return await store.save(LoginSubmission(snapshot.defaults, 'replace', snapshot.revision,
            credential=SecretValue('offline-no-tools-key'), provider_profile=provider))
    import asyncio
    assert asyncio.run(save()).defaults_saved
    sent = []
    original = cli.deepseek_plugin
    monkeypatch.setattr(cli, 'deepseek_plugin', lambda *a, **kw: original(*a, **kw,
        transport=httpx.MockTransport(lambda request: sent.append(request))))
    code, _, _ = invoke(monkeypatch, ['--workspace', str(tmp_path), 'must reject'], store.root)
    assert code == 2 and not sent


async def test_opencode_go_key_only_routes_main_summary_new_and_restored_sessions(tmp_path):
    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys
    from cicada.ui.login_wizard import make_login_wizard
    from cicada.provider_presets import OPENCODE_GO_BASE_URL, OPENCODE_GO_MODEL
    from cicada.plugins.openai_chat.opencode_go import GoConfig

    root = _repo(tmp_path)
    store = FileUserConfigStore(tmp_path / 'user')
    wizard = make_login_wizard(store.read_view(), initial_set_default=True)
    wizard.handle(KeyPress(Keys.BracketedPaste, 'OpenCode Go'))
    wizard.handle(KeyPress(Keys.ControlM, '\r'))
    assert wizard.step == 'edit' and wizard._edit_field == 'api_key'
    wizard.handle(KeyPress(Keys.BracketedPaste, 'offline-go-key'))
    assert 'offline-go-key' not in ''.join(wizard.render_lines(80, 24))
    assert wizard.handle(KeyPress(Keys.ControlM, '\r')) == 'submit'
    submission = wizard.submission()
    assert submission.defaults.model == OPENCODE_GO_MODEL and submission.set_default
    assert submission.defaults.model == 'deepseek-v4.1-flash'
    assert submission.defaults.num_ctx == 1000000 and submission.defaults.max_output_tokens == 32768
    assert (await store.save(submission)).defaults_saved
    wizard.clear_secret()
    provider = next(p for p in store.read_profiles() if p.id == 'opencode-go')
    assert provider.base_url == OPENCODE_GO_BASE_URL
    assert store.load().credential.reveal() == 'offline-go-key'
    with pytest.raises(ValueError, match='unsupported'):
        GoConfig('opencode-go', 'bad', 'kimi-k3', OPENCODE_GO_BASE_URL, 262144, 32768)
    with pytest.raises(ValueError, match='unsupported'):
        GoConfig('opencode-go', OPENCODE_GO_MODEL, OPENCODE_GO_MODEL, OPENCODE_GO_BASE_URL, 1000001, 32768)
    legacy = GoConfig('opencode-go', 'kimi-k2.6', 'kimi-k2.6', OPENCODE_GO_BASE_URL, 262144, 32768)
    assert legacy.identity.model == 'kimi-k2.6'
    seen = []
    def handle(request):
        seen.append((request, json.loads(request.content)))
        body = seen[-1][1]
        calls = (('go-read', 'read', {'path': 'a.txt'}),) if len(seen) == 1 else ()
        return httpx.Response(200, content=response(
            'go thought' if body['thinking']['type'] == 'enabled' else None, 'offline', calls=calls))
    transport = httpx.MockTransport(handle)
    selected = 'opencode-go/' + OPENCODE_GO_MODEL
    legacy_model = replace(provider.models[0], key='kimi-k2.6', model_id='kimi-k2.6',
                           think=False, context_window=262144)
    legacy_provider = replace(provider, models=(legacy_model,), default_model=legacy_model.key)
    legacy_root = tmp_path / 'legacy'
    legacy_root.mkdir()
    legacy_session = await open_application(application_config(
        _repo(legacy_root), transport, (legacy_provider,), 'opencode-go/kimi-k2.6'))
    try:
        assert not legacy_session.reasoning_effort.enabled
        assert seen == []
    finally:
        await legacy_session.aclose()
    prepared = application_config(root, transport, (provider,), selected)
    plugin = model_catalog_plugin((provider,), selected, transport=transport,
        credential=lambda pid, key: store.load(provider_id=pid, model_key=key))
    prepared = replace(prepared, model_plugin=plugin)
    session = await open_application(prepared)
    try:
        original = session.persistence.session_id
        assert session.reasoning_effort.current == 'high'
        await session.execute('first')
        assert session.reasoning_effort.levels == ('low', 'high', 'max')
        assert next(m for m in seen[1][1]['messages'] if m.get('tool_calls'))['reasoning_content'] == 'go thought'
        for effort in ('low', 'high', 'max'):
            await session.set_reasoning_effort(effort)
            await session.execute('effort ' + effort)
            assert seen[-1][1]['reasoning_effort'] == effort
        await session.execute('second')
        await session.compact_context()
        assert seen[-1][1]['max_tokens'] == 4096 and 'tools' not in seen[-1][1]
        assert all(req.headers['x-opencode-session'] == original for req, _ in seen)
        await session.submit_new_session().wait()
        fresh = session.persistence.session_id
        await session.execute('new conversation')
        assert fresh != original and seen[-1][0].headers['x-opencode-session'] == fresh
        await session.submit_resume_session(original).wait()
        assert session.reasoning_effort.current == 'max'
        await session.execute('resumed conversation')
        assert seen[-1][0].headers['x-opencode-session'] == original
        for request, body in seen:
            assert str(request.url) == OPENCODE_GO_BASE_URL + '/chat/completions'
            assert request.headers['authorization'] == 'Bearer offline-go-key'
            assert request.headers['user-agent'].startswith('cicada/')
            assert body['model'] == OPENCODE_GO_MODEL and body['thinking'] == {'type': 'enabled' if 'tools' in body else 'disabled'}
            assert body['stream_options'] == {'include_usage': True}
        assert dict(session.config.persistence.identity.options)['compatibility_profile'] == 'opencode-go-v1'
    finally:
        await session.aclose()
    stored = read_session_identity(directory=root / '.cicada' / 'sessions', session_id=original, workspace=root)
    restored_model = replace(provider.models[0], think=stored.launch.think,
                             reasoning_effort=stored.launch.options.get('reasoning_effort', 'high'))
    restored_provider = replace(provider, models=(restored_model,))
    restored_plugin = model_catalog_plugin((restored_provider,), selected, transport=transport,
        credential=lambda pid, key: store.load(provider_id=pid, model_key=key))
    restored_config = replace(prepared, model_plugin=restored_plugin,
        persistence=replace(prepared.persistence, identity=stored.launch, session_id=original))
    restored = await open_application(restored_config)
    try:
        assert restored.reasoning_effort.current == 'max'
        await restored.execute('process restart')
        assert any(m.get('reasoning_content') == 'go thought' for m in seen[-1][1]['messages'])
        assert seen[-1][0].headers['x-opencode-session'] == original
    finally:
        await restored.aclose()


def test_cli_go_key_only_startup_wizard_and_resume(monkeypatch, tmp_path):
    import asyncio
    import cicada.ui
    from cicada import __main__ as cli
    from cicada.ui import terminal_io
    from prompt_toolkit.key_binding.key_processor import KeyPress
    from prompt_toolkit.keys import Keys
    from test_cli import _CaptureStream
    from test_user_config_startup import invoke

    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)
    entered = []
    class StartupTerminal(terminal_io.FakeTerminal):
        def start(self, *args):
            super().start(*args)
            entered.append(self)
            asyncio.get_running_loop().call_soon(self.emit, [
                KeyPress(Keys.BracketedPaste, 'OpenCode Go'), KeyPress(Keys.ControlM, '\r'),
                KeyPress(Keys.BracketedPaste, 'offline-cli-go-key'), KeyPress(Keys.ControlM, '\r')])
    monkeypatch.setattr(terminal_io, 'PromptToolkitTerminal', StartupTerminal)
    ids, seen = [], []
    async def tui(session, *, initial_prompt):
        ids.append(session.persistence.session_id)
        try:
            return (await session.execute(initial_prompt)).exit_code
        finally:
            await session.aclose()
    def handle(request):
        seen.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, content=response(
            'cli thought' if body['thinking']['type'] == 'enabled' else None))
    original = cli.model_catalog_plugin
    monkeypatch.setattr(cicada.ui, 'run_tui_session', tui)
    monkeypatch.setattr(cli, 'model_catalog_plugin', lambda *a, **kw: original(
        *a, **kw, transport=httpx.MockTransport(handle)))
    user = tmp_path / 'user'
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', 'first'], user)
    assert code == 0, out + err
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '--resume', ids[0], 'second'], user)
    assert code == 0, out + err
    assert len(entered) == 1 and entered[0].stopped
    assert 'offline-cli-go-key' not in entered[0].output_text + out + err
    assert ids[0] == ids[1]
    assert all(request.headers['x-opencode-session'] == ids[0] for request in seen)
    assert all(request.headers['authorization'] == 'Bearer offline-cli-go-key' for request in seen)
