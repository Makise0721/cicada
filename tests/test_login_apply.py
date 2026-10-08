"""04 保存并应用：真实应用/SQLite/用户文件，HTTP 仅在系统边界受控 (A11–A15)。"""

import asyncio
import json
import sqlite3
import threading
from dataclasses import replace

import httpx
import pytest

from cicada.application import ConfigurationChanged, LoginCompleted, open_application
from cicada.models import ConfigurationRequest, ModelConfigurationPort
from cicada.plugins.deepseek import DeepSeekConfig
from cicada.plugins.deepseek.model import DeepSeekModel
from cicada.plugins.session_store.store import read_session_identity
from cicada.plugins.user_config import FileUserConfigStore
from cicada.plugins.user_config import store as files
from cicada.user_config import LoginSubmission, SecretValue, UserConfigError, UserDefaults
from test_application import _repo
from test_deepseek_application import config, response
from test_login_defaults import open_session, wait_for_thread


def recorded_transport(recorded):
    """捕获 Authorization 与请求体；thinking 关闭的通道不给 reasoning。"""
    def handle(req):
        body = json.loads(req.content)
        recorded.append((req.headers.get('authorization'), body))
        if body['thinking']['type'] == 'disabled':
            return httpx.Response(200, content=response(None, '关闭通道答案'))
        return httpx.Response(200, content=response(' 新思考\n ', '思考答案'))
    return httpx.MockTransport(handle)


def submission(view, *, action='keep', credential=None, think=None, effort=None,
               target='save-and-apply'):
    return LoginSubmission(
        UserDefaults(model=view.model,
                     think=view.think if think is None else think,
                     reasoning_effort=effort or view.reasoning_effort,
                     num_ctx=view.num_ctx, max_output_tokens=view.max_output_tokens),
        action, view.revision, target,
        SecretValue(credential) if credential is not None else None)


def test_prepare_configuration_is_local_shares_client_and_keeps_old_binding():
    calls = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: calls.append(req) or httpx.Response(200)))
    old = DeepSeekModel(client, DeepSeekConfig(), 'old-sentinel')
    assert isinstance(old, ModelConfigurationPort)
    before = old.initial_binding
    request = ConfigurationRequest(
        replace(before.identity, think=False, options=(('num_ctx', 65536), ('max_tokens', 4096),
                                                       ('reasoning_effort', 'low'))),
        'replace', SecretValue('new-sentinel'))
    binding = old.prepare_configuration(request)
    assert calls == []  # 准备零网络
    assert binding.model.client is client and binding.summary_model.client is client  # 不新建 client
    assert old.config.think and old._api_key == 'old-sentinel'  # 旧对象不变
    assert binding.identity.think is False and dict(binding.identity.options)['num_ctx'] == 65536
    summary_options = dict(binding.summary_identity.options)
    assert binding.summary_identity.think is False and summary_options['max_tokens'] == 2048
    assert summary_options['num_ctx'] == 65536  # 隐藏窗口不重置
    assert binding.model._api_key == 'new-sentinel' and binding.summary_model._api_key == 'new-sentinel'
    kept = old.prepare_configuration(ConfigurationRequest(before.identity, 'keep'))
    assert kept.model._api_key == 'old-sentinel'
    with pytest.raises(ValueError):
        old.prepare_configuration(ConfigurationRequest(
            replace(before.identity, provider='ollama'), 'keep'))
    with pytest.raises(ValueError):
        old.prepare_configuration(ConfigurationRequest(
            replace(before.identity, options=(('num_ctx', 100), ('max_tokens', 4096),
                                              ('reasoning_effort', 'low'))), 'keep'))
    with pytest.raises(ValueError):
        ConfigurationRequest(before.identity, 'replace')  # replace 必须携带经校验秘密


async def test_apply_replaces_key_and_thinking_across_requests_summary_model_and_restart(tmp_path):
    recorded = []
    session, store = await open_session(tmp_path, transport=recorded_transport(recorded))
    try:
        await session.execute('第一请求')
        assert recorded[-1] == ('Bearer offline-test-key', recorded[-1][1])
        assert recorded[-1][1]['thinking'] == {'type': 'enabled'}
        assert session.budget.anchor is not None
        before_epoch = session.persistence.epoch_id
        before_history = session.history
        updates = []
        session.subscribe(updates.append)
        view = session.login_view()
        assert view.actions == ('save-and-apply', 'defaults-only')
        result = await session.configure_login(
            submission(view, action='replace', credential='new-sentinel', think=False))
        assert result.defaults_saved and result.session_applied
        assert result.application_outcome == 'applied' and result.error_code is None
        changed = [u for u in updates if isinstance(u, ConfigurationChanged)]
        assert len(changed) == 1 and changed[0].identity.think is False
        epoch = session.persistence.epoch_id
        assert changed[0].epoch_id == epoch and epoch != before_epoch
        assert session.budget.anchor is None  # 真实应用使计量锚失效
        assert session.history == before_history  # 历史、摘要与验证状态原样保留
        assert session.config.persistence.identity.sources.get('credentials') == 'login'
        await session.execute('应用后请求')
        authorization, body = recorded[-1]
        assert authorization == 'Bearer new-sentinel'
        assert body['thinking'] == {'type': 'disabled'} and 'reasoning_effort' not in body
        await session.compact_context()
        authorization, summary = recorded[-1]
        assert authorization == 'Bearer new-sentinel'
        assert summary['thinking'] == {'type': 'disabled'} and summary['max_tokens'] == 2048
        # 后继 /model 从最新接受绑定准备, 不恢复启动 key/think/effort
        await session.select_model('deepseek-v4-pro')
        await session.execute('选择后请求')
        authorization, body = recorded[-1]
        assert authorization == 'Bearer new-sentinel'
        assert body['model'] == 'deepseek-v4-pro' and body['thinking'] == {'type': 'disabled'}
        # 后继 /login 初值来自最新绑定且仍提供应用
        view = session.login_view()
        assert view.model == 'deepseek-v4-pro' and not view.think
        assert 'save-and-apply' in view.actions
        session.cancel_login()
        session_id = session.persistence.session_id
        db_bytes = session._store.db_path.read_bytes()
        config_toml = (store.root / 'config.toml').read_bytes()
        credentials = (store.root / 'credentials.json').read_text(encoding='utf-8')
    finally:
        await session.aclose()
    # A15: 秘密只在凭据文件与 HTTP Authorization; epoch/存档/事件/普通配置无泄漏
    assert b'new-sentinel' not in db_bytes and b'new-sentinel' not in config_toml
    assert 'new-sentinel' in credentials and 'new-sentinel' not in repr(updates)
    assert 'new-sentinel' not in repr(session.history)
    saved = read_session_identity(directory=tmp_path / 'repo' / '.cicada' / 'sessions',
                                  session_id=session_id, workspace=tmp_path / 'repo')
    assert saved.launch.think is False and saved.launch.model == 'deepseek-v4-pro'
    assert saved.launch.sources.get('credentials') == 'login'
    reloaded = FileUserConfigStore(store.root).load()
    assert reloaded.credential_source == 'file' and reloaded.credential.reveal() == 'new-sentinel'
    assert reloaded.defaults.think is False


async def test_keep_with_identical_settings_is_noop_without_epoch_or_anchor_loss(tmp_path):
    recorded = []
    session, store = await open_session(tmp_path, transport=recorded_transport(recorded))
    try:
        await session.execute('先建立锚')
        assert session.budget.anchor is not None
        before = session.persistence
        updates = []
        session.subscribe(updates.append)
        result = await session.configure_login(submission(session.login_view()))
        assert result.defaults_saved and not result.session_applied
        assert result.application_outcome == 'unchanged'
        assert session.persistence == before and session.budget.anchor is not None
        assert not [u for u in updates if isinstance(u, ConfigurationChanged)]
        assert FileUserConfigStore(store.root).load().defaults.think  # 默认值仍已保存
    finally:
        await session.aclose()


async def test_same_model_think_change_and_pure_replace_each_write_epoch(tmp_path):
    recorded = []
    session, store = await open_session(tmp_path, transport=recorded_transport(recorded))
    try:
        first = session.persistence.epoch_id
        result = await session.configure_login(submission(session.login_view(), think=False))
        assert result.application_outcome == 'applied'
        second = session.persistence.epoch_id
        assert second != first
        assert 'credentials' not in session.config.persistence.identity.sources
        result = await session.configure_login(
            submission(session.login_view(), action='replace', credential='other-sentinel'))
        assert result.application_outcome == 'applied'
        assert session.persistence.epoch_id not in (first, second)
        assert session.config.persistence.identity.sources['credentials'] == 'login'
        await session.execute('验证凭据')
        assert recorded[-1][0] == 'Bearer other-sentinel'
    finally:
        await session.aclose()


async def test_history_incompatible_with_thinking_blocks_apply_but_saves_defaults(tmp_path):
    recorded = []
    root = _repo(tmp_path)
    store = FileUserConfigStore(tmp_path / 'user')

    def handle(req):
        body = json.loads(req.content)
        recorded.append(body)
        assert body['thinking'] == {'type': 'disabled'}
        if len(recorded) == 1:
            return httpx.Response(200, content=response(None, '使用工具', calls=(
                ('call-1', 'write', {'path': 'note.txt', 'content': '甲'}),)))
        return httpx.Response(200, content=response(None, '完成'))

    session = await open_application(
        config(root, httpx.MockTransport(handle), None, DeepSeekConfig(think=False)),
        login_store_factory=lambda: store, login_environment=lambda: 'environment-sentinel')
    try:
        await session.execute('写入文件')  # 非思考工具历史: assistant 无 reasoning
        before = (session.persistence.epoch_id, session.history)
        view = session.login_view()
        assert not view.think and 'save-and-apply' in view.actions
        result = await session.configure_login(submission(view, think=True))
        assert result.defaults_saved and not result.session_applied
        assert result.application_outcome == 'blocked' and result.error_code == 'history-incompatible'
        assert session.persistence.epoch_id == before[0] and session.history == before[1]
        assert session._model_route.binding.identity.think is False  # 旧运行路由保留
        await session.execute('旧绑定仍可续写')
        assert recorded[-1]['thinking'] == {'type': 'disabled'}
        assert FileUserConfigStore(store.root).load().defaults.think  # 已保存事实单独成立
    finally:
        await session.aclose()


@pytest.mark.parametrize('latch', ['view-only', 'storage-failed', 'process-uncertain'])
async def test_blocked_sessions_do_not_offer_or_accept_apply(tmp_path, latch):
    session, store = await open_session(tmp_path)
    if latch == 'view-only':
        session._view_only_reason = 'restored-unproven'
    elif latch == 'storage-failed':
        session._storage_failed = 'controlled-storage-failure'
    else:
        session._process_uncertain = True
    try:
        view = session.login_view()
        assert view.actions == ('defaults-only',)
        with pytest.raises(UserConfigError, match='unsupported-target'):
            await session.configure_login(submission(view))
        result = await session.configure_login(submission(view, target='defaults-only'))
        assert result.defaults_saved and result.application_outcome == 'not-requested'
    finally:
        await session.aclose()


async def test_cancel_after_save_before_acceptance_writes_zero_epoch(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    atomic = files._atomic_replace

    def pause(path, raw, cancelled):
        atomic(path, raw, cancelled)
        if path.name == 'config.toml':
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr(files, '_atomic_replace', pause)
    session, store = await open_session(tmp_path)
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(args)
        raise AssertionError('cancelled apply must not reach the epoch transaction')

    monkeypatch.setattr(session._store, 'submit_model_change', forbidden)
    try:
        before = session.persistence.epoch_id
        view = session.login_view()
        task = asyncio.create_task(session.configure_login(submission(view, think=False)))
        await wait_for_thread(entered)  # config 已发布, epoch 事务尚未接受
        task.cancel()
        release.set()
        result = await task
        assert result.defaults_saved and not result.session_applied
        assert result.application_outcome == 'blocked' and result.error_code == 'cancelled'
        assert session.persistence.epoch_id == before and forbidden_calls == []
        assert session._model_route.binding.identity.think  # 旧绑定保留
    finally:
        release.set()
        await session.aclose()


async def test_close_waits_for_accepted_apply_commit_and_publication(tmp_path, monkeypatch):
    session, store = await open_session(tmp_path)
    accepted, release = asyncio.Event(), asyncio.Event()
    real_submit = session._store.submit_model_change
    calls = []

    def submit(*args, **kwargs):
        calls.append(args)

        class Delayed:
            async def wait(self):
                accepted.set()
                await release.wait()
                return 'accepted-epoch'

        return Delayed()

    monkeypatch.setattr(session._store, 'submit_model_change', submit)
    try:
        view = session.login_view()
        task = asyncio.create_task(session.configure_login(
            submission(view, action='replace', credential='close-sentinel', think=False)))
        await asyncio.wait_for(accepted.wait(), 5)
        assert calls  # 事务已被接受: close 必须等同一生产者落定
        closing = asyncio.create_task(session.aclose())
        await asyncio.sleep(0.1)
        assert not closing.done() and not session._closed
        with pytest.raises(RuntimeError, match='progress'):
            await session.execute('应用窗口内拒绝请求')
        release.set()
        result = await task
        await asyncio.wait_for(closing, 5)
        assert result.session_applied and result.application_outcome == 'applied'
        assert session._epoch_id == 'accepted-epoch'
        assert session._model_route.binding.identity.think is False
        assert session.state == 'closed'
    finally:
        release.set()
        if not session._closed:
            await session.aclose()


async def test_apply_storage_failure_keeps_old_route_and_latches_storage(tmp_path):
    session, store = await open_session(tmp_path)
    try:
        before = (session.persistence.epoch_id, session._model_route.binding.identity)
        session._store._conn.execute(
            "CREATE TRIGGER reject_epoch BEFORE INSERT ON epoch BEGIN SELECT RAISE(ABORT, 'injected'); END")
        session._store._conn.commit()
        result = await session.configure_login(submission(session.login_view(), think=False))
        assert result.defaults_saved and not result.session_applied
        assert result.application_outcome == 'failed' and result.error_code == 'storage-failed'
        assert session.persistence.epoch_id == before[0]
        assert session._model_route.binding.identity == before[1]
        assert session.persistence.storage_failed is not None
        with pytest.raises(RuntimeError, match='storage_failed'):
            await session.execute('锁存拒绝新请求')
        assert not FileUserConfigStore(store.root).load().defaults.think  # 保存事实不回滚
        rows = sqlite3.connect(session._store.db_path).execute(
            'SELECT COUNT(*) FROM epoch').fetchone()
        assert rows[0] == 1  # 失败事务全回滚, 无半截 epoch
    finally:
        await session.aclose()


async def test_apply_after_model_change_prepares_from_latest_binding(tmp_path):
    recorded = []
    session, store = await open_session(tmp_path, transport=recorded_transport(recorded))
    try:
        await session.select_model('deepseek-v4-pro')
        view = session.login_view()
        assert view.model == 'deepseek-v4-pro'  # 初值来自最新绑定, 非启动缓存
        result = await session.configure_login(submission(view, think=False))
        assert result.application_outcome == 'applied'
        await session.execute('验证模型与凭据')
        authorization, body = recorded[-1]
        assert authorization == 'Bearer offline-test-key'  # keep 保留当前已接受 key
        assert body['model'] == 'deepseek-v4-pro' and body['thinking'] == {'type': 'disabled'}
    finally:
        await session.aclose()


async def test_ephemeral_session_applies_without_epoch(tmp_path):
    recorded = []
    root = _repo(tmp_path)
    store = FileUserConfigStore(tmp_path / 'user')
    initial = replace(config(root, recorded_transport(recorded)), persistence=None)
    session = await open_application(initial, login_store_factory=lambda: store,
                                     login_environment=lambda: 'environment-sentinel')
    try:
        view = session.login_view()
        assert 'save-and-apply' in view.actions
        result = await session.configure_login(
            submission(view, action='replace', credential='ephemeral-sentinel'))
        assert result.application_outcome == 'applied' and session.persistence is None
        await session.execute('无持久会话的应用后请求')
        assert recorded[-1][0] == 'Bearer ephemeral-sentinel'
    finally:
        await session.aclose()
