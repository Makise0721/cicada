"""登录经真实应用、用户文件和 SQLite；HTTP 仅在系统边界受控。"""

from dataclasses import replace
import asyncio
import threading

import httpx
import pytest

from cicada.application import open_application
from cicada.plugins.user_config import FileUserConfigStore
from cicada.user_config import LoginSubmission, SecretValue, UserDefaults
from cicada.plugins.user_config import store as files
from test_application import _repo
from test_deepseek_application import config, response


async def test_save_defaults_preserves_current_session_and_reopens_new_values(tmp_path):
    root = _repo(tmp_path)
    store = FileUserConfigStore(tmp_path / 'user')
    session = await open_application(config(root, httpx.MockTransport(
        lambda _: httpx.Response(200, content=response('thought')))),
        login_store_factory=lambda: store, login_environment=lambda: 'environment-sentinel')
    try:
        await session.select_model('deepseek-v4-pro')
        before = session.persistence
        budget = session.budget
        updates = []
        session.subscribe(updates.append)
        view = session.login_view()
        assert view.model == 'deepseek-v4-pro' and view.think
        submission = LoginSubmission(UserDefaults(model=view.model, think=False),
                                     'replace', view.revision, credential=SecretValue('new-sentinel'))
        result = await session.configure_login(submission)
        assert result.defaults_saved and not result.session_applied
        assert session.persistence == before and session.budget == budget
        assert session.config.model_label == 'deepseek-v4-pro' and session.history == ()
        snapshot = FileUserConfigStore(store.root).load()
        assert snapshot.defaults.model == 'deepseek-v4-pro' and not snapshot.defaults.think
        assert snapshot.credential.reveal() == 'new-sentinel'
        for forbidden in ('new-sentinel', snapshot.defaults.credential_ref, view.revision):
            assert forbidden not in repr(updates) + repr(session.history)
            assert forbidden.encode() not in session._store.capture().db_path.read_bytes()
    finally:
        await session.aclose()


async def open_session(tmp_path, *, store=None, environment=lambda: 'environment-sentinel', transport=None):
    store = store or FileUserConfigStore(tmp_path / 'user')
    session = await open_application(config(_repo(tmp_path), transport or httpx.MockTransport(
        lambda _: httpx.Response(200, content=response('thought')))),
        login_store_factory=lambda: store, login_environment=environment)
    return session, store


def keep(view):
    return LoginSubmission(UserDefaults(model=view.model, think=view.think,
        reasoning_effort=view.reasoning_effort, num_ctx=view.num_ctx,
        max_output_tokens=view.max_output_tokens), 'keep', view.revision)


@pytest.mark.parametrize('operation', ['execute', 'clear', 'compact', 'model', 'login'])
async def test_login_editing_blocks_application_operations_before_save(tmp_path, operation):
    session, store = await open_session(tmp_path)
    try:
        session.login_view()
        with pytest.raises((RuntimeError, ValueError), match='progress'):
            if operation == 'execute':
                await session.execute('should not start')
            elif operation == 'clear':
                await session.clear_context()
            elif operation == 'compact':
                await session.compact_context()
            elif operation == 'model':
                await session.select_model('deepseek-v4-pro')
            else:
                session.login_view()
        assert session.history == () and not store.root.exists()
        session.cancel_login()
        assert (await session.execute('now allowed')).exit_code == 0
    finally:
        await session.aclose()


@pytest.mark.parametrize('latch', ['view-only', 'storage-failed', 'process-uncertain'])
async def test_save_defaults_preserves_blocked_session_latches(tmp_path, latch):
    session, store = await open_session(tmp_path)
    # Controlled application fault states, as specified by the login acceptance seam.
    if latch == 'view-only':
        session._view_only_reason = 'restored-unproven'
    elif latch == 'storage-failed':
        session._storage_failed = 'controlled-storage-failure'
    else:
        session._process_uncertain = True
    before = (session.view_only_reason, session._storage_failed, session.has_process_uncertain())
    try:
        result = await session.configure_login(keep(session.login_view()))
        assert result.defaults_saved and not result.session_applied
        assert (session.view_only_reason, session._storage_failed, session.has_process_uncertain()) == before
        assert not (store.root / 'credentials.json').exists()
    finally:
        await session.aclose()


async def test_config_failure_keeps_session_storage_and_route_usable(tmp_path):
    session, store = await open_session(tmp_path)
    try:
        submission = keep(session.login_view())
        store.root.mkdir()
        (store.root / 'config.toml').write_text('version=1\n', encoding='utf-8')
        result = await session.configure_login(submission)
        assert not result.defaults_saved and result.error_code == 'configuration-changed'
        assert session.persistence.storage_failed is None
        assert (await session.execute('still usable')).exit_code == 0
    finally:
        await session.aclose()


async def wait_for_thread(event):
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize('published', [False, True])
@pytest.mark.parametrize('action', ['cancel', 'close'])
async def test_cancel_and_close_wait_for_file_worker_true_publication(tmp_path, monkeypatch, published, action):
    entered, release = threading.Event(), threading.Event()
    atomic = files._atomic_replace

    def pause(path, raw, cancelled):
        if path.name != 'config.toml':
            return atomic(path, raw, cancelled)
        if published:
            atomic(path, raw, cancelled)
        entered.set()
        assert release.wait(10)
        if not published:
            atomic(path, raw, cancelled)

    monkeypatch.setattr(files, '_atomic_replace', pause)
    session, store = await open_session(tmp_path)
    task = asyncio.create_task(session.configure_login(keep(session.login_view())))
    closing = None
    try:
        await wait_for_thread(entered)
        if action == 'cancel':
            task.cancel()
            task.cancel()
        else:
            closing = asyncio.create_task(session.aclose())
        await asyncio.sleep(0.05)
        assert not task.done() and session.login_busy
        if closing:
            assert not closing.done()
        release.set()
        result = await asyncio.wait_for(task, 5)
        assert result.defaults_saved is published and not result.session_applied
        assert (store.root / 'config.toml').exists() is published
        assert not list(store.root.glob('*.tmp'))
        if closing:
            await closing
            assert session.state == 'closed'
        else:
            assert (await session.execute('after cancellation')).exit_code == 0
    finally:
        release.set()
        await session.aclose()


async def test_execute_reservation_refuses_login_before_http_finishes(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    async def handle(request):
        entered.set()
        await release.wait()
        return httpx.Response(200, content=response('thought'))
    session, store = await open_session(tmp_path, transport=httpx.MockTransport(handle))
    running = asyncio.create_task(session.execute('running'))
    try:
        await entered.wait()
        with pytest.raises(ValueError, match='operation-in-progress'):
            session.login_view()
        assert not store.root.exists()
    finally:
        release.set()
        await running
        await session.aclose()


async def test_login_status_reads_latest_effective_binding_without_file_or_key_reads(tmp_path):
    reads = []
    def environment():
        reads.append(True)
        return 'status-sentinel'
    session, store = await open_session(tmp_path, environment=environment)
    try:
        session.login_view()
        session.cancel_login()
        await session.select_model('deepseek-v4-pro')
        before = session.persistence
        assert 'deepseek-v4-pro/high' in ' '.join(session.login_status)
        assert reads == [True] and not store.root.exists()
        assert session.persistence == before
        result = await session.configure_login(replace(keep(session.login_view()),
            defaults=UserDefaults(model='deepseek-v4-pro', think=False)))
        assert result.defaults_saved
        assert 'deepseek-v4-pro/high' in ' '.join(session.login_status)
        assert 'status-sentinel' not in repr(session.login_status)
    finally:
        await session.aclose()
