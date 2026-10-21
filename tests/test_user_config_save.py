"""可靠保存的公开文件接缝：重开只选 config 引用的凭据。"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from cicada.plugins.user_config import FileUserConfigStore
from cicada.plugins.user_config import store as files
from cicada.user_config import (
    LoginSubmission, SecretValue, UserConfigError, UserDefaults, ModelProfile, ProviderProfile,
)


def test_missing_config_is_initialized_once_without_credentials(tmp_path):
    root = tmp_path / 'user'
    store = FileUserConfigStore(root)
    assert store.ensure_default_config()
    first = store.load()
    path = root / 'config.toml'
    expected = files._config_bytes(UserDefaults())
    assert path.read_bytes() == expected
    assert first.defaults == UserDefaults() and first.fields == files.DEFAULT_FIELDS
    assert first.credential is None and not (root / 'credentials.json').exists()
    path.write_bytes(expected + b'\n')
    assert not store.ensure_default_config()
    assert store.load().defaults == UserDefaults()
    assert path.read_bytes() == expected + b'\n'


def test_default_creation_keeps_config_created_by_another_writer(monkeypatch, tmp_path):
    root = tmp_path / 'user'
    path = root / 'config.toml'

    def competing_publish(source, target):
        assert Path(target) == path
        path.write_text('version = 1\n[defaults]\nthink = false\n', encoding='utf-8')
        raise FileExistsError

    monkeypatch.setattr(os, 'link', competing_publish)
    store = FileUserConfigStore(root)
    assert store.ensure_default_config()
    snapshot = store.load()
    assert not snapshot.defaults.think and path.read_text(encoding='utf-8').endswith('think = false\n')
    assert not list(root.glob('*.tmp'))


def test_default_creation_reports_write_failure_without_partial_config(monkeypatch, tmp_path):
    root = tmp_path / 'user'
    monkeypatch.setattr(os, 'link', lambda *_: (_ for _ in ()).throw(OSError('private path')))
    with pytest.raises(UserConfigError, match='config: write-failed'):
        FileUserConfigStore(root).ensure_default_config()
    assert not (root / 'config.toml').exists() and not list(root.glob('*.tmp'))


def test_default_creation_cleanup_failure_has_fixed_error(monkeypatch, tmp_path):
    root = tmp_path / 'user'
    original_unlink = Path.unlink

    def blocked_unlink(path, *args, **kwargs):
        if path.suffix == '.tmp':
            raise OSError('private path')
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', blocked_unlink)
    with pytest.raises(UserConfigError, match='config: write-failed') as error:
        FileUserConfigStore(root).ensure_default_config()
    assert 'private path' not in str(error.value)
    assert (root / 'config.toml').read_bytes() == files._config_bytes(UserDefaults())


async def seed(root, key='old-sentinel'):
    store = FileUserConfigStore(root)
    snapshot = store.load()
    result = await store.save(LoginSubmission(UserDefaults(), 'replace', snapshot.revision,
                                             credential=SecretValue(key)))
    assert result.defaults_saved
    return store


def replacing(snapshot, key='new-sentinel'):
    return LoginSubmission(replace(snapshot.defaults, think=False), 'replace', snapshot.revision,
                           credential=SecretValue(key))


async def test_replace_publishes_defaults_and_referenced_secret(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    before = store.load()
    result = await store.save(LoginSubmission(
        UserDefaults(think=False, num_ctx=65536, max_output_tokens=4096),
        'replace', before.revision, credential=SecretValue('save-sentinel')))
    assert result.defaults_saved and not result.session_applied
    after = FileUserConfigStore(store.root).load()
    assert after.credential.reveal() == 'save-sentinel'
    assert not after.defaults.think and after.defaults.num_ctx == 65536
    assert after.defaults.max_output_tokens == 4096
    assert b'save-sentinel' not in (store.root / 'config.toml').read_bytes()
    assert not list(store.root.glob('*.tmp'))


@pytest.mark.parametrize('point', ['credentials', 'config', 'fsync'])
@pytest.mark.parametrize('existing', [False, True])
async def test_write_fault_keeps_old_reference_or_first_run_missing(monkeypatch, tmp_path, point, existing):
    root = tmp_path / 'user'
    store = await seed(root) if existing else FileUserConfigStore(root)
    snapshot = store.load()
    original = os.replace

    def replace_fault(source, destination):
        if Path(destination).name == point + '.json' or (point == 'config' and Path(destination).name == 'config.toml'):
            raise OSError('new-sentinel injected secret must stay private')
        original(source, destination)

    if point == 'fsync':
        monkeypatch.setattr(os, 'fsync', lambda _: (_ for _ in ()).throw(OSError('new-sentinel')))
    else:
        monkeypatch.setattr(os, 'replace', replace_fault)
    result = await store.save(replacing(snapshot))
    assert not result.defaults_saved and result.error_code == 'write-failed'
    after = FileUserConfigStore(root).load()
    assert after.credential.reveal() == 'old-sentinel' if existing else after.credential is None
    assert after.defaults.think and not list(root.glob('*.tmp'))
    assert 'new-sentinel' not in repr(result)


async def test_keep_environment_never_reads_or_modifies_unused_credentials(monkeypatch, tmp_path):
    store = await seed(tmp_path / 'user')
    path = store.root / 'credentials.json'
    path.write_bytes(b'bad-secret-sentinel')
    snapshot = store.load('environment-sentinel')
    original_open = Path.open

    def guarded(path_arg, *args, **kwargs):
        if path_arg == path:
            pytest.fail('unselected credentials accessed')
        return original_open(path_arg, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    result = await store.save(LoginSubmission(replace(snapshot.defaults, think=False), 'keep', snapshot.revision))
    assert result.defaults_saved
    after = store.load('environment-sentinel')
    assert not after.defaults.think and after.defaults.credential_ref == snapshot.defaults.credential_ref
    assert 'environment-sentinel' not in (store.root / 'config.toml').read_text(encoding='utf-8')
    with original_open(path, 'rb') as stream:
        assert stream.read() == b'bad-secret-sentinel'


async def test_keep_environment_without_reference_does_not_copy_secret(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    snapshot = store.load('environment-sentinel')
    result = await store.save(LoginSubmission(snapshot.defaults, 'keep', snapshot.revision))
    assert result.defaults_saved and not (store.root / 'credentials.json').exists()
    assert FileUserConfigStore(store.root).load().credential is None


@pytest.mark.parametrize('changed', ['config.toml', 'credentials.json'])
async def test_stale_observation_cannot_overwrite_manual_changes(tmp_path, changed):
    store = await seed(tmp_path / 'user')
    snapshot = store.load()
    path = store.root / changed
    path.write_bytes(path.read_bytes() + b'\n')
    expected = path.read_bytes()
    result = await store.save(replacing(snapshot))
    assert not result.defaults_saved and result.error_code == 'configuration-changed'
    assert path.read_bytes() == expected
    assert FileUserConfigStore(store.root).load().credential.reveal() == 'old-sentinel'


async def test_explicit_replace_refuses_broken_old_reference_even_with_environment(tmp_path):
    store = await seed(tmp_path / 'user')
    (store.root / 'credentials.json').write_bytes(b'broken')
    snapshot = store.load('env-sentinel')
    before = (store.root / 'config.toml').read_bytes()
    result = await store.save(replacing(snapshot))
    assert not result.defaults_saved and result.error_code == 'invalid-format'
    assert (store.root / 'config.toml').read_bytes() == before


async def test_replacements_prune_only_previously_unreferenced_entries(tmp_path):
    store = await seed(tmp_path / 'user')
    for key in ('second-sentinel', 'third-sentinel'):
        assert (await store.save(replacing(store.load(), key))).defaults_saved
    entries = json.loads((store.root / 'credentials.json').read_bytes())['entries']
    assert {e['api_key'] for e in entries.values()} == {'second-sentinel', 'third-sentinel'}
    assert store.load().credential.reveal() == 'third-sentinel'


async def test_slow_reader_survives_two_real_saves_pruning_initial_reference(monkeypatch, tmp_path):
    writer = await seed(tmp_path / 'user')
    reader = FileUserConfigStore(writer.root)
    entered, release = threading.Event(), threading.Event()
    original = files._read
    reader_thread = None

    def paused(path, kind):
        nonlocal reader_thread
        if kind == 'credentials' and reader_thread is None:
            reader_thread = threading.get_ident()
            entered.set()
            assert release.wait(5)
        return original(path, kind)

    monkeypatch.setattr(files, '_read', paused)
    reading = asyncio.create_task(asyncio.to_thread(reader.load))
    await wait_event(entered)
    try:
        for key in ('second-sentinel', 'third-sentinel'):
            assert (await writer.save(replacing(writer.load(), key))).defaults_saved
    finally:
        release.set()
    snapshot = await reading
    assert snapshot.credential.reveal() == 'third-sentinel'


async def wait_event(event):
    async with asyncio.timeout(5):
        while not event.is_set():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize('published', [False, True])
async def test_repeated_cancel_waits_for_worker_and_reports_actual_publication(monkeypatch, tmp_path, published):
    store = await seed(tmp_path / 'user')
    entered, release = threading.Event(), threading.Event()
    original = os.replace

    def paused(source, destination):
        if Path(destination).name == ('config.toml' if published else 'credentials.json'):
            original(source, destination)
            entered.set()
            assert release.wait(5)
        else:
            original(source, destination)

    monkeypatch.setattr(os, 'replace', paused)
    task = asyncio.create_task(store.save(replacing(store.load())))
    await wait_event(entered)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        busy = await store.save(replacing(store.load()))
        assert busy.error_code == 'operation-in-progress'
        # 另一 Adapter 也受同一个 OS 锁保护。
        competitor = FileUserConfigStore(store.root)
        second = asyncio.create_task(competitor.save(replacing(competitor.load(), 'other-sentinel')))
        await asyncio.sleep(0.03)
        assert not second.done() and not task.done()
        second.cancel()
        assert not (await second).defaults_saved
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)
    assert result.defaults_saved == published
    after = FileUserConfigStore(store.root).load()
    assert after.credential.reveal() == ('new-sentinel' if published else 'old-sentinel')
    assert not list(store.root.glob('*.tmp'))


async def test_lock_wait_has_two_second_bound_and_crash_releases_os_lock(tmp_path):
    store = await seed(tmp_path / 'user')
    snapshot = store.load()
    lock_path = store.root / 'config.lock'
    # 使用真正的另进程 OS lock，不能用同一对象的 busy 标记替代。
    script = ('import os,msvcrt,sys,time; fd=os.open(sys.argv[1],os.O_RDWR); '
              'msvcrt.locking(fd,msvcrt.LK_NBLCK,1); print("ready",flush=True); time.sleep(20)')
    process = subprocess.Popen([sys.executable, '-c', script, str(lock_path)], stdout=subprocess.PIPE)
    try:
        assert await asyncio.to_thread(process.stdout.readline) == b'ready\r\n'
        started = time.monotonic()
        result = await store.save(replacing(snapshot))
        elapsed = time.monotonic() - started
        assert result.error_code == 'lock-timeout' and 1.8 <= elapsed < 4
    finally:
        process.kill()
        await asyncio.to_thread(process.wait)
        process.stdout.close()
    assert (await store.save(replacing(snapshot))).defaults_saved


@pytest.mark.parametrize('point', ['credentials.json', 'config.toml'])
@pytest.mark.parametrize('existing', [False, True])
async def test_process_exit_at_replace_reopens_by_reference(tmp_path, point, existing):
    root = tmp_path / 'user'
    if existing:
        await seed(root)
    script = '''import asyncio,os,sys
from pathlib import Path
from cicada.plugins.user_config import FileUserConfigStore
from cicada.user_config import LoginSubmission,SecretValue,UserDefaults
root=Path(sys.argv[1]); point=sys.argv[2]; original=os.replace
def crash(source,destination):
    original(source,destination)
    if Path(destination).name == point: os._exit(17)
os.replace=crash
store=FileUserConfigStore(root); snapshot=store.load()
asyncio.run(store.save(LoginSubmission(UserDefaults(think=False),'replace',snapshot.revision,credential=SecretValue('crash-sentinel'))))
'''
    environment = dict(os.environ, PYTHONPATH=str(Path('src').resolve()))
    completed = await asyncio.to_thread(subprocess.run,
        [sys.executable, '-c', script, str(root), point], env=environment, capture_output=True, timeout=10)
    assert completed.returncode == 17, completed.stderr
    reopened = FileUserConfigStore(root)
    after = reopened.load()
    if point == 'config.toml':
        assert after.credential.reveal() == 'crash-sentinel' and not after.defaults.think
    elif existing:
        assert after.credential.reveal() == 'old-sentinel' and after.defaults.think
    else:
        assert after.credential is None
    assert not list(root.glob('*.tmp'))
    assert (await reopened.save(replacing(after))).defaults_saved


def gateway(*, api_key_env=None):
    return ProviderProfile('work.gateway', '工作网关', 'openai-chat-completions',
                           'https://gateway.example/v1',
                           (ModelProfile('coder.v1', 'org/remote-model', context_window=65536,
                                         max_output_tokens=4096),), 'coder.v1', api_key_env=api_key_env)


async def test_profile_save_preserves_other_provider_credentials_and_default(monkeypatch, tmp_path):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    store = await seed(tmp_path / 'user')
    initial = store.load()
    chosen = UserDefaults('work.gateway', 'coder.v1', False, 'high', 65536, 4096)
    saved = await store.save(LoginSubmission(chosen, 'replace', initial.revision,
        credential=SecretValue('work-sentinel'), provider_profile=gateway(), set_default=False))
    assert saved.defaults_saved
    assert store.load().defaults.provider == 'deepseek'
    assert store.load().credential.reveal() == 'old-sentinel'
    selected = store.load(provider_id='work.gateway')
    assert selected.defaults == replace(chosen, credential_ref=selected.defaults.credential_ref)
    assert selected.credential.reveal() == 'work-sentinel'
    profile = next(provider for provider in selected.providers if provider.id == 'work.gateway')
    assert profile.models[0].model_id == 'org/remote-model'
    view = store.read_view(provider_id='work.gateway')
    assert all(provider.credential_ref is None for provider in view.providers)
    assert 'work-sentinel' not in repr(view)
    assert (await store.save(LoginSubmission(chosen, 'keep', view.revision,
        provider_profile=gateway(), set_default=True))).defaults_saved
    assert store.load().defaults.provider == 'work.gateway'
    assert store.load().credential.reveal() == 'work-sentinel'
    assert store.load(provider_id='deepseek').credential.reveal() == 'old-sentinel'


async def test_profile_environment_is_selected_per_instance_and_invalid_does_not_fall_back(monkeypatch, tmp_path):
    store = await seed(tmp_path / 'user')
    initial = store.load()
    chosen = UserDefaults('work.gateway', 'coder.v1', False, 'high', 65536, 4096)
    assert (await store.save(LoginSubmission(chosen, 'replace', initial.revision,
        credential=SecretValue('work-file-sentinel'),
        provider_profile=gateway(api_key_env='WORK_LLM_KEY')))).defaults_saved
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-env-sentinel')
    monkeypatch.setenv('WORK_LLM_KEY', 'work-env-sentinel')
    assert store.load('wrong-global-sentinel').credential.reveal() == 'work-env-sentinel'
    assert store.load(provider_id='deepseek').credential.reveal() == 'deepseek-env-sentinel'
    with pytest.raises(UserConfigError, match='environment: invalid-key'):
        store.load('', provider_id='deepseek')
    monkeypatch.setenv('WORK_LLM_KEY', '')
    with pytest.raises(UserConfigError, match='environment: invalid-key'):
        store.load()
    monkeypatch.delenv('WORK_LLM_KEY')
    assert store.load().credential.reveal() == 'work-file-sentinel'


async def test_profile_credentials_keep_old_refs_and_oversize_save_publishes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    store = await seed(tmp_path / 'user')
    initial = store.load()
    provider = initial.providers[0]
    assert (await store.save(LoginSubmission(initial.defaults, 'replace', initial.revision,
        credential=SecretValue('second-sentinel'), provider_profile=provider))).defaults_saved
    for key in ('third-sentinel', 'fourth-sentinel'):
        current = store.load()
        assert (await store.save(replacing(current, key))).defaults_saved
    path = store.root / 'credentials.json'
    entries = json.loads(path.read_bytes())['entries']
    assert {entry['api_key'] for entry in entries.values()} == {
        'old-sentinel', 'second-sentinel', 'third-sentinel', 'fourth-sentinel'}
    # 文件已经接近上限时，下次凭据增长须在发布任何文件前拒绝。
    current = store.load()
    before_config = (store.root / 'config.toml').read_bytes()
    before_credentials = path.read_bytes()
    monkeypatch.setattr(files, 'FILE_LIMIT', max(len(before_config), len(before_credentials)))
    result = await store.save(replacing(current, 'x' * 4096))
    assert not result.defaults_saved and result.error_code == 'file-too-large'
    assert path.read_bytes() == before_credentials
    assert (store.root / 'config.toml').read_bytes() == before_config
