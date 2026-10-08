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
from cicada.user_config import LoginSubmission, SecretValue, UserDefaults


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
