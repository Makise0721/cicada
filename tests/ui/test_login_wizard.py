"""共享向导通过生产 Terminal 宿主和实际 VT 验证。"""

import asyncio
import threading
import pytest

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.plugins.user_config import FileUserConfigStore
from cicada.ui.login_wizard import run_startup_login
from cicada.ui.terminal_io import FakeTerminal
from cicada.user_config import LoginSubmission, SecretValue
from cicada.plugins.user_config import store as files
from vt_screen import VtScreen

ENTER = KeyPress(Keys.ControlM, '\r')


async def test_first_run_masks_secret_and_saves_once(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal()
    screen = VtScreen(columns=80, rows=24)
    terminal.feed_screen = screen.feed
    task = asyncio.create_task(run_startup_login(terminal, store, store.read_view()))
    await asyncio.sleep(0)
    terminal.emit([ENTER])
    terminal.emit([KeyPress(Keys.BracketedPaste, 'wizard-sentinel')])
    assert 'wizard-sentinel' not in terminal.output_text
    assert '*' in '\n'.join(screen.screen_text())
    terminal.emit([ENTER, ENTER, ENTER, ENTER])
    result, errors = await asyncio.wait_for(task, 5)
    assert result.defaults_saved and errors == []
    assert store.load().credential.reveal() == 'wizard-sentinel'
    assert terminal.started and terminal.stopped
    assert 'wizard-sentinel' not in terminal.output_text


async def start(terminal, store, **kwargs):
    view = store.read_view(kwargs.pop('environment_key', None))
    task = asyncio.create_task(run_startup_login(terminal, store, view, **kwargs))
    await asyncio.sleep(0)
    return task


@pytest.mark.parametrize('paste', ['first\nsecond', '\nwrapped\n', 'two words', 'control\x1bsecret', 'x' * 4097])
async def test_invalid_paste_stays_masked_in_wizard_and_can_cancel(tmp_path, paste):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal()
    task = await start(terminal, store)
    terminal.emit([ENTER, KeyPress(Keys.BracketedPaste, paste), ENTER])
    assert not task.done() and 'API key:' in terminal.output_text
    assert paste not in terminal.output_text
    terminal.emit([KeyPress(Keys.ControlC, '\x03')])
    result, errors = await task
    assert result is None and errors == [] and not store.root.exists()


async def test_existing_environment_keep_displays_source_without_copying_secret(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal()
    task = await start(terminal, store, environment_key='environment-sentinel')
    terminal.emit([ENTER])
    assert '环境变量' in terminal.output_text and '保留凭据' in terminal.output_text
    assert 'environment-sentinel' not in terminal.output_text
    terminal.emit([ENTER, ENTER, ENTER])
    result, _ = await task
    assert result.defaults_saved and not (store.root / 'credentials.json').exists()
    assert store.load().credential is None


async def test_existing_file_replace_can_go_back_without_prefill(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    snapshot = store.load()
    await store.save(LoginSubmission(
        snapshot.defaults, 'replace', snapshot.revision, credential=SecretValue('old-sentinel')))
    terminal = FakeTerminal()
    task = await start(terminal, store)
    terminal.emit([ENTER, KeyPress(Keys.Down, ''), KeyPress('n', 'n'), KeyPress('e', 'e'),
                   KeyPress('w', 'w'), KeyPress(Keys.ControlH, '\b'), KeyPress('x', 'x'), ENTER])
    terminal.emit([KeyPress(Keys.ControlI, '\t')])
    assert 'old-sentinel' not in terminal.output_text and 'nex' not in terminal.output_text
    terminal.emit([KeyPress(Keys.Down, ''), KeyPress(Keys.BracketedPaste, 'replacement-sentinel'),
                   ENTER, ENTER, ENTER])
    result, _ = await task
    assert result.defaults_saved and store.load().credential.reveal() == 'replacement-sentinel'


async def test_narrow_window_resize_restores_cursor_and_does_not_scroll_secret(tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal(columns=18, rows=6)
    screen = VtScreen(columns=18, rows=6)
    terminal.feed_screen = screen.feed
    task = await start(terminal, store)
    terminal.emit([ENTER, KeyPress(Keys.BracketedPaste, 'narrow-sentinel')])
    assert not screen.cursor_visible and '*' in '\n'.join(screen.screen_text())
    resized = VtScreen(columns=90, rows=24)
    terminal.feed_screen = resized.feed
    terminal.set_size(90, 24)
    terminal.emit([ENTER, ENTER])
    assert '确认保存' in '\n'.join(resized.screen_text())
    terminal.emit([KeyPress(Keys.Escape, '\x1b')])
    result, errors = await task
    assert result is None and errors == [] and resized.cursor_visible
    assert 'narrow-sentinel' not in '\n'.join(screen.history_text() + resized.history_text())
    assert not store.root.exists()


@pytest.mark.parametrize('phase', ['start', 'edit', 'resize'])
async def test_terminal_failure_clears_input_and_restores_modes(tmp_path, phase):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal()
    if phase == 'start':
        terminal.fail_write = True
    task = await start(terminal, store)
    if phase != 'start':
        terminal.emit([ENTER, KeyPress(Keys.BracketedPaste, 'failure-sentinel')])
        terminal.fail_write = True
        if phase == 'edit':
            terminal.emit([KeyPress('x', 'x')])
        else:
            terminal.set_size(30, 8)
    result, errors = await task
    assert not result.defaults_saved and result.error_code == 'terminal-failed'
    assert errors and terminal.stopped and not store.root.exists()
    assert 'failure-sentinel' not in terminal.output_text + repr(result) + repr(errors)


async def test_submitting_silences_keys_and_close_waits_for_actual_file_result(monkeypatch, tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal()
    entered, release = threading.Event(), threading.Event()
    original = files.os.replace

    def pause(source, destination):
        original(source, destination)
        if destination.name == 'config.toml':
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(files.os, 'replace', pause)
    task = await start(terminal, store)
    terminal.emit([ENTER, KeyPress(Keys.BracketedPaste, 'close-sentinel'), ENTER, ENTER, ENTER, ENTER])
    async with asyncio.timeout(5):
        while not entered.is_set():
            await asyncio.sleep(0.005)
    try:
        terminal.emit([ENTER, KeyPress(Keys.Escape, ''), KeyPress(Keys.ControlC, '')])
        assert not task.done() and not terminal.stopped
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not terminal.stopped
    finally:
        release.set()
    result, _ = await asyncio.wait_for(task, 5)
    assert result.defaults_saved and terminal.stopped
    assert store.load().credential.reveal() == 'close-sentinel'


async def test_terminal_failure_during_save_waits_worker_before_stop(monkeypatch, tmp_path):
    store = FileUserConfigStore(tmp_path / 'user')
    terminal = FakeTerminal()
    entered, release = threading.Event(), threading.Event()
    original = files.os.replace

    def pause(source, destination):
        original(source, destination)
        if destination.name == 'credentials.json':
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(files.os, 'replace', pause)
    task = await start(terminal, store)
    terminal.emit([ENTER, KeyPress(Keys.BracketedPaste, 'fault-sentinel'), ENTER, ENTER, ENTER])
    async with asyncio.timeout(5):
        while not entered.is_set():
            await asyncio.sleep(0.005)
    try:
        terminal.fail_write = True
        terminal.set_size(30, 8)
        await asyncio.sleep(0.01)
        assert not task.done() and not terminal.stopped
    finally:
        release.set()
    result, errors = await task
    assert not result.defaults_saved and result.error_code == 'terminal-failed'
    assert terminal.stopped and errors and store.load().credential is None
    assert not list(store.root.glob('*.tmp'))
