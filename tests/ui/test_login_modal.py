"""A07/A09/A10/A15：实际主屏 VT、文件保存、关闭 worker 与后继启动。"""

import asyncio
import threading
import json

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.application import open_application
from cicada.plugins.user_config import FileUserConfigStore
from cicada.plugins.user_config import store as files
from cicada.ui import run_tui_session
from cicada.ui.commands import SLASH_COMMANDS
from cicada.ui.main_screen import visible_width, strip_terminal_sequences
from cicada.ui.terminal_io import FakeTerminal
from test_application import _repo
from test_conversation_view import _config, _script
from test_login_defaults import open_session, wait_for_thread
from test_model_menu import TUIHarness, ENTER, ESCAPE, DOWN, text_keys
from tests.ui.vt_screen import VtScreen


PASTE = KeyPress(Keys.BracketedPaste, 'modal-sentinel')
CTRL_C = KeyPress(Keys.ControlC, '\x03')
CTRL_D = KeyPress(Keys.ControlD, '\x04')


async def settled(h):
    async with asyncio.timeout(5):
        while h.view.login_modal.is_open:
            await asyncio.sleep(0.01)
    await h.settle()


async def test_real_vt_login_replace_masks_secret_saves_once_and_preserves_draft(tmp_path):
    session, store = await open_session(tmp_path)
    h = TUIHarness(session, columns=70, rows=18)
    before = session.persistence
    try:
        await h.start()
        h.view.editor.set_text('原聊天草稿\n第二行')
        h.view.editor.move_cursor(0, -2)
        position = (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col)
        assert h.view.on_submit('/login')
        assert session.login_busy and h.view._gate.login_active
        assert not h.view.on_submit('must be refused')
        assert not h.view.on_submit('/clear') and not h.view.on_submit('/exit')
        # Secret paste isolated from ordinary editor, undo and transcript.
        h.terminal.emit([ENTER, DOWN, PASTE])
        await h.settle()
        assert h.find('API key: ********') and 'modal-sentinel' not in h.terminal.output_text
        assert h.view.editor.get_text() == '原聊天草稿\n第二行'
        h.terminal.emit([CTRL_D, KeyPress(Keys.ControlO, '\x0f')])
        assert h.exit_requested == 0
        h.terminal.emit([ENTER, DOWN, ENTER])  # high -> max
        await h.settle()
        assert h.find('下次新会话生效') and h.find('后续启动仍优先环境变量')
        assert h.find('保存思考：max')
        h.terminal.emit([ENTER, ENTER, ENTER])
        await settled(h)
        snapshot = FileUserConfigStore(store.root).load()
        assert snapshot.defaults.reasoning_effort == 'max'
        assert snapshot.credential.reveal() == 'modal-sentinel'
        assert len(json.loads((store.root / 'credentials.json').read_text())['entries']) == 1
        assert session.persistence == before and session.history == ()
        assert h.view.editor.get_text() == '原聊天草稿\n第二行'
        assert (h.view.editor.state.cursor_line, h.view.editor.state.cursor_col) == position
        h.view.on_submit('/status')
        h.view.on_submit('/help')
        await h.settle()
        for needle in ('modal-sentinel', snapshot.defaults.credential_ref, h.view._session._login_view.revision):
            assert needle not in h.terminal.output_text + repr(h.view.transcript) + repr(session.history)
            assert needle.encode() not in session._store.db_path.read_bytes()
    finally:
        await h.stop()
        await session.aclose()
        h.view.close()


@pytest.mark.parametrize('cancel', [ESCAPE, CTRL_C])
async def test_cancel_discards_remainder_of_input_batch_and_restores_cursor(tmp_path, cancel):
    session, store = await open_session(tmp_path)
    h = TUIHarness(session)
    try:
        await h.start()
        h.view.editor.set_text('draft')
        h.view.editor.move_cursor(0, -2)
        position = h.view.editor.state.cursor_col
        h.view.on_submit('/login')
        h.terminal.emit([ENTER, DOWN, PASTE, cancel, ENTER, PASTE, CTRL_D])
        await h.settle()
        assert not session.login_busy and not h.view.login_modal.is_open
        assert h.view.editor.get_text() == 'draft' and h.view.editor.state.cursor_col == position
        assert h.exit_requested == 0 and session.history == () and not store.root.exists()
        assert 'modal-sentinel' not in h.terminal.output_text
        assert '›' in h.screen()[h.vt.cursor_row]
    finally:
        await h.stop()
        await session.aclose()
        h.view.close()


async def test_slash_help_parameter_rejection_and_escape_are_local(tmp_path):
    session, store = await open_session(tmp_path)
    h = TUIHarness(session)
    try:
        await h.start()
        assert any(c.command == '/login' for c in SLASH_COMMANDS)
        h.terminal.emit(text_keys('/l'))
        await h.settle()
        assert h.find('login')
        h.terminal.emit([ENTER])
        assert h.view.login_modal.is_open
        h.terminal.emit([ESCAPE])
        h.view.on_submit('/login tail-secret')
        h.view.on_submit('/help')
        await h.settle()
        assert 'tail-secret' not in h.terminal.output_text
        assert '不接受参数' in strip_terminal_sequences(h.terminal.output_text)
        assert session.history == () and not store.root.exists()
        assert h.view.on_submit('//login')
        async with asyncio.timeout(5):
            while not session.history:
                await asyncio.sleep(0.01)
        assert session.history[0].text == '/login'
    finally:
        await h.stop()
        await session.aclose()
        h.view.close()


async def test_narrow_resize_and_invalid_paste_stay_in_modal(tmp_path):
    session, store = await open_session(tmp_path)
    h = TUIHarness(session, columns=24, rows=8)
    try:
        await h.start()
        h.view.on_submit('/login')
        h.terminal.emit([ENTER, DOWN, KeyPress(Keys.BracketedPaste, 'bad\nsecret'), ENTER])
        await h.settle()
        assert h.view.login_modal.wizard.step == 'credential'
        assert all(visible_width(row) <= 24 for row in h.screen())
        h.vt = VtScreen(45, 12)
        h.terminal.feed_screen = h.vt.feed
        h.terminal.set_size(45, 12)
        await h.settle()
        assert all(visible_width(row) <= 45 for row in h.screen())
        assert h.view.editor.is_empty() and session.history == ()
        assert 'secret' not in h.terminal.output_text
        h.terminal.emit([ESCAPE])
    finally:
        await h.stop()
        await session.aclose()
        h.view.close()


async def test_fake_configuration_is_lazy_until_explicit_login(tmp_path):
    reads = []
    store = FileUserConfigStore(tmp_path / 'user')
    def environment():
        reads.append(True)
        return 'env-sentinel'
    session = await open_application(_config(_repo(tmp_path), _script({'text': 'fake answer'})),
        login_store_factory=lambda: store, login_environment=environment)
    h = TUIHarness(session)
    try:
        await h.start()
        h.view.on_submit('/status')
        assert reads == []
        h.view.on_submit('/login')
        assert reads == [True] and h.view.login_modal.wizard.view.model == 'deepseek-flash'
        h.terminal.emit([ENTER, ENTER, ENTER, ENTER, ENTER])
        await settled(h)
        assert session.config.model_label == 'fake' and session.history == ()
        assert not (store.root / 'credentials.json').exists()
    finally:
        await h.stop()
        await session.aclose()
        h.view.close()


@pytest.mark.parametrize('failure', ['input', 'write', 'cancel-host'])
async def test_production_host_failure_waits_for_saved_result_and_one_terminal_owner(tmp_path, monkeypatch, failure):
    entered, release = threading.Event(), threading.Event()
    atomic = files._atomic_replace
    def pause(path, raw, cancelled):
        atomic(path, raw, cancelled)
        if path.name == 'config.toml':
            entered.set()
            assert release.wait(10)
    monkeypatch.setattr(files, '_atomic_replace', pause)
    session, store = await open_session(tmp_path)
    class CountingTerminal(FakeTerminal):
        starts = stops = 0
        def start(self, *args):
            self.starts += 1
            super().start(*args)
        def stop(self):
            self.stops += 1
            return super().stop()
    terminal = CountingTerminal()
    runner = asyncio.create_task(run_tui_session(session, terminal=terminal))
    try:
        await asyncio.sleep(0.05)
        terminal.emit(text_keys('/login') + [ENTER, ENTER, ENTER, ENTER, ENTER, ENTER])
        await wait_for_thread(entered)
        if failure == 'input':
            terminal.input_failed = True
        elif failure == 'write':
            terminal.fail_write = True
            terminal.set_size(79, 23)
        else:
            runner.cancel()
        await asyncio.sleep(0.35)
        assert not runner.done() and session.login_busy
        release.set()
        if failure == 'cancel-host':
            with pytest.raises(asyncio.CancelledError):
                await runner
        else:
            assert await runner == 1
        assert session.state == 'closed' and not session.login_busy
        assert FileUserConfigStore(store.root).load('env-sentinel').defaults.think
        assert terminal.starts == terminal.stops == 1
    finally:
        release.set()
        if not runner.done():
            runner.cancel()
        await session.aclose()
