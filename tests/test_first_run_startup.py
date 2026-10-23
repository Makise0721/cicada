"""首次向导经过实际启动、Terminal、文件、SQLite 与 mock HTTP。"""

import asyncio
import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada import __main__ as cli
from cicada.application import ApplicationOpenError
from cicada.plugins.session_store import read_session_identity
from cicada.plugins.user_config import FileUserConfigStore
from cicada.ui.terminal_io import FakeTerminal
from test_cli import _CaptureStream, _runtime
from test_user_config_startup import invoke, requests, write_config

ENTER = KeyPress(Keys.ControlM, '\r')
PASTE = KeyPress(Keys.BracketedPaste, 'startup-sentinel')
DOWN = KeyPress(Keys.Down, '')
UP = KeyPress(Keys.Up, '')
DEEPSEEK = KeyPress(Keys.BracketedPaste, 'DeepSeek')


def startup_keys():
    # 搜索现有 DeepSeek，输入凭据，再经设置页与确认页保存。
    return [DEEPSEEK, ENTER, PASTE, ENTER, DOWN, DOWN, ENTER, ENTER]


def credentials_only_keys():
    return [DOWN, ENTER, PASTE, ENTER, ENTER]


class ScriptTerminal(FakeTerminal):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.starts = self.stops = 0
        self.driver = None
        self.timed_out = False

    def start(self, on_input, on_resize):
        self.starts += 1
        super().start(on_input, on_resize)
        self.driver = asyncio.create_task(self._drive())

    async def _drive(self):
        await asyncio.sleep(0)
        self.emit(self.keys)
        await asyncio.sleep(5)
        self.timed_out = True
        self.emit([KeyPress(Keys.Escape, '\x1b')])

    def stop(self):
        self.stops += 1
        if self.driver is not None:
            self.driver.cancel()
        return super().stop()


@pytest.fixture
def interactive(monkeypatch):
    import cicada.ui
    terminal = ScriptTerminal(startup_keys())
    monkeypatch.setattr('cicada.ui.terminal_io.PromptToolkitTerminal', lambda: terminal)
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)
    sessions = []

    async def tui(session, *, initial_prompt):
        assert terminal.stopped and terminal.starts == terminal.stops == 1
        sessions.append(session)
        try:
            if initial_prompt is not None:
                return (await session.execute(initial_prompt)).exit_code
            return 3 if session.view_only_reason else 0
        finally:
            await session.aclose()

    monkeypatch.setattr(cicada.ui, 'run_tui_session', tui)
    yield terminal, sessions
    assert not terminal.timed_out, "当前向导按键脚本未在限时内完成"


def test_first_run_saves_before_open_and_initial_prompt_once(monkeypatch, tmp_path, requests, interactive):
    root = tmp_path / 'user'
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '首问'], root)
    assert code == 0, err
    assert len(requests) == 1 and requests[0][0] == 'Bearer startup-sentinel'
    assert requests[0][1]['model'] == 'deepseek-flash'
    assert requests[0][1]['reasoning_effort'] == 'high'
    terminal, sessions = interactive
    assert FileUserConfigStore(root).load().credential.reveal() == 'startup-sentinel'
    archive = next((_runtime(tmp_path) / 'sessions').glob('*.sqlite3'))
    identity = read_session_identity(directory=archive.parent, session_id=archive.stem, workspace=tmp_path)
    assert identity.launch.sources['model'] == 'cli'  # 向导选择用于本次启动
    assert identity.launch.provider == 'deepseek' and identity.launch.model == 'deepseek-flash'
    for needle in ('startup-sentinel', FileUserConfigStore(root).load().defaults.credential_ref):
        assert needle not in out + err + terminal.output_text + repr(sessions[0].history)
        assert needle.encode() not in archive.read_bytes()


def test_wizard_cancel_keeps_default_config_without_application_or_epoch(monkeypatch, tmp_path, requests, interactive):
    terminal, sessions = interactive
    terminal.keys = [DEEPSEEK, ENTER, PASTE, KeyPress(Keys.Escape, '\x1b')]
    root = tmp_path / 'user'
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '首问'], root)
    assert code == 0 and sessions == [] and requests == [], err
    assert (root / 'config.toml').exists() and not (root / 'credentials.json').exists()
    assert not (tmp_path / '.cicada').exists()
    assert terminal.stopped and 'startup-sentinel' not in terminal.output_text


def test_saved_defaults_remain_when_application_open_fails(monkeypatch, tmp_path, requests, interactive):
    async def fail(*args, **kwargs):
        raise ApplicationOpenError('injected open error')
    monkeypatch.setattr(cli, 'open_application', fail)
    root = tmp_path / 'user'
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '首问'], root)
    assert code == 2 and '已保存' in err and requests == []
    assert FileUserConfigStore(root).load().credential.reveal() == 'startup-sentinel'
    assert not (tmp_path / '.cicada').exists()


def test_explicit_think_overrides_saved_choice_and_preserves_limits(monkeypatch, tmp_path, requests, interactive):
    terminal, _ = interactive
    terminal.keys = [DEEPSEEK, ENTER, PASTE, ENTER, UP, UP, ENTER, DOWN, ENTER,
        *([DOWN] * 3), ENTER, KeyPress(Keys.BracketedPaste, '65536'), ENTER,
        *([DOWN] * 4), ENTER, KeyPress(Keys.BracketedPaste, '4096'), ENTER,
        *([DOWN] * 6), ENTER, DOWN, DOWN, ENTER,
        *([DOWN] * 4), ENTER, ENTER]
    root = tmp_path / 'user'
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '--think',
        '--model', 'deepseek-v4-pro', '--num-ctx', '65536', '--max-output-tokens', '4096', '首问'], root)
    assert code == 0, err
    saved = FileUserConfigStore(root).load().defaults
    assert not saved.think and saved.num_ctx == 65536 and saved.max_output_tokens == 4096
    assert saved.model == 'deepseek-v4-pro' and requests[0][1]['thinking']['type'] == 'enabled'
    assert '本次启动仍按显式 CLI 参数覆盖' in terminal.output_text and '65536/4096' in terminal.output_text


@pytest.mark.parametrize('adopt', [False, True])
def test_resume_missing_key_preserves_archive_or_explicitly_adopts(monkeypatch, tmp_path, requests, interactive, adopt):
    root = tmp_path / 'user'
    write_config(root, 'model = "deepseek-v4-pro"\nthink = false\nreasoning_effort = "low"\n'
        'num_ctx = 65536\nmax_output_tokens = 4096\n')
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', '旧问'], root)
    assert code == 0, err
    session_id = out.split('[session id=')[1].split(' ')[0]
    (root / 'config.toml').write_text('version = 1\n[defaults]\nreasoning_effort = "max"\n', encoding='utf-8')
    (root / 'credentials.json').unlink()
    interactive[0].keys = startup_keys() if adopt else credentials_only_keys()
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--resume', session_id, '--tui',
        *(['--resume-with-current-config'] if adopt else []), '续问'], root)
    assert code == 0, err
    body = requests[-1][1]
    assert requests[-1][0] == 'Bearer startup-sentinel'
    saved = FileUserConfigStore(root).load().defaults
    assert saved.model == 'deepseek-flash' and saved.reasoning_effort == 'max'
    if adopt:
        assert body['model'] == 'deepseek-flash' and body['thinking']['type'] == 'enabled'
    else:
        assert body['model'] == 'deepseek-v4-pro' and body['thinking']['type'] == 'disabled'
        assert body['max_tokens'] == 4096 and '仅补凭据，存档配置保持' in interactive[0].output_text


@pytest.mark.parametrize('flags', [
    ['--model', 'bad-model'], ['--num-ctx', '2048', '--max-output-tokens', '4096'],
    ['--instructions-file', 'missing-file'], ['--skills-dir', 'missing-dir'],
    ['--check-command', 'Write-Output ok'], ['--resume', 'bad-id'],
])
def test_invalid_local_inputs_fail_before_wizard(monkeypatch, tmp_path, requests, interactive, flags):
    root = tmp_path / 'user'
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', *flags, '问'], root)
    assert code == 2 and requests == [] and interactive[0].starts == 0, err
    if flags == ['--check-command', 'Write-Output ok']:
        assert (root / 'config.toml').exists()
    else:
        assert not root.exists()


def test_non_tty_never_starts_wizard(monkeypatch, tmp_path, requests, interactive):
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: False)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '问'], tmp_path / 'user')
    assert code == 2 and interactive[0].starts == 0 and requests == []


def test_valid_credentials_and_bad_files_do_not_trigger_missing_only_wizard(monkeypatch, tmp_path, requests, interactive):
    root = tmp_path / 'user'
    write_config(root)
    # 使用本票 fake 主屏接缝时允许直接启动（没有向导 terminal 生命周期）。
    async def close(session, **kwargs):
        await session.aclose()
        return 0
    monkeypatch.setattr('cicada.ui.run_tui_session', close)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui'], root)
    assert code == 0 and interactive[0].starts == 0, err
    (root / 'credentials.json').write_bytes(b'broken-sentinel')
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui'], root)
    assert code == 2 and interactive[0].starts == 0 and 'broken-sentinel' not in err
    (root / 'config.toml').write_bytes(b'broken-sentinel')
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui'], root)
    assert code == 2 and interactive[0].starts == 0 and requests == []


@pytest.mark.parametrize('adopt', [False, True])
def test_credentials_only_resume_keeps_recovery_block(monkeypatch, tmp_path, requests, interactive, adopt):
    from cicada.plugins.session_store import open_session_store

    root = tmp_path / 'user'
    write_config(root)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', '旧问'], root)
    assert code == 0, err
    session_id = out.split('[session id=')[1].split(' ')[0]
    directory = _runtime(tmp_path) / 'sessions'

    async def unfinished():
        store = open_session_store(directory=directory, session_id=session_id, workspace=tmp_path)
        try:
            epoch = await store.start_epoch(store.recovery_snapshot().identity)
            request_id = await store.begin_request(epoch, '中断问')
            await store.begin_process(request_id, tool_name='powershell', call_id='unfinished',
                                      arguments={'command': 'Write-Output mock'})
        finally:
            await store.aclose()

    asyncio.run(unfinished())
    (root / 'config.toml').unlink()
    (root / 'credentials.json').unlink()
    before = len(requests)
    interactive[0].keys = startup_keys() if adopt else credentials_only_keys()
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--resume', session_id, '--tui',
        *(['--resume-with-current-config'] if adopt else [])], root)
    assert code == 3 and len(requests) == before, err
    session = interactive[1][-1]
    assert session.view_only_reason is not None
    assert FileUserConfigStore(root).load().credential.reveal() == 'startup-sentinel'


def test_selector_then_wizard_then_main_screen_sequence(monkeypatch, tmp_path, requests, interactive):
    root = tmp_path / 'user'
    write_config(root)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', '旧问'], root)
    assert code == 0, err
    session_id = out.split('[session id=')[1].split(' ')[0]
    (root / 'config.toml').unlink()
    (root / 'credentials.json').unlink()
    events = []
    terminal, _ = interactive
    start, stop = terminal.start, terminal.stop

    def wizard_start(*args):
        events.append('wizard-start')
        start(*args)

    def wizard_stop():
        events.append('wizard-stop')
        return stop()

    monkeypatch.setattr(terminal, 'start', wizard_start)
    monkeypatch.setattr(terminal, 'stop', wizard_stop)

    async def selector(workspace):
        from cicada.application import make_session_catalog_reader
        from cicada.ui.selector import run_session_selector
        picker = ScriptTerminal([])
        directory = _runtime(workspace) / 'sessions'
        reader = make_session_catalog_reader(directory=directory)

        async def observe(cancel):
            snapshot = await reader.observe(cancel)
            asyncio.get_running_loop().call_soon(picker.emit, [ENTER])
            return snapshot

        events.append('selector-start')
        try:
            picked, errors = await asyncio.wait_for(
                run_session_selector(picker, directory, observe=observe), timeout=10,
            )
        finally:
            await reader.aclose()
        assert errors == [] and picker.stopped and terminal.starts == 0
        events.append('selector-stop')
        return picked

    terminal.keys = credentials_only_keys()
    monkeypatch.setattr(cli, '_launch_session_selector', selector)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '--resume'], root)
    assert code == 0, err
    assert events == ['selector-start', 'selector-stop', 'wizard-start', 'wizard-stop']
