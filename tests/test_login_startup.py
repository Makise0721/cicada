"""生产 CLI→主屏保存→下次新建 CLI/mock HTTP，fake 入口保持惰性读取。"""

import asyncio

import pytest
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada import __main__ as cli
from cicada.plugins.user_config import FileUserConfigStore
from cicada.ui import run_tui_session
from cicada.ui.terminal_io import FakeTerminal
from test_cli import _CaptureStream
from test_user_config_startup import invoke, requests, write_config
from tests.ui.test_frontend_headless import ENTER, CTRL_C, text_keys


def test_login_saved_defaults_reach_next_real_cli_request(monkeypatch, tmp_path, requests):
    root = tmp_path / 'user'
    write_config(root, 'model="deepseek-v4-pro"\n', key='old-sentinel')
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)
    sessions = []
    class ScriptTerminal(FakeTerminal):
        def start(self, on_input, on_resize):
            super().start(on_input, on_resize)
            down = KeyPress(Keys.Down, '')
            up = KeyPress(Keys.Up, '')
            # DeepSeek → 新凭据 → pro 模型参数 low → 设为下次默认 → 只保存。
            asyncio.get_running_loop().call_soon(self.emit, text_keys('/login') + [
                ENTER, KeyPress(Keys.BracketedPaste, 'DeepSeek'), ENTER,
                KeyPress(Keys.BracketedPaste, 'next-start-sentinel'), ENTER,
                up, up, ENTER, down, ENTER, *([down] * 7), ENTER, ENTER, down, ENTER,
                *([down] * 3), ENTER, down, ENTER, down, ENTER])

    terminal = ScriptTerminal()
    async def tui(session, *, initial_prompt):
        sessions.append(session)
        async def quit_after_save():
            async with asyncio.timeout(5):
                while True:
                    loaded = FileUserConfigStore(root).load()
                    if loaded.credential.reveal() == 'next-start-sentinel':
                        break
                    await asyncio.sleep(0.01)
                while session.login_busy:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)  # 等 UI 等待者落定并恢复聊天草稿。
            terminal.emit([CTRL_C, *text_keys('/exit'), ENTER])
        closer = asyncio.create_task(quit_after_save())
        try:
            return await asyncio.wait_for(run_tui_session(session, terminal=terminal,
                initial_prompt=initial_prompt), 8)
        finally:
            await closer
    monkeypatch.setattr('cicada.ui.run_tui_session', tui)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui'], root)
    assert code == 0 and requests == [], err
    assert sessions[0].history == () and sessions[0].config.model_label == 'DeepSeek / deepseek-v4-pro'
    assert sessions[0].config.persistence.identity.think is True
    loaded = FileUserConfigStore(root).load()
    assert loaded.defaults.reasoning_effort == 'low'
    for needle in ('next-start-sentinel', loaded.defaults.credential_ref):
        assert needle not in out + err + terminal.output_text
        assert needle.encode() not in sessions[0]._store.capture().db_path.read_bytes()
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', 'next question'], root)
    assert code == 0 and len(requests) == 1, err
    authorization, body = requests[0]
    assert authorization == 'Bearer next-start-sentinel'
    assert body['model'] == 'deepseek-v4-pro' and body['reasoning_effort'] == 'low'


def test_fake_tui_launch_reads_no_config_or_environment_key(monkeypatch, tmp_path):
    script = tmp_path / 'script.json'
    script.write_text('[{"text":"answer"}]', encoding='utf-8')
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('fake startup must skip credentials')
    monkeypatch.setattr(FileUserConfigStore, 'load', forbidden)
    original = cli.os.environ.get
    monkeypatch.setattr(cli.os.environ, 'get', lambda name, *args: forbidden() if name == 'DEEPSEEK_API_KEY' else original(name, *args))
    async def tui(session, **kwargs):
        try:
            assert session.login_status == ()
            return (await session.execute('fake question')).exit_code
        finally:
            await session.aclose()
    monkeypatch.setattr('cicada.ui.run_tui_session', tui)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--script', str(script), '--tui'], tmp_path / 'user')
    assert code == 0, err
