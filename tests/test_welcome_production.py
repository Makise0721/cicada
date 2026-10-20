"""tui-startup-welcome 票据 03/04 生产组合检查: 真实 CLI → 真实 run_tui_session.

A13/A10 三场景经真实入口组合 (cicada.__main__ → _run_tui →
cicada.ui.run_tui_session → MainScreenFrontend), 终端经 PromptToolkitTerminal
工厂接缝注入探针终端 (VT 屏镜像 + 观察-再-退出驱动); mock HTTP 零真实网络。
断言真实 VT 画面含启动欢迎 (r2 Pi 风格: 新会话完整块段组, 无图标 / 恢复
三行短版), 新请求体与 SQLite 存档不含欢迎文字 (A10), 首次配置场景向导
终端先停、主屏终端后启动的顺序 owner 交接。生产代码零改动。
"""

from __future__ import annotations

import asyncio
import json

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

import cicada.ui
from cicada import __main__ as cli
from cicada.plugins.user_config import FileUserConfigStore
from cicada.ui.main_screen import strip_terminal_sequences
from cicada.ui.terminal_io import FakeTerminal
from test_cli import _CaptureStream, _runtime
from test_user_config_startup import invoke, requests, write_config
from tests.ui.vt_screen import VtScreen

ENTER = KeyPress(Keys.ControlM, '\r')
PASTE = KeyPress(Keys.BracketedPaste, 'startup-sentinel')
CTRL_D = KeyPress(Keys.ControlD, '')
BACKSPACE = KeyPress(Keys.ControlH, '\b')
_LOGO_CHARS = '█▄▀'
_WELCOME_LEAKS = ('启动快照', '启动模型', '输入 / 查看命令')
_BUSY_MARKERS = ('正在输出…', '正在收尾…', '正在取消…',
                 '正在执行…', '等待模型…', '正在思考…')  # 票据 02: 忙态文案全集
_BUSY_STATUS_LINES = ('working',)  # 2026-10-19 模型三段统一; 按整行匹配,
# 不按子串: 工作区路径或回答正文里出现同拼写会被误判成一直在忙


def text_keys(text: str) -> list[KeyPress]:
    return [KeyPress(ch, ch) for ch in text]


def _force_tty(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(_CaptureStream, 'isatty', lambda self: True, raising=False)


def _force_color_env(monkeypatch) -> None:
    """欢迎 plain 策略捕获自环境: 固定有色路径, 不受运行环境 NO_COLOR/TERM 影响。"""
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')


def _assert_body_has_no_welcome(body: dict) -> None:
    serialized = json.dumps(body, ensure_ascii=False)
    for needle in _WELCOME_LEAKS:
        assert needle not in serialized


def _assert_archive_has_no_welcome(workspace) -> None:
    archives = list((_runtime(workspace) / 'sessions').glob('*.sqlite3'))
    assert len(archives) == 1
    raw = archives[0].read_bytes()
    for needle in _WELCOME_LEAKS:
        assert needle.encode('utf-8') not in raw


class MainProbeTerminal(FakeTerminal):
    """主屏探针终端: VT 屏镜像 + 事件记录; start 后按脚本驱动 (观察回答后再发退出键)。"""

    def __init__(self, script, events, tag='main', columns=96, rows=24):
        super().__init__(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.feed_screen = self.vt.feed
        self.notes: dict[str, object] = {}
        self._script = script
        self._events = events
        self._tag = tag
        self._driver: asyncio.Task | None = None

    def seen(self, needle: str) -> bool:
        if self.vt.find_row(needle) is not None:
            return True
        if any(needle in line for line in self.vt.history_text()):
            return True
        # 自有视口 (紧凑交互 01): 帧 ≤ rows 不产生滚动历史, 曾被窗口呈现的行
        # 只留在累计写出流; 剥离控制序列后检索即"曾上屏"语义
        return needle in strip_terminal_sequences(self.output_text)

    async def wait_seen(self, needle: str, timeout: float = 15.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not self.seen(needle):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'VT 屏超时未出现: {needle}')
            await asyncio.sleep(0.05)

    async def wait_idle(self, timeout: float = 15.0) -> None:
        """等可见屏忙状态行消退 (请求落定); 忙碌时退出键会被屏障静默。"""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            screen = '\n'.join(self.vt.screen_text(strip_blank_tail=False))
            busy = any(marker in screen for marker in _BUSY_MARKERS)
            if not busy:
                # working 状态行独占一行 (无 spinner/文案前缀): 按整行判定,
                # 不让路径或回答里的同拼写误判
                busy = any(line.strip() in _BUSY_STATUS_LINES
                           for line in self.vt.screen_text(strip_blank_tail=False))
            if not busy:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('VT 屏忙状态行未消退')
            await asyncio.sleep(0.05)

    def start(self, on_input, on_resize):
        super().start(on_input, on_resize)
        self._events.append(f'{self._tag}-start')
        self._driver = asyncio.get_running_loop().create_task(self._drive())

    def stop(self):
        self._events.append(f'{self._tag}-stop')
        driver, self._driver = self._driver, None
        if driver is not None and not driver.done():
            driver.cancel()
        return super().stop()

    async def _drive(self):
        try:
            await asyncio.wait_for(self._script(self), timeout=25.0)
        except (Exception, asyncio.CancelledError):
            pass  # 流程异常由 invoke 后的断言暴露; 此处只保证尽力退出
        finally:
            for _ in range(10):  # 先等脚本自己的退出键生效
                if self.stopped:
                    return
                await asyncio.sleep(0.05)
            for _ in range(120):  # 失败兜底: 清草稿后反复 Ctrl+D 直至前端停止
                if self.stopped:
                    return
                self.emit([BACKSPACE] * 64)
                self.emit([CTRL_D])
                await asyncio.sleep(0.1)


class WizardProbeTerminal(FakeTerminal):
    """向导探针终端: start 后经事件循环一次性 emit 预排按键。"""

    def __init__(self, keys, events, tag='wizard', columns=96, rows=24):
        super().__init__(columns, rows)
        self.vt = VtScreen(columns, rows)
        self.feed_screen = self.vt.feed
        self._keys = keys
        self._events = events
        self._tag = tag

    def start(self, on_input, on_resize):
        super().start(on_input, on_resize)
        self._events.append(f'{self._tag}-start')
        asyncio.get_running_loop().call_soon(self.emit, list(self._keys))

    def stop(self):
        self._events.append(f'{self._tag}-stop')
        return super().stop()


class _OrderedTerminalFactory:
    """同一有序工厂补丁向导 (terminal_io 模块级) 与主屏 (cicada.ui 模块级) 两处接缝。"""

    def __init__(self, *terminals):
        self._terminals = list(terminals)
        self.calls: list[FakeTerminal] = []

    def __call__(self):
        terminal = self._terminals[len(self.calls)]
        self.calls.append(terminal)
        return terminal


async def _ask_and_exit(term: MainProbeTerminal, prompt: str) -> None:
    term.emit(text_keys(prompt))
    term.emit([ENTER])
    await term.wait_seen('离线答复')
    await term.wait_idle()
    term.notes['answer_seen'] = True
    term.emit([CTRL_D])


async def _new_session_script(term: MainProbeTerminal) -> None:
    await term.wait_seen('新会话 · 启动快照')
    await asyncio.sleep(0.1)  # 等首帧整体落屏再取样
    term.notes['welcome_text'] = '\n'.join(term.vt.screen_text())
    term.notes['title_line'] = term.vt.line_text(term.vt.find_row('新会话 · 启动快照'))
    await _ask_and_exit(term, '验收问题')


async def _resume_script(term: MainProbeTerminal) -> None:
    await term.wait_seen('恢复会话 · 启动快照')
    await asyncio.sleep(0.1)
    term.notes['welcome_text'] = '\n'.join(term.vt.screen_text())
    term.notes['title_line'] = term.vt.line_text(term.vt.find_row('恢复会话 · 启动快照'))
    await term.wait_seen('已恢复历史')  # 恢复投影头与旧记录条目
    await term.wait_seen('旧问')
    term.notes['restored_seen'] = True
    await _ask_and_exit(term, '续问探针')


async def _initial_prompt_script(term: MainProbeTerminal) -> None:
    await term.wait_seen('新会话 · 启动快照')
    term.notes['welcome_seen'] = True
    await term.wait_seen('离线答复')  # 首问沿既有准入提交
    await term.wait_idle()
    term.notes['answer_seen'] = True
    term.emit([CTRL_D])


def test_new_session_welcome_reaches_real_tui_screen(monkeypatch, tmp_path, requests):
    """A13/A10 场景 1: 新建会话真实 CLI → 真实 run_tui_session。"""
    _force_tty(monkeypatch)
    _force_color_env(monkeypatch)
    root = tmp_path / 'user'
    write_config(root)
    events: list[str] = []
    main = MainProbeTerminal(_new_session_script, events)
    monkeypatch.setattr(cicada.ui, 'PromptToolkitTerminal', lambda: main)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui'], root)
    assert code == 0, err
    assert events == ['main-start', 'main-stop']
    welcome = main.notes.get('welcome_text')
    assert welcome is not None, '主屏未出现新会话欢迎'
    for needle in ('新会话 · 启动快照', '[工作区]', '[启动模型]', '[Skills]', '输入 / 查看命令'):
        assert needle in welcome
    assert not any(mark in welcome for mark in _LOGO_CHARS), 'r2 起全宽度无标志'
    title = main.notes['title_line']
    assert title.lstrip().startswith('Cicada '), 'r2 完整块标题无左侧图标缩进'
    assert main.notes.get('answer_seen'), 'mock 回答未上屏'
    assert len(requests) == 1 and requests[0][0] == 'Bearer file-sentinel-key'
    _assert_body_has_no_welcome(requests[0][1])
    _assert_archive_has_no_welcome(tmp_path)


def test_resumed_session_welcome_has_no_logo_and_replies_again(monkeypatch, tmp_path, requests):
    """A13/A10 场景 2: 恢复会话真实 CLI → 真实 run_tui_session (短欢迎无标志)。"""
    _force_tty(monkeypatch)
    _force_color_env(monkeypatch)
    root = tmp_path / 'user'
    write_config(root)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--save-session', '旧问'], root)
    assert code == 0, err
    session_id = out.split('[session id=')[1].split(' ')[0]
    events: list[str] = []
    main = MainProbeTerminal(_resume_script, events)
    monkeypatch.setattr(cicada.ui, 'PromptToolkitTerminal', lambda: main)
    code, _, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--resume', session_id, '--tui'], root)
    assert code == 0, err
    assert events == ['main-start', 'main-stop']
    welcome = main.notes.get('welcome_text')
    assert welcome is not None, '主屏未出现恢复欢迎'
    for needle in ('恢复会话 · 启动快照', '启动模型: deepseek-flash', '输入 / 查看命令'):
        assert needle in welcome
    assert not any(mark in welcome for mark in _LOGO_CHARS), '恢复短欢迎无标志'
    assert main.notes.get('restored_seen'), '恢复投影的旧记录不可见'
    assert main.notes.get('answer_seen'), '恢复后会话无法再提交新请求'
    assert len(requests) == 2 and requests[-1][0] == 'Bearer file-sentinel-key'
    _assert_body_has_no_welcome(requests[-1][1])
    _assert_archive_has_no_welcome(tmp_path)


def test_first_run_wizard_hands_terminal_to_main_screen_in_order(monkeypatch, tmp_path, requests):
    """A13 场景 3: 首次配置真实向导 → 主屏的顺序 owner 交接 (向导 stop 先于主屏 start)。"""
    _force_tty(monkeypatch)
    _force_color_env(monkeypatch)
    root = tmp_path / 'user'
    events: list[str] = []
    wizard = WizardProbeTerminal([ENTER, PASTE, ENTER, ENTER, ENTER, ENTER], events)
    main = MainProbeTerminal(_initial_prompt_script, events)
    factory = _OrderedTerminalFactory(wizard, main)
    monkeypatch.setattr('cicada.ui.terminal_io.PromptToolkitTerminal', factory)
    monkeypatch.setattr(cicada.ui, 'PromptToolkitTerminal', factory)
    code, out, err = invoke(monkeypatch, ['--workspace', str(tmp_path), '--tui', '首问'], root)
    assert code == 0, err
    assert factory.calls == [wizard, main], '工厂恰好被调用 2 次且顺序为向导→主屏'
    assert events == ['wizard-start', 'wizard-stop', 'main-start', 'main-stop']
    assert events.index('wizard-stop') < events.index('main-start')
    assert '首次配置' in wizard.output_text and '*' in wizard.output_text
    assert main.notes.get('welcome_seen'), '主屏未出现新会话欢迎'
    assert main.notes.get('answer_seen'), '首问回答未上屏'
    assert len(requests) == 1
    assert requests[0][0] == 'Bearer startup-sentinel'
    assert requests[0][1]['model'] == 'deepseek-flash'
    ref = FileUserConfigStore(root).load().defaults.credential_ref
    leak_surface = out + err + wizard.output_text + main.output_text
    assert 'startup-sentinel' not in leak_surface
    assert ref not in leak_surface
