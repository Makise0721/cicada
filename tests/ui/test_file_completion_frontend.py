"""@ 文件补全经真实 MainScreenFrontend + FakeTerminal 的组合检查 (C6/C7)。

通过 VT 屏幕与按键批次观察键位、屏障与生命周期; 菜单内部状态由
test_file_menu.py 覆盖, 此处不重复。
"""

import asyncio

from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from cicada.file_completion import FileCompletionCandidate, FileCompletionResult
from cicada.ui.file_menu import FileCompletionMenu
from tests.ui.test_frontend_headless import ENTER, ESCAPE, Harness, text_keys

UP = KeyPress(Keys.Up, "\x1b[A")
DOWN = KeyPress(Keys.Down, "\x1b[B")
TAB = KeyPress(Keys.Tab, "\t")

FILE = FileCompletionCandidate("src/main.py", False)
OTHER = FileCompletionCandidate("src/other.py", False)
DIR = FileCompletionCandidate("src/", True)


def _instant(*candidates):
    async def complete(query, cancel):
        return FileCompletionResult(tuple(candidates), "git")
    return complete


def _harness(complete, *, enter_barrier=None, **kwargs) -> Harness:
    h = Harness(**kwargs)
    # 菜单必须绑定前端实际持有的 Editor 实例, 故在构造后注入组合面
    menu = FileCompletionMenu(h.editor, complete, enter_barrier=enter_barrier)
    menu.on_update = h.frontend.request_render_now  # 异步候选落定触发重绘
    h.frontend._file_menu = menu
    return h


async def test_type_at_shows_candidates_selection_and_tab_accepts():
    h = _harness(_instant(FILE, OTHER))
    try:
        await h.start()
        h.terminal.emit(text_keys("@src"))
        await h.settle()
        assert h.find("src/main.py")  # 候选渲染在编辑器下方
        h.terminal.emit([DOWN])
        await h.settle()
        assert h.find("src/other.py") and not h.find("→ src/main.py")
        h.terminal.emit([TAB])  # 有候选: 接受当前项
        await h.settle()
        assert h.editor.get_text() == "@src/other.py "
        assert not h.find("→ src/other.py")  # 完成文件后菜单关闭 (草稿中的路径仍在)
        h.terminal.emit([ENTER])
        await h.settle()
        assert h.submitted == ["@src/other.py"]  # 原文提交, 无附件展开
    finally:
        await h.stop()


async def test_enter_completes_candidate_then_next_enter_submits_original_text():
    h = _harness(_instant(FILE))
    try:
        await h.start()
        h.terminal.emit(text_keys("修改 @src/ma"))
        await h.settle()
        assert h.find("src/main.py")
        h.terminal.emit([ENTER])  # 菜单展示候选时 Enter 完成当前项并返回
        await h.settle()
        assert h.editor.get_text() == "修改 @src/main.py "
        assert h.submitted == []
        h.terminal.emit(text_keys("请检查") + [ENTER])  # 第二次 Enter 才走原准入
        await h.settle()
        assert h.submitted == ["修改 @src/main.py 请检查"]
    finally:
        await h.stop()


async def test_escape_dismisses_without_cancel_and_reopens_on_change():
    h = _harness(_instant(FILE))
    try:
        await h.start()
        h.terminal.emit(text_keys("@s"))
        await h.settle()
        assert h.find("src/main.py")
        h.terminal.emit([ESCAPE])
        await h.settle()
        assert not h.find("src/main.py")
        assert h.editor.get_text() == "@s"  # Esc 只收起候选, 草稿保留
        assert h.escape_count == 0  # 不触发会话取消
        h.frontend.request_render()
        await h.settle()
        assert not h.find("src/main.py")  # 相同草稿/光标不因重绘重开
        h.terminal.emit(text_keys("r"))
        await h.settle()
        assert h.find("src/main.py")  # 文本变化解锁
    finally:
        await h.stop()


async def test_enter_while_waiting_submits_and_late_result_does_not_reopen():
    release = asyncio.Event()

    async def slow(query, cancel):
        await release.wait()
        return FileCompletionResult((FILE,), "git")

    h = _harness(slow)
    try:
        await h.start()
        h.terminal.emit(text_keys("@s"))
        await h.settle()
        h.terminal.emit([ENTER])  # 等待查询但无候选: 使查询失效, 走原提交
        await h.settle()
        assert h.submitted == ["@s"]
        release.set()
        await h.settle()
        assert not h.find("src/main.py")  # 迟到结果不可重开菜单
    finally:
        release.set()
        await h.stop()


async def test_busy_barrier_keeps_candidates_and_silent():
    submitted = []

    def on_submit(text):
        submitted.append(text)
        return False  # 模拟准入拒绝路径

    h = _harness(_instant(FILE), on_submit=on_submit, enter_barrier=lambda: True)
    try:
        await h.start()
        h.terminal.emit(text_keys("@s"))
        await h.settle()
        h.terminal.emit([ENTER])
        await h.settle()
        # busy 屏障优先: 即使有候选也不确认; Enter 落入原静默屏障 (拒绝), 草稿保留
        assert submitted == ["@s"]
        assert h.editor.get_text() == "@s"
        assert h.find("→ src/main.py")  # 候选保持未被消费
    finally:
        await h.stop()


async def test_slash_command_task_suffix_completes_files():
    h = _harness(_instant(FILE), commands=True)
    try:
        await h.start()
        h.terminal.emit(text_keys("/skill 写 @s"))
        await h.settle()
        assert not h.find("→ skill")  # 斜线菜单不因任务后缀打开
        assert h.find("src/main.py")  # 任务文本中的独立 @ 可补全
        h.terminal.emit([TAB])
        await h.settle()
        assert h.editor.get_text() == "/skill 写 @src/main.py "
    finally:
        await h.stop()


async def test_unavailable_reason_shows_short_status_without_stderr():
    async def complete(query, cancel):
        return FileCompletionResult((), "unavailable", reason="root_mismatch")

    h = _harness(complete)
    try:
        await h.start()
        h.terminal.emit(text_keys("@s"))
        await h.settle()
        assert h.find("候选不可用：工作区不是 Git 根目录")
        assert not h.find("Git failed")  # 不显示任意 stderr
    finally:
        await h.stop()


async def test_directory_accept_reopens_menu_for_continue():
    async def complete(query, cancel):
        if query == "src":
            return FileCompletionResult((DIR,), "git")
        return FileCompletionResult((FILE,), "git")

    h = _harness(complete)
    try:
        await h.start()
        h.terminal.emit(text_keys("@src"))
        await h.settle()
        assert h.find("src/")
        h.terminal.emit([TAB])  # 接受目录 -> "@src/"
        await h.settle()
        assert h.editor.get_text() == "@src/"
        assert h.find("src/main.py")  # 目录继续: 新 query 的候选
        h.terminal.emit([TAB])
        await h.settle()
        assert h.editor.get_text() == "@src/main.py "
    finally:
        await h.stop()


async def test_exit_cancels_pending_query_and_stops_cleanly():
    release = asyncio.Event()

    async def slow(query, cancel):
        await release.wait()
        return FileCompletionResult((FILE,), "git")

    h = _harness(slow)
    try:
        await h.start()
        h.terminal.emit(text_keys("@s"))
        await h.settle()
        errors = await h.stop()  # 前端退出先取消并等待查询任务
        assert errors == []
    finally:
        release.set()
