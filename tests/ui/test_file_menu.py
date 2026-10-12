"""@ 文件补全菜单状态机: token 解析、一次替换与查询生命周期。

时间敏感路径 (20ms 防抖/旧工作落定) 用真实短睡眠驱动; 迟到/取消边界用
可控 fake 端口证明。
"""

import asyncio

import pytest

from cicada.core.cancel import CancelToken
from cicada.file_completion import FileCompletionCandidate, FileCompletionResult
from cicada.ui.editor import Editor
from cicada.ui.file_menu import FileCompletionMenu, parse_at_token

FILE = FileCompletionCandidate("src/main.py", False)
OTHER = FileCompletionCandidate("src/other.py", False)
DIR = FileCompletionCandidate("src/", True)
SPACED = FileCompletionCandidate("my 文件.txt", False)


def _result(*candidates, truncated=False):
    return FileCompletionResult(tuple(candidates), "git", truncated=truncated)


def _editor(text: str, col: int | None = None) -> Editor:
    editor = Editor()
    editor.set_text(text)
    if col is not None:
        editor.state.cursor_col = col
    return editor


def _menu(editor: Editor, complete=None, *, enter_barrier=None) -> FileCompletionMenu:
    if complete is None:
        async def complete(query, cancel):
            return _result()
    return FileCompletionMenu(editor, complete, enter_barrier=enter_barrier)


# --- token 解析 -------------------------------------------------------------------

@pytest.mark.parametrize("line,cursor,expected", [
    ("@", 1, ""),
    ("@src", 4, "src"),
    ("看 @src/ma", 9, "src/ma"),
    ("@a @b", 5, "b"),
    ('@"src/ma', 8, "src/ma"),
    ('@"src/main.py', 13, "src/main.py"),  # 未闭合右引号前
    ('@"src/main.py"', 13, "src/main.py"),  # 带右引号 token 内
    ("/skill 修 @src", 13, "src"),  # slash 命令任务后缀
])
def test_parse_at_token_recognizes_independent_tokens(line, cursor, expected):
    token = parse_at_token(line, cursor)
    assert token is not None and token.query == expected
    assert line[token.start] == "@"
    assert token.end >= cursor


@pytest.mark.parametrize("line,cursor", [
    ("a@b", 3),      # 邮箱整体
    ("a@", 2),       # 邮箱输入中
    ("hi a@b", 6),
    ("email@example.com", 5),
    ("say \"hello", 10),   # 非 @" 未闭合引号
    ("@", 0),        # 光标在 @ 之前
    ("@a\x01b", 4),  # 控制字符
    ('@"a"b"c', 7),  # 非法引号排列
])
def test_parse_at_token_rejects_non_trigger_forms(line, cursor):
    assert parse_at_token(line, cursor) is None


def test_parse_at_token_extends_end_past_cursor_for_tail_replacement():
    token = parse_at_token("@src/main.py", 5)
    assert (token.start, token.end) == (0, 12)
    quoted = parse_at_token('@"src/main.py"', 8)
    assert (quoted.start, quoted.end) == (0, 14)  # 含右引号


# --- 一次替换与接受 -----------------------------------------------------------------

def test_replace_span_single_undo_and_bounds():
    editor = _editor("hello world", col=5)
    editor.replace_span(0, 6, 11, "python")
    assert editor.get_text() == "hello python"
    assert editor.state.cursor_col == 12
    editor.undo()
    assert editor.get_text() == "hello world"
    with pytest.raises(ValueError):
        editor.replace_span(1, 0, 1, "x")
    with pytest.raises(ValueError):
        editor.replace_span(0, 4, 2, "x")
    with pytest.raises(ValueError):
        editor.replace_span(0, 0, 1, "x", cursor_offset=5)


async def test_accept_file_appends_space_and_single_undo_restores():
    editor = _editor("看 @src/ma")
    calls: list[str] = []

    async def complete(query, cancel):
        calls.append(query)
        return _result(FILE)

    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.selected == FILE
    assert menu.accept() is True
    assert editor.get_text() == "看 @src/main.py "
    assert editor.state.cursor_col == len(editor.get_text())
    editor.undo()
    assert editor.get_text() == "看 @src/ma"  # 一次接受一次 undo
    assert menu.candidates == ()  # 完成文件后关闭


async def test_accept_reuses_existing_trailing_space_and_quotes_spaced_paths():
    editor = _editor("@a.py ")
    editor.state.cursor_col = 5

    async def complete(query, cancel):
        return _result(FILE)

    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    menu.accept()
    assert editor.get_text() == "@src/main.py "  # 复用右侧空格不重复

    editor2 = _editor("@a")
    async def complete2(query, cancel):
        return _result(SPACED)
    menu2 = _menu(editor2, complete2)
    menu2.sync(editor2, eligible=True)
    await asyncio.sleep(0.05)
    menu2.accept()
    assert editor2.get_text() == '@"my 文件.txt" '  # 含空格/中文路径包双引号


async def test_accept_directory_continues_with_new_query_and_quoted_cursor():
    queries: list[str] = []

    async def complete(query, cancel):
        queries.append(query)
        return _result(DIR) if query == "src" else _result(FILE)

    editor = _editor("@src")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.candidates == (DIR,)
    menu.accept()
    assert editor.get_text() == "@src/"
    await asyncio.sleep(0.05)
    assert queries[-1] == "src/"  # 完成目录生成新 query
    assert menu.candidates == (FILE,)  # 目录继续: 立即按新 query 刷新候选

    editor2 = _editor('@"src')
    async def complete2(query, cancel):
        return _result(DIR)
    menu2 = _menu(editor2, complete2)
    menu2.sync(editor2, eligible=True)
    await asyncio.sleep(0.05)
    menu2.accept()
    assert editor2.get_text() == '@"src/"'
    assert editor2.state.cursor_col == len('@"src/')  # 带引号光标停在右引号前


async def test_accept_second_reference_keeps_other_tokens():
    editor = _editor("看 @src/ma 和 @docs/x.md", col=9)
    async def complete(query, cancel):
        return _result(FILE)
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    menu.accept()
    assert editor.get_text() == "看 @src/main.py 和 @docs/x.md"  # 复用右侧空格
    editor.undo()
    assert editor.get_text() == "看 @src/ma 和 @docs/x.md"  # 其它引用原样保留


async def test_accept_mid_token_replaces_whole_token_tail_included():
    editor = _editor("@src/main.py 尾部", col=5)  # 光标在 token 中段
    async def complete(query, cancel):
        return _result(OTHER)
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.accept() is True
    assert editor.get_text() == "@src/other.py 尾部"  # 光标后尾部一并替换, 外部文字保留
    editor.undo()
    assert editor.get_text() == "@src/main.py 尾部"


# --- 键位语义 -----------------------------------------------------------------------

async def test_enter_with_candidates_consumes_and_barrier_defers():
    editor = _editor("@src/ma")
    async def complete(query, cancel):
        return _result(FILE)
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.handle_enter() is True  # 第一次 Enter 完成当前项
    assert editor.get_text() == "@src/main.py "

    editor2 = _editor("@src/ma")
    blocked = _menu(editor2, complete, enter_barrier=lambda: True)
    blocked.sync(editor2, eligible=True)
    await asyncio.sleep(0.05)
    assert blocked.candidates
    assert blocked.handle_enter() is False  # busy 屏障: 即使有候选也不确认
    assert blocked.candidates  # 候选保持, 由原静默屏障处理


async def test_enter_without_candidates_invalidates_in_flight_query():
    release = asyncio.Event()

    async def complete(query, cancel):
        await release.wait()
        return _result(FILE)

    editor = _editor("@src")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.handle_enter() is False  # 等待查询无候选: 走原提交
    release.set()
    await asyncio.sleep(0.05)
    assert not menu.visible  # 迟到结果不可重开菜单


async def test_tab_without_results_bypasses_debounce_and_consumes_key():
    queries: list[str] = []

    async def complete(query, cancel):
        queries.append(query)
        return _result(FILE)

    editor = _editor("@src")
    menu = _menu(editor, complete)
    assert menu.handle_tab() is True  # 有 token 无快照: 立即启动并消费
    await asyncio.sleep(0.01)
    assert queries == ["src"]
    assert menu.handle_tab() is True and menu.candidates == ()  # 在途: 消费不重复启动


def test_tab_without_at_token_falls_through_to_original_editing():
    editor = _editor("plain")
    menu = _menu(editor)
    assert menu.handle_tab() is False  # 不消费: 原编辑行为 (制表符)


async def test_escape_locks_same_draft_until_change_or_tab():
    queries: list[str] = []

    async def complete(query, cancel):
        queries.append(query)
        return _result(FILE)

    editor = _editor("@src")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.dismiss() is True
    assert not menu.visible
    menu.sync(editor, eligible=True)  # 相同草稿/光标: 不重开
    assert not menu.visible and not menu.candidates
    editor.insert_text("r")  # 文本变化解锁
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.visible

    editor2 = _editor("@src")
    queries2: list[str] = []
    async def complete2(query, cancel):
        queries2.append(query)
        return _result(FILE)
    menu2 = _menu(editor2, complete2)
    menu2.sync(editor2, eligible=True)
    await asyncio.sleep(0.05)
    menu2.dismiss()
    menu2.handle_tab()  # 显式 Tab 解锁并立即查询
    assert queries2[-1] == "src"


# --- 查询生命周期 -------------------------------------------------------------------

async def test_debounce_coalesces_rapid_input_to_latest_query():
    queries: list[str] = []

    async def complete(query, cancel):
        queries.append(query)
        return _result(FILE)

    editor = _editor("@s")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    editor.insert_text("r")
    menu.sync(editor, eligible=True)
    editor.insert_text("c")
    menu.sync(editor, eligible=True)
    assert queries == []  # 20ms 内连续输入不启动查询
    await asyncio.sleep(0.08)
    assert queries == ["src"]


async def test_old_work_settles_before_latest_pending_starts():
    started = asyncio.Event()
    release_first = asyncio.Event()

    async def complete(query, cancel):
        if query == "a":
            started.set()
            try:
                await release_first.wait()
            except asyncio.CancelledError:
                raise
            return _result(FILE, DIR)
        return _result(FILE)

    editor = _editor("@a")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.wait_for(started.wait(), 1)
    editor.insert_text("b")  # 旧查询在途时输入变化
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)  # 新查询防抖到期但旧工作未落定
    release_first.set()
    await asyncio.sleep(0.1)  # 旧工作落定后最新待执行才启动
    assert menu.candidates == (FILE,)  # 旧结果 (FILE, DIR) 被丢弃, 新结果生效


async def test_ineligible_invalidation_cancels_work_and_drops_results():
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(query, cancel):
        started.set()
        await release.wait()
        return _result(FILE)

    editor = _editor("@src")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.wait_for(started.wait(), 1)
    menu.sync(editor, eligible=False)  # 模态/文档焦点优先: 候选失效并取消
    release.set()
    await asyncio.sleep(0.05)
    assert not menu.visible
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.08)
    assert menu.visible  # 重新获得资格后按新查询启动


async def test_unavailable_reason_maps_to_short_status():
    async def complete(query, cancel):
        return FileCompletionResult((), "unavailable", reason="not_git")

    editor = _editor("@src")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    assert menu.visible and menu.candidates == ()
    rendered = menu.render(40, 4)
    assert any("不是 Git 工作区" in line for line in rendered)


@pytest.mark.parametrize("scope,truncated,label", [
    ("git", True, "仅显示前 20 项"),
    ("directory", True, "当前目录 · 范围不完整"),
    ("directory", False, "当前目录"),
])
async def test_truncated_annotation_shown_with_candidates(scope, truncated, label):
    many = tuple(FileCompletionCandidate(f"f{i:02}.py", False) for i in range(20))

    async def complete(query, cancel):
        return FileCompletionResult(many, scope, truncated=truncated)

    editor = _editor("@")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.sleep(0.05)
    rendered = menu.render(40, 4)
    assert any(label in line for line in rendered)
    assert len([line for line in rendered if "f" in line]) == 3  # 最多 4 行含状态行


async def test_close_awaits_running_task_and_refuses_restart():
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(query, cancel):
        started.set()
        await release.wait()
        return _result(FILE)

    editor = _editor("@src")
    menu = _menu(editor, complete)
    menu.sync(editor, eligible=True)
    await asyncio.wait_for(started.wait(), 1)
    closing = asyncio.ensure_future(menu.close())
    await asyncio.sleep(0.05)
    assert not closing.done()  # 等待实际落定
    release.set()
    await asyncio.wait_for(closing, 1)
    menu.sync(editor, eligible=True)
    assert not menu.visible  # 封闭入口后不再启动
