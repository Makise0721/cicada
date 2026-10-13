"""启动 Terminal adapter；拥有观察任务，搜索/分页/render 使用纯内存状态。

旧无 observe 调用通过应用组合 helper 创建任意目录的 legacy reader；生产
调用显式注入绑定工作区的观察方法。取消先排空观察，再恢复终端并返回。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from cicada.core.cancel import CancelToken
from cicada.persistence.contracts import LIST_SESSIONS_PAGE_SIZE, SessionCatalogSnapshot
from cicada.ui.session_selector import SELECTOR_TITLE, SessionSelectorState
from cicada.ui.terminal_io import Terminal
from cicada.ui.display_style import DisplayStyle

_CLEAR_SCREEN = "\x1b[2J\x1b[H"


async def run_session_selector(
    terminal: Terminal, directory: Path, *, limit: int = LIST_SESSIONS_PAGE_SIZE,
    observe: Callable[[CancelToken], Awaitable[SessionCatalogSnapshot]] | None = None,
) -> tuple[str | None, list[str]]:
    reader = None
    if observe is None:
        from cicada.application import make_session_catalog_reader

        reader = make_session_catalog_reader(directory=Path(directory))
        observe = reader.observe
    state = SessionSelectorState(limit=limit)
    done = asyncio.Event()
    result = None
    active: asyncio.Task | None = None
    token: CancelToken | None = None
    generation = 0
    closed = False
    refresh_pending = False
    failure: Exception | None = None
    display_style = DisplayStyle.capture()

    def render() -> None:
        nonlocal failure, closed, generation
        if not closed:
            try:
                with display_style.use():
                    lines = state.render_lines(terminal.columns, terminal.rows)
                terminal.write(_CLEAR_SCREEN + "\n".join(lines) + "\n")
            except Exception as exc:
                failure, closed = exc, True
                generation += 1
                if token is not None:
                    token.cancel()
                done.set()

    async def load(version: int, cancel: CancelToken) -> None:
        nonlocal active, refresh_pending
        try:
            snapshot = await observe(cancel)
            if not closed and generation == version and not cancel.cancelled:
                state.accept_observation(snapshot)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not closed and generation == version and not cancel.cancelled:
                state.fail_observation(f"{type(exc).__name__}: {exc}")
        finally:
            active = None
            if not closed:
                if refresh_pending:
                    refresh_pending = False
                    start_observation()
                render()

    def start_observation() -> None:
        nonlocal active, token, generation, refresh_pending
        generation += 1
        state.begin_observation()
        if active is not None:
            refresh_pending = True
            token.cancel()
            return
        token = CancelToken()
        active = asyncio.create_task(load(generation, token))

    def on_input(batch) -> None:
        nonlocal closed, result, generation
        if closed:
            return
        action, session_id = state.handle_batch(batch)
        if action in ("cancel", "select"):
            closed = True
            generation += 1
            result = session_id
            if token is not None:
                token.cancel()
            done.set()
            return
        if action == "refresh":
            start_observation()
        render()

    errors = []
    try:
        terminal.start(on_input, render)
        render()
        if not closed:
            start_observation()
        await done.wait()
        if failure is not None:
            raise failure
    finally:
        closed = True
        generation += 1
        if token is not None:
            token.cancel()
        cancelled = False
        owned = active
        if owned is not None:
            while not owned.done():
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError:
                    cancelled = True
            owned.result()
        try:
            if reader is not None:
                await reader.aclose()
        finally:
            errors = terminal.stop()
        if cancelled:
            raise asyncio.CancelledError
    return result, list(errors)


__all__ = ["SELECTOR_TITLE", "run_session_selector"]
