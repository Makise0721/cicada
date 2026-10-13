"""主屏会话模态：复用纯选择状态，拥有查询与恢复等待，不写终端。"""
from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.keys import Keys

from cicada.application import ApplicationSession, SessionManagementError
from cicada.core.cancel import CancelToken
from cicada.ui.main_screen import CURSOR_MARKER, truncate_to_width
from cicada.ui.rendering import sanitize_display
from cicada.ui.session_selector import SessionSelectorState


class SessionResumeModal:
    def __init__(self, session: ApplicationSession, *, on_update: Callable[[], None],
                 on_error: Callable[[str], None]) -> None:
        self.session = session
        self.on_update = on_update
        self.on_error = on_error
        self.state = SessionSelectorState()
        self.visible = False
        self._closed = False
        self._generation = 0
        self._session_generation = 0
        self._task: asyncio.Task | None = None
        self._token: CancelToken | None = None
        self._submission = None
        self._refresh_pending = False
        self._conflict: tuple[str, str] | None = None

    @property
    def active(self) -> bool:
        return self.visible or self._task is not None

    def open(self) -> None:
        if self._closed or self.active:
            return
        self._session_generation = self.session.session_generation
        self.state = SessionSelectorState(current_session_id=self.session.persistence.session_id
                                          if self.session.persistence else None)
        self.visible = True
        self.refresh()

    def _current(self, generation: int) -> bool:
        return (not self._closed and self.visible and generation == self._generation
                and self._session_generation == self.session.session_generation)

    def refresh(self) -> None:
        self._generation += 1
        self._conflict = None
        self.state.begin_observation()
        if self._task is not None:
            self._refresh_pending = True
            if self._token:
                self._token.cancel()
            return
        self._token = CancelToken()
        self._task = asyncio.create_task(self._observe(self._generation, self._token))
        self.on_update()

    async def _observe(self, generation: int, token: CancelToken) -> None:
        try:
            snapshot = await self.session.observe_session_catalog(token)
            if self._current(generation) and not token.cancelled:
                self.state.accept_observation(snapshot)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._current(generation) and not token.cancelled:
                self.state.fail_observation(str(exc))
        finally:
            self._task = None
            self._token = None
            if self._refresh_pending and self.visible and not self._closed:
                self._refresh_pending = False
                self.refresh()
            self.on_update()

    def cancel(self) -> None:
        self.visible = False
        self._generation += 1
        self._conflict = None
        self._refresh_pending = False
        if self._token:
            self._token.cancel()
        if self._submission:
            self._submission.cancel()
        self.on_update()

    def handle_batch(self, batch) -> None:
        if not self.visible:
            return  # 取消收尾期仍捕获所有输入，不落入聊天。
        if self._submission is not None:
            if any(key.key in (Keys.Escape, Keys.ControlC) for key in batch):
                self.cancel()
            return
        if self._conflict is not None:
            for key in batch:
                if key.key in (Keys.Escape, Keys.ControlC):
                    self.cancel()
                    break
                if key.data in ("c", "C"):
                    self._resume(self._conflict[0], True)
                    break
                if key.key == Keys.ControlR:
                    self.refresh()
                    break
            return
        action, session_id = self.state.handle_batch(batch)
        if action == "cancel":
            self.cancel()
        elif action == "refresh":
            self.refresh()
        elif action == "select" and self._task is None:
            self._resume(session_id, False)
        self.on_update()

    def _resume(self, session_id: str, adopt: bool) -> None:
        try:
            submission = self.session.submit_resume_session(session_id, adopt_current_config=adopt)
        except SessionManagementError as exc:
            self.cancel()
            self.on_error(str(exc))
            return
        self._conflict = None
        self._submission = submission
        self._task = asyncio.create_task(self._wait_resume(submission, session_id, self._generation))
        self.on_update()

    async def _wait_resume(self, submission, session_id: str, generation: int) -> None:
        waiter = asyncio.create_task(submission.wait())
        try:
            while not waiter.done():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    submission.cancel()
                except Exception:
                    break
            try:
                waiter.result()
                self.visible = False
            except asyncio.CancelledError:
                self.visible = False
            except SessionManagementError as exc:
                if exc.code == "config-conflict" and self._current(generation):
                    self._conflict = (session_id, str(exc))
                else:
                    self.visible = False
                    if not self._closed:
                        self.on_error(str(exc))
        finally:
            self._submission = None
            self._task = None
            self.on_update()

    def render(self, width: int, rows: int) -> list[str]:
        if not self.visible or self._submission is not None:
            return [truncate_to_width("正在切换会话…" if self._submission else "正在结束查询…", width)]
        if self._conflict:
            session_id, message = self._conflict
            lines = ["目标配置冲突", "ID: " + session_id, sanitize_display(message),
                     "C 采用当前配置  Ctrl+R 返回列表  Esc 取消"]
            return [truncate_to_width(line, width) for line in lines][:max(1, rows)]
        return self.state.render_lines(width, rows, cursor_marker=CURSOR_MARKER)

    async def close(self) -> None:
        self._closed = True
        self.cancel()
        owned = self._task
        cancelled = False
        if owned is not None:
            while not owned.done():
                try:
                    await asyncio.shield(owned)
                except asyncio.CancelledError:
                    cancelled = True
                    self.cancel()
            owned.result()
        if cancelled:
            raise asyncio.CancelledError
