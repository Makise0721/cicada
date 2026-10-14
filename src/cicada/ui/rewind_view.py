"""/rewind 查看、计划与确认页；输入和写屏由主屏 frontend 统一拥有。"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.keys import Keys

from cicada.application import ApplicationSession
from cicada.restore_points import RewindError, RewindListView, RewindPointView, RewindPreview, SCOPE_NOTE
from cicada.ui import theme
from cicada.ui.display_style import current_style
from cicada.ui.document_layout import DisplayAction, DocumentLayout, DocumentViewport, HitSpan
from cicada.ui.editor import Editor
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.rendering import sanitize_display
from cicada.ui.review_view import ReviewView

ACTION_REWIND = "rewind"
_AVAILABILITY = {"available": "可用", "expired": "已清理", "incomplete": "不完整"}
_SOURCE = {"committed": "完整请求", "interrupted": "中断请求", "unknown": "请求来源未知"}
_RELATION = {"clean": "与请求后相符", "later_request": "后继请求", "external": "请求外改动",
             "missing": "当前文件缺失", "unknown": "当前内容未知"}


class RewindView:
    def __init__(self, session: ApplicationSession, editor: Editor,
                 on_update: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None) -> None:
        self._session, self._editor = session, editor
        self._on_update, self._on_error = on_update, on_error
        self.active = self.closing = False
        self.generation = 0
        self.snapshot: RewindListView | None = None
        self.point: RewindPointView | None = None
        self.preview: RewindPreview | None = None
        self.plan = self.result = None
        self.message = ""
        self._task: asyncio.Task | None = None
        self._started = False
        self._selected = 0
        self._actions: tuple[DisplayAction, ...] = ()
        self._visible_actions: tuple[DisplayAction, ...] = ()
        self._top = 0
        self._height = 1
        self._nav_count = 2
        self._viewport = DocumentViewport()
        self._viewport.follow = False
        self._body = DocumentLayout((), (), ())

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def open(self) -> bool:
        if self.active:
            return False
        self.active, self.closing = True, False
        self.generation += 1
        self.snapshot = self.point = self.preview = None
        self.plan = self.result = None
        self._top = self._selected = 0
        self.message = "读取还原点…"
        self._start("open", consume=True)
        return True

    def _start(self, operation: str, target: str = "", *, consume=False) -> None:
        self._started = False
        self._task = asyncio.create_task(self._read(operation, target, self.generation, consume))
        self._task.add_done_callback(self._settled)
        self._changed()

    def _settled(self, task):
        if not task.cancelled():
            task.exception()
        if self.closing and not self._session.rewind_active:
            self.active = False
            self.generation += 1
        self._changed()

    async def _read(self, operation, target, generation, consume):
        self._started = True
        try:
            if operation == "open":
                value = await self._session.open_rewind()
            elif operation == "point":
                value = await self._session.read_rewind_point(target)
            elif operation == "plan":
                value = await self._session.plan_rewind_restore(target)
            elif operation == "confirm":
                value = await self._session.execute_rewind_restore(target)
            else:
                value = await self._session.read_rewind_preview(self.point.point_id, target)
            if self.closing or generation != self.generation:
                return
            if operation == "open":
                self.snapshot = value
                if consume:
                    self._editor.clear()
                self.message = "选择还原点查看文件" if value.points else "本会话无可还原改动（可能未改文件或捕获已关闭）"
            elif operation == "point":
                self.point = value
                self.message = "选择文件查看 before → after" if value.entries else "该点无可还原条目"
            elif operation in ("plan", "confirm"):
                if operation == "plan":
                    self.plan = value
                    lines = ["确认后将覆盖以下路径："]
                    lines.extend(f"{p.relative_path}: {p.op} · {_RELATION[p.current_relation]}"
                                 for p in value.paths)
                    lines.append("将删除：")
                    lines.extend(value.deletions or ("无",))
                    lines.append("后继影响：")
                    lines.extend(value.successor_impacts or ("无",))
                    self.message = "核对路径、删除与后继影响后，选择确认还原"
                else:
                    self.plan = None
                    self.result = value
                    lines = [f"还原备份点：{value.backup_point_id or '未生成'}"]
                    lines.extend(f"{p.relative_path}: {p.state}"
                                 + (f" · winerror={p.winerror}" if p.winerror is not None else "")
                                 + (f" · {p.notice}" if p.notice else "") for p in value.files)
                    lines.append(value.notice or "既有检查回执已过期；交付前请重新检查。")
                    self.message = "还原已结束；逐文件结果如下"
                self._body = DocumentLayout(tuple(sanitize_display(line) for line in lines), (),
                                            (None,) * len(lines))
                self._viewport = DocumentViewport()
                self._viewport.follow = False
            else:
                self.preview = value
                lines = tuple(sanitize_display(line) for line in value.text.splitlines())
                self._body = DocumentLayout(lines, (), (None,) * len(lines))
                self._viewport = DocumentViewport()
                self._viewport.follow = False
                self.message = value.notice or value.state
        except RewindError as exc:
            if self.closing:
                return
            self.message = str(exc)
            self.plan = None
            if consume or not self._session.rewind_active:
                self.closing = True
                self._session.cancel_rewind()
                if self._on_error:
                    self._on_error(self.message)
        except asyncio.CancelledError:
            self.closing = True
            self._session.cancel_rewind()
            raise
        except Exception:
            if not self.closing and self._on_error:
                self._on_error("还原点读取失败，请重新打开")
            self.message = "还原点读取失败，请重新打开"
            self.closing = True
            self._session.cancel_rewind()
        finally:
            # 监督/调用方取消不等于线程已释放；保留页面输入屏障到实际落定。
            self._changed()
            while self.closing and self._session.rewind_active:
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    self._session.cancel_rewind()

    def cancel(self) -> None:
        if not self.active:
            return
        self.generation += 1
        self.closing = True
        self.message = "正在结束读取…"
        self._session.cancel_rewind()
        if self.busy and not self._started:
            self._task.cancel()
        if not self.busy and not self._session.rewind_active:
            self.active = False
        self._changed()

    def _action(self, operation, target=""):
        return DisplayAction(ACTION_REWIND, (str(self.generation), operation, target))

    def execute(self, action: DisplayAction) -> bool:
        if not self.active or action.kind != ACTION_REWIND or len(action.target_id) != 3:
            return False
        generation, operation, target = action.target_id
        if generation != str(self.generation):
            return False
        if operation == "back":
            self.cancel()
            return True
        if self.closing or self.busy:
            return False
        if operation == "point" and not (self.snapshot and any(p.point_id == target for p in self.snapshot.points)):
            return False
        if operation == "file" and not (self.point and any(e.entry_id == target for e in self.point.entries)):
            return False
        if operation == "plan" and not (self.point and target == self.point.point_id):
            return False
        if operation == "confirm" and not (self.plan and target == self.plan.plan_id):
            return False
        if operation not in ("refresh", "list", "point", "file", "plan", "confirm"):
            return False
        self.generation += 1
        self.preview = None
        self.plan = self.result = None
        self._body = DocumentLayout((), (), ())
        self._top = self._selected = 0
        if operation in ("refresh", "list"):
            self.point = None
            self.message = "选择还原点查看文件"
            if operation == "refresh":
                self.snapshot = None
                self._start("open")
        else:
            if operation == "point":
                self.point = None
            self.message = "读取中…"
            self._start(operation, target)
        self._changed()
        return True

    def handle(self, key) -> None:
        if key.key in (Keys.Escape, Keys.ControlC) or key.data in ("\x1b", "\x03"):
            self.cancel()
            return
        if self.closing or self.busy:
            return
        if key.data in ("r", "R"):
            self.execute(self._action("refresh"))
        elif key.key in (Keys.Up, Keys.Down) and self._actions:
            self._selected = (self._selected + (-1 if key.key is Keys.Up else 1)) % len(self._actions)
            if self._selected >= self._nav_count:
                self._top = max(0, self._selected - self._nav_count - self._height // 2)
        elif key.key is Keys.ControlM or key.data == "\r":
            if self._actions and self._actions[self._selected] in self._visible_actions:
                self.execute(self._actions[self._selected])
        elif key.key in (Keys.PageUp, Keys.PageDown):
            self.wheel(1 if key.key is Keys.PageUp else -1)
        elif key.key is Keys.Home:
            self._top = 0
            self._viewport.go_home()
        elif key.key is Keys.End:
            self._top = len(self.point.entries if self.point else self.snapshot.points) if self.snapshot else 0
            self._viewport.go_end()
        self._changed()

    def wheel(self, notches: int) -> None:
        if not self.closing:
            if self.preview or self.plan or self.result:
                self._viewport.wheel_scroll(notches)
            else:
                self._top = max(0, self._top - notches * 3)
            self._changed()

    def layout(self, width: int, rows: int) -> DocumentLayout:
        width, rows = max(1, width), max(1, rows)
        lines, hits = [], []
        nav = [("返回", self._action("back")), ("刷新", self._action("refresh"))]
        if self.point:
            nav.append(("还原点列表", self._action("list")))
            if self.plan:
                nav.append(("确认还原", self._action("confirm", self.plan.plan_id)))
            elif not self.result and self.point.availability != "expired" and self.point.entries:
                nav.append(("计划还原此点", self._action("plan", self.point.point_id)))
        items = []
        if self.plan or self.result:
            pass
        elif self.point:
            for entry in self.point.entries:
                relation = _RELATION[entry.current_relation]
                if entry.current_relation == "later_request":
                    relation += " #" + (str(entry.later_seq) if entry.later_seq is not None else "未知")
                items.append((f"{'新建' if entry.op == 'create' else '修改'} {entry.relative_path} · {entry.size_bytes} B · {relation}",
                              self._action("file", entry.entry_id)))
        elif self.snapshot:
            for point in self.snapshot.points:
                label = "还原备份点" if point.kind == "pre_restore" else "请求"
                seq = str(point.seq) if point.seq is not None else "未知"
                items.append((f"{label} #{seq} · {_AVAILABILITY[point.availability]} · {_SOURCE[point.source]}"
                              + (" · " + point.note if point.note else "")
                              + f" · {point.entry_count} 文件 · {point.created_at}", self._action("point", point.point_id)))
        self._nav_count = len(nav)
        self._actions = tuple(action for _, action in nav + items)
        self._selected = min(self._selected, max(0, len(self._actions) - 1))
        selected = self._actions[self._selected]
        def add(text, action=None, color=theme.ASSISTANT_TEXT):
            if len(lines) >= rows:
                return
            focused = action == selected
            text = sanitize_display(text).replace("\t", "    ").replace("\n", " ")
            text = truncate_to_width(("→ " if focused else "") + text, width)
            if action and text:
                hits.append(HitSpan(len(lines), 0, width, action))
            style = theme.fg_open(color)
            if focused:
                style = theme.bg_open(theme.MENU_SEL_BG) + style
            lines.append(style + text + (" " * (width - visible_width(text)) if focused else "") + current_style().sgr("0"))
        # 固定范围声明优先于列表/预览，不被长正文挤走。
        for label, action in nav:
            add(label, action, theme.MD_LINK)
        add(SCOPE_NOTE, color=theme.TOOL_HINT_FG)
        if self.closing:
            add(self.message, color=theme.MUTED_GRAY)
        else:
            add("请求还原点" + (" · " + _AVAILABILITY[self.point.availability] if self.point else ""), color=theme.EDITOR_BORDER)
            self._height = max(1, min(6 if self.preview else rows, (rows - len(lines) - 2) // (3 if self.preview else 1)))
            self._top = min(self._top, max(0, len(items) - self._height))
            for label, action in items[self._top:self._top + self._height]:
                add(label, action)
            add(self.message, color=theme.MUTED_GRAY)
            if self.preview:
                add("before → after · " + self.preview.state + " · " + self.preview.observed_at, color=theme.MD_LINK)
                lines.extend(ReviewView._diff_line(line, width) for line in self._viewport.frame(self._body, rows - len(lines)))
            elif self.plan or self.result:
                lines.extend(truncate_to_width(line, width)
                             for line in self._viewport.frame(self._body, rows - len(lines)))
        self._visible_actions = tuple(hit.action for hit in hits)
        return DocumentLayout(tuple(lines), tuple(hits), (None,) * len(lines))

    def _changed(self):
        if self._on_update:
            self._on_update()
