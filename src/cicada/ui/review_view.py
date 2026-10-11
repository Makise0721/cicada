"""独立审阅页：拥有串行读取与显示状态，由现有 frontend 唯一写屏。"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.keys import Keys

from cicada.application import ApplicationSession
from cicada.review import ReviewError, ReviewPreview, ReviewSnapshot
from cicada.ui import theme
from cicada.ui.document_layout import DisplayAction, DocumentLayout, DocumentViewport, HitSpan
from cicada.ui.editor import Editor
from cicada.ui.display_style import current_style
from cicada.ui.main_screen import truncate_to_width, visible_width
from cicada.ui.rendering import sanitize_display

ACTION_REVIEW = "review"
_LABELS = {"unstaged": "未暂存", "staged": "已暂存", "untracked": "未跟踪"}
_STATUS = {"added": "+", "modified": "M", "deleted": "-", "type_changed": "T",
           "conflict": "!", "metadata": "?"}


class ReviewView:
    def __init__(self, session: ApplicationSession, editor: Editor,
                 on_update: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None) -> None:
        self._session = session
        self._editor = editor
        self._on_update = on_update
        self._on_error = on_error
        self.active = False
        self.generation = 0
        self.closing = False
        self.snapshot: ReviewSnapshot | None = None
        self.preview: ReviewPreview | None = None
        self.group = "unstaged"
        self.target: str | None = None
        self.message = ""
        self._task: asyncio.Task | None = None
        self._started = False
        self._selected = 0
        self._actions: tuple[DisplayAction, ...] = ()
        self._visible_actions: tuple[DisplayAction, ...] = ()
        self._body_mode = False
        self._body_lines: tuple[str, ...] = ()
        self._body_layout: DocumentLayout | None = None
        self._list_top = 0
        self._list_height = 1
        self._list_scroll_dirty = False
        self._viewport = DocumentViewport()
        self._viewport.follow = False

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def open(self) -> bool:
        if self.active:
            return False
        # 同步 reservation 先于 create_task；frontend 从这一刻接管所有输入。
        self.active = True
        self.closing = False
        self.generation += 1
        self.snapshot = None
        self.preview = None
        self._body_lines = ()
        self._body_layout = None
        self.message = "读取文件清单…"
        self._selected = 0
        self._body_mode = False
        self._start(self._open(self.generation, consume_command=True))
        return True

    def _start(self, operation) -> None:
        self._started = False
        self._task = asyncio.create_task(operation)
        self._task.add_done_callback(self._settled)
        self._changed()

    def _settled(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()
        if self.closing:
            self._session.cancel_review()
            if not self._session.review_active:
                self.active = False
                self.generation += 1
        self._changed()

    async def _open(self, generation: int, *, consume_command: bool = False) -> None:
        self._started = True
        try:
            snapshot = await self._session.open_review()
            if not self.active or self.closing or generation != self.generation:
                self._session.cancel_review()
                return
            self.snapshot = snapshot
            self.group = next((g.group for g in snapshot.groups if g.state == "complete"
                               and any(i.group == g.group for i in snapshot.items)), "unstaged")
            self.target = None
            self.preview = None
            self.message = "选择文件查看差异"
            if consume_command:
                self._editor.clear()
        except ReviewError as exc:
            if self.closing:
                return
            if consume_command:
                # 准入失败不消费命令，短错误由页退回后显示到原草稿旁。
                self.message = str(exc)
                self.active = False
                self.generation += 1
                if self._on_error:
                    self._on_error(self.message)
            else:
                self.message = str(exc)
        except asyncio.CancelledError:
            self._session.cancel_review()
            raise
        except Exception:
            self.message = "审阅读取失败"
            if self._on_error and not self.closing:
                self._on_error(self.message)
            self._session.cancel_review()
            self.closing = True
        finally:
            self._changed()

    def cancel(self) -> None:
        if not self.active:
            return
        self.generation += 1
        self.closing = True
        self.message = "正在结束读取…"
        self._session.cancel_review()
        # 起跑前取消拥有 open task，不能只取消尚 inactive 的应用 lease。
        if self.busy and not self._started:
            self._task.cancel()
        if not self.busy and not self._session.review_active:
            self.active = False
        self._changed()

    async def _read(self, generation: int, snapshot: ReviewSnapshot, target: str) -> None:
        self._started = True
        try:
            preview = await self._session.read_review(snapshot.review_id, target)
            if (self.active and not self.closing and generation == self.generation
                    and self.snapshot is snapshot and self.target == target):
                self.preview = preview
                # 一次净化，逻辑行只在可见窗口按宽度截断，不每帧全量 wrap。
                self._body_lines = tuple(sanitize_display(preview.text).splitlines())
                self._body_layout = DocumentLayout(self._body_lines, (), (None,) * len(self._body_lines))
                self._viewport = DocumentViewport()
                self._viewport.follow = False
                self.message = ""
        except ReviewError as exc:
            if not self.closing:
                self.message = str(exc)
        except asyncio.CancelledError:
            self._session.cancel_review()
            raise
        except Exception:
            if not self.closing:
                self.message = "文件读取失败，请刷新"
        finally:
            self._changed()

    def _action(self, operation: str, target: str = "") -> DisplayAction:
        identity = self.snapshot.review_id if self.snapshot is not None else str(self.generation)
        return DisplayAction(ACTION_REVIEW, (identity, operation, target))

    def execute(self, action: DisplayAction) -> bool:
        if not self.active or action.kind != ACTION_REVIEW or len(action.target_id) != 3:
            return False
        identity, operation, target = action.target_id
        if identity != self._action(operation, target).target_id[0]:
            return False
        if operation == "back":
            self.cancel()
            return True
        if self.closing or self.busy:
            return False
        self.generation += 1
        if operation == "refresh":
            self.message = "刷新文件清单…"
            self.preview = None
            self._body_lines = ()
            self._body_layout = None
            self._body_mode = False
            self._start(self._open(self.generation))
        elif operation == "group" and target in _LABELS:
            self.group = target
            self.target = None
            self.preview = None
            self._body_lines = ()
            self._body_layout = None
            self._body_mode = False
            self.message = "选择文件查看差异"
            self._selected = 0
        elif operation == "list":
            self._body_mode = False
        elif operation == "file" and self.snapshot is not None:
            if not any(i.item_id == target and i.group == self.group for i in self.snapshot.items):
                return False
            self.target = target
            self.preview = None
            self._body_lines = ()
            self._body_layout = None
            self.message = "读取文件…"
            self._body_mode = True
            self._start(self._read(self.generation, self.snapshot, target))
        else:
            return False
        self._changed()
        return True

    def handle(self, key) -> None:
        data = key.data
        if key.key in (Keys.Escape, Keys.ControlC) or data in ("\x1b", "\x03"):
            self.cancel()
            return
        if self.closing or self.busy:
            return
        if data in ("r", "R"):
            self.execute(self._action("refresh"))
        elif key.key in (Keys.Up, Keys.Down):
            self._list_scroll_dirty = False
            if self._actions:
                self._selected = (self._selected + (-1 if key.key is Keys.Up else 1)) % len(self._actions)
        elif key.key in (Keys.Left, Keys.Right):
            groups = tuple(_LABELS)
            index = groups.index(self.group)
            self.execute(self._action("group", groups[(index + (-1 if key.key is Keys.Left else 1)) % 3]))
        elif key.key is Keys.ControlM or data == "\r":
            if self._actions:
                action = self._actions[min(self._selected, len(self._actions)-1)]
                if action in self._visible_actions:
                    self.execute(action)
        elif key.key in (Keys.PageUp, Keys.PageDown):
            direction = -1 if key.key is Keys.PageUp else 1
            if self.preview is None and self.snapshot is not None:
                self._scroll_list(direction * self._list_height)
            else:
                self._viewport.page(direction)
        elif key.key is Keys.Home:
            if self.preview is None:
                self._scroll_list(-self._list_top)
            else:
                self._viewport.go_home()
        elif key.key is Keys.End:
            if self.preview is None and self.snapshot is not None:
                self._scroll_list(len(self.snapshot.items))
            else:
                self._viewport.go_end()
        self._changed()

    def wheel(self, notches: int) -> None:
        if not self.closing:
            if self.preview is None and self.snapshot is not None:
                self._scroll_list(-notches * 3)
            else:
                self._viewport.wheel_scroll(notches)
            self._changed()

    def _scroll_list(self, delta: int) -> None:
        count = sum(i.group == self.group for i in self.snapshot.items) if self.snapshot else 0
        self._list_top = max(0, min(self._list_top + delta, count - self._list_height))
        self._list_scroll_dirty = True

    def layout(self, width: int, rows: int) -> DocumentLayout:
        width, rows = max(1, width), max(1, rows)
        lines: list[str] = []
        hits: list[HitSpan] = []
        files = tuple(i for i in self.snapshot.items if i.group == self.group) if self.snapshot else ()
        compact_body = self._body_mode and (width < 60 or rows < 14)
        compact_list = width < 60 or rows < 14
        previous = (self._actions[min(self._selected, len(self._actions)-1)] if self._actions else None)
        nav = [self._action("back"), self._action("refresh")]
        if compact_body:
            nav.append(self._action("list"))
        elif self.snapshot:
            nav.extend(self._action("group", group) for group in (_LABELS if not compact_list else (self.group,)))
            nav.extend(self._action("file", item.item_id) for item in files)
        if previous in nav:
            self._selected = nav.index(previous)
        self._actions = tuple(nav)
        self._selected = min(self._selected, max(0, len(nav)-1))
        selected = nav[self._selected] if nav else None

        def add(text: str, action: DisplayAction | None = None) -> None:
            if len(lines) >= rows:
                return
            text = truncate_to_width(sanitize_display(text).replace("\t", "    ").replace("\n", " "), width)
            if action is not None:
                if selected == action and current_style().plain:
                    text = truncate_to_width("> " + text, width)
                if selected == action:
                    text = theme.bg_open(theme.MENU_SEL_BG) + text + current_style().sgr("0")
                hits.append(HitSpan(len(lines), 0, visible_width(text), action))
            lines.append(text)

        # 顶部操作一行、极窄时缩短；刷新同时保留 R 键入口。
        labels = ("返回", "刷新") if width >= 12 else ("返", "刷")
        top = ""
        for operation, label in zip(("back", "refresh"), labels):
            start = visible_width(top)
            action = self._action(operation)
            label = (">" if action == selected else " ") + label
            fragment = truncate_to_width(label, max(0, width-start))
            if fragment:
                hits.append(HitSpan(0, start, start + visible_width(fragment), action))
                top += fragment
            top += " "
        lines.append(truncate_to_width(top.rstrip(), width))
        if self.closing or self.snapshot is None:
            add(self.message)
            self._visible_actions = tuple(h.action for h in hits)
            return DocumentLayout(tuple(lines), tuple(hits), (None,) * len(lines))
        if rows >= 12:
            add("审阅 · " + self.snapshot.workspace_label)
        if compact_body:
            add("返回列表", self._action("list"))
        else:
            # 每组有独立状态；未知不填零，不用一组空值推断整个工作区无改动。
            for group in self.snapshot.groups:
                if compact_list and group.group != self.group:
                    continue
                count = sum(i.group == group.group for i in self.snapshot.items)
                state = str(count) if group.state == "complete" else group.reason or group.state
                marker = "> " if self.group == group.group else "  "
                hint = " ←/→" if compact_list else ""
                add(marker + _LABELS[group.group] + "  " + state + hint, self._action("group", group.group))
            list_height = max(1, min(6, (rows - len(lines)) // 3))
            selected_file = next((n for n, item in enumerate(files)
                                  if selected == self._action("file", item.item_id)), 0)
            self._list_height = list_height
            if self._list_scroll_dirty:
                start = max(0, min(self._list_top, len(files) - list_height))
            else:
                start = max(0, min(selected_file - list_height//2, len(files) - list_height))
            self._list_top = start
            for item in files[start:start+list_height]:
                add(f"{_STATUS[item.kind]} {item.relative_path}", self._action("file", item.item_id))
            if not files:
                state = next(g for g in self.snapshot.groups if g.group == self.group)
                add("该组无改动" if state.state == "complete" else "该组不可用，请查看原因")
        if rows - len(lines) > 4:
            add(self.snapshot.scope)
            add("检查：未启用" if self.snapshot.checks_state == "not_enabled" else "检查记录：尚未提供详情")
        if self.preview:
            preview = self.preview
            if rows >= 5:
                add(preview.title + " · " + preview.comparison)
            if rows >= 8:
                add(preview.state + " · " + (preview.notice or ""))
            if rows - len(lines) > 2:
                add("正文观察 " + (preview.observed_at or "未知") + " · 长行按窗口截断")
            height = rows - len(lines)
            body_frame = self._viewport.frame(self._body_layout or DocumentLayout((), (), ()), height)
            for line in body_frame:
                line = truncate_to_width(line.replace("\t", "    "), width)
                color = (theme.DIFF_ADDED if line.startswith("+") else
                         theme.DIFF_REMOVED if line.startswith("-") else theme.MUTED_GRAY)
                lines.append(theme.fg(color, line))
        else:
            add(self.message)
        self._visible_actions = tuple(h.action for h in hits)
        return DocumentLayout(tuple(lines), tuple(hits), (None,) * len(lines))

    def _changed(self) -> None:
        if self._on_update:
            self._on_update()
