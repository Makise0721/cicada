"""独立审阅页：拥有串行读取与显示状态，由现有 frontend 唯一写屏。

窗格 list/file 沿 01；checks（检查记录详情）与 artifact（累计变更工件正文）
沿同一 lease/target/有界读取路径（02, C8），不建立第二 renderer。
"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.keys import Keys

from cicada.application import ApplicationSession
from cicada.review import ReviewCheckDetails, ReviewError, ReviewPreview, ReviewSnapshot
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
_INTEGRITY = {"verified": "核验相符", "missing": "缺失", "mismatch": "与记录不符",
              "unknown": "未能核验", "not_recorded": "未记录"}


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
        self.checks: ReviewCheckDetails | None = None
        self.group = "unstaged"
        self.target: str | None = None
        self.message = ""
        self._task: asyncio.Task | None = None
        self._started = False
        self._selected = 0
        self._actions: tuple[DisplayAction, ...] = ()
        self._visible_actions: tuple[DisplayAction, ...] = ()
        self._pane = "list"
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
        self.checks = None
        self._pane = "list"
        self._body_lines = ()
        self._body_layout = None
        self.message = "读取文件清单…"
        self._selected = 0
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
            self.checks = None
            self._pane = "list"
            self._body_lines = ()
            self._body_layout = None
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

    def _accept(self, generation: int, snapshot: ReviewSnapshot, target: str) -> bool:
        """读取返回验收：页面 generation/目标身份与旧异步结果纪律沿 01 (C4)。"""
        return (self.active and not self.closing and generation == self.generation
                and self.snapshot is snapshot and self.target == target)

    def _set_body(self, lines: list[str]) -> None:
        self._body_lines = tuple(sanitize_display(line) for line in lines)
        self._body_layout = DocumentLayout(self._body_lines, (), (None,) * len(self._body_lines))
        self._viewport = DocumentViewport()
        self._viewport.follow = False

    async def _read(self, generation: int, snapshot: ReviewSnapshot, target: str) -> None:
        self._started = True
        try:
            preview = await self._session.read_review(snapshot.review_id, target)
            if self._accept(generation, snapshot, target):
                self.preview = preview
                # 一次净化，逻辑行只在可见窗口按宽度截断，不每帧全量 wrap。
                self._set_body(preview.text.splitlines())
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

    async def _read_checks(self, generation: int, snapshot: ReviewSnapshot, target: str) -> None:
        self._started = True
        try:
            details = await self._session.read_review(snapshot.review_id, target)
            if self._accept(generation, snapshot, target):
                self.checks = details
                self._set_body(self._checks_lines(details))
                self.message = ""
        except ReviewError as exc:
            if not self.closing:
                self.message = str(exc)
        except asyncio.CancelledError:
            self._session.cancel_review()
            raise
        except Exception:
            if not self.closing:
                self.message = "检查详情读取失败，请重试"
        finally:
            self._changed()

    async def _read_artifact(self, generation: int, snapshot: ReviewSnapshot, target: str) -> None:
        self._started = True
        try:
            preview = await self._session.read_review(snapshot.review_id, target)
            if self._accept(generation, snapshot, target):
                self.preview = preview
                self._set_body(preview.text.splitlines())
                self.message = ""
        except ReviewError as exc:
            if not self.closing:
                self.message = str(exc)
        except asyncio.CancelledError:
            self._session.cancel_review()
            raise
        except Exception:
            if not self.closing:
                self.message = "工件读取失败，请重试"
        finally:
            self._changed()

    def _checks_lines(self, details: ReviewCheckDetails) -> list[str]:
        """详情行：历史执行、freshness、工件完整性与阻断原因分别呈现。"""
        lines = [
            f"验证 run {details.verification_run_id}",
            "当前快照 " + (details.snapshot_ref or "不可用 · freshness 只能是 unknown")
            + " · scope " + (details.scope_id or "未知"),
            "process_uncertain：" + ("是（锁存未解除）" if details.process_uncertain else "否"),
        ]
        for row in details.rows:
            execution = row.execution_status or "尚未运行"
            verdict = row.verification_status or "尚未运行"
            lines.append(
                f"{row.check_id} · 执行 {execution} · 判定 {verdict} · freshness {row.freshness}"
                f" · 工件 {_INTEGRITY.get(row.artifact_integrity, row.artifact_integrity)}")
            lines.append("  命令 " + row.command)
            if row.receipt_id:
                lines.append("  回执 " + row.receipt_id)
            if row.reason:
                lines.append("  " + row.reason)
        if not details.rows:
            lines.append("计划内没有检查定义")
        if details.blocking_reasons:
            lines.append("阻断原因：")
            lines.extend("  " + reason for reason in details.blocking_reasons)
        lines.append("检查范围与上方 Git 审阅范围（排除/忽略）可能不同")
        lines.append("详情为进入页面时读取；返回清单重新进入可刷新")
        return lines

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
            self.checks = None
            self._pane = "list"
            self._body_lines = ()
            self._body_layout = None
            self._start(self._open(self.generation))
        elif operation == "group" and target in _LABELS:
            self.group = target
            self.target = None
            self.preview = None
            self._pane = "list"
            self._body_lines = ()
            self._body_layout = None
            self.message = "选择文件查看差异"
            self._selected = 0
        elif operation == "list":
            if self._pane in ("checks", "artifact"):
                self.preview = None  # 离开检查/工件窗格不携带旧正文
            self._pane = "list"
        elif operation == "back_checks" and self.checks is not None:
            self._pane = "checks"
            # 返回详情重建正文：工件窗格覆盖过的 body 不能带进检查窗格。
            self._set_body(self._checks_lines(self.checks))
        elif operation == "checks" and self.snapshot is not None:
            if target != self.snapshot.checks_target_id:
                return False
            self.target = target
            self.preview = None
            self.checks = None
            self._pane = "checks"
            self._body_lines = ()
            self._body_layout = None
            self.message = "读取检查记录…"
            self._start(self._read_checks(self.generation, self.snapshot, target))
        elif operation == "artifact" and self.snapshot is not None and self.checks is not None:
            if target != self.checks.artifact_target_id:
                return False
            self.target = target
            self.preview = None
            self._pane = "artifact"
            self._body_lines = ()
            self._body_layout = None
            self.message = "读取累计变更工件…"
            self._start(self._read_artifact(self.generation, self.snapshot, target))
        elif operation == "file" and self.snapshot is not None:
            if not any(i.item_id == target and i.group == self.group for i in self.snapshot.items):
                return False
            self.target = target
            self.preview = None
            self._pane = "file"
            self._body_lines = ()
            self._body_layout = None
            self.message = "读取文件…"
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
            if self._pane == "list" and self.snapshot is not None:
                self._scroll_list(direction * self._list_height)
            else:
                self._viewport.page(direction)
        elif key.key is Keys.Home:
            if self._pane == "list":
                self._scroll_list(-self._list_top)
            else:
                self._viewport.go_home()
        elif key.key is Keys.End:
            if self._pane == "list" and self.snapshot is not None:
                self._scroll_list(len(self.snapshot.items))
            else:
                self._viewport.go_end()
        self._changed()

    def wheel(self, notches: int) -> None:
        if not self.closing:
            if self._pane == "list" and self.snapshot is not None:
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
        snapshot = self.snapshot
        files = tuple(i for i in snapshot.items if i.group == self.group) if snapshot else ()
        compact = width < 60 or rows < 14
        pane = self._pane
        previous = (self._actions[min(self._selected, len(self._actions)-1)] if self._actions else None)
        nav = [self._action("back"), self._action("refresh")]
        if pane in ("list", "file") and snapshot is not None:
            if pane == "file" and compact:
                nav.append(self._action("list"))
            else:
                if snapshot.checks_state == "available" and snapshot.checks_target_id:
                    nav.append(self._action("checks", snapshot.checks_target_id))
                if compact:
                    nav.append(self._action("group", self.group))
                else:
                    nav.extend(self._action("group", group) for group in _LABELS)
                nav.extend(self._action("file", item.item_id) for item in files)
        elif pane == "checks" and self.checks is not None:
            nav.append(self._action("list"))
            if self.checks.artifact_target_id:
                nav.append(self._action("artifact", self.checks.artifact_target_id))
        elif pane == "artifact" and self.preview is not None:
            nav.append(self._action("back_checks"))
        if previous in nav:
            self._selected = nav.index(previous)
        self._actions = tuple(nav)
        self._selected = min(self._selected, max(0, len(nav)-1))
        selected = nav[self._selected] if nav else None

        pad = "  " if width >= 20 else ""
        content_width = max(1, width - len(pad) * 2)

        def add(text: str, action: DisplayAction | None = None, *,
                color: str = theme.ASSISTANT_TEXT, bold: bool = False,
                active: bool = False) -> None:
            if len(lines) >= rows:
                return
            text = sanitize_display(text).replace("\t", "    ").replace("\n", " ")
            focused = action is not None and selected == action
            highlighted = focused or active
            text = truncate_to_width(("→ " if focused else "› " if active else "") + text, content_width)
            used = len(pad) + visible_width(text)
            if action is not None and text:
                hits.append(HitSpan(len(lines), len(pad), width if highlighted else used, action))
            # 背景行内只开样式，行尾一次复位；选中态与斜线菜单一致。
            styled = theme.fg_open(theme.EDITOR_BORDER if focused else color)
            if bold:
                styled += current_style().sgr("1")
            if highlighted:
                styled = theme.bg_open(theme.MENU_SEL_BG) + styled
            lines.append(styled + pad + text + (" " * (width - used) if highlighted else "")
                         + current_style().sgr("0"))

        def section(title: str) -> None:
            add(title + " " + "─" * max(0, content_width - visible_width(title) - 1),
                color=theme.EDITOR_BORDER, bold=True)

        # 顶部操作一行、极窄时缩短；刷新同时保留 R 键入口。
        labels = ("返回", "刷新") if width >= 12 else ("返", "刷")
        top = ""
        for operation, label in zip(("back", "refresh"), labels):
            start = visible_width(top)
            action = self._action(operation)
            label = ("→ " if action == selected else "  ") + label
            fragment = truncate_to_width(label, max(0, width-start))
            if fragment:
                hits.append(HitSpan(0, start, start + visible_width(fragment), action))
                if action == selected:
                    top += (theme.bg_open(theme.MENU_SEL_BG) + theme.fg_open(theme.EDITOR_BORDER)
                            + fragment + current_style().sgr("0"))
                else:
                    top += theme.fg(theme.MD_LINK, fragment)
            top += " "
        lines.append(truncate_to_width(top.rstrip(), width))
        if self.closing or snapshot is None:
            add(self.message, color=theme.MUTED_GRAY)
            self._visible_actions = tuple(h.action for h in hits)
            return DocumentLayout(tuple(lines), tuple(hits), (None,) * len(lines))
        if pane in ("list", "file"):
            if rows >= 12 and not (pane == "file" and compact):
                section("改动审阅")
                add(snapshot.workspace_label, color=theme.MUTED_GRAY)
            if pane == "file" and compact:
                add("返回列表", self._action("list"))
            else:
                # 每组有独立状态；未知不填零，不用一组空值推断整个工作区无改动。
                for group in snapshot.groups:
                    if compact and group.group != self.group:
                        continue
                    count = sum(i.group == group.group for i in snapshot.items)
                    state = str(count) if group.state == "complete" else group.reason or group.state
                    marker = "› " if self.group == group.group else "  "
                    hint = " ←/→" if compact else ""
                    add(marker + _LABELS[group.group] + "  " + state + hint,
                        self._action("group", group.group),
                        color=theme.EDITOR_BORDER if self.group == group.group else theme.MUTED_GRAY)
                if not compact:
                    add("")
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
                    color = (theme.DIFF_ADDED if item.kind == "added" else
                             theme.DIFF_REMOVED if item.kind in ("deleted", "conflict") else theme.MD_LINK)
                    add(f"{_STATUS[item.kind]}  {item.relative_path}",
                        self._action("file", item.item_id), color=color, active=self.target == item.item_id)
                if not files:
                    state = next(g for g in snapshot.groups if g.group == self.group)
                    add("该组无改动" if state.state == "complete" else "该组不可用，请查看原因")
            if rows - len(lines) > 4 and not (pane == "file" and compact):
                add(snapshot.scope, color=theme.TOOL_HINT_FG)
                # 三状态准确：未启用/本进程尚无记录只显示事实，有记录才可点击 (C8)。
                if snapshot.checks_state == "not_enabled":
                    add("检查：未启用", color=theme.MUTED_GRAY)
                elif snapshot.checks_state == "available" and snapshot.checks_target_id:
                    add("检查记录 · 点击查看", self._action("checks", snapshot.checks_target_id), color=theme.MD_LINK)
                else:
                    add("检查记录：本进程尚无检查记录", color=theme.MUTED_GRAY)
            if self.preview:
                if rows - len(lines) > 6:
                    add("")
                self._layout_preview(add, width, rows, lines)
            else:
                add(self.message, color=theme.MUTED_GRAY)
        elif pane == "checks":
            self._layout_checks(add, width, rows, lines)
        elif pane == "artifact":
            self._layout_artifact(add, width, rows, lines)
        self._visible_actions = tuple(h.action for h in hits)
        return DocumentLayout(tuple(lines), tuple(hits), (None,) * len(lines))

    def _layout_preview(self, add, width: int, rows: int, lines: list[str]) -> None:
        preview = self.preview
        if rows >= 5:
            add(preview.title + " · " + preview.comparison, color=theme.MD_LINK, bold=True)
        if rows >= 8:
            add(preview.state + " · " + (preview.notice or ""), color=theme.MUTED_GRAY)
        if rows - len(lines) > 2:
            add("正文观察 " + (preview.observed_at or "未知") + " · 长行按窗口截断", color=theme.TOOL_HINT_FG)
        height = rows - len(lines)
        body_frame = self._viewport.frame(self._body_layout or DocumentLayout((), (), ()), height)
        for line in body_frame:
            lines.append(self._diff_line(line, width))

    def _layout_checks(self, add, width: int, rows: int, lines: list[str]) -> None:
        details = self.checks
        add("检查记录 " + "─" * max(0, width - (4 if width >= 20 else 0) - 9),
            color=theme.EDITOR_BORDER, bold=True)
        if details is None:
            add(self.message, color=theme.MUTED_GRAY)
            return
        add("返回列表", self._action("list"), color=theme.MD_LINK)
        if details.artifact_target_id:
            add("查看" + (details.artifact_label or "累计变更工件"),
                self._action("artifact", details.artifact_target_id), color=theme.MD_LINK)
        if rows - len(lines) > 8:
            add("")
        height = rows - len(lines)
        body_frame = self._viewport.frame(self._body_layout or DocumentLayout((), (), ()), height)
        for line in body_frame:
            add(line, color=theme.TOOL_OUTPUT_FG)

    def _layout_artifact(self, add, width: int, rows: int, lines: list[str]) -> None:
        preview = self.preview
        if preview is None:
            add(self.message, color=theme.MUTED_GRAY)
            return
        if rows >= 5:
            add(preview.title + " · " + preview.comparison, color=theme.MD_LINK, bold=True)
        if rows >= 8:
            add(preview.state + " · " + _INTEGRITY.get(preview.integrity, preview.integrity)
                + " · " + (preview.notice or ""), color=theme.MUTED_GRAY)
        if rows - len(lines) > 2:
            add("返回检查详情", self._action("back_checks"), color=theme.MD_LINK)
        height = rows - len(lines)
        body_frame = self._viewport.frame(self._body_layout or DocumentLayout((), (), ()), height)
        for line in body_frame:
            lines.append(self._diff_line(line, width))

    @staticmethod
    def _diff_line(line: str, width: int) -> str:
        """单栏差异：复用工具卡增删底色、菜单分块与主屏文本亮度。"""
        pad = "  " if width >= 20 else ""
        text = truncate_to_width(line.replace("\t", "    "), max(1, width - len(pad) * 2))
        color, background = theme.ASSISTANT_TEXT, None
        if line.startswith(("+++", "---")):
            color = theme.MD_LINK
        elif line.startswith("+"):
            color, background = theme.DIFF_ADDED, theme.DIFF_ADDED_BG
        elif line.startswith("-"):
            color, background = theme.DIFF_REMOVED, theme.DIFF_REMOVED_BG
        elif line.startswith("@@"):
            color, background = theme.MD_LINK, theme.MENU_SEL_BG
        elif line.startswith(("diff ", "index ", "\\")):
            color = theme.MUTED_GRAY
        styled = theme.fg_open(color)
        if background:
            styled = theme.bg_open(background) + styled
        used = len(pad) + visible_width(text)
        return styled + pad + text + (" " * (width - used) if background else "") + current_style().sgr("0")

    def _changed(self) -> None:
        if self._on_update:
            self._on_update()
