"""cicada.ui: Pi 主屏前端生产组合 (票据 05).

--tui 公开入口的唯一前端: ConversationView (02 显示文档/准入/斜线命令) +
ToolDocument (03 工具卡) + CheckProjection (05 检查事实) + FooterComponent
(04 底栏) 注入 MainScreenFrontend (01 冻结: 唯一写屏 owner)。终端经 Terminal
协议注入: 缺省 PromptToolkitTerminal (真实终端), headless 检查注入 FakeTerminal。

组合层职责:
- 编辑器聚焦 (硬件光标/IME 跟随) 与 Ctrl+O 工具展开接线 (toggle + 立即帧);
  紧凑交互票据 01: 注入同源 layout_document 与 execute_display_action,
  前端自有文档视口 (帧 ≤ rows、F6 文档焦点、键盘回看) 随之启用;
- 生命周期: frontend 退出 (正常或不可恢复失败) 后 await session.aclose()
  (本函数是唯一 close owner); 关闭结果观察者与 ConversationView 订阅保留到
  ApplicationClosed 送达 (session 收尾清空观察者) 再退订;
- 退出码: 启动失败 2 (CLI 层); frontend 不可恢复失败/会话不可恢复崩溃/关闭抛错/
  ApplicationClosed.error 非空优先 1; 正常关闭取最近请求退出码, 无请求 0;
  frontend 异常与终端恢复错误经 stderr 如实留痕 (renderer 已停, 不写屏);
- renderer 停止后由同一终端适配独占写一次晚到关闭短错误 (不重启 renderer,
  不存在同时写屏的两个 owner)。
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Callable

from cicada.ui.conversation_view import ConversationView
from cicada.ui.commands import SlashCommandMenu
from cicada.ui.file_menu import FileCompletionMenu
from cicada.ui.frontend import MainScreenFrontend
from cicada.ui.terminal_io import PromptToolkitTerminal, Terminal

if TYPE_CHECKING:
    from cicada.application import ApplicationSession, ApplicationUpdate

_LATE_ERROR_MAX_CHARS = 200


async def run_tui_session(
    session: "ApplicationSession",
    *,
    initial_prompt: str | None = None,
    terminal: Terminal | None = None,
    diagnostics: Callable[[], list[str]] | None = None,
) -> int:
    """运行 Pi 主屏前端直至退出并完成应用关闭收尾; 返回进程退出码。

    session 由调用方 open; 关闭顺序: frontend 停止 (恢复终端模式) ->
    await session.aclose() (关闭监督) -> 退订观察者。diagnostics 为 /status
    注入的只读诊断行 (CLI 提供的不可变对象行化, 限额归 /status 组装层)。
    """
    if terminal is None:
        terminal = PromptToolkitTerminal()

    refs: dict[str, MainScreenFrontend] = {}

    def _request_render() -> None:
        frontend = refs.get("frontend")
        if frontend is not None:
            frontend.request_render()

    view = ConversationView(
        session,
        on_exit=lambda: _request_exit(refs),
        on_update=_request_render,
        diagnostics=diagnostics,
    )
    view.editor.focused = True  # 硬件光标标记: IME 候选窗跟随编辑光标
    # 会话不可恢复崩溃 (execute 异常且 state 已入 closing/closed): 退出码优先 1
    state: dict[str, bool] = {"unrecoverable": False}

    def _flag_unrecoverable() -> None:
        state["unrecoverable"] = True
        _request_exit(refs)

    view.on_unrecoverable = _flag_unrecoverable

    def _toggle_tools() -> None:
        view.handle_ctrl_o()
        refs["frontend"].request_render_now()

    def _mouse_unavailable_hint() -> None:
        # 票据 02 (C4): 鼠标启用失败退键盘, 一次短提示经显示文档如实成条
        error = getattr(terminal, "mouse_setup_error", None) or "unavailable"
        view.transcript.append_system(
            f"鼠标不可用（{error}）：已回退键盘操作（F6 回看，Ctrl+O 展开）。"
        )
        refs["frontend"].request_render_now()

    frontend = MainScreenFrontend(
        terminal,
        document_lines=view.document_lines,
        layout_document=view.layout_document,  # 紧凑交互 01: 同源布局 + 自有视口
        on_display_action=view.execute_display_action,
        on_mouse_unavailable=_mouse_unavailable_hint,  # 紧凑交互 02: 键盘回退提示
        editor=view.editor,
        on_submit=view.on_submit,
        on_escape=view.handle_escape,
        on_ctrl_c=view.handle_ctrl_c,
        on_ctrl_o=_toggle_tools,
        on_exit=view.handle_exit,
        footer_lines=view.footer.render,
        footer_badge_spans=lambda: view.footer.badge_spans,  # 票据 03: 徽标命中列
        on_shift_tab=view.handle_shift_tab,  # 票据 03: 思考强度循环 (C6)
        command_menu=SlashCommandMenu(),
        status_lines=view.status_lines,
        model_menu=view.model_menu,
        on_model_confirm=view.confirm_model_selection,
        modal=view.login_modal,
        review=view.review_view,
        display_style=view.display_style,
        # 文件补全 (project-context-v1 C3/C7): 候选端口注入应用只读出口;
        # busy Enter 屏障事实来自 view 的准入闸门; 异步候选落定经 _request_render
        file_menu=FileCompletionMenu(
            view.editor,
            complete=session.complete_files,
            enter_barrier=view.enter_barrier,
            on_update=_request_render,
        ),
    )
    refs["frontend"] = frontend
    view.on_session_prepare = frontend.prepare_session_change
    view.on_session_changed = frontend.reset_session_display

    # 关闭结果观察者: 独立于前端, 保留订阅直到 ApplicationClosed 送达
    close_state = _CloseState()

    def observe_close(update: "ApplicationUpdate") -> None:
        from cicada.application import ApplicationClosed, RequestCompleted, SessionChanged

        if isinstance(update, RequestCompleted):
            close_state.last_exit_code = update.outcome.exit_code
        elif isinstance(update, SessionChanged):
            close_state.last_exit_code = None
        elif isinstance(update, ApplicationClosed):
            close_state.closed_error = update.error

    unsubscribe_close = session.subscribe(observe_close)
    try:
        runner = asyncio.ensure_future(frontend.run_async())
        if initial_prompt is not None and initial_prompt.strip():
            await asyncio.sleep(0)  # 让 run_async 先行: terminal.start + 首帧
            view.on_submit(initial_prompt.strip())
        try:
            restore_errors = await runner
        except asyncio.CancelledError:
            await _close_application(session)
            view.close()
            raise
        except Exception as exc:
            # frontend 自身异常: 终端恢复由 run_async 收尾尽力而为, 应用关闭监督收尾;
            # 异常细节经 stderr 如实留痕 (renderer 已停, 不写屏), 退出 1
            await _close_application(session)
            view.close()
            _report_stderr('login frontend failure' if frontend.sensitive_input_used else f"frontend failure: {exc!r}")
            return 1
        if frontend.terminal_failed:
            await _close_application(session)
            view.close()
            return 1
        # frontend 正常退出: 应用关闭监督收尾; 晚到关闭错误优先退出 1
        close_failure = await _close_application(session)
        view.close()
        if close_state.closed_error is None:
            close_state.closed_error = close_failure
    finally:
        unsubscribe_close()

    # 停止阶段的终端恢复错误: renderer 已停, stderr 如实留痕 (不写屏、不改退出码)
    if restore_errors:
        _report_stderr("terminal restore errors: " + "; ".join(restore_errors))
    if close_state.closed_error is not None:
        _write_late_close_error(terminal, close_state.closed_error)
        return 1
    if state["unrecoverable"]:
        return 1
    return close_state.last_exit_code if close_state.last_exit_code is not None else 0


class _CloseState:
    """关闭观察者的可变状态 (RequestCompleted 退出码 / ApplicationClosed 错误)。"""

    def __init__(self) -> None:
        self.last_exit_code: int | None = None
        self.closed_error: str | None = None


def _request_exit(refs: dict[str, MainScreenFrontend]) -> None:
    frontend = refs.get("frontend")
    if frontend is not None:
        frontend.request_exit()


async def _close_application(session: "ApplicationSession") -> str | None:
    """应用关闭监督收尾; 返回关闭抛错文本 (无则 None)。

    aclose 返回/抛错都不代表资源必然全部释放: 监督超时由 ApplicationClosed.error
    与本返回值如实呈现, 不当作资源已释放。
    """
    try:
        await session.aclose()
    except Exception as exc:
        return f"close failed: {exc!r}"
    return None


def _report_stderr(message: str) -> None:
    """renderer 已停止后的组合层留痕: stderr 单行 (不写屏、不重启 renderer)。"""
    print(f"[cicada] {message}", file=sys.stderr)


def _write_late_close_error(terminal: Terminal, message: str) -> None:
    """renderer 停止后的唯一晚到错误出口: 同一终端适配独占写一次短错误。

    输出永久失效时不再启用 renderer (尽力而为, 不伪称屏幕已恢复)。
    """
    short = message if len(message) <= _LATE_ERROR_MAX_CHARS else message[:_LATE_ERROR_MAX_CHARS - 3] + "..."
    try:
        terminal.write(f"\n[cicada] 关闭错误: {short}\n")
    except Exception:
        pass


__all__ = ["run_tui_session"]
