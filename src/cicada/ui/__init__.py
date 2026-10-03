"""cicada.ui: 终端交互与展示投影 (TUI 第一版).

02 发布共享数据契约 (contracts); 终端执行器 (terminal, 03) 与展示投影
(presenter/rendering, 04) 的实现进入子模块. 本模块是 05 生产组合:
共享应用 + 正式 presenter + Rich 渲染 + 终端执行器, 只由 `--tui` 分支导入
(普通单次模式无 UI 导入要求); 组合不新增接口语义, 只做三件事:
- ProjectionPresenter (04) 作为 TuiPresenter 注入 TerminalExecutor (03);
- RenderingSink 把 PresentationRecord 经 record_fragments 渲染为
  prompt_toolkit styled fragments, ANSI 串不作普通文本 print;
- ProductionTerminalExecutor 覆盖记录渲染 / 活动预览 / 状态栏投影, 并在
  最终回答记录追加时收起活动预览 (正式 presenter 的记录键为 run:turn:assistant).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from prompt_toolkit.shortcuts import print_formatted_text

from cicada.ui.contracts import (
    PresentationRecord,
    PresentationUpdate,
    StatusSnapshot,
)
from cicada.ui.terminal import TerminalExecutor

if TYPE_CHECKING:
    from cicada.application import ApplicationSession, ApplicationUpdate


class RenderingSink:
    """生产输出 seam: 记录经渲染管线输出, 行文本原样; 全部经 prompt_toolkit 协调."""

    def __init__(self) -> None:
        # 已渲染记录的可见纯文本 (供测试/诊断断言, 不含样式)
        self.rendered: list[str] = []

    def print_line(self, text: str = "") -> None:
        print_formatted_text(text)

    def print_record(self, record: PresentationRecord) -> None:
        from cicada.ui.rendering import record_fragments

        fragments = record_fragments(record)
        self.rendered.append("".join(text for _, text in fragments))
        print_formatted_text(fragments)


class ProductionTerminalExecutor(TerminalExecutor):
    """正式 presenter 组合: 记录键/状态字段与 03 基础 presenter 不同, 覆盖投影."""

    def _apply(self, update: PresentationUpdate) -> None:
        # 正式 presenter 的最终回答记录 (键 run:turn:assistant) 追加时收起活动预览;
        # 基础 presenter 的键 assistant:run:turn 由父类自身的收起逻辑处理
        for record in update.appended:
            if (
                record.role == "assistant"
                and self._display_preview is not None
                and record.key.startswith(f"{self._display_preview.run_id}:")
            ):
                self._display_preview = None
        for record in update.appended:
            if record.key in self._printed_keys:
                continue
            self._printed_keys.add(record.key)
            self._sink.print_record(record)  # type: ignore[attr-defined]
        if update.preview is not None:
            self._display_preview = update.preview
        if update.status is not None:
            self._display_status = update.status

    def _get_preview_lines(self) -> Any:
        preview = self._display_preview
        if preview is None:
            return super()._get_preview_lines()
        from cicada.ui.rendering import preview_fragments

        return preview_fragments(preview)

    def _get_status_line(self) -> Any:
        from cicada.ui.rendering import status_fragments

        snapshot = self._display_status
        if snapshot is None:
            # 首个请求前的回退快照取 session 只读事实, 口径与 /status 一致:
            # 检查模式启动窗口不得把已有验证运行显示成"未启用检查"
            snapshot = StatusSnapshot(
                model=self._session.config.model_label or None,
                workspace=str(self._session.config.workspace),
                phase=self._session.state,
                verification_run_id=self._session.verification_run_id,
            )
        return status_fragments(snapshot)

    def _status_summary(self) -> str:
        """/status 的纯文本摘要: 在基础字段上补检查事实 (口径与渲染一致)."""
        base = super()._status_summary()
        snapshot = self._display_status
        if snapshot is None:
            return base
        check = snapshot.check_status
        if check is None:
            check = "未启用检查" if snapshot.verification_run_id is None else "不可得"
        extras = [f"check={check}"]
        if snapshot.change_scope is not None:
            extras.append(snapshot.change_scope)
        return f"{base} {' '.join(extras)}"


def build_production_executor(
    session: "ApplicationSession",
    *,
    input: Any = None,
    output: Any = None,
    sink: RenderingSink | None = None,
) -> ProductionTerminalExecutor:
    """按生产组合构建执行器: 正式 presenter + RenderingSink (input/output/sink 可注入)."""
    from cicada.ui.presenter import ProjectionPresenter

    config = session.config
    presenter = ProjectionPresenter(
        model=config.model_label or None,
        workspace=str(config.workspace),
        verification_run_id=(
            config.plan.verification_run_id if config.plan is not None else None
        ),
    )
    return ProductionTerminalExecutor(
        session,
        presenter=presenter,
        sink=sink if sink is not None else RenderingSink(),
        input=input,
        output=output,
        use_patch_stdout=input is None and output is None,
    )


async def run_tui_session(
    session: "ApplicationSession",
    *,
    initial_prompt: str | None = None,
    input: Any = None,
    output: Any = None,
    sink: RenderingSink | None = None,
) -> int:
    """运行生产组合 TUI 直至退出并完成关闭收尾; 返回最后一条请求的退出码.

    session 由调用方 open; 关闭顺序由 TerminalExecutor.run_async 负责
    (恢复终端模式 -> 退订 -> 停 mailbox 消费 -> session.aclose), 本函数不重复关闭.
    input/output 注入时 (headless 测试) 不启用 patch_stdout.
    """
    executor = build_production_executor(session, input=input, output=output, sink=sink)
    last_exit_codes: list[int] = []

    def track(update: "ApplicationUpdate") -> None:
        from cicada.application import RequestCompleted

        if isinstance(update, RequestCompleted):
            last_exit_codes.append(update.outcome.exit_code)

    unsubscribe = session.subscribe(track)
    try:
        runner = asyncio.ensure_future(executor.run_async())
        if initial_prompt:
            # 等订阅与消费任务就绪后再提交首问 (executor 在 run_async 内订阅)
            while executor._consumer_task is None:
                await asyncio.sleep(0.01)
            executor._accept_prompt(initial_prompt, None)
        try:
            await runner
        except BaseException:
            # run_async 收尾未完成的异常路径兜底; aclose 幂等
            await session.aclose()
            raise
    finally:
        unsubscribe()
    return last_exit_codes[-1] if last_exit_codes else 0


__all__ = [
    "ProductionTerminalExecutor",
    "RenderingSink",
    "build_production_executor",
    "run_tui_session",
]
