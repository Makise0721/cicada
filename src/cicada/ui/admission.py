"""输入准入闸门 (票据 02): 同步 reservation 先于 execute 任务的创建.

语义 (设计 §准入与键位表 / 规格 §输入与安静的界面):
- 接受编码请求时在同一个同步输入回调内先占用 reservation, 再创建 execute
  任务; 不能仅凭 session.state == idle 判断下一次 Enter。reservation 保留至
  本次拥有任务终结且 session 为 idle, 再立即释放; 不等待 process_uncertain
  等编码策略重新允许, closing/closed 不重新开放输入。
- executing/cancelling/finalizing 屏障内 Enter 在调用方 (conversation_view)
  直接无动作; reservation 未起跑窗口由本闸门拒绝, 同批二次 Enter/斜线命令
  零动作 (任务未起跑时 state 仍是 idle, 只有 reservation 能挡住)。
- 取消意图绑定本次任务: reservation 窗口内 Ctrl+C/Escape 记录意图, 任务实际
  起跑 (phase=executing) 后交公共 session.cancel(); 任务拒绝或终结即清除,
  不带到下次请求。不触碰独立 finalize token。
- execute 的异常 (如 process_uncertain 拒绝、内核异常) 经 on_error 回调如实
  成条; 抛出不被吞没成"成功"。
"""

from __future__ import annotations

import asyncio
from typing import Callable

from cicada.application import ApplicationSession, RequestOutcome


class AdmissionGate:
    """串行请求准入: 至多一个拥有任务; 同步占位 → create_task → 终结释放。"""

    def __init__(self, session: ApplicationSession) -> None:
        self._session = session
        self._task: asyncio.Task | None = None
        self._reservation_open = False
        self._pending_cancel = False
        # 票据 05: 持久清空提交中 (OperationBusy("clear") -> HistoryCleared/失败落定);
        # 期间 Enter/斜线命令/退出屏障, 草稿保持可编辑
        self._clearing = False
        # 票据 02: 持久压缩提交中 (OperationBusy("compact") -> ContextCompacted/
        # 失败落定); 屏障语义与清空同型
        self._compacting = False
        # DeepSeek 阶段票据 04: 模型选择提交中 (OperationBusy("model") ->
        # ModelChanged/失败落定); 屏障语义与清空/压缩同型, 菜单确认时同步置位
        self._model_selecting = False
        self._login_active = False
        self._review_active = False
        # 宿主注入: execute 异常 (如 process_uncertain 拒绝) 的如实记录出口
        self.on_error: Callable[[str], None] | None = None

    # --- 状态 ----------------------------------------------------------------------

    @property
    def pending_cancel(self) -> bool:
        return self._pending_cancel

    @property
    def clearing(self) -> bool:
        """持久清空提交进行中 (输入屏障; 与请求 reservation 独立)。"""
        return self._clearing

    @property
    def compacting(self) -> bool:
        """持久压缩提交进行中 (输入屏障; 与请求 reservation 独立)。"""
        return self._compacting

    @property
    def model_selecting(self) -> bool:
        """模型选择提交进行中 (输入屏障; 与请求 reservation 独立, 04)。"""
        return self._model_selecting

    def busy(self) -> bool:
        """reservation 存在即忙 (含 execute 任务尚未起跑的同步窗口)。"""
        return self._reservation_open

    # --- 准入 ----------------------------------------------------------------------

    def try_admit(self, prompt: str) -> asyncio.Task | None:
        """同步准入: 可执行则先占 reservation 再创建任务, 返回任务; 否则 None。

        拒绝条件: 已有 reservation / 持久清空/压缩提交中 / 模型选择提交中 /
        session 非 idle (含 closing/closed) / 空提示。返回 None 表示拒绝, 调用方不得伪造用户条目。
        """
        if self._reservation_open:
            return None
        if self._clearing or self._compacting or self._model_selecting or self._login_active or self._review_active or getattr(self._session, "review_active", False):
            return None
        if self._session.state != "idle":
            return None
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        # 同步先占 reservation (先于 create_task): 同批后续 Enter 在任务起跑前
        # 也必须被本闸门挡住, 不能依赖尚未生效的 session.state
        self._reservation_open = True
        task = asyncio.get_running_loop().create_task(self._run_request(prompt))
        self._task = task
        task.add_done_callback(self._on_task_done)
        return task

    async def _run_request(self, prompt: str) -> RequestOutcome | None:
        try:
            return await self._session.execute(prompt)
        except Exception as exc:
            # execute 的同步前置检查与内核异常都从协程内抛出: 如实记录,
            # 不向上杀宿主; state 由此进入 closing 或保持 idle (策略拒绝)
            if self.on_error is not None:
                self.on_error(str(exc))
            return None

    def _on_task_done(self, task: asyncio.Task) -> None:
        # 取消意图绑定本次任务: 任务终结即清除, 不带到下次请求
        self._pending_cancel = False
        self._release_if_idle()

    def _release_if_idle(self) -> None:
        """本次拥有任务已终结且 session 真正 idle 才释放 reservation。"""
        if (
            self._reservation_open
            and self._task is not None
            and self._task.done()
            and self._session.state == "idle"
        ):
            self._reservation_open = False
            self._task = None

    # --- 应用事件联动 -----------------------------------------------------------------

    def note_request_completed(self) -> None:
        """RequestCompleted 后核对释放条件 (正常路径 state 已回 idle)。"""
        self._release_if_idle()

    # --- 持久清空屏障 (票据 05) ---------------------------------------------------------

    def note_clear_started(self) -> None:
        """清空提交开始 (命令处置时同步置位, 先于任务起跑窗口); 幂等。"""
        self._clearing = True

    def note_clear_settled(self) -> None:
        """清空提交落定 (成功由 HistoryCleared 投影, 失败由等待任务释放); 幂等。"""
        self._clearing = False

    def note_compact_started(self) -> None:
        """压缩提交开始 (命令处置时同步置位, 先于任务起跑窗口); 幂等 (02)。"""
        self._compacting = True

    def note_compact_settled(self) -> None:
        """压缩提交落定 (成功由 ContextCompacted 投影, 失败/放弃由等待任务释放); 幂等。"""
        self._compacting = False

    # --- 模型选择屏障 (DeepSeek 阶段票据 04) -------------------------------------------

    def note_model_started(self) -> None:
        """模型选择提交开始 (菜单确认时同步置位, 先于任务创建); 幂等。"""
        self._model_selecting = True

    @property
    def login_active(self) -> bool:
        return self._login_active

    def note_login_started(self) -> None:
        self._login_active = True

    def note_login_settled(self) -> None:
        self._login_active = False

    @property
    def review_active(self) -> bool:
        return self._review_active

    def note_review_started(self) -> None:
        self._review_active = True

    def note_review_settled(self) -> None:
        self._review_active = False

    def note_model_settled(self) -> None:
        """模型选择落定 (成功由 ModelChanged 投影, 失败/取消由等待任务释放); 幂等。"""
        self._model_selecting = False

    def note_phase(self, phase: str) -> None:
        """阶段通知: reservation 窗口记录的取消意图在任务实际执行时交给公共 cancel。"""
        if self._pending_cancel and phase == "executing":
            self._pending_cancel = False
            self._session.cancel()

    def note_closed(self) -> None:
        """ApplicationClosed: 清理全部准入状态; closing/closed 不重开输入由
        try_admit 的 state 检查保证, 此处只做状态卫生。"""
        self._pending_cancel = False
        self._reservation_open = False
        self._task = None
        self._clearing = False
        self._compacting = False
        self._model_selecting = False
        self._login_active = False

    # --- 取消意图 ----------------------------------------------------------------------

    def arm_pending_cancel(self) -> None:
        """reservation 窗口内 (任务尚未起跑) 记录取消意图。"""
        self._pending_cancel = True


def cancel_if_active(session: ApplicationSession) -> bool:
    """对 executing/cancelling 会话调用公共 cancel (幂等); 返回是否施加。

    finalizing/closing/closed 静默无动作, 不输出"取消无效"类说明。
    """
    if session.state in ("executing", "cancelling"):
        session.cancel()
        return True
    return False


def is_enter_barrier(session: ApplicationSession, gate: AdmissionGate) -> bool:
    """Enter 屏障: 屏障内 Enter 无动作 (不排队、不执行命令、不提示)。

    覆盖 reservation 未起跑窗口、持久清空提交 (05)、持久压缩提交 (02)、
    模型选择提交 (DeepSeek 04) 与 executing/cancelling/finalizing/closing/closed。
    """
    if gate.busy():
        return True
    if gate.clearing or gate.compacting or gate.model_selecting or gate.login_active or gate.review_active or getattr(session, "review_active", False):
        return True
    return session.state in (
        "executing",
        "cancelling",
        "finalizing",
        "closing",
        "closed",
    )


__all__ = [
    "AdmissionGate",
    "cancel_if_active",
    "is_enter_barrier",
]
