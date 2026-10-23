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

"谁在改会话"不在这里回答 (P0-3): 事务所有权归 `session.busy_operation`。闸门只
保留会话不可能知道的三件事——请求 reservation 的同步窗口、同批输入余下的静默、
以及打开中的会话模态页; 已提交但尚未占槽的清空/压缩/模型选择另有一个同步槽位
(`_submitted`), 它们由命令处置时置位、事件落定时清除。
"""

from __future__ import annotations

import asyncio
from typing import Callable

from cicada.application import ApplicationSession, RequestOutcome

#: 命令处置时同步占位、事件落定时清除的提交槽位; 三者互斥 (命令在屏障内不排队)。
_SUBMITTED_OPERATIONS = frozenset({"clear", "compaction", "model"})


class AdmissionGate:
    """串行请求准入: 至多一个拥有任务; 同步占位 → create_task → 终结释放。"""

    def __init__(self, session: ApplicationSession) -> None:
        self._session = session
        self._task: asyncio.Task | None = None
        self._reservation_open = False
        self._pending_cancel = False
        # 已提交、但会话尚未占槽的提交 (清空/压缩/模型选择): 命令处置时同步置位,
        # 落定事件或等待任务释放; 只有一个槽位, 因为屏障内命令不排队
        self._submitted: str | None = None
        # 会话模态页打开或会话切换/命名在途: 前者是纯 UI 页, 后者会话侧也有租约
        self._session_managing = False
        self._session_rejected_in_batch = False
        # 宿主注入: execute 异常 (如 process_uncertain 拒绝) 的如实记录出口
        self.on_error: Callable[[str], None] | None = None

    # --- 状态 ----------------------------------------------------------------------

    @property
    def pending_cancel(self) -> bool:
        return self._pending_cancel

    @property
    def clearing(self) -> bool:
        """持久清空提交进行中 (输入屏障; 与请求 reservation 独立)。"""
        return self._submitted == "clear"

    @property
    def compacting(self) -> bool:
        """持久压缩提交进行中 (输入屏障; 与请求 reservation 独立)。"""
        return self._submitted == "compaction"

    @property
    def model_selecting(self) -> bool:
        """模型选择提交进行中 (输入屏障; 与请求 reservation 独立, 04)。"""
        return self._submitted == "model"

    @property
    def login_active(self) -> bool:
        """登录编辑窗打开或保存中: 会话持有 login 租约 (同步占槽)。"""
        return self._session.busy_operation == "login"

    @property
    def submitting(self) -> bool:
        """命令已提交、会话尚未占槽 (清空/压缩/模型选择的同步窗口)。"""
        return self._submitted is not None

    def busy(self) -> bool:
        """reservation 存在即忙 (含 execute 任务尚未起跑的同步窗口)。"""
        return self._reservation_open

    @property
    def session_managing(self) -> bool:
        return self._session_managing

    def note_session_started(self) -> None:
        self._session_managing = True

    def note_session_settled(self) -> None:
        self._session_managing = False

    @property
    def session_rejected_in_batch(self) -> bool:
        return self._session_rejected_in_batch

    def note_session_rejected(self) -> None:
        """释放管理占位，但当前同步输入批余下 Enter 仍静默。"""
        self._session_managing = False
        self._session_rejected_in_batch = True
        asyncio.get_running_loop().call_soon(self._settle_session_input_batch)

    def _settle_session_input_batch(self) -> None:
        self._session_rejected_in_batch = False

    # --- 准入 ----------------------------------------------------------------------

    def other_transaction_active(self) -> bool:
        """会话里有本闸门之外的独占事务在途.

        请求的屏障由 reservation 回答 (该任务归闸门所有); 清空/压缩/模型/登录/
        审阅/还原点/目录直接问会话的所有权令牌, 不再各自镜像。
        """
        return self._session.busy_operation not in (None, "request")

    def try_admit(self, prompt: str) -> asyncio.Task | None:
        """同步准入: 可执行则先占 reservation 再创建任务, 返回任务; 否则 None。

        拒绝条件: 已有 reservation / 同批已被拒 / 会话模态或切换在途 / 会话里另有
        独占事务 / 已提交未占槽的提交 / session 非 idle (含 closing/closed) /
        空提示。返回 None 表示拒绝, 调用方不得伪造用户条目。
        """
        if self._reservation_open:
            return None
        if self._session_rejected_in_batch or self._session_managing:
            return None
        if self.other_transaction_active():
            return None
        if self._submitted is not None:
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

    # --- 提交槽位 (清空 / 压缩 / 模型选择) ----------------------------------------------

    def _reserve(self, operation: str) -> None:
        """命令处置时同步占位, 先于任务起跑窗口."""
        assert operation in _SUBMITTED_OPERATIONS
        self._submitted = operation

    def _release(self, operation: str) -> None:
        """落定 (成功事件或等待任务终结) 后释放; 幂等."""
        if self._submitted == operation:
            self._submitted = None

    def note_clear_started(self) -> None:
        self._reserve("clear")

    def note_clear_settled(self) -> None:
        self._release("clear")

    def note_compact_started(self) -> None:
        self._reserve("compaction")

    def note_compact_settled(self) -> None:
        self._release("compaction")

    def note_model_started(self) -> None:
        self._reserve("model")

    def note_model_settled(self) -> None:
        self._release("model")

    def note_phase(self, phase: str) -> None:
        """阶段通知: reservation 窗口记录的取消意图在任务实际执行时交给公共 cancel。"""
        if self._pending_cancel and phase == "executing":
            self._pending_cancel = False
            self._session.cancel()

    def note_closed(self) -> None:
        """ApplicationClosed: 清理全部准入状态; closing/closed 不重开输入由
        try_admit 的 state 检查保证, 此处只做状态卫生。"""
        self._pending_cancel = False
        self._session_managing = False
        self._session_rejected_in_batch = False
        self._reservation_open = False
        self._task = None
        self._submitted = None

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

    覆盖 reservation 未起跑窗口、同批已拒输入、会话模态/切换、任一事务持有
    所有权 (清空/压缩/模型/登录/审阅/还原点/目录) 与 executing/cancelling/
    finalizing/closing/closed。
    """
    if gate.busy():
        return True
    if gate.session_rejected_in_batch or gate.session_managing:
        return True
    if gate.submitting:
        return True
    if gate.other_transaction_active():
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
