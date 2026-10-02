"""TUI 共享展示数据契约 (02 冻结; 03 终端执行器与 04 展示投影共同消费).

职责边界:
- presenter 只消费 ApplicationUpdate, 只改自己的 UI 状态, 不执行工具、
  不写会话、不计算自制的 PASS, 不改应用结果.
- 终端由 prompt_toolkit 唯一控制; presenter 产出可展示的记录/预览/状态快照.
- 记录有稳定 key、角色、标题、状态及展示正文; 活动预览使用 run/turn key;
  最终完成替换预览并只把最终回答追加一次.

有界性属于实现层 (工具卡 ≤32 行/8192 bytes, 活动预览 ≤12 行/4096 bytes,
mailbox 1024 条, 30 FPS 合并), 由 03/04 按契约实现; 本文件只冻结数据形状.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from cicada.application import ApplicationUpdate

RecordRole = Literal["user", "assistant", "tool", "system", "error"]
RecordStatus = Literal["pending", "running", "done", "failed", "cancelled"]


@dataclass(frozen=True)
class PresentationRecord:
    """一条已完成的展示记录; key 稳定, 重复追加由实现层按键去重."""

    key: str
    role: RecordRole
    title: str
    status: RecordStatus
    body: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActivePreview:
    """活动 assistant 预览; text 为合并后的当前文本, truncated 标明截断事实."""

    run_id: str
    turn_index: int
    text: str
    truncated: bool = False


@dataclass(frozen=True)
class StatusSnapshot:
    """状态栏快照; None 表示不可得, 不是 0."""

    model: str | None = None
    workspace: str | None = None
    phase: str | None = None
    turn_index: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    verification_run_id: str | None = None
    check_status: str | None = None
    change_scope: str | None = None


@dataclass(frozen=True)
class PresentationUpdate:
    """presenter.accept(update) 的返回: 可追加记录、可替换预览、状态快照.

    preview=None 表示本更新不携带预览变化; 需要收起活动预览时由 presenter
    以"最终记录已追加"的语义自行处理 (最终完成替换预览并只追加一次).
    status=None 表示状态不变.
    """

    appended: tuple[PresentationRecord, ...] = ()
    preview: ActivePreview | None = None
    status: StatusSnapshot | None = None


class TuiPresenter(Protocol):
    """只读展示投影协议; 实现层不得反向调用应用执行接口."""

    def accept(self, update: "ApplicationUpdate") -> PresentationUpdate:
        """消费一条应用更新, 返回本轮展示变化; 必须同步快速返回."""
        ...
