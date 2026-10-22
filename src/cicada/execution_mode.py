"""执行档 (P0-1 v2): 只读 / 审批 / 完全, 运行时可切换.

档位是进程级限制, 不写会话存档; 默认 `full` 与既有行为一致. 分派闸门对每个工具
生效: 只读档拒绝写入/命令/检查, 审批档只让 powershell 等待用户决定. 因为档位可在
会话内切换, 工具表始终是完整集合; 被拒绝的调用返回程序错误结果, 不产生副作用.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from cicada.approval import DENIED_CONTENT, ApprovalBroker, Decision
from cicada.core.cancel import CancelToken
from cicada.core.messages import ToolResult
from cicada.core.ports import Tool, ToolContext, ToolSpec

Mode = Literal["read-only", "approval", "full"]
MODES: tuple[Mode, ...] = ("read-only", "approval", "full")
MODE_LABELS: dict[str, str] = {"read-only": "只读", "approval": "审批", "full": "完全"}
READ_ONLY_TOOLS = frozenset({"read", "glob", "grep", "load_skill"})
READ_ONLY_DENIED = "read-only mode: tool '{name}' is unavailable"


def is_mode(value: str) -> bool:
    return value in MODES


class ExecutionModeController:
    """会话持有的当前档位; 工具包装器每调用读一次, 切换立即生效."""

    def __init__(self, mode: Mode = "full") -> None:
        if not is_mode(mode):
            raise ValueError(f"unknown execution mode: {mode!r}")
        self._mode: Mode = mode
        self.broker = ApprovalBroker()

    @property
    def mode(self) -> Mode:
        return self._mode

    def set_mode(self, mode: str) -> bool:
        """设置档位; 返回是否发生变化. 非法值抛 ValueError, 不改变当前档."""
        if not is_mode(mode):
            raise ValueError(f"unknown execution mode: {mode!r}")
        if mode == self._mode:
            return False
        self._mode = mode  # type: ignore[assignment]
        return True

    def allows(self, tool_name: str) -> bool:
        return self._mode != "read-only" or tool_name in READ_ONLY_TOOLS


class ExecutionModeTool:
    """Tool 装饰器: 只读拒绝、审批等待、完全放行; 其它语义保持内层不变."""

    def __init__(self, inner: Tool, controller: ExecutionModeController) -> None:
        self._inner = inner
        self._controller = controller

    @property
    def spec(self) -> ToolSpec:
        return self._inner.spec

    async def execute(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self._inner.spec.name
        if not self._controller.allows(name):
            return _denied(name, ctx, READ_ONLY_DENIED.format(name=name))
        decision: Decision | None = None
        if name == "powershell" and self._controller.mode == "approval":
            decision = await self._controller.broker.request(
                name, str(arguments.get("command", "")), ctx.cancel
            )
            if decision == "deny":
                return _denied(name, ctx, DENIED_CONTENT)
        result = await self._inner.execute(arguments, ctx)
        if decision is None:
            return result
        details = dict(result.details) if isinstance(result.details, dict) else {}
        details["approval"] = decision
        return replace(result, details=details)


def _denied(name: str, ctx: ToolContext, content: str) -> ToolResult:
    return ToolResult(
        call_id=ctx.call_id, name=name, content=content, is_error=True,
        details={"approval": "deny"} if content == DENIED_CONTENT else {},
    )
