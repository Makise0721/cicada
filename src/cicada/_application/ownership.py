"""独占事务的单一所有权令牌 (P0-3).

同一时刻只允许一个事务改动规范历史、存档或模型绑定。过去这件事由 8 个
in-flight 布尔分散回答, 每个入口自己组合一遍: 布尔之间没有共同真相, 释放点
分散在各个 finally, 而且"请求起始窗"这种子阶段只能靠额外布尔补齐。

这里用一枚令牌回答"现在谁在改会话": 入口先问 `owner`, 再 `claim`, 结束
`release`。请求内部的自动压缩是唯一允许的嵌套 (请求临时占住压缩槽), 因此
令牌是一个栈而不是单值; 栈顶即对外的答案。
"""

from __future__ import annotations

#: 每个独占事务的诊断短语; 各入口据此构造自己领域的拒绝文案。
OWNER_PHRASES: dict[str, str] = {
    "request": "a request is in progress",
    "clear": "a persistent clear is in progress",
    "compaction": "a compaction is in progress",
    "model": "model selection is in progress",
    "session": "session management is in progress",
    "login": "login configuration is in progress",
    "review": "review is in progress",
    "rewind": "restore-point viewing is in progress",
    "catalog": "session catalog observation is in progress",
}

#: 请求/清空/压缩之外的事务; 这三类入口对它们统一拒绝 (各自保留具体窗口文案)。
FOREIGN_OWNERS = frozenset({"session", "catalog", "review", "rewind", "login", "model"})


def busy_refusal(owner: str, action: str) -> str:
    """owner 已占槽时对 action 的统一拒绝文案; 异常类型由入口决定."""
    return f"{OWNER_PHRASES.get(owner, owner)}; {action} is refused"


class OwnershipError(RuntimeError):
    """账本使用错误 (在途时重复占用); 属编程错误, 不是用户可见的拒绝."""


class OperationLedger:
    """独占事务账本; 栈顶是当前持有者."""

    def __init__(self) -> None:
        self._stack: list[str] = []

    @property
    def owner(self) -> str | None:
        """最内层事务; None = 没有事务在途."""
        return self._stack[-1] if self._stack else None

    def holds(self, operation: str) -> bool:
        """operation 是否在栈上 (页面类租约的释放点据此幂等)."""
        return operation in self._stack

    def claim(self, operation: str, *, inside: str | None = None) -> None:
        """占住独占槽; inside 声明唯一允许的父事务 (请求内的自动压缩)."""
        current = self.owner
        if current is not None and current != inside:
            raise OwnershipError(
                f"ownership ledger: cannot claim {operation!r} while {current!r} is in flight")
        self._stack.append(operation)

    def release(self, operation: str) -> None:
        """清理入口: 从不失败, 未持有则忽略 (finally 中调用)."""
        try:
            index = len(self._stack) - 1 - self._stack[::-1].index(operation)
        except ValueError:
            return
        del self._stack[index]
