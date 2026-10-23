"""独占事务的单一所有权令牌 (P0-3).

同一时刻只允许一个事务改动规范历史、存档或模型绑定。过去这件事由 8 个
in-flight 布尔分散回答, 每个入口自己组合一遍: 布尔之间没有共同真相, 释放点
分散在各个 finally, 而且"请求起始窗"这种子阶段只能靠额外布尔补齐。

这里用一枚令牌回答"现在谁在改会话": 入口先问 `owner`, 再 `claim`, 结束
`release`。请求内部的自动压缩是唯一允许的嵌套 (请求临时占住压缩槽), 因此
令牌是一个栈而不是单值; 栈顶即对外的答案。

每次占用同时登记一个落定信号: `release` 就是"该事务已落定", 于是 close 的
监督等待不必再各自持有 future, 令牌本身就是唯一的落定来源。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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


def _completion() -> asyncio.Future | None:
    """本次占用的落定信号; 无运行循环时 (纯同步单元) 不提供, 互斥照常."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.create_future()


@dataclass
class _Claim:
    operation: str
    completion: asyncio.Future | None


class OwnershipError(RuntimeError):
    """账本使用错误 (在途时重复占用); 属编程错误, 不是用户可见的拒绝."""


class OperationLedger:
    """独占事务账本; 栈顶是当前持有者, 每次占用带一个落定信号."""

    def __init__(self) -> None:
        self._claims: list[_Claim] = []

    @property
    def owner(self) -> str | None:
        """最内层事务; None = 没有事务在途."""
        return self._claims[-1].operation if self._claims else None

    def holds(self, operation: str) -> bool:
        """operation 是否在栈上 (页面类租约的释放点据此幂等)."""
        return any(claim.operation == operation for claim in self._claims)

    def claim(self, operation: str, *, inside: str | None = None) -> asyncio.Future | None:
        """占住独占槽; inside 声明唯一允许的父事务 (请求内的自动压缩)."""
        current = self.owner
        if current is not None and current != inside:
            raise OwnershipError(
                f"ownership ledger: cannot claim {operation!r} while {current!r} is in flight")
        claim = _Claim(operation, _completion())
        self._claims.append(claim)
        return claim.completion

    def release(self, operation: str) -> None:
        """清理入口: 从不失败, 未持有则忽略; 同时落定该事务的完成信号."""
        index = next(
            (i for i in range(len(self._claims) - 1, -1, -1)
             if self._claims[i].operation == operation), None)
        if index is None:
            return
        completion = self._claims.pop(index).completion
        if completion is not None and not completion.done():
            completion.set_result(None)

    def completion(self, operation: str) -> asyncio.Future | None:
        """该事务当前在途的落定信号; 不在途或已落定为 None."""
        for claim in reversed(self._claims):
            if claim.operation != operation:
                continue
            if claim.completion is not None and not claim.completion.done():
                return claim.completion
            return None
        return None

    def completions(self, *operations: str) -> tuple[asyncio.Future, ...]:
        """按事务名取在途落定信号; 供 close 的监督等待."""
        return tuple(
            future for operation in operations
            if (future := self.completion(operation)) is not None
        )
