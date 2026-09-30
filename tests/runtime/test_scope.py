import asyncio

import pytest

from cicada.runtime.scope import ResourceScope, ScopeClosedError


async def test_lifo_order():
    scope = ResourceScope("s")
    order = []
    for i in (1, 2, 3):
        scope.defer(lambda i=i: order.append(i))  # 默认参数捕获, 避免循环变量陷阱
    await scope.aclose()
    assert order == [3, 2, 1]


async def test_cleanup_failure_aggregates_and_continues():
    scope = ResourceScope("s")
    ran = []

    def boom():
        raise ValueError("cleanup x")

    scope.defer(lambda: ran.append("first-registered"))
    scope.defer(boom)
    scope.defer(lambda: ran.append("last-registered"))
    with pytest.raises(ExceptionGroup) as exc_info:
        await scope.aclose()
    assert ran == ["last-registered", "first-registered"]
    assert len(exc_info.value.exceptions) == 1


async def test_aclose_idempotent_and_concurrent():
    scope = ResourceScope("s")
    count = 0

    async def cleanup():
        nonlocal count
        count += 1

    scope.defer(cleanup)
    await asyncio.gather(scope.aclose(), scope.aclose())
    await scope.aclose()
    assert count == 1


async def test_repeated_aclose_replays_cleanup_errors():
    scope = ResourceScope("s")

    def boom():
        raise ValueError("cleanup x")

    scope.defer(boom)
    with pytest.raises(ExceptionGroup):
        await scope.aclose()
    with pytest.raises(ExceptionGroup):
        await scope.aclose()


async def test_defer_after_close_raises():
    scope = ResourceScope("s")
    await scope.aclose()
    with pytest.raises(ScopeClosedError):
        scope.defer(lambda: None)
