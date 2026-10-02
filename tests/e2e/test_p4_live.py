"""P4 07 阶段验收 live 用例: 固定 fixture + qwen3.5:9b 完成一次可核对的小改动.

协议 (spec Live 验收协议):
- gate /api/version + /api/tags, 不可达 skip (开发回归); 阶段关闭必须有完整通过证据.
- 每次调用执行一个 attempt (环境变量 CICADA_P4_LIVE_ATTEMPT, 默认 1); 最多两次
  全新 fixture、相同 prompt/profile 的尝试, 证据全部落盘后才断言, 失败也保留.
- 600 秒 episode 预算与请求/响应 tee 都由 harness (p4_fixture) 控制, 不是 CLI 能力.
"""

from __future__ import annotations

import os

import pytest

from p4_fixture import gate, run_attempt, validate_attempt

_GATE = gate()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_GATE is not None, reason=_GATE or ""),
]


def test_p4_live_small_change_delivery():
    attempt = int(os.environ.get("CICADA_P4_LIVE_ATTEMPT", "1"))
    result = run_attempt(attempt)
    validate_attempt(result)
