"""P4 07 阶段验收 live 用例: 固定 fixture + qwen3.5:9b 完成一次可核对的小改动.

协议 (spec Live 验收协议):
- gate /api/version + /api/tags, 不可达 skip (开发回归); 阶段关闭必须有完整通过证据.
- 每次调用执行一个 attempt (环境变量 CICADA_P4_LIVE_ATTEMPT, 默认 1); 最多两次
  全新 fixture、相同 prompt/profile 的尝试, 证据全部落盘后才断言, 失败也保留.
- 600 秒 episode 预算与请求/响应 tee 都由 harness (p4_fixture) 控制, 不是 CLI 能力.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from p4_fixture import (
    MODEL,
    PROMPT_EVAL_PEAK_LIMIT,
    gate,
    run_attempt,
)

_GATE = gate()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_GATE is not None, reason=_GATE or ""),
]


def test_p4_live_small_change_delivery():
    attempt = int(os.environ.get("CICADA_P4_LIVE_ATTEMPT", "1"))
    result = run_attempt(attempt)  # 证据已在返回前全部落盘

    assert not result["timed_out"], "600s episode 预算内未完成"
    assert result["proxy_errors"] == [], f"tee 代理出错: {result['proxy_errors']}"
    assert result["returncode"] == 0, (
        f"CLI 退出码 {result['returncode']}; 证据: {result['evidence_dir']}")

    stdout = result["stdout"]
    assert "[delivery can_deliver=true model_stopped=true]" in stdout
    assert "- check-1 status=passed freshness=current" in stdout

    # 模型实际走了 搜索定位 -> read -> 修改 -> check -> 汇报 的路径
    assert (">> grep (" in stdout) or (">> glob (" in stdout), "模型未使用 glob/grep 定位"
    assert ">> read (" in stdout
    assert (">> edit (" in stdout) or (">> write (" in stdout)
    assert ">> check (" in stdout

    # 独立 checker 与固定工件未被模型改动
    assert result["checker_exit"] == 0, "独立 checker 对最终代码未通过"
    assert result["checker_unchanged"], "checker 被改动, 尝试无效"

    # 变更仅为必要目标实现; .cicada 工件目录属运行输出, __pycache__ 属验证范围政策
    # 排除项 (且 launcher 禁写字节码, 出现即为 checker 进程产物, 不是模型改动)
    allowed_prefixes = ("?? .cicada/",)
    changed = [
        line for line in result["changed_git_lines"]
        if not line.startswith(allowed_prefixes) and "__pycache__" not in line
    ]
    assert changed == [" M src/cicada/reporting.py"], f"出现了目标外的改动: {changed}"
    assert "tools_ok" in result["git_diff"]

    # 单次 prompt_eval_count 峰值在 32K 窗口的 75% 内
    peak = result["prompt_eval_count_peak"]
    assert peak is not None, "代理未捕获任何 done 行计量"
    assert peak <= PROMPT_EVAL_PEAK_LIMIT, f"prompt_eval_count 峰值 {peak} 超过 {PROMPT_EVAL_PEAK_LIMIT}"

    # 请求触发加载后 /api/ps 核对模型在场 (digest 全文在 ps-after.json/identity.json)
    ps = json.loads(
        (Path(result["evidence_dir"]) / "ps-after.json").read_text(encoding="utf-8"))
    assert any(MODEL in json.dumps(entry) for entry in ps.get("models", [])), (
        f"/api/ps 中未见 {MODEL}: {ps}"
    )
