"""只读 UI 诊断对象 (票据 05): CLI 提供不可变事实, /status 据此追溯.

- 诊断对象由 CLI 组装 (__main__ 的 ApplicationConfig/prompt_info/plan 已发布
  字段), 经 run_tui_session(diagnostics=...) 注入; UI 不读取 provider 闭包、
  不读环境外的可变状态。
- 字段限额与截断标记由 ui.checks.bound_status_lines 在 /status 组装时统一
  施加 (20 行/8192 bytes 总限额、512 bytes 单字段限额, P12); 本模块只提供
  固定字段的行化。
"""

from __future__ import annotations

from dataclasses import dataclass

from cicada.prompting import INSTRUCTION_MODE_LABELS


@dataclass(frozen=True)
class UIDiagnostics:
    """CLI 提供的不可变 UI 诊断快照; None 字段显示为不可得, 不虚构。"""

    workspace: str
    model_label: str | None
    prompt_version: str
    prompt_sha256: str | None
    instructions_path: str | None
    instructions_sha256: str | None
    verification_run_id: str | None
    check_ids: tuple[str, ...] = ()
    check_timeout_s: float | None = None
    max_turns: int | None = None
    instructions_mode: str | None = None


def diagnostics_lines(diagnostics: UIDiagnostics) -> list[str]:
    """诊断快照 -> /status 附加行 (固定字段; 限额归 /status 组装层)。"""
    lines = [
        f"工作区: {diagnostics.workspace}",
        f"模型: {diagnostics.model_label or '不可得'}",
        f"prompt 版本: {diagnostics.prompt_version}",
        f"prompt sha256: {diagnostics.prompt_sha256 or '不可得'}",
    ]
    if diagnostics.instructions_path is not None:
        mode = INSTRUCTION_MODE_LABELS.get(diagnostics.instructions_mode or "")
        mode_text = f" 模式={mode}" if mode else ""
        lines.append(
            f"指令文件: {diagnostics.instructions_path}{mode_text}"
            f" sha256={diagnostics.instructions_sha256 or '不可得'}"
        )
    if diagnostics.verification_run_id is not None:
        lines.append(f"检查配置: ids={','.join(diagnostics.check_ids) or '无'}")
        if diagnostics.check_timeout_s is not None:
            lines.append(f"检查超时: {diagnostics.check_timeout_s:g}s")
    if diagnostics.max_turns is not None:
        lines.append(f"轮次上限: {diagnostics.max_turns}")
    return lines


__all__ = ["UIDiagnostics", "diagnostics_lines"]
