"""检查事实投影 (票据 05): RequestOutcome -> 显示行, 口径与旧 presenter 一致.

迁移自旧 ui/presenter.py 的检查格式化段 (随旧渲染链退场):
- 来源四分类: 本轮执行 / 仍有效历史 PASS / 中性历史 / 未运行; "仍有效" 只属于
  快照没变且仍为最新尝试的旧 PASS, 失败/stale/未知 freshness 的历史回执不得
  宣称为仍有效。
- receipt_id 经 CHECK_TOOL_NAME 的 ToolCompleted.details 收集 (口径同旧
  presenter), 每请求开始清零, 用于区分"本次检查"与历史来源。
- 检查结论条目只在 decision 非空 (检查模式) 时追加; 正常完成保持安静。
- 进程未知锁存显示取 session.has_process_uncertain() (应用层单调事实),
  不再由展示层自存副本。
- /status 检查事实与限额 (20 行/8192 bytes 总限额、512 bytes 单字段限额与
  截断标记) 也在此实现 (P12)。

本模块只读: 不执行工具、不写会话、不计算自制的 PASS; 检查/交付事实一律取自
outcome.view / outcome.evidence / outcome.decision (程序判定)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cicada.core.events import Event, ToolCompleted

if TYPE_CHECKING:
    from cicada.application import ApplicationSession, RequestOutcome

CHECK_TOOL_NAME = "check"

# 检查/状态文案 (程序事实的中文标签; 不含 PASS/FAIL 之类自制判定; 同旧 presenter)
NORMAL_MODE_CHECK_LABEL = "未启用检查"
HISTORICAL_PASS_LABEL = "此前检查，仍有效"
HISTORICAL_NEUTRAL_LABEL = "此前检查"
CURRENT_RUN_CHECK_LABEL = "本次检查"
PROCESS_UNCERTAIN_LABEL = "进程终结未知（锁存）"
NOT_RUN_CHECK_LABEL = "未运行"

# 阻断原因与 /status 的有界性 (P12; 阻断原因沿用旧工具卡限额)
BLOCKING_MAX_LINES = 32
BLOCKING_MAX_BYTES = 8192
STATUS_MAX_LINES = 20
STATUS_MAX_BYTES = 8192
STATUS_FIELD_MAX_BYTES = 512


def clip_lines(text: str, max_lines: int, max_bytes: int) -> tuple[tuple[str, ...], int, int, bool]:
    """按 UTF-8 bytes 与行数截断; 返回 (保留行, 原行数, 原 bytes, 是否截断).

    单行超限时退到完整字符边界; 只在确有内容被丢弃时标记截断.
    (自旧 ui/presenter.py 原样移植。)
    """
    lines = text.splitlines() or [""]
    original_bytes = len(text.encode("utf-8"))
    kept: list[str] = []
    budget = max_bytes
    truncated = False
    for line in lines:
        if len(kept) >= max_lines:
            truncated = True
            break
        encoded = line.encode("utf-8")
        if len(encoded) > budget:
            clipped = encoded[: max(budget, 0)].decode("utf-8", errors="ignore")
            kept.append(clipped)
            truncated = True
            break
        kept.append(line)
        budget -= len(encoded) + 1  # 预留换行符
    return tuple(kept), len(lines), original_bytes, truncated


def truncation_note(
    kept: tuple[str, ...], original_lines: int, original_bytes: int, truncated: bool
) -> tuple[str, ...]:
    """截断标记行 (同旧 presenter 文案)。"""
    if not truncated:
        return ()
    kept_bytes = len("\n".join(kept).encode("utf-8"))
    return (f"…[已截断: 显示 {len(kept)}/{original_lines} 行, {kept_bytes}/{original_bytes} UTF-8 bytes]",)


def artifact_line(details: dict | None) -> str | None:
    """回执引用的输出工件: 只显示路径与 hash, 不读入内容. (旧 presenter 移植)"""
    if not isinstance(details, dict):
        return None
    path = details.get("output_artifact_path")
    if not isinstance(path, str) or not path:
        return None
    sha = details.get("output_artifact_sha256")
    sha_text = sha if isinstance(sha, str) and sha else "unverified"
    return f"输出工件: {path} sha256={sha_text}"


def delivery_lines(
    outcome: "RequestOutcome",
    session: "ApplicationSession",
    receipts_this_request: set[str],
) -> list[str]:
    """检查结论行 (旧 _delivery_record 移植): 程序判定 + 来源分类 + 累计口径。"""
    decision = outcome.decision
    view = outcome.view
    evidence = outcome.evidence
    lines: list[str] = [
        "程序判定（非模型自然语言）: " + ("可交付" if decision.can_deliver else "不可交付")
    ]
    if view is not None:
        lines.append(f"verification_run_id={view.verification_run_id}")
        if view.process_uncertain or session.has_process_uncertain():
            lines.append(PROCESS_UNCERTAIN_LABEL)
        for state in view.checks:
            if state.receipt is None:
                source = NOT_RUN_CHECK_LABEL
            elif state.receipt.receipt_id in receipts_this_request:
                source = CURRENT_RUN_CHECK_LABEL
            elif state.status == "passed" and state.freshness == "current":
                # "仍有效" 只属于快照没变且仍为最新尝试的旧 PASS
                source = HISTORICAL_PASS_LABEL
            else:
                # 失败/stale/未知 freshness 的历史回执不得宣称为仍有效
                source = HISTORICAL_NEUTRAL_LABEL
            line = f"check {state.check_id}: status={state.status} freshness={state.freshness} 来源={source}"
            if state.receipt is not None:
                line += f" receipt_id={state.receipt.receipt_id} exit_code={state.receipt.exit_code}"
            lines.append(line)
    if evidence is not None:
        if evidence.artifact_path is not None:
            sha = evidence.artifact_sha256 or "unverified"
            lines.append(f"累计变更工件: {evidence.artifact_path} sha256={sha}")
        lines.append(f"相对启动基线累计改动: {len(evidence.changes)} 项")
    if not decision.can_deliver:
        kept, total_lines, size, truncated = clip_lines(
            "\n".join(decision.blocking_reasons), BLOCKING_MAX_LINES, BLOCKING_MAX_BYTES
        )
        lines.append("不能交付的原因:")
        lines.extend(f"- {reason}" for reason in kept)
        lines.extend(truncation_note(kept, total_lines, size, truncated))
    return lines


def check_status_text(outcome: "RequestOutcome", session: "ApplicationSession") -> str | None:
    """检查状态摘要 (旧 _check_status_text 移植); view 为 None 即未启用检查。"""
    if outcome.view is None:
        return NORMAL_MODE_CHECK_LABEL
    parts: list[str] = []
    if session.has_process_uncertain():
        parts.append(PROCESS_UNCERTAIN_LABEL)
    if outcome.view.checks:
        parts.extend(
            f"{state.check_id}={state.status}/{state.freshness}" for state in outcome.view.checks
        )
    else:
        parts.append(NOT_RUN_CHECK_LABEL)
    return " ".join(parts)


def change_scope_text(outcome: "RequestOutcome") -> str | None:
    """累计改动口径 (旧 _change_scope_text 移植); 无证据为 None。"""
    evidence = outcome.evidence
    if evidence is None:
        return None
    if not evidence.complete:
        detail = f"（不完整: {evidence.failure_kind}）" if evidence.failure_kind else "（不完整）"
        return f"累计改动口径: 相对启动基线{detail}"
    return f"累计改动口径: 相对启动基线 {len(evidence.changes)} 项变更"


class CheckProjection:
    """检查/交付事实的显示投影: 回执收集 + 结论行 + /status 事实 (票据 05)。

    订阅侧职责: 每条内核事件经 observe_event 收集 check 回执身份; 阶段通知经
    note_phase 在新请求开始时清零本轮回执; RequestCompleted 经
    on_request_completed 取结论行 (decision 非空即检查模式, 正常完成返回 None
    保持安静)。锁存显示读 session 单调事实, 不自存副本。
    """

    def __init__(self, session: "ApplicationSession") -> None:
        self._session = session
        self._receipts_this_request: set[str] = set()
        self._last_completed: "RequestOutcome | None" = None

    # --- 订阅侧入口 ------------------------------------------------------------------

    def note_phase(self, run_id: str | None, phase: str) -> None:
        """新请求开始清零本轮回执 (run_id None 即请求级 executing, 同旧口径)。"""
        if phase == "executing" and run_id is None:
            self._receipts_this_request = set()

    def observe_event(self, event: Event) -> None:
        """check 工具完成事件的 receipt_id 收集 (口径同旧 presenter)。"""
        if (
            isinstance(event, ToolCompleted)
            and event.result.name == CHECK_TOOL_NAME
            and event.result.details
        ):
            receipt_id = event.result.details.get("receipt_id")
            if isinstance(receipt_id, str) and receipt_id:
                self._receipts_this_request.add(receipt_id)

    def on_request_completed(self, outcome: "RequestOutcome") -> list[str] | None:
        """记录最近完成 outcome; 检查模式返回结论行, 正常模式返回 None (安静)。"""
        self._last_completed = outcome
        if outcome.decision is None:
            return None
        return ["检查结论", *delivery_lines(outcome, self._session, self._receipts_this_request)]

    # --- /status 事实 -----------------------------------------------------------------

    def status_lines(self) -> list[str]:
        """/status 的检查事实: 验证运行 + 最近完成请求的检查摘要与累计口径。"""
        lines: list[str] = []
        if self._session.verification_run_id is not None:
            lines.append(f"验证运行: {self._session.verification_run_id}")
        if self._last_completed is None:
            if self._session.verification_run_id is not None:
                lines.append(f"检查: 不可得")
            return lines
        check = check_status_text(self._last_completed, self._session)
        if check is not None:
            lines.append(f"检查: {check}")
        scope = change_scope_text(self._last_completed)
        if scope is not None:
            lines.append(scope)
        return lines


def clip_status_field(line: str) -> str:
    """单字段 512 UTF-8 bytes 限额: 超限按字符边界截断并给截断标记 (P12)。"""
    encoded = line.encode("utf-8")
    if len(encoded) <= STATUS_FIELD_MAX_BYTES:
        return line
    clipped = encoded[:STATUS_FIELD_MAX_BYTES].decode("utf-8", errors="ignore")
    return f"{clipped}…[已截断: {STATUS_FIELD_MAX_BYTES}/{len(encoded)} UTF-8 bytes]"


def bound_status_lines(lines: list[str]) -> list[str]:
    """/status 总限额 (20 行/8192 UTF-8 bytes) + 单字段限额与截断标记 (P12)。"""
    bounded = [clip_status_field(line) for line in lines]
    total = "\n".join(bounded)
    if len(bounded) <= STATUS_MAX_LINES and len(total.encode("utf-8")) <= STATUS_MAX_BYTES:
        return bounded
    kept, original_lines, original_bytes, truncated = clip_lines(
        total, STATUS_MAX_LINES, STATUS_MAX_BYTES
    )
    return [*kept, *truncation_note(kept, original_lines, original_bytes, truncated)]


__all__ = [
    "BLOCKING_MAX_BYTES",
    "BLOCKING_MAX_LINES",
    "CHECK_TOOL_NAME",
    "CURRENT_RUN_CHECK_LABEL",
    "CheckProjection",
    "HISTORICAL_NEUTRAL_LABEL",
    "HISTORICAL_PASS_LABEL",
    "NORMAL_MODE_CHECK_LABEL",
    "NOT_RUN_CHECK_LABEL",
    "PROCESS_UNCERTAIN_LABEL",
    "STATUS_FIELD_MAX_BYTES",
    "STATUS_MAX_BYTES",
    "STATUS_MAX_LINES",
    "artifact_line",
    "bound_status_lines",
    "change_scope_text",
    "check_status_text",
    "clip_lines",
    "clip_status_field",
    "delivery_lines",
    "truncation_note",
]
