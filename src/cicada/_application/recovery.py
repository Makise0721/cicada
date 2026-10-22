"""恢复事实的准入、证明、投影与身份计算；不打开存储或发布会话。"""

from __future__ import annotations

import hashlib
import json

from cicada.boot import App
from cicada.context_policy import SummaryBlock, canonical_prefix_sha256, summary_block_message
from cicada.core.messages import ToolResultMessage
from cicada.persistence.codec import dumps_strict, encode_messages
from cicada.persistence.contracts import (
    RecoverySnapshot,
    StoredReceipt,
    conflicting_launch_fields,
    receipt_terminal_trustworthy,
)
from cicada.run_policy import HistoricalPrefixProof, HistoricalProjection, PrefixEntry

from .contracts import (
    ApplicationOpenError,
    PersistenceConfig,
)


def _tool_fingerprint(app: App, *, legacy_output_paths: bool = False) -> str:
    """bootstrap 解析后的实际工具声明指纹: 稳定顺序 + 完整 schema 内容."""
    tools = getattr(app.agent, "_tools", {})
    specs = []
    for name in sorted(tools):
        spec = tools[name].spec
        description = spec.description
        if legacy_output_paths and name == "edit":
            description = description.replace(
                "写入用户级运行状态目录的唯一文件",
                "写入 workspace 的 .cicada/outputs/ 唯一文件",
            )
        elif legacy_output_paths and name == "powershell":
            description = description.replace(
                "超限时只保留有界 tail, 完整输出持续写入用户级运行状态目录下的工件 (最多 16 MiB).",
                "超限时只保留有界 tail, 完整输出持续写入工作区 .cicada/outputs 下的工件 (最多 16 MiB).",
            )
        specs.append({"name": spec.name, "description": description,
                      "parameters": spec.parameters})
    encoded = json.dumps(specs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verification_checks_signature(checks) -> str:
    """检查定义的稳定摘要; 进入 LaunchIdentity.checks_signature 参与恢复比对 (03)."""
    payload = [
        {"check_id": check.check_id, "command": check.command, "timeout_s": check.timeout_s}
        for check in checks
    ]
    return hashlib.sha256(dumps_strict(payload).encode("utf-8")).hexdigest()


def _receipt_fingerprint(check_id: str, command: str, timeout_s: float) -> str:
    return hashlib.sha256(dumps_strict(
        {"check_id": check_id, "command": command, "timeout_s": timeout_s}
    ).encode("utf-8")).hexdigest()


def _receipt_facts_mismatch(receipt: StoredReceipt, details: dict) -> str | None:
    """消息 details 的终结事实与回执记录不一致 -> 返回原因; 一致返回 None.

    契约 §5「事实不匹配……属于恢复格式/关系错误」: 缺键不比 (None 容忍),
    显式事实冲突 (exit_code/timed_out/cancelled/output_complete) 拒绝恢复.
    """
    exit_code = details.get("exit_code")
    if (
        isinstance(exit_code, int) and not isinstance(exit_code, bool)
        and receipt.exit_code != exit_code
    ):
        return f"exit_code {exit_code} but the receipt records {receipt.exit_code}"
    for key in ("timed_out", "cancelled", "output_complete"):
        value = details.get(key)
        if isinstance(value, bool) and getattr(receipt, key) != value:
            return f"{key} is {value} but the receipt records {getattr(receipt, key)}"
    return None


def _build_prefix_proof(snapshot: RecoverySnapshot) -> HistoricalProjection:
    """从恢复事实构建前缀证明容器; 回执引用缺失/错挂/事实不符属恢复关系错误, 模型零请求."""
    receipts_by_id = {receipt.receipt_id: receipt for receipt in snapshot.restored_receipts}
    entries: list[PrefixEntry] = []
    for request in snapshot.requests:
        if request.context_generation != snapshot.context_generation:
            continue
        for message in request.messages:
            receipt_id: str | None = None
            verification_run_id: str | None = None
            if isinstance(message, ToolResultMessage) and message.result.name == "check":
                details = (
                    message.result.details if isinstance(message.result.details, dict) else {}
                )
                candidate = details.get("receipt_id")
                if isinstance(candidate, str) and candidate:
                    receipt = receipts_by_id.get(candidate)
                    if receipt is None:
                        raise ApplicationOpenError(
                            f"restored history references check receipt {candidate} "
                            "that is not in the archive (relation error)"
                        )
                    if receipt.request_id != request.request_id:
                        raise ApplicationOpenError(
                            f"restored receipt {candidate} does not belong to its request"
                        )
                    mismatch = _receipt_facts_mismatch(receipt, details)
                    if mismatch is not None:
                        raise ApplicationOpenError(
                            f"restored check result contradicts receipt {candidate}: {mismatch}"
                        )
                    receipt_id = candidate
                    verification_run_id = receipt.verification_run_id
            entries.append(PrefixEntry(
                epoch_id=request.epoch_id,
                request_id=request.request_id,
                seq=request.seq,
                message_sha256=hashlib.sha256(
                    encode_messages((message,)).encode("utf-8")
                ).hexdigest(),
                receipt_id=receipt_id,
                verification_run_id=verification_run_id,
            ))
    # 显式关系校验: 证明逐位置背书的内容必须就是实际注入的规范历史
    # (隐式依赖 store 两条推导路径一致不够; 分叉按 fail-closed 拒绝, 见 A09)
    if len(entries) != len(snapshot.canonical_history):
        raise ApplicationOpenError(
            "restored requests do not reconstruct the canonical history (relation error)"
        )
    for entry, message in zip(entries, snapshot.canonical_history):
        observed = hashlib.sha256(
            encode_messages((message,)).encode("utf-8")
        ).hexdigest()
        if observed != entry.message_sha256:
            raise ApplicationOpenError(
                "restored requests do not reconstruct the canonical history (relation error)"
            )
    return HistoricalProjection(HistoricalPrefixProof(entries=tuple(entries)))


def _rebuild_context_projection(snapshot: RecoverySnapshot) -> tuple[SummaryBlock, ...]:
    """按摘要记录确定性重建投影块 (02, D6/A08); 关系/指纹不符属恢复错误, 模型零请求.

    逐块核对: generation 与当前代次一致、seq 引用当前代次已提交完整请求、
    覆盖前缀 [0:end) 的规范编码 sha256 与规范历史逐位一致。块消息由
    summary_block_message 程序生成, 与压缩时字节逐位一致。
    """
    generation_messages = snapshot.canonical_history
    seqs = {
        request.seq
        for request in snapshot.requests
        if request.context_generation == snapshot.context_generation
    }
    blocks: list[SummaryBlock] = []
    for record in snapshot.summaries:
        if record.context_generation != snapshot.context_generation:
            raise ApplicationOpenError(
                f"restored summary {record.summary_id} belongs to generation "
                f"{record.context_generation}, not the current {snapshot.context_generation}"
            )
        for seq in (record.first_request_seq, record.last_request_seq):
            if seq not in seqs:
                raise ApplicationOpenError(
                    f"restored summary {record.summary_id} references request {seq} "
                    "outside the current generation (relation error)"
                )
        if record.end_message_index > len(generation_messages):
            raise ApplicationOpenError(
                f"restored summary {record.summary_id} coverage end exceeds the "
                "canonical history (relation error)"
            )
        observed = canonical_prefix_sha256(generation_messages[: record.end_message_index])
        if observed != record.prefix_sha256:
            raise ApplicationOpenError(
                f"restored summary {record.summary_id} coverage does not match the "
                "canonical history (relation error)"
            )
        blocks.append(SummaryBlock(
            message=summary_block_message(
                first_request_seq=record.first_request_seq,
                last_request_seq=record.last_request_seq,
                message_count=record.message_count,
                text=record.text,
            ),
            end_message_index=record.end_message_index,
            first_request_seq=record.first_request_seq,
            last_request_seq=record.last_request_seq,
            message_count=record.message_count,
            prefix_sha256=record.prefix_sha256,
        ))
    return tuple(blocks)


def _admit_persistent_resume(
    persistence: PersistenceConfig, snapshot: RecoverySnapshot
) -> str | None:
    """恢复准入: 配置冲突在 bootstrap 前拒绝 (无副作用); 返回 view-only 原因或 None.

    无未证终结活动的中断请求可以继续 (pending intent 只标中断并显示);
    存在未证终结/未闭合活动时恢复为查看状态, 不能经模式或配置覆盖解除 (A15).
    """
    if persistence.session_id is None:
        return None
    saved_identity = snapshot.identity
    if saved_identity is None:
        raise ApplicationOpenError(
            "the session archive has no saved launch identity to compare against"
        )
    if not persistence.adopt_current_config:
        conflicts = conflicting_launch_fields(saved_identity.launch, persistence.identity)
        if conflicts:
            raise ApplicationOpenError(
                "restored session config conflicts with the current launch: " + ", ".join(conflicts)
            )
    unproven = snapshot.unproven_process_activities()
    if unproven:
        first = unproven[0]
        return (
            f"restored session has {len(unproven)} process activity without proven "
            f"termination (first: {first.tool_name} call {first.call_id}, "
            + (
                f"reason: {first.terminal.reason}"
                if first.terminal is not None and first.terminal.reason
                else "no terminal record"
            ) + ")"
        )
    untrusted = [
        receipt for receipt in snapshot.restored_receipts
        if not receipt_terminal_trustworthy(receipt)
    ]
    if untrusted:
        # 存档级 process_uncertain 证据 (契约 D3): 检查回执终结不可信, 仅可查看
        return (
            f"restored archive records process_uncertain "
            f"({len(untrusted)} check receipt without trustworthy terminal facts; "
            f"first: receipt {untrusted[0].receipt_id})"
        )
    return None
