"""已有事实的本地报告；不读取工作区、执行检查或恢复交付权威。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class ReportError(ValueError):
    pass


def _metrics(calls) -> dict:
    records = list(calls)
    metrics = [record.metrics for record in records]
    result = {'calls': len(records)}
    for field in ('input_tokens', 'output_tokens', 'cache_hit_input_tokens', 'cache_miss_input_tokens'):
        values = [getattr(item, field) for item in metrics if item is not None and getattr(item, field) is not None]
        result[field] = {'known_total': sum(values) if values else None, 'known_calls': len(values)}
    return result


def build_report(*, config, persistence, generation: int, outcome, request_model: str | None, model_identity,
                 ordinary_calls, summary_calls, cost, restored_receipts) -> dict:
    """白名单字段序列化；不递归导出模型、异常、消息或工具对象。"""
    try:
        cicada_version = version('cicada')
    except PackageNotFoundError:
        cicada_version = None
    latest = None
    if outcome is not None:
        view, evidence, decision = outcome.view, outcome.evidence, outcome.decision
        latest = {
            'run_id': outcome.result.run_id,
            'model_at_request': request_model,
            'stop_reason': outcome.result.stop_reason,
            'exit_code': outcome.exit_code,
            'error_present': outcome.result.error is not None,
            'saved': outcome.persistence.saved if outcome.persistence is not None else None,
            'delivery_at_completion': ({'model_stopped': decision.model_stopped,
                'can_deliver': decision.can_deliver, 'blocking_reason_count': len(decision.blocking_reasons)}
                if decision is not None else None),
            'snapshot_at_completion': view.snapshot_ref if view is not None else None,
            'verification_run_id': view.verification_run_id if view is not None else None,
            'process_uncertain': view.process_uncertain if view is not None else None,
            'checks_at_completion': [{
                'check_id': state.check_id, 'status': state.status,
                'freshness_at_completion': state.freshness,
                'receipt_id': state.receipt.receipt_id if state.receipt else None,
                'verification_run_id': state.receipt.verification_run_id if state.receipt else None,
                'snapshot_before': state.receipt.snapshot_before if state.receipt else None,
                'snapshot_after': state.receipt.snapshot_after if state.receipt else None,
                'execution_status': state.receipt.execution_status if state.receipt else None,
                'exit_code': state.receipt.exit_code if state.receipt else None,
                'failure_kind': state.receipt.failure_kind if state.receipt else None,
            } for state in view.checks] if view is not None else [],
            'changes_at_completion': [{
                'path': change.relative_path, 'kind': change.kind,
                'before_sha256': change.before_sha256, 'after_sha256': change.after_sha256,
            } for change in evidence.changes] if evidence is not None else [],
            'change_evidence_complete': evidence.complete if evidence is not None else None,
            'change_evidence_failure_kind': evidence.failure_kind if evidence is not None else None,
        }
    return {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'runtime': {'cicada_version': cicada_version,
                    'python_version': '.'.join(str(part) for part in sys.version_info[:3])},
        'observation': 'recorded facts; not revalidated at export; no current delivery authority',
        'workspace': str(config.workspace.expanduser().resolve()),
        'session_id': persistence.session_id if persistence is not None else None,
        'session_generation': generation,
        'current_configuration': {
            'model_label': config.model_label, 'max_turns': config.max_turns,
            'model_identity': asdict(model_identity) if model_identity is not None else None,
            'prompt_version': config.prompt_info.version,
            'prompt_sha256': config.prompt_info.prompt_sha256,
            'instructions_sha256': config.prompt_info.instructions_sha256,
            'context_budget': asdict(config.budget) if config.budget is not None else None,
        },
        'latest_request': latest,
        'current_process_metrics': {'ordinary': _metrics(ordinary_calls), 'summary': _metrics(summary_calls)},
        'cost': asdict(cost) if cost is not None else None,
        'historical_receipt_count': len(restored_receipts),
        'historical_receipts': [{
            'request_id': receipt.request_id, 'epoch_id': receipt.epoch_id,
            'verification_run_id': receipt.verification_run_id, 'receipt_id': receipt.receipt_id,
            'definition_fingerprint': receipt.definition_fingerprint,
            'execution_status': receipt.execution_status, 'exit_code': receipt.exit_code,
            'timed_out': receipt.timed_out, 'cancelled': receipt.cancelled,
            'output_complete': receipt.output_complete,
        } for receipt in restored_receipts],
        'historical_receipts_have_current_authority': False,
        'content_policy': 'no prompts, conversation, reasoning, tool bodies, raw errors or credentials',
    }


def validate_report_path(path: Path) -> None:
    if path.suffix.lower() not in ('.json', '.md'):
        raise ReportError('report path must end in .json or .md')
    if path.exists() or path.is_symlink():
        raise ReportError('report destination already exists; choose a new path')


def write_report(path: Path, report: dict) -> None:
    """独占创建；不覆盖已有文件，写入失败清除本次新建的不完整文件。"""
    validate_report_path(path)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if path.suffix.lower() == '.md':
        fence = '```'
        while fence in payload:
            fence += '`'
        latest = report['latest_request']
        summary = (f"最近请求：{latest['stop_reason']}，退出码 {latest['exit_code']}。"
                   if latest is not None else '当前会话没有本进程完成的请求。')
        payload = ('# Cicada 任务诊断报告\n\n' + summary + '\n\n'
                   '以下为记录时事实，导出未重新检查工作区，不证明当前可交付。'
                   '不含对话、思考或工具正文。\n\n' + fence + 'json\n' + payload + '\n' + fence + '\n')
    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            created = True
            handle.write(payload)
    except (OSError, ValueError):
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise ReportError('report export failed; check destination and permissions') from None
