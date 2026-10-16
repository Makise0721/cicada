"""PV02 live 的本地授权记录校验；本模块不读取凭据或发送网络请求。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cicada.costs import CostProfile
from tests.support.swebench_budget import AttemptBudgetError


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def candidate_binding(*, manifest_path: Path, prepared_root: Path,
                      attempt_root: Path, profile: CostProfile) -> dict:
    """绑定候选事实；结果本身不表示获得了执行授权。"""
    from tests.support import swebench_lifecycle as lifecycle

    try:
        manifest_path = Path(manifest_path).resolve(strict=True)
        prepared_root = Path(prepared_root).resolve(strict=True)
        attempt_root = Path(attempt_root).resolve()
        lock = prepared_root / "lock.json"
        receipt = prepared_root / "prepare-receipt.json"
        plan = lifecycle.rehearsal_plan(prepared_root, manifest_path)
        manifest = plan["manifest"]
        project = Path(__file__).resolve().parents[2]

        def git(ref):
            result = subprocess.run(["git", "rev-parse", ref], cwd=project,
                                    check=True, capture_output=True, text=True)
            value = result.stdout.strip()
            if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError
            return value

        # 读取存在性及 hash，不读取 private 评估材料或 credentials。
        return {
            "schema_version": 1,
            "task_key": manifest["task_key"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "prepared_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "prepare_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "profile_sha256": profile.sha256,
            "profile_checked_at": profile.checked_at,
            "runtime_sha256": lifecycle.runtime_identity(),
            "git_head": git("HEAD"),
            "git_tree": git("HEAD^{tree}"),
            "prompt_sha256": plan["prompt_sha256"],
            "attempt_root": str(attempt_root),
            "prepared_root": str(prepared_root),
            "model": manifest["model"],
            "limits": manifest["limits"],
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError,
            subprocess.SubprocessError):
        raise AttemptBudgetError("authorization candidate evidence invalid") from None


def _utc_timestamp(value):
    if not isinstance(value, str):
        raise ValueError
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError
    return parsed


def validate_authorization(path: Path | None, expected: dict) -> dict:
    """严格匹配一次显式授权及有效价格核对；任何失败均零发送拒绝。"""
    if path is None:
        raise AttemptBudgetError("live authorization required")
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError
        record = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                            parse_constant=_reject_constant)
        if (not isinstance(record, dict) or set(record) != {
                "schema_version", "authorized", "approval_reference", "expires_at",
                "pricing_checked_at", "binding"}
                or type(record["schema_version"]) is not int
                or record["schema_version"] != 1
                or record["authorized"] is not True
                or not isinstance(record["approval_reference"], str)
                or not record["approval_reference"].strip()
                or not isinstance(record["binding"], dict)):
            raise ValueError
        # JSON 字节比较区分 true/1 与 1/1.0；Python 字典等号不能区分。
        if _json_bytes(record["binding"]) != _json_bytes(expected):
            raise ValueError
        now = datetime.now(timezone.utc)
        expires = _utc_timestamp(record["expires_at"])
        checked = _utc_timestamp(record["pricing_checked_at"])
        if expires <= now or not 0 <= (now - checked).total_seconds() <= 86_400:
            raise ValueError
        shanghai = timezone(timedelta(hours=8))
        if checked.astimezone(shanghai).date().isoformat() != expected["profile_checked_at"]:
            raise ValueError
        return record
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise AttemptBudgetError("live authorization invalid or stale") from None
