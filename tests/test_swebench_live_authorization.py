"""授权闸的关键拒绝代表；有效授权只在 pytest 临时目录中构造。"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from cicada.costs import CostProfile
from tests.support.swebench_authorization import candidate_binding, validate_authorization
from tests.support.swebench_budget import AttemptBudgetError


def _record():
    now = datetime.now(timezone.utc)
    binding = {"schema_version": 1, "task_key": "fixture-only",
               "profile_checked_at": now.astimezone(timezone(timedelta(hours=8))).date().isoformat(),
               "runtime_sha256": {"fixture.py": "a" * 64}}
    return binding, {
        "schema_version": 1, "authorized": True,
        "approval_reference": "pytest fixture only; no human authorization",
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "pricing_checked_at": now.isoformat(), "binding": binding.copy(),
    }


def test_authorization_requires_an_explicit_record(tmp_path):
    with pytest.raises(AttemptBudgetError):
        validate_authorization(None, {})
    with pytest.raises(AttemptBudgetError):
        validate_authorization(tmp_path / "absent.json", {})


def test_authorization_accepts_the_exact_current_binding(tmp_path):
    expected, record = _record()
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert validate_authorization(path, expected) == record


def test_candidate_binds_present_evidence_and_detects_missing_receipt(tmp_path):
    prepared = tmp_path / "prepared"
    (prepared / "public").mkdir(parents=True)
    (prepared / "public/instruction.md").write_text("fixture instruction", encoding="utf-8")
    (prepared / "lock.json").write_text(json.dumps({"instance": {
        "allowed_paths": ["fixture.py"], "base_commit": "a" * 40}}), encoding="utf-8")
    receipt = prepared / "prepare-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"task_key": "fixture-only", "model": {"normal": {}},
                                    "limits": {"run_timeout_s": 900}}), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"schema_version": 1,
        "source": "https://example.test/pricing", "checked_at": "2026-10-16",
        "models": {"deepseek-flash": {"input_cny_per_million": "2",
                                       "output_cny_per_million": "8"}}}), encoding="utf-8")
    arguments = dict(manifest_path=manifest, prepared_root=prepared,
                     attempt_root=tmp_path / "attempt", profile=CostProfile.load(profile_path))
    binding = candidate_binding(**arguments)
    assert binding["prepared_root"] == str(prepared.resolve())
    assert binding["attempt_root"] == str((tmp_path / "attempt").resolve())
    assert len(binding["git_head"]) == 40
    assert binding["runtime_sha256"]
    receipt.write_text('{"changed":true}', encoding="utf-8")
    assert candidate_binding(**arguments)["prepare_receipt_sha256"] != binding["prepare_receipt_sha256"]
    receipt.unlink()
    with pytest.raises(AttemptBudgetError, match="^authorization candidate evidence invalid$"):
        candidate_binding(**arguments)


@pytest.mark.parametrize("defect", [
    "false", "drift", "expired", "duplicate", "unknown", "missing",
    "future-price", "stale-price", "naive-time", "profile-date", "boolean-identity",
])
def test_authorization_refuses_invalid_or_stale_records(tmp_path, defect):
    expected, record = _record()
    if defect == "false":
        record["authorized"] = False
    elif defect == "drift":
        record["binding"]["runtime_sha256"] = {"fixture.py": "b" * 64}
    elif defect == "expired":
        record["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    elif defect == "unknown":
        record["other"] = "private-data-must-not-be-echoed"
    elif defect == "missing":
        del record["approval_reference"]
    elif defect == "future-price":
        record["pricing_checked_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    elif defect == "stale-price":
        record["pricing_checked_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    elif defect == "naive-time":
        record["expires_at"] = "2099-01-01T00:00:00"
    elif defect == "profile-date":
        expected["profile_checked_at"] = record["binding"]["profile_checked_at"] = "2000-01-01"
    elif defect == "boolean-identity":
        record["binding"]["schema_version"] = True
    raw = json.dumps(record)
    if defect == "duplicate":
        raw = raw.replace('"authorized": true', '"authorized": false, "authorized": true')
    path = tmp_path / "authorization.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(AttemptBudgetError, match="^live authorization invalid or stale$"):
        validate_authorization(path, expected)
