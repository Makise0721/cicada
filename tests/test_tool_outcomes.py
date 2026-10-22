"""结果载荷契约的最小回归: 标注/解码/版本与 legacy 行为 (跨层契约).

只证明契约本身的可观察行为; 各消费者的既有测试继续覆盖其判定路径.
"""

from __future__ import annotations

import pytest

from cicada.tool_outcomes import (
    CONTRACT,
    CONTRACT_VERSION,
    check_status_details,
    mutation_details,
    mutation_facts,
    process_details,
    process_facts,
)


def test_process_details_roundtrip_preserves_exhibition_fields() -> None:
    details = process_details(
        {"exit_code": 0, "output_complete": True, "timed_out": False, "cancelled": False,
         "receipt_id": "r-1", "verification_run_id": "v-1",
         "artifact_path": "out.txt", "artifact_sha256": "sha"},
        extras={"truncated": False},
    )
    assert details["contract"] == CONTRACT and details["contract_version"] == CONTRACT_VERSION
    assert details["truncated"] is False
    facts = process_facts(details)
    assert facts is not None and facts.legacy is False
    assert facts.receipt_id == "r-1" and facts.termination_known
    assert facts.artifact_path == "out.txt" and facts.artifact_sha256 == "sha"


def test_legacy_artifact_keys_decode() -> None:
    facts = process_facts({"output_artifact_path": "old", "output_artifact_sha256": "h"})
    assert facts is not None and facts.artifact_path == "old" and facts.artifact_sha256 == "h"
    mutation = mutation_facts({"mutation": {"state": "committed"}, "full_diff_path": "d.diff"})
    assert mutation is not None and mutation.artifact_path == "d.diff"


def test_legacy_details_still_decode() -> None:
    facts = process_facts(
        {"exit_code": 0, "output_complete": True, "timed_out": False, "cancelled": False}
    )
    assert facts is not None and facts.legacy is True and facts.termination_known


def test_unknown_version_is_not_trusted() -> None:
    details = process_details({"exit_code": 0, "output_complete": True})
    details["contract_version"] = CONTRACT_VERSION + 1
    assert process_facts(details) is None


def test_status_payload_is_not_process_facts() -> None:
    assert process_facts(check_status_details({"receipt_ids": []})) is None


def test_mutation_details_roundtrip_and_legacy() -> None:
    mutation = {
        "version": 1, "state": "committed", "reason": "ok", "canonical_path": "a.txt",
        "current_matches": True, "caller_cancelled": False, "winerror": None,
        "retained_paths": [], "cleanup_warning": None,
    }
    facts = mutation_facts(mutation_details(mutation, extras={"diff": ""}))
    assert facts is not None and facts.state == "committed" and facts.legacy is False
    legacy = mutation_facts({"mutation": {"state": "unknown"}})
    assert legacy is not None and legacy.state == "unknown" and legacy.legacy


def test_producer_rejects_wrong_field_type() -> None:
    with pytest.raises(TypeError):
        process_details({"exit_code": True})
