"""Offline controls for the fixed P4 checker and immutable attempt evidence."""

from __future__ import annotations

import os
import json
import subprocess

import pytest

import p4_fixture as harness


# 代表值覆盖两个拒绝分支: 越下界/越上界 (范围) 与非 int 类型


def test_acceptance_rejects_observer_recording_failure(tmp_path):
    observer = {"evidence_complete": False, "recording_errors": ["injected write failure"]}
    (tmp_path / "observer-result.json").write_text(json.dumps(observer), encoding="utf-8")
    result = {"evidence_dir": str(tmp_path), "timed_out": False,
              "episode_within_budget": True, "proxy_errors": [], "returncode": 0}
    with pytest.raises(AssertionError, match="injected write failure"):
        harness.validate_attempt(result)
    assert json.loads((tmp_path / "acceptance.json").read_text())["accepted"] is False


def test_build_fixture_refuses_to_replace_an_existing_directory(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    marker = fixture / "source.py"
    marker.write_bytes(b"original fixture")
    with pytest.raises(FileExistsError):
        harness.build_fixture(fixture)
    assert marker.read_bytes() == b"original fixture"


@pytest.mark.parametrize("change,expected_exit", [
    ("signature-only", 1), ("always-zero", 1), ("success-count", 0), ("wrong-shape", 1),
])
def test_behavior_checker_rejects_noop_and_wrong_counts(tmp_path, change, expected_exit):
    fixture = tmp_path / "fixture"
    facts = harness.build_fixture(fixture)
    assert facts["checker_baseline_exit"] == 1
    assert facts["checker_baseline_stderr_empty"]
    assert "CHECK_FAIL" in facts["checker_baseline_stdout"]
    source = fixture / "src/cicada/reporting.py"
    original = source.read_text(encoding="utf-8")
    if change == "signature-only":
        modified = original.replace(
            "def build_run_summary(result: RunResult) -> str:",
            "def build_run_summary(result: RunResult, tools_ok: bool = False) -> str:",
        )
        assert modified != original
    else:
        # Positive/negative controls live only in this private fixture.
        count = "0" if change == "always-zero" else (
            "sum(not m.result.is_error for m in result.messages if isinstance(m, ToolResultMessage))"
        )
        suffix = " + '\\nextra'" if change == "wrong-shape" else ""
        modified = original + (
            "\n_original_summary = build_run_summary\n"
            "def build_run_summary(result):\n"
            "    lines = _original_summary(result).splitlines()\n"
            f"    count = {count}\n"
            "    lines[0] = lines[0][:-1] + f' tools_ok={count}]'\n"
            f"    return '\\n'.join(lines){suffix}\n"
        )
    source.write_text(modified, encoding="utf-8", newline="\n")
    result = subprocess.run(
        [str(harness.VENV_PYTHON), harness.CHECKER_NAME], cwd=fixture,
        capture_output=True, text=True, encoding="utf-8", timeout=15,
        env={**os.environ, "PYTHONPATH": str(harness.SRC), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert result.stderr == ""
    assert result.stdout.startswith("CHECK_PASS" if expected_exit == 0 else "CHECK_FAIL")
    assert harness.sha256_file(fixture / harness.CHECKER_NAME) == facts["checker_sha256"]
