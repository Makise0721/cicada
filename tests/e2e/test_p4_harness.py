"""Offline controls for the fixed P4 checker and immutable attempt evidence."""

from __future__ import annotations

import os
import subprocess

import pytest

import p4_fixture as harness


@pytest.mark.parametrize("attempt", [-1, 0, 3, True, False, "1", 1.0, None])
def test_invalid_attempt_stops_before_creating_evidence(tmp_path, monkeypatch, attempt):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "EVIDENCE_ROOT", tmp_path / "evidence")

    def forbidden(*args):
        pytest.fail("invalid numbering must stop before fixture/live work")

    monkeypatch.setattr(harness, "build_fixture", forbidden)
    monkeypatch.setattr(harness, "ollama_digest", forbidden)
    with pytest.raises(ValueError, match="attempt must be"):
        harness.run_attempt(attempt)
    assert not harness.EVIDENCE_ROOT.exists()


def test_existing_attempt_is_preserved_before_any_fixture_or_live_work(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "EVIDENCE_ROOT", tmp_path / "evidence")
    attempt = harness.EVIDENCE_ROOT / "attempt-1"
    attempt.mkdir(parents=True)
    marker = attempt / "raw-evidence.bin"
    marker.write_bytes(b"original evidence")

    def forbidden(*args):
        pytest.fail("an existing attempt must stop before fixture/live work")

    monkeypatch.setattr(harness, "build_fixture", forbidden)
    monkeypatch.setattr(harness, "ollama_digest", forbidden)
    with pytest.raises(FileExistsError):
        harness.run_attempt(1)
    assert marker.read_bytes() == b"original evidence"


def test_existing_live_fixture_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "EVIDENCE_ROOT", tmp_path / "evidence")
    fixture = tmp_path / ".scratch/p4-implementation/temp/I-0607-live/attempt-1/fixture"
    fixture.mkdir(parents=True)
    marker = fixture / "source.py"
    marker.write_bytes(b"original fixture")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        harness.run_attempt(1)
    assert marker.read_bytes() == b"original fixture"
    assert not harness.EVIDENCE_ROOT.exists()


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
