"""Offline controls for the fixed P4 checker and immutable attempt evidence."""

from __future__ import annotations

import os
import json
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
    fixture = tmp_path / harness.FIXTURE_RELATIVE / "attempt-1/fixture"
    fixture.mkdir(parents=True)
    marker = fixture / "source.py"
    marker.write_bytes(b"original fixture")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        harness.run_attempt(1)
    assert marker.read_bytes() == b"original fixture"
    assert not harness.EVIDENCE_ROOT.exists()


def test_campaign_freeze_rejects_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "SRC", tmp_path / "src")
    monkeypatch.setattr(harness, "EVIDENCE_ROOT", tmp_path / "campaign/live")
    observer = tmp_path / "observer.py"
    observer.write_text("# observation only", encoding="utf-8")
    monkeypatch.setattr(harness, "OBSERVER", observer)
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *a, **kw: "fixed-head\n")
    harness.SRC.mkdir()
    (harness.SRC / "source.py").write_text("# frozen source", encoding="utf-8")
    first = harness.freeze_campaign()
    manifest = harness.EVIDENCE_ROOT.parent / "campaign-manifest.json"
    original = manifest.read_bytes()
    assert harness.freeze_campaign() == first
    monkeypatch.setattr(harness, "TASK_PROMPT", "a changed prompt")
    with pytest.raises(ValueError, match="campaign inputs changed"):
        harness.freeze_campaign()
    assert manifest.read_bytes() == original


def test_failed_attempt_is_saved_and_cannot_be_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(harness, "freeze_campaign", lambda: {})

    def fail(*args):
        raise RuntimeError("offline failure before any live call")

    monkeypatch.setattr(harness, "_execute_attempt", fail)
    with pytest.raises(RuntimeError, match="offline failure"):
        harness.run_attempt(1)
    error_path = harness.EVIDENCE_ROOT / "attempt-1/attempt-error.json"
    original = error_path.read_bytes()
    assert json.loads(original)["error_type"] == "RuntimeError"
    with pytest.raises(FileExistsError):
        harness.run_attempt(1)
    assert error_path.read_bytes() == original


def test_acceptance_rejects_observer_recording_failure(tmp_path):
    observer = {"evidence_complete": False, "recording_errors": ["injected write failure"]}
    (tmp_path / "observer-result.json").write_text(json.dumps(observer), encoding="utf-8")
    result = {"evidence_dir": str(tmp_path), "timed_out": False,
              "episode_within_budget": True, "proxy_errors": [], "returncode": 0}
    with pytest.raises(AssertionError, match="injected write failure"):
        harness.validate_attempt(result)
    assert json.loads((tmp_path / "acceptance.json").read_text())["accepted"] is False


def test_proxy_is_closed_when_post_creation_identity_write_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PROJECT", tmp_path)
    monkeypatch.setattr(harness, "EVIDENCE_ROOT", tmp_path / "campaign/live")
    campaign = harness.EVIDENCE_ROOT.parent
    campaign.mkdir()
    (campaign / "campaign-manifest.json").write_text("{}", encoding="utf-8")
    evidence = harness.EVIDENCE_ROOT / "attempt-1"
    evidence.mkdir(parents=True)
    facts = {
        "main_head": "fixed", "fixture_commit": "fixture", "copied_py_files": 0,
        "checker_sha256": "checker", "checker_baseline_exit": 1,
        "checker_baseline_stderr_empty": True, "checker_baseline_stdout": "CHECK_FAIL",
        "import_probe": "private/src/cicada/__init__.py", "baseline_manifest": {},
    }
    monkeypatch.setattr(harness, "build_fixture", lambda *a: facts)
    monkeypatch.setattr(harness, "ollama_digest", lambda: {})
    operations = []

    class Proxy:
        url = "http://offline.invalid"

        def __init__(self, *args):
            operations.append("create")

        def stop(self):
            operations.append("stop")

        def save(self):
            operations.append("save")

    monkeypatch.setattr(harness, "_TeeProxy", Proxy)
    original_write = harness._write_json

    def fail_identity(path, value):
        if path.name == "identity.json" and "argv" in value:
            raise OSError("post-proxy identity write failed")
        original_write(path, value)

    monkeypatch.setattr(harness, "_write_json", fail_identity)
    frozen = {"launcher_sha256": "launcher", "observer_sha256": "observer",
              "main_head": "fixed", "src_manifest": {}, "prompt_sha256": "prompt"}
    with pytest.raises(OSError, match="post-proxy identity write failed"):
        harness._execute_attempt(1, evidence, tmp_path / "fixture", frozen, harness.time.monotonic())
    assert operations == ["create", "stop", "save"]


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
