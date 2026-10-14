"""私有基准 harness 的补丁/评测安全边界；不调用 Docker 或模型。"""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.support.swebench_bridge import (
    GitCLI, PatchScopeError, export_patch, grade_from_artifacts,
)
from tests.support.swebench_owner import EvaluationOwner


def repository(tmp_path: Path):
    root = tmp_path / "repository"
    root.mkdir()
    git = GitCLI(root)
    git.run("init", check=True)
    git.run("config", "user.email", "test@example.invalid", check=True)
    git.run("config", "user.name", "test", check=True)
    git.run("config", "core.autocrlf", "false", check=True)
    (root / "source.txt").write_text("base\n", encoding="utf-8")
    git.run("add", ".", check=True)
    git.run("commit", "-m", "base", check=True)
    return root, git, git.run("rev-parse", "HEAD", check=True).stdout.strip()


def test_large_patch_new_file_and_deletion_round_trip(tmp_path):
    root, git, base = repository(tmp_path)
    index = (root / ".git/index").read_bytes()
    (root / "source.txt").unlink()
    content = "".join(f"line {i}: payload\n" for i in range(8000))
    (root / "new.txt").write_text(content, encoding="utf-8", newline="\n")
    outputs = root / ".cicada/outputs"
    outputs.mkdir(parents=True)
    (outputs / "check.txt").write_text("runtime evidence", encoding="utf-8")
    exported = export_patch(root, base, ("source.txt", "new.txt"), tmp_path, git)
    assert len(exported.patch_bytes) > 20000
    assert b".cicada" not in exported.patch_bytes
    assert (root / ".git/index").read_bytes() == index
    patch = tmp_path / "export.patch"
    patch.write_bytes(exported.patch_bytes)
    # 只回退本测试拥有的源码，验证完整补丁能重建最终内容。
    (root / "new.txt").unlink()
    git.run("checkout", "--", "source.txt", check=True)
    git.run("apply", str(patch), check=True)
    assert not (root / "source.txt").exists()
    assert (root / "new.txt").read_text(encoding="utf-8") == content


def test_outside_scope_including_ignored_file_is_rejected(tmp_path):
    root, git, base = repository(tmp_path)
    (root / ".git/info/exclude").write_text("secret.txt\n", encoding="utf-8")
    (root / "secret.txt").write_text("unexplained", encoding="utf-8")
    with pytest.raises(PatchScopeError):
        export_patch(root, base, ("source.txt",), tmp_path, git)


def test_official_reward_cannot_replace_required_coverage():
    output = json.dumps({"tests": [{"name": "regression", "status": "PASSED"}]}).encode()
    assert not grade_from_artifacts(0, b"1", output, ["required"], ["regression"])["grader_pass"]
    assert not grade_from_artifacts(0, b"1", output, [], ["regression"])["grader_pass"]
    assert not grade_from_artifacts(0, b"1", b"not-json", ["required"], [])["grader_pass"]
    assert grade_from_artifacts(1, b"0", None, ["required"], [])["parse_failed"]
    assert grade_from_artifacts(0, b"1", output, ["regression"], [])["grader_pass"]


def test_owner_seals_and_cleans_after_checker_is_killed(tmp_path):
    started = threading.Event()
    stopped = threading.Event()
    resource = tmp_path / "owned-resource"

    def evaluate(cancel):
        resource.write_text("active", encoding="utf-8")
        started.set()
        if cancel.wait(10):
            stopped.set()
        return 2, {"result": "anomaly"}

    def cleanup():
        resource.unlink(missing_ok=True)
        return [{"outcome": "confirmed"}]

    with EvaluationOwner(tmp_path, evaluate, cleanup) as owner:
        code = "from pathlib import Path; from tests.support.swebench_owner import request_check; import sys; request_check(Path(sys.argv[1]))"
        checker = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert started.wait(10)
            checker.kill()
            checker.wait(timeout=5)
            assert stopped.wait(5)
            assert owner.stopping.is_set()
        finally:
            if checker.poll() is None:
                checker.kill()
            checker.communicate(timeout=5)
    assert not resource.exists()
    assert not owner.thread.is_alive() and not owner.watch.is_alive()
