"""改动审阅 reader 捕获分组的聚焦验证 (03 收口应修回归).

真实 Git/Windows fixture 覆盖两处收口修复的公开行为: 大小写改名的删除＋新增
展开, 以及 filter 重核对自身失败只降级 unstaged。
"""

import pytest

from cicada.core.cancel import CancelToken
from cicada.plugins.coding.change_review import ChangeReview
from cicada.plugins.coding.git_io import GitIOError
from cicada.plugins.coding.workspace import Workspace
from tests.support.plugins import git


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "baseline")
    return root


def make(repo):
    return ChangeReview(Workspace.create(repo))


def group_of(snapshot, name):
    return next(state for state in snapshot.groups if state.group == name)


async def test_case_only_rename_stages_as_delete_and_add(repo):
    # git mv Foo.txt foo.txt 暂存后 --no-renames 展开为同 casefold 的删除＋新增对；
    # C2 要求按删除＋新增登记两个条目，而不是整组 alias_conflict。
    (repo / "Foo.txt").write_text("content\n", encoding="utf-8")
    git(repo, "add", "Foo.txt")
    git(repo, "commit", "-q", "-m", "add Foo.txt")
    git(repo, "mv", "Foo.txt", "foo.txt")
    snapshot = await make(repo).capture("review-1", CancelToken())
    staged = group_of(snapshot, "staged")
    assert staged.state == "complete", staged.reason
    staged_items = sorted(item.relative_path for item in snapshot.items if item.group == "staged")
    assert staged_items == ["Foo.txt", "foo.txt"]
    kinds = {item.relative_path: item.kind for item in snapshot.items if item.group == "staged"}
    assert kinds == {"Foo.txt": "deleted", "foo.txt": "added"}


async def test_filter_recheck_failure_degrades_only_unstaged(repo, monkeypatch):
    # 重核对自身失败（如元数据预算耗尽）时无法续证 unstaged 的 filter 保证：
    # 仅 unstaged 降级，已完整的 staged/untracked 保留（C3）。
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    review = make(repo)
    original = review._filters
    calls = 0

    async def flaky(observation, paths):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise GitIOError("output_limit", "metadata budget exhausted")
        return await original(observation, paths)

    monkeypatch.setattr(review, "_filters", flaky)
    snapshot = await review.capture("review-1", CancelToken())
    unstaged = group_of(snapshot, "unstaged")
    assert unstaged.state == "unknown" and unstaged.reason == "output_limit"
    assert group_of(snapshot, "staged").state == "complete"
    assert group_of(snapshot, "untracked").state == "complete"
    assert all(item.group != "unstaged" for item in snapshot.items)
    assert any(item.group == "untracked" for item in snapshot.items)
