"""file_completion 中立契约的纯逻辑检查: 查询规范、评分、排序与记录不变量。"""

import pytest

from cicada.file_completion import (
    MAX_CANDIDATES,
    MAX_QUERY_BYTES,
    FileCompletionCandidate,
    FileCompletionResult,
    UnsupportedPathError,
    is_separator,
    normalize_query,
    rank_candidates,
    score_candidate,
)


def test_normalize_query_keeps_relative_paths_and_folds_backslash():
    assert normalize_query("") == ""
    assert normalize_query("src/主 文件.py") == "src/主 文件.py"
    assert normalize_query("src\\util\\x.py") == "src/util/x.py"
    assert normalize_query("a~b.txt") == "a~b.txt"  # 文件名中部的 ~ 合法


@pytest.mark.parametrize("query", [
    "C:", "C:x", "C:/x", "\\\\server/share", "//server/share",
    "~", "~/x", "../x", "a/../b", "a\x01b", "a\x7fb", "x" * (MAX_QUERY_BYTES + 1),
    "/outside", "a/x:stream", 'a/"x', "a\x80b",
])
def test_normalize_query_rejects_unsafe_forms(query):
    with pytest.raises(UnsupportedPathError):
        normalize_query(query)


def test_normalize_query_rejects_unrepresentable_text():
    with pytest.raises(UnsupportedPathError):
        normalize_query("\udc80")  # 孤立代理对无法编码 UTF-8


def test_score_candidate_filenames_with_casefold_and_directory_bonus():
    assert score_candidate("src/Readme.md", False, "readme.md") == 100
    assert score_candidate("src/readme.md", False, "rea") == 80
    assert score_candidate("src/an-extra.md", False, "xtr") == 50
    assert score_candidate("src/deep/other.md", False, "deep") == 30
    assert score_candidate("src/", True, "SRC") == 110
    assert score_candidate("src/other.md", False, "zzz") == 0
    assert score_candidate("src/", True, "zzz") == 0  # 无匹配不因加分入选
    assert score_candidate("src/a.md", False, "") == 0  # 空 fragment 基础分 0


def test_rank_candidates_orders_by_score_depth_length_then_paths():
    pool = [
        ("a/y-dir/readme.md", False),  # 文件名无 y, 完整路径包含 30
        ("yll.md", False),             # 文件名前缀 80
        ("ay.md", False),              # 文件名包含 50
        ("y.md", False),               # 精确 100
        ("y/", True),                  # 精确 100 + 目录 10 = 110
        ("b/y.md", False),             # 精确 100, 深度 2 排在同分 depth 1 之后
    ]
    candidates, truncated = rank_candidates(pool, "y")
    assert [c.relative_path for c in candidates] == [
        "y/", "y.md", "yll.md", "b/y.md", "ay.md", "a/y-dir/readme.md",
    ]
    assert truncated is False


def test_rank_candidates_empty_fragment_sorts_everything_by_depth_then_path():
    pool = [("b/b.md", False), ("a/x.md", False), ("a/", True), ("b/", True)]
    candidates, truncated = rank_candidates(pool, "")
    assert [c.relative_path for c in candidates] == ["a/", "b/", "a/x.md", "b/b.md"]
    assert truncated is False


def test_rank_candidates_caps_at_twenty_and_marks_truncation():
    pool = [(f"f{i:03}.txt", False) for i in range(MAX_CANDIDATES + 5)]
    candidates, truncated = rank_candidates(pool, "")
    assert len(candidates) == MAX_CANDIDATES
    assert candidates[0].relative_path == "f000.txt"
    assert truncated is True
    candidates, truncated = rank_candidates(pool[:MAX_CANDIDATES], "")
    assert len(candidates) == MAX_CANDIDATES and truncated is False


def test_candidate_record_shape_is_enforced():
    FileCompletionCandidate("src/main.py", False)
    FileCompletionCandidate("src/", True)
    with pytest.raises(ValueError):
        FileCompletionCandidate("src\\main.py", False)
    with pytest.raises(ValueError):
        FileCompletionCandidate("src/main.py", True)  # 文件不得带尾部斜杠
    with pytest.raises(ValueError):
        FileCompletionCandidate("src", True)  # 目录必须带尾部斜杠
    with pytest.raises(ValueError):
        FileCompletionCandidate("/abs", False)


def test_result_record_scope_and_reason_invariants():
    result = FileCompletionResult((), "git")
    assert result.truncated is False and result.reason is None
    FileCompletionResult((), "unavailable", reason="not_git")
    with pytest.raises(ValueError):
        FileCompletionResult((), "bogus")
    with pytest.raises(ValueError):
        FileCompletionResult((FileCompletionCandidate("a", False),), "unavailable")
    with pytest.raises(ValueError):
        FileCompletionResult((), "git", reason="not_git")  # reason 只属 unavailable
    with pytest.raises(ValueError):
        FileCompletionResult(
            (FileCompletionCandidate("a", False),) * (MAX_CANDIDATES + 1), "git")


def test_separator_set_covers_pi_quotes_equals_and_cjk_punctuation():
    for ch in (" ", "\t", '"', "'", "=", "，", "。", "”", "《", "…", "—"):
        assert is_separator(ch)
    for ch in ("a", "@", "/", "_", "中"):
        assert not is_separator(ch)
