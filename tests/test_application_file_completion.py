"""complete_files 应用只读出口: 真实默认装配、关闭语义与原文请求 (C3/C8)。

真实内核 + 真实插件组装 + 项目内隔离 Git 工作区; 模型用确定性 fake, 零 live。
"""

import asyncio
from pathlib import Path

import pytest

from cicada.application import ApplicationSession, open_application
from cicada.boot import bootstrap
from cicada.core.cancel import CancelToken
from cicada.plugins.coding.inventory import GitInventory
from cicada.plugins.coding.workspace import Workspace
from cicada.plugins.fake_model import FakeModel, fake_model_plugin

from tests.support.app import prompt_info as _prompt_info, repo as _repo


def _config(root: Path) -> object:
    from cicada.application import ApplicationConfig

    return ApplicationConfig(
        workspace=root,
        system_prompt="test system prompt",
        prompt_info=_prompt_info(root),
        model_plugin=fake_model_plugin(FakeModel([])),
        max_turns=6,
        model_label="fake",
    )


async def test_default_assembly_serves_git_candidates_via_readonly_exit(tmp_path):
    root = _repo(tmp_path)
    (root / "src").mkdir()
    (root / "src/main.py").write_text("x", encoding="utf-8")
    session = await open_application(_config(root))
    try:
        result = await session.complete_files("", CancelToken())
        assert result.scope == "git"
        assert "sample.py" in {c.relative_path for c in result.candidates}
        result = await session.complete_files("src/", CancelToken())
        assert "src/main.py" in {c.relative_path for c in result.candidates}
        # 进行中的编码不阻止只读候选; 手写请求原文不按候选规则改写
        outcome = await session.execute("@src/main.py 说明这个文件")
        assert outcome is not None
        assert session.history[-2].text == "@src/main.py 说明这个文件"
    finally:
        await session.aclose()


async def test_complete_files_after_close_reports_session_closed(tmp_path):
    session = await open_application(_config(_repo(tmp_path)))
    await session.aclose()
    result = await session.complete_files("", CancelToken())
    assert (result.scope, result.reason) == ("unavailable", "session_closed")


async def test_session_without_capability_reports_capability_unavailable(tmp_path):
    # 旧直接构造未装配 coding.file_completion: 出口返回短原因, 不抛错
    app = await bootstrap(
        [fake_model_plugin(FakeModel([]))], tool_capabilities=(), system_prompt="t", max_turns=6,
    )
    session = ApplicationSession(app=app, config=_config(_repo(tmp_path)), service=None, policy=None)
    try:
        result = await session.complete_files("", CancelToken())
        assert (result.scope, result.reason) == ("unavailable", "capability_unavailable")
    finally:
        await session.aclose()


async def test_completion_internal_inventory_waits_for_real_process_reclaim(tmp_path):
    """取消在途查询后立即再查询仍成功: 资源实际回收, 旧查询不占用所有权。"""
    root = _repo(tmp_path)
    for i in range(200):
        (root / f"filler-{i:03}.py").write_text("x", encoding="utf-8")
    from tests.support.app import git as run_git

    run_git(root, "add", ".")
    session = await open_application(_config(root))
    try:
        token = CancelToken()
        task = asyncio.create_task(session.complete_files("", token))
        await asyncio.sleep(0.005)
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        result = await session.complete_files("sample", CancelToken())
        assert [c.relative_path for c in result.candidates] == ["sample.py"]
        assert GitInventory(Workspace.create(root)).cleanup_timeout_s == 1.0
    finally:
        await session.aclose()
