"""A05 崩溃夹具子进程: 真实共享应用 + 剧本假模型, 由父进程按存档状态 kill.

用法:
    python crash_harness.py run    <workspace> <new>   <script.json>
    python crash_harness.py resume <workspace> <uuid4>  <script.json>

握手输出 (flush, 每行一个事实, 供父进程 readline 等待):
    [session id=... mode=... restored=N]
    [view_only=<reason|->]
    [interrupted=N]
    [request1 committed] / [request2 committed]
    [outcome exit=N saved=<bool>]   (仅 resume 且可续写)
    [refused]                        (resume 且 view-only; 退出码 3)

退出码: 正常 0; view-only 拒绝 3; 打开失败 2.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cicada.__main__ import parse_script
from cicada.application import (
    ApplicationConfig,
    PersistenceConfig,
    open_application,
)
from cicada.persistence.contracts import LaunchIdentity
from cicada.plugins.fake_model import FakeModel, fake_model_plugin
from cicada.plugins.session_store.store import SQLiteSessionStore
from cicada.prompting import DEFAULT_TOOLS, compose_system_prompt


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'resume'))
    parser.add_argument('workspace')
    parser.add_argument('session_id')
    parser.add_argument('script_path')
    parser.add_argument('--before-commit-ready', type=Path)
    args = parser.parse_args()
    mode, workspace, session_id, script_path = (
        args.mode, args.workspace, args.session_id, args.script_path,
    )
    if args.before_commit_ready is not None:
        assert mode == 'run', '提交前屏障仅用于 run 子进程'
        original_commit = SQLiteSessionStore.commit_request
        commit_calls = 0

        async def held_commit(store, *commit_args, **commit_kwargs):
            nonlocal commit_calls
            commit_calls += 1
            if commit_calls == 2:
                args.before_commit_ready.write_text('ready', encoding='utf-8')
                await asyncio.Event().wait()
            return await original_commit(store, *commit_args, **commit_kwargs)

        SQLiteSessionStore.commit_request = held_commit
    script = parse_script(Path(script_path).read_text(encoding="utf-8"))
    prompt, info = compose_system_prompt(Path(workspace), None, tools=DEFAULT_TOOLS, checks=())
    identity = LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_request_bytes=None, max_turns=50,
        prompt_version=info.version, prompt_sha256=info.prompt_sha256,
        instructions_sha256=info.instructions_sha256, checks_signature=None, sources={},
    )
    config = ApplicationConfig(
        workspace=Path(workspace),
        system_prompt=prompt,
        prompt_info=info,
        model_plugin=fake_model_plugin(FakeModel(script)),
        max_turns=50,
        model_label="fake",
        persistence=PersistenceConfig(
            session_id=None if session_id == "new" else session_id, identity=identity,
        ),
    )
    try:
        session = await open_application(config)
    except RuntimeError as exc:
        print(f"[open failed: {exc}]", flush=True)
        return 2
    facts = session.persistence
    assert facts is not None
    print(f"[session id={facts.session_id} mode={facts.mode} restored={facts.restored_request_count}]", flush=True)
    view_only = session.view_only_reason or "-"
    print(f"[view_only={view_only}]", flush=True)
    print(f"[interrupted={len(session.restored_interrupted)}]", flush=True)
    if mode == "resume":
        if session.view_only_reason is not None:
            print("[refused]", flush=True)
            await session.aclose()
            return 3
        outcome = await session.execute("恢复后的第三问")
        saved = outcome.persistence.saved if outcome.persistence is not None else False
        print(f"[outcome exit={outcome.exit_code} saved={saved}]", flush=True)
        await session.aclose()
        return 0
    await session.execute("第一问")
    print("[request1 committed]", flush=True)
    await session.execute("第二问")
    print("[request2 committed]", flush=True)
    await session.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
