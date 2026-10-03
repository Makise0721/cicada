"""06 阶段验收: 固定 fake fixture 经公开入口 (--tui 生产组合) 的完整交互序列.

序列 (plan §3-06): 首问(工具+检查) → 追问 → 修改(检查过期) → 重检 → 取消 →
清空 → 退出。全部经 run_tui_session (公开 --tui 组合) 驱动, 断言各阶段的
展示事实、程序判定、退出码与资源回收; 无真实模型调用。
真实终端人工核对 (T01: 中文 IME/粘贴/resize/恢复) 由用户在 Windows Terminal
按阶段报告清单执行, 自动输入不能代签。
"""

from __future__ import annotations

import asyncio

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from cicada.application import open_application
from cicada.ui import run_tui_session
from tui_fixture import (
    StageSink,
    check_plan,
    done_count,
    git_repo,
    model_of,
    stage_config,
    stage_script,
    wait_until,
)


async def test_stage_acceptance_full_interaction(tmp_path):
    root = git_repo(tmp_path)
    plan = check_plan(root)
    session = await open_application(stage_config(root, stage_script(), plan=plan))
    sink = StageSink()
    try:
        with create_pipe_input() as pipe:
            task = asyncio.ensure_future(
                run_tui_session(session, input=pipe, output=DummyOutput(), sink=sink)
            )

            # 1) 首问: 工具卡 + 检查通过, 程序判定可交付, 来源=本次检查
            pipe.send_text("改好 sample.py 并运行检查\r")
            assert await wait_until(lambda: done_count(sink) >= 1)
            assert any("工具 edit" in text for text in sink.rendered), "工具卡经生产渲染输出"
            assert any("工具 check" in text for text in sink.rendered)
            assert any("来源=本次检查" in text for text in sink.rendered)
            assert any("可交付" in text for text in sink.rendered)
            assert await wait_until(lambda: session.state == "idle")
            assert len(model_of(session).requests) == 2, "首问 = 工具轮 + 文本轮"

            # 2) 追问: 第二请求携带一次且仅一次此前完整历史 (T02), 历史 PASS 标注来源 (T08)
            history_after_first = len(session.history)
            assert history_after_first > 0
            pipe.send_text("追问: 刚才做了什么?\r")
            assert await wait_until(lambda: done_count(sink) >= 2)
            assert len(model_of(session).requests) == 3
            second_request_messages = model_of(session).requests[-1].messages
            # 追问请求 = 完整历史 (首轮后缀) + 本次 UserMessage; 只携带一次
            assert len(second_request_messages) == history_after_first + 1
            assert any("此前检查，仍有效" in text for text in sink.rendered)

            # 3) 修改但不重检: 检查过期, 程序判定不可交付, 退出码 3 (T08/T10)
            pipe.send_text("再改一处但不重检\r")
            assert await wait_until(lambda: done_count(sink) >= 3)
            assert any("不可交付" in text for text in sink.rendered)
            assert any("stale" in text for text in sink.rendered)

            # 4) 重检: 再次通过, 恢复可交付 (T08 生命周期可重复)
            pipe.send_text("重新运行检查\r")
            assert await wait_until(lambda: done_count(sink) >= 4)
            assert any("来源=本次检查" in text for text in sink.rendered)
            assert await wait_until(lambda: session.state == "idle")
            assert session.has_process_uncertain() is False

            # 5) 取消: 忙碌 Ctrl+C 协作取消, 不伪造 idle, aborted 记录 (T11)
            pipe.send_text("慢慢回答这个问题\r")
            assert await wait_until(lambda: session.state == "executing")
            pipe.send_text("\x03")
            assert await wait_until(lambda: done_count(sink) >= 5)
            assert any("aborted" in text for text in sink.rendered)
            assert await wait_until(lambda: session.state == "idle")

            # 6) 清空: 模型历史清空, 验证事实保留 (T09/T08 连续口径)
            history_before_clear = len(session.history)
            assert history_before_clear > 0
            pipe.send_text("/clear\r")
            assert await wait_until(
                lambda: any("历史已清空" in text for text in sink.rendered)
            )
            assert len(session.history) == 0
            assert session.verification_run_id == plan.verification_run_id

            # 7) 退出: 关闭顺序收尾, 进程退出码 = 最后一条请求退出码 (aborted = 1)
            pipe.send_text("/exit\r")
            code = await asyncio.wait_for(task, timeout=20)
            assert code == 1
            assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()
