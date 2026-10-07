"""05/06 阶段验收: 固定 fake fixture 经公开入口 (--tui 新前端) 的完整交互序列.

序列 (plan §3-06): 首问(工具→长流式→展开→检查) → 追问 → 修改(stale) →
重检 → 取消 → 清空 → /status → 退出。全部经 run_tui_session (公开 --tui
生产组合) 驱动, 断言 VT 屏可见事实 / model 请求级事实 / 退出码; 无真实
模型调用。长摘要行 (检查来源/工件 hash) 允许折行, 一律用 flat() 拼平匹配。
真实终端人工核对 (P16: IME/缩放/复制/恢复) 由用户在 Windows Terminal
按清单执行, 自动输入不能代签。
"""

from __future__ import annotations

import asyncio

from cicada.application import open_application
from cicada.ui import run_tui_session
from tui_fixture import (
    CTRL_C,
    CTRL_O,
    ENTER,
    StageTerminal,
    check_plan,
    git_repo,
    model_of,
    stage_config,
    stage_script,
    text_keys,
    wait_until,
)


async def test_stage_acceptance_full_interaction(tmp_path):
    root = git_repo(tmp_path)
    plan = check_plan(root)
    session = await open_application(stage_config(root, stage_script(), plan=plan))
    stage = StageTerminal()
    try:
        task = asyncio.ensure_future(run_tui_session(session, terminal=stage.terminal))

        # 0) 正常启动安静: 无内部 phase/run/hash 诊断行 (P12)
        assert await wait_until(lambda: stage.contains("fake")), "底栏模型标签可见"
        assert "状态:" not in stage.text() and "run=" not in stage.text()
        assert "prompt_sha256" not in stage.text()

        # 1) 首问: 工具卡(powershell/edit/check) + 长流式回答同一条目原位更新 (P02/P04)
        stage.emit(text_keys("改好 sample.py 并运行检查"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("edit  sample.py")), "edit 工具卡可见"  # §B: 状态符号标题, 名称与参数两个空格
        assert await wait_until(lambda: stage.contains("检查报告")), "长回答开始流式"
        assert stage.count("检查报告") == 1, "流式期间同一条目原位更新, 无重复"
        assert await wait_until(lambda: len(model_of(session).requests) == 2), "首问 = 工具轮 + 文本轮"
        assert await wait_until(lambda: session.state == "idle")
        assert await wait_until(lambda: stage.flat().count("结论: 第一轮修改完成并通过检查。") == 1), (
            "完成校准无二次打印"
        )
        assert stage.count("这是追问的回答。") == 0
        # 检查结论: 程序判定可交付, 来源=本次检查 (P14)
        assert await wait_until(lambda: "程序判定（非模型自然语言）: 可交付" in stage.flat())
        assert "来源=本次检查" in stage.flat()
        assert f"verification_run_id={plan.verification_run_id}" in stage.flat()

        # 2) 工具输出默认 10 行折叠, Ctrl+O 展开/收起只改展示 (P05)
        assert stage.contains("powershell-line-10") and stage.contains("powershell-line-01")
        assert not stage.contains("powershell-line-14"), "折叠态不显示第 14 行"
        assert stage.contains("(Ctrl+O 展开)")
        before = len(model_of(session).requests)
        stage.emit([CTRL_O])
        assert await wait_until(lambda: stage.contains("powershell-line-14"))
        assert not stage.contains("(Ctrl+O 展开)"), "展开后无折叠提示"
        assert len(model_of(session).requests) == before, "展开不触发请求"
        stage.emit([CTRL_O])
        assert await wait_until(lambda: not stage.contains("powershell-line-14"))
        assert stage.contains("(Ctrl+O 展开)")

        # 3) 追问: 完整历史携带一次; 历史 PASS 标注"此前检查，仍有效" (P14)
        history_after_first = len(session.history)
        assert history_after_first > 0
        stage.emit(text_keys("追问: 刚才做了什么?"))
        stage.emit([ENTER])
        assert await wait_until(lambda: len(model_of(session).requests) == 3)
        assert await wait_until(lambda: stage.flat().count("这是追问的回答。") == 1)
        second_request_messages = model_of(session).requests[-1].messages
        assert len(second_request_messages) == history_after_first + 1, "追问只携带一次历史"
        assert await wait_until(lambda: session.state == "idle")
        assert await wait_until(lambda: stage.flat().count("此前检查，仍有效") == 1), (
            "历史 current PASS 的来源标注"
        )
        assert stage.flat().count("来源=本次检查") == 1, "第二轮无 check 工具, 不标本次"

        # 4) 修改但不重检: stale, 程序判定不可交付 (P14)
        stage.emit(text_keys("再改一处但不重检"))
        stage.emit([ENTER])
        assert await wait_until(lambda: len(model_of(session).requests) == 5)
        assert await wait_until(lambda: "不可交付" in stage.flat())
        assert "freshness=stale" in stage.flat()
        assert stage.flat().count("此前检查，仍有效") == 1, "stale 历史 PASS 不改标仍有效"

        # 5) 重检: 再次通过, 恢复可交付 (P14 生命周期可重复)
        stage.emit(text_keys("重新运行检查"))
        stage.emit([ENTER])
        assert await wait_until(lambda: len(model_of(session).requests) == 7)
        assert await wait_until(lambda: session.state == "idle")
        assert await wait_until(lambda: stage.flat().count("来源=本次检查") == 2), "重检后新来源标注"
        assert session.has_process_uncertain() is False

        # 6) 取消: 忙碌 Ctrl+C 协作取消, 取消条目如实 (P07/P14)
        stage.emit(text_keys("慢慢回答这个问题"))
        stage.emit([ENTER])
        assert await wait_until(lambda: session.state == "executing")
        stage.emit([CTRL_C])
        assert await wait_until(lambda: stage.contains("已取消"))
        assert await wait_until(lambda: session.state == "idle")
        # 取消可能落在模型流起点之前 (kernel 级): 计数只作下界, 事实由"已取消"与退出码承载
        assert len(model_of(session).requests) >= 7

        # 7) 清空: 模型历史清空, 验证状态保留, 显示文档保留 (P14)
        history_before_clear = len(session.history)
        assert history_before_clear > 0
        stage.emit(text_keys("/clear"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("已清空模型历史"))
        assert len(session.history) == 0
        assert session.verification_run_id == plan.verification_run_id
        assert stage.contains("第一轮修改完成并通过检查。"), "已显示记录保留"

        # 8) /status: 空闲可用, 有界事实可追溯 (P12/P14)
        stage.emit(text_keys("/status"))
        stage.emit([ENTER])
        assert await wait_until(lambda: stage.contains("状态: idle"))
        assert "检查: check-1=passed/current" in stage.flat()
        assert stage.contains("计量:")

        # 9) 退出: 关闭顺序收尾, 进程退出码 = 最后一条请求退出码 (aborted = 1) (P15)
        stage.emit(text_keys("/exit"))
        stage.emit([ENTER])
        code = await asyncio.wait_for(task, timeout=20)
        assert code == 1
        assert session.state == "closed"
    finally:
        if session.state != "closed":
            await session.aclose()
