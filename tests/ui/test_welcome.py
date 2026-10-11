"""tui-startup-welcome 检查: 文字欢迎区贯通 (票据 01) 与 r2 Pi 风格排版 (票据 04).

覆盖规格 A01–A07 与 A10: 新建/恢复/空存档/view-only 启动快照、缺失值降级、
思考决策表、宽度降级与行宽有界、控制字符清理、模型切换后启动快照保留、
clear/compact/initial_prompt 不重复条目; 欢迎不发模型请求、不写历史/持久存档、
不调用 login_view。r2: Pi 风格段组行序与样式 (标题加粗/低亮行/段标题默认前景)、
>=48 各宽度同一无图标布局、资产模块退场。真实 ApplicationSession + fake 模型 /
mock HTTP (DeepSeek), 零真实模型调用。
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cicada.application import PersistenceConfig, open_application
from cicada.persistence.contracts import LaunchIdentity
from cicada.plugins.fake_model import FakeModel
from cicada.ui.main_screen import strip_terminal_sequences, visible_width
from cicada.ui.theme import format_workspace
from cicada.ui.welcome import WelcomeEntry, WelcomeFacts, capture_welcome_facts

from test_conversation_persistence import _persistence_config
from test_conversation_view import (
    Harness,
    _config,
    _prompt_info,
    _repo,
    _script,
    _wait_until,
)
from test_deepseek_application import config as ds_config
from test_deepseek_application import response as ds_response
from test_skill_display import make_config as skills_config
from test_skill_display import make_snapshot, write_skill


def _welcome_entries(h: Harness) -> list[WelcomeEntry]:
    return [e for e in h.entries() if isinstance(e, WelcomeEntry)]


def _text(h: Harness, width: int = 80) -> str:
    return strip_terminal_sequences("\n".join(h.lines(width)))


def _facts(**overrides) -> WelcomeFacts:
    base = dict(
        version="0.1.0",
        workspace="~/codex_space/cicada",
        model_label="deepseek-flash",
        thinking_label="high",
        instructions_name="AGENTS.md",
        skills_count=3,
        resumed=False,
    )
    base.update(overrides)
    return WelcomeFacts(**base)


# --- A01: 新会话首条且唯一欢迎; 零模型请求、零历史增量 --------------------------------------


async def test_new_session_welcome_first_single_no_request_no_history(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True})))
    h = Harness(session)
    try:
        welcomes = _welcome_entries(h)
        assert len(welcomes) == 1
        assert isinstance(h.entries()[0], WelcomeEntry)
        assert session.history == ()
        assert h.model().requests == [], "欢迎不发模型请求"
        lines = h.lines(80)
        # Transcript 去掉文档起始空行: 首行即标题 (无全部空白前导)
        assert strip_terminal_sequences(lines[0]).startswith("Cicada ")
        text = _text(h)
        assert "新会话 · 启动快照" in text
        # r2: Pi 风格段组 (段标题 + 缩进内容行)
        assert "[工作区]" in text
        # 私有 basetemp 路径可能超过 80 列；完整路径事实在足够宽的视图核对。
        workspace = format_workspace(str(root))
        wide_text = _text(h, width=max(80, visible_width(workspace) + 2))
        assert f"  {workspace}" in wide_text
        assert "[启动模型]" in text and "  fake · 思考 不可得" in text  # 无持久身份: 思考不可得
        assert "[指令]" in text and "  未指定" in text
        assert "[Skills]" in text and "  未启用" in text
        assert "/help 全部命令 · /model 模型 · /login 配置 · /status 状态" in text
        assert "Enter 发送 · Shift+Enter 换行" in text
    finally:
        h.close()
        await session.aclose()


async def test_welcome_stays_single_and_first_after_requests(tmp_path):
    """initial_prompt 形态 (启动后经同一 on_submit 准入): 欢迎不重复且仍在用户条目前。"""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True})))
    h = Harness(session)
    try:
        assert h.submit("第一问")
        await h.wait_request_done()
        assert len(_welcome_entries(h)) == 1
        entries = h.entries()
        assert isinstance(entries[0], WelcomeEntry)
        assert type(entries[1]).__name__ == "UserEntry"
        assert "新会话 · 启动快照" in _text(h)
    finally:
        h.close()
        await session.aclose()


# --- A02: 指令与 Skills 事实 ------------------------------------------------------------


async def test_instructions_filename_shown(tmp_path):
    root = _repo(tmp_path)
    config = dc_replace(
        _config(root, _script()),
        prompt_info=dc_replace(_prompt_info(root), instructions_path=root / "AGENTS.md"),
    )
    session = await open_application(config)
    h = Harness(session)
    try:
        text = _text(h)
        assert "[指令]" in text and "  AGENTS.md" in text
    finally:
        h.close()
        await session.aclose()


async def test_skills_count_and_skill_entry_hint(tmp_path):
    root = _repo(tmp_path)
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "alpha")
    write_skill(skills_dir, "beta")
    snapshot = make_snapshot(root, skills_dir)
    session = await open_application(skills_config(root, FakeModel(_script()), snapshot))
    h = Harness(session)
    try:
        text = _text(h)
        assert "  2 个 · /skill 查看目录" in text
        assert "[指令]" in text and "  未指定" in text
    finally:
        h.close()
        await session.aclose()


async def test_empty_skills_dir_shows_zero_without_skill_command(tmp_path):
    root = _repo(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    snapshot = make_snapshot(root, skills_dir)
    session = await open_application(skills_config(root, FakeModel(_script()), snapshot))
    h = Harness(session)
    try:
        text = _text(h)
        assert "  0 个" in text
        assert "/skill 查看目录" not in text
    finally:
        h.close()
        await session.aclose()


# --- A03: 思考决策表 / 版本缺失 / 不调用 login_view -----------------------------------------


def _identity_config(root: Path, script: list, *, provider: str, think, options: dict | None):
    info = _prompt_info(root)
    identity = LaunchIdentity(
        provider=provider, model="fake", base_url=None, think=think, options=options or {},
        max_turns=6, prompt_version=info.version, prompt_sha256=info.prompt_sha256,
        instructions_sha256=None, checks_signature=None, sources={},
    )
    return dc_replace(
        _config(root, script),
        persistence=PersistenceConfig(session_id=None, identity=identity),
    )


@pytest.mark.parametrize(
    ("provider", "think", "options", "expected"),
    [
        ("fake", None, None, "不可得"),
        ("fake", False, None, "off"),
        ("deepseek", True, {"reasoning_effort": "low"}, "low"),
        ("deepseek", True, {"reasoning_effort": "high"}, "high"),
        ("deepseek", True, {"reasoning_effort": "max"}, "max"),
        ("deepseek", True, {}, "不可得"),
        ("deepseek", True, {"reasoning_effort": "medium"}, "不可得"),
        ("fake", True, None, "开启"),
    ],
)
async def test_thinking_decision_table(tmp_path, provider, think, options, expected):
    root = _repo(tmp_path)
    session = await open_application(
        _identity_config(root, _script(), provider=provider, think=think, options=options)
    )
    h = Harness(session)
    try:
        assert f"  fake · 思考 {expected}" in _text(h)
    finally:
        h.close()
        await session.aclose()


async def test_missing_version_shows_unavailable(tmp_path, monkeypatch):
    def _missing(name: str):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script()))
    h = Harness(session)
    try:
        assert "Cicada 版本不可得 · 新会话 · 启动快照" in _text(h)
    finally:
        h.close()
        await session.aclose()


async def test_welcome_capture_never_calls_login_view(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script()))
    monkeypatch.setattr(
        session,
        "login_view",
        lambda: (_ for _ in ()).throw(AssertionError("欢迎事实提取调用了 login_view")),
    )
    h = Harness(session)  # 构造即捕获; 触发 login_view 立即失败
    try:
        assert "新会话 · 启动快照" in _text(h)
    finally:
        h.close()
        await session.aclose()


# --- A04: 模型发布后欢迎保留启动值, 底栏与确认条目反映新值 -----------------------------------


async def _wait_model_settled(h: Harness) -> bool:
    return await _wait_until(lambda: not h.view._gate.model_selecting)


async def test_model_change_keeps_startup_snapshot(tmp_path):
    root = _repo(tmp_path)
    bodies: list = []

    async def handle(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, content=next(answers))

    answers = iter([ds_response("思考", "答")])
    session = await open_application(ds_config(root, httpx.MockTransport(handle)))
    h = Harness(session)
    try:
        assert "  deepseek-flash · 思考 high" in _text(h)
        assert h.submit("/model")
        h.view.confirm_model_selection("deepseek-v4-pro")
        assert await _wait_model_settled(h)
        rendered = _text(h)
        assert "  deepseek-flash · 思考 high" in rendered  # 启动快照保留启动值
        assert "已切换模型：deepseek-flash → deepseek-v4-pro" in rendered  # 确认条目反映新值
        footer = strip_terminal_sequences("\n".join(h.view.footer.render(80, 20)))
        assert "deepseek-v4-pro" in footer  # 底栏为有效配置
        assert len(_welcome_entries(h)) == 1
        # A10: 欢迎文字不进入 mock HTTP 请求体
        assert h.submit("问一句")
        await h.wait_request_done()
        assert bodies, "模型请求真实发生"
        payload = json.dumps(bodies, ensure_ascii=False)
        assert "启动快照" not in payload and "Shift+Enter" not in payload
    finally:
        h.close()
        await session.aclose()


# --- A05: 恢复 (非空 / 空存档) 走短欢迎; 旧显示保留 -------------------------------------------


async def test_resume_shows_short_welcome_before_restored_history(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "第一轮回答", "stop": True}), session_id=None)
    )
    first = Harness(session)
    assert first.submit("第一问")
    await first.wait_request_done()
    session_id = session.persistence.session_id
    first.close()
    await session.aclose()

    resumed_session = await open_application(
        _persistence_config(root, _script(), session_id=session_id)
    )
    resumed = Harness(resumed_session)
    try:
        assert isinstance(resumed.entries()[0], WelcomeEntry)
        assert len(_welcome_entries(resumed)) == 1
        rendered = _text(resumed)
        assert "恢复会话 · 启动快照" in rendered
        assert "/help 帮助 · /status 状态" in rendered
        assert "Enter 发送" not in rendered, "恢复短欢迎无键位行"
        # 恢复投影保留且在欢迎之后
        assert rendered.index("恢复会话") < rendered.index("已恢复历史：1 条请求")
        assert "第一问" in rendered and "第一轮回答" in rendered
    finally:
        resumed.close()
        await resumed_session.aclose()


async def test_resume_empty_archive_is_resume_welcome_not_new(tmp_path):
    """零历史存档不按空集合误判: mode=resume 仍是恢复短欢迎 (规格 D4)。"""
    root = _repo(tmp_path)
    session = await open_application(_persistence_config(root, _script(), session_id=None))
    session_id = session.persistence.session_id
    await session.aclose()

    resumed_session = await open_application(
        _persistence_config(root, _script(), session_id=session_id)
    )
    resumed = Harness(resumed_session)
    try:
        assert resumed_session.restored_requests == ()
        assert resumed_session.restored_interrupted == ()
        rendered = _text(resumed)
        assert "恢复会话 · 启动快照" in rendered
        assert "新会话" not in rendered
        assert "已恢复历史" not in rendered
    finally:
        resumed.close()
        await resumed_session.aclose()


# --- A06: 零恢复记录的查看状态原因可见, 新请求仍被拒绝 -----------------------------------------


async def test_view_only_reason_visible_with_zero_restored_records(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_persistence_config(root, _script(), session_id=None))
    session_id = session.persistence.session_id
    await session.aclose()

    resumed_session = await open_application(
        _persistence_config(root, _script(), session_id=session_id)
    )
    # 模拟零恢复记录的查看状态 (真实来源是恢复准入发现的未证终结证据; 规格 D4
    # 只要求该说明不依附"有恢复记录"条件, 不改应用准入)
    resumed_session._view_only_reason = (
        "restored archive records process_uncertain (simulated zero-record)"
    )
    resumed = Harness(resumed_session)
    try:
        rendered = _text(resumed)
        assert "恢复为查看状态（新请求被拒绝）" in rendered
        assert resumed.submit("新请求")
        assert await _wait_until(lambda: "view-only" in _text(resumed)), "准入拒绝如实成条"
    finally:
        resumed.close()
        await resumed_session.aclose()


# --- A07: clear/compact 不重复欢迎条目 --------------------------------------------------------


async def test_persistent_clear_and_compact_never_add_welcome(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "答", "stop": True}), session_id=None)
    )
    h = Harness(session)
    try:
        assert h.submit("第一问")
        await h.wait_request_done()
        assert len(_welcome_entries(h)) == 1
        assert h.submit("/clear")
        assert await _wait_until(lambda: "已清空模型历史" in _text(h))
        assert await _wait_until(lambda: not h.view._gate.clearing)
        assert len(_welcome_entries(h)) == 1
        assert h.submit("/compact")  # fake 无预算: 只出提示条目
        assert await _wait_until(lambda: "压缩需要启用预算" in _text(h))
        assert len(_welcome_entries(h)) == 1
    finally:
        h.close()
        await session.aclose()


async def test_nonpersistent_clear_keeps_single_welcome(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script({"text": "答", "stop": True})))
    h = Harness(session)
    try:
        assert h.submit("/clear")
        assert "已清空模型历史" in _text(h)
        assert len(_welcome_entries(h)) == 1
    finally:
        h.close()
        await session.aclose()


# --- D3: 纯文字宽度降级 (条目级排版; 成品行宽有界) ----------------------------------------------


@pytest.mark.parametrize("width", [48, 120])
def test_full_block_r2_layout_within_width(width):
    """r2: 新会话完整块 (Pi 风格段组) 在 >=48 宽度同一 13 行布局, 行宽有界, 无标志字符。

    (v1 五宽度版与 r2_same_layout/a08 重档合并: 档位边界由
    test_a08_boundary_widths_bounded 双向覆盖, 无图标断言并入本条。)
    """
    lines = WelcomeEntry(_facts(), plain=True).render(width)
    assert len(lines) == 13  # 前导空行 + 标题/键位/命令/空行 + 四段组各两行
    assert lines[0] == ""
    assert "启动快照" in lines[1]
    visible = [strip_terminal_sequences(line) for line in lines]
    assert not any(mark in line for line in visible for mark in "█▄▀")
    assert all(visible_width(line) <= width for line in lines)


def test_narrow_block_four_lines_within_width():
    lines = WelcomeEntry(_facts(), plain=True).render(47)
    assert len(lines) == 5  # 前导空行 + 标题/启动模型/资源简述//help
    assert "启动模型: deepseek-flash" in lines[2]
    assert strip_terminal_sequences(lines[4]).startswith("/help")
    assert all(visible_width(line) <= 47 for line in lines)


def test_narrow_resource_brief_fixed_order_at_47():
    lines = WelcomeEntry(_facts(), plain=True).render(47)
    assert "指令: AGENTS.md · Skills: 3 个" in lines[3]  # 资源简述固定顺序
    # 未启用 Skills 的简述
    lines = WelcomeEntry(_facts(skills_count=None), plain=True).render(47)
    assert "指令: AGENTS.md · Skills: 未启用" in lines[3]


def test_minimal_block_two_lines_within_width():
    width = 23
    lines = WelcomeEntry(_facts(), plain=True).render(width)
    assert len(lines) == 3  # 前导空行 + Cicada + /help
    assert all(visible_width(line) <= width for line in lines)


def test_nonpositive_width_returns_no_lines():
    assert WelcomeEntry(_facts(), plain=True).render(0) == []


@pytest.mark.parametrize("plain", [True, False])
def test_long_cjk_emoji_fields_stay_bounded(plain):
    """超长 CJK/emoji 字段在窄档 (四行块/最小块) 行宽有界; 宽档 13 行截断
    路径由 test_r2_long_cjk_emoji_fields_bounded 覆盖。"""
    facts = _facts(
        workspace="~/工作区/" + "蝉" * 60 + "🪲" * 20,
        model_label="模型🪲" + "x" * 200,
    )
    for width in (47, 23):
        lines = WelcomeEntry(facts, plain=plain).render(width)
        assert all(visible_width(line) <= width for line in lines)


async def test_document_lines_strip_leading_blank_and_bound_width(tmp_path):
    """组合层出口: document_lines 首行即标题; 各档宽度行宽有界。"""
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script()))
    h = Harness(session)
    try:
        for width in (80, 47, 24, 23, 1):
            lines = h.lines(width)
            assert lines, width
            assert all(visible_width(line) <= width for line in lines)
        assert strip_terminal_sequences(h.lines(80)[0]).startswith("Cicada ")
    finally:
        h.close()
        await session.aclose()


# --- D5: plain 零 ESC; 有色行末复位 ------------------------------------------------------------


def test_plain_entry_outputs_no_escape_at_any_width():
    entry = WelcomeEntry(_facts(), plain=True)
    for width in (120, 80, 47, 24, 23, 1):
        for line in entry.render(width):
            assert "\x1b" not in line


def test_color_entry_resets_every_styled_line():
    """r2: 标题加粗, 键位/命令/段内容行 MUTED_GRAY 低亮, 段标题默认前景; 行末复位。"""
    lines = WelcomeEntry(_facts(), plain=False).render(80)
    assert lines[1].startswith("\x1b[1m") and lines[1].endswith("\x1b[0m")  # 标题加粗
    styled = [line for line in lines if "\x1b" in line]
    assert len(styled) == 7  # 标题 + 键位行 + 命令行 + 四个段内容行
    assert all(line.endswith("\x1b[0m") for line in styled)
    dim = "\x1b[38;2;128;128;128m"  # theme.MUTED_GRAY
    assert all(line.startswith(dim) for line in styled[1:]), "标题外着色行均为低亮"
    for heading in ("[工作区]", "[启动模型]", "[指令]", "[Skills]"):
        row = next(line for line in lines if heading in line)
        assert "\x1b" not in row, "段标题默认前景零样式"


async def test_capture_plain_policy_from_environment(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    session = await open_application(_config(root, _script()))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("TERM", raising=False)
    h = Harness(session)
    try:
        for line in _welcome_entries(h)[0].render(80):
            assert "\x1b" not in line
    finally:
        h.close()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    h = Harness(session)
    try:
        for line in _welcome_entries(h)[0].render(80):
            assert "\x1b" not in line
    finally:
        h.close()
        await session.aclose()


# --- A10: 显示清理; 欢迎文字不进入持久存档 -------------------------------------------------------


async def test_control_characters_in_model_label_are_cleaned(tmp_path):
    root = _repo(tmp_path)
    label = "fake\x1b[31m红\n绿\ty"
    session = await open_application(dc_replace(_config(root, _script()), model_label=label))
    h = Harness(session)
    try:
        raw = "\n".join(h.lines(80))
        assert "\x1b[31m" not in raw, "控制序列不得进入终端"
        # 换行/Tab 折成空格: 单行内展示 (r2: 启动模型段内容行)
        assert "  fake红 绿 y · 思考 不可得" in _text(h)
    finally:
        h.close()
        await session.aclose()


async def test_welcome_text_not_in_model_requests_or_archive(tmp_path):
    root = _repo(tmp_path)
    session = await open_application(
        _persistence_config(root, _script({"text": "答", "stop": True}), session_id=None)
    )
    h = Harness(session)
    try:
        assert h.submit("持久问题")
        await h.wait_request_done()
        requests = h.model().requests
        assert requests, "模型请求真实发生"
        visible = json.dumps(
            [getattr(m, "text", "") for req in requests for m in req.messages],
            ensure_ascii=False,
        ) + json.dumps([req.system_prompt for req in requests], ensure_ascii=False)
        assert "启动快照" not in visible and "Shift+Enter" not in visible
    finally:
        h.close()
        await session.aclose()
    archive = root / ".cicada" / "sessions"
    blobs = [p.read_bytes() for p in archive.rglob("*") if p.is_file()]
    assert blobs, "存档真实写入"
    for marker in ("启动快照", "Shift+Enter", "[工作区]", "/help 全部命令"):
        assert all(marker.encode("utf-8") not in blob for blob in blobs)


# --- C2: 无持久化时的恢复兜底判定 (单元级, 不经过存储) -------------------------------------------


def test_resume_fallback_without_persistence_uses_restored_facts():
    config = SimpleNamespace(
        workspace=Path("/tmp/ws"),
        model_label="fake",
        persistence=None,
        prompt_info=SimpleNamespace(instructions_path=None),
    )
    stub = SimpleNamespace(
        config=config,
        skills=None,
        persistence=None,
        restored_requests=(object(),),
        restored_interrupted=(),
    )
    assert capture_welcome_facts(stub).resumed is True
    stub.restored_requests = ()
    assert capture_welcome_facts(stub).resumed is False


# --- 票据 04 (r2): Pi 风格完整块行序、段组样式与资产退场 -------------------------------------


def test_r2_full_block_line_order_and_indent():
    """r2 行序: 标题/键位/命令/空行 + [工作区]/[启动模型]/[指令]/[Skills] 段组 (内容缩进 2 列)。"""
    lines = WelcomeEntry(_facts(), plain=True).render(96)
    visible = [strip_terminal_sequences(line) for line in lines]
    assert visible[1].startswith("Cicada 0.1.0 · 新会话 · 启动快照")
    assert visible[2].startswith("Enter 发送 · Shift+Enter 换行")
    assert visible[3] == "/help 全部命令 · /model 模型 · /login 配置 · /status 状态"
    assert visible[4] == ""
    assert visible[5] == "[工作区]" and visible[6] == "  ~/codex_space/cicada"
    assert visible[7] == "[启动模型]" and visible[8] == "  deepseek-flash · 思考 high"
    assert visible[9] == "[指令]" and visible[10] == "  AGENTS.md"
    assert visible[11] == "[Skills]" and visible[12] == "  3 个 · /skill 查看目录"


@pytest.mark.parametrize(
    ("skills_count", "expected"),
    [(None, "  未启用"), (0, "  0 个"), (5, "  5 个 · /skill 查看目录")],
)
def test_r2_skills_section_three_states(skills_count, expected):
    """Skills 段只显示三态计数, 不列名称 (用户故事 7)。"""
    lines = WelcomeEntry(_facts(skills_count=skills_count), plain=True).render(96)
    assert lines[12] == expected


@pytest.mark.parametrize("plain", [True, False])
def test_r2_resume_short_block_has_no_sections(plain):
    """恢复保持三行短版: 无段组、无键位行, 宽度档文案同 v1 (含 40/10 窄档)。"""
    entry = WelcomeEntry(_facts(resumed=True), plain=plain)
    lines = entry.render(96)
    visible = [strip_terminal_sequences(line) for line in lines]
    assert len(lines) == 4
    assert "恢复会话 · 启动快照" in visible[1]
    assert visible[2] == "启动模型: deepseek-flash · 思考 high"
    assert visible[3] == "/help 帮助 · /status 状态"
    assert not any(line.startswith("[") for line in visible)
    if plain:
        assert all("\x1b" not in line for line in lines)
    narrow = [strip_terminal_sequences(line) for line in entry.render(40)]
    assert narrow[1:] == ["Cicada · 恢复会话", "/help 帮助"]
    minimal = [strip_terminal_sequences(line) for line in entry.render(10)]
    assert minimal[1:] == ["Cicada", "/help"]


@pytest.mark.parametrize("width", [1, 23, 24, 47, 48, 120])
@pytest.mark.parametrize("plain", [True, False])
def test_a08_boundary_widths_bounded(width, plain):
    """A08 阈值两侧全部有界; 行数按宽度档固定 (r2: >=48 为 13 行完整块)。"""
    lines = WelcomeEntry(_facts(), plain=plain).render(width)
    expected = 13 if width >= 48 else (5 if width >= 24 else 3)
    assert len(lines) == expected
    assert all(visible_width(line) <= width for line in lines)


@pytest.mark.parametrize("width", [48, 120])
@pytest.mark.parametrize("plain", [True, False])
def test_r2_long_cjk_emoji_fields_bounded(width, plain):
    facts = _facts(
        workspace="~/工作区/" + "蝉" * 60 + "🪲" * 20,
        model_label="模型🪲" + "x" * 200,
    )
    lines = WelcomeEntry(facts, plain=plain).render(width)
    assert len(lines) == 13
    assert all(visible_width(line) <= width for line in lines)
    assert "…" in strip_terminal_sequences(lines[6] + lines[8])  # 超长字段截断保留省略标记


def test_r2_no_asset_module_or_image_dependency():
    """r2: 图标资产整体退场——无 welcome_assets 模块, 欢迎模块无设计目录/图片引用。"""
    import importlib.util

    assert importlib.util.find_spec("cicada.ui.welcome_assets") is None
    import cicada.ui.welcome as welcome_module

    source = Path(welcome_module.__file__).read_text(encoding="utf-8")
    for marker in (".scratch", "PIL", "Pillow", "pillow", ".png", ".PNG", "welcome_assets"):
        assert marker not in source, marker
    pyproject = Path(__file__).parents[2].joinpath("pyproject.toml").read_text(encoding="utf-8")
    assert "pillow" not in pyproject.lower()
