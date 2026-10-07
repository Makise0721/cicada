"""local-skills-v1 票据 03 TUI 组合检查: /skill 命令、有界目录、Skill 正文显示.

覆盖 A06 (/skill 提交经原准入与实际请求)、A07 (无目录/未知/超限/忙碌/双 Enter
拒绝且草稿保留、零伪条目)、A10 (默认收起、Ctrl+O 展开、窄窗与控制字符、坏封装
回退、恢复解码来自记录)。真实 ApplicationSession + fake 模型, 零真实模型调用。
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from cicada.application import (
    ApplicationConfig,
    PersistenceConfig,
    open_application,
)
from cicada.core.cancel import CancelToken
from cicada.core.ports import StreamDone
from cicada.plugins.fake_model import fake_model_plugin
from cicada.plugins.skills import load_skill_catalog
from cicada.prompting import compose_system_prompt
from cicada.persistence.contracts import LaunchIdentity
from cicada.skills import (
    SKILL_REQUEST_PREFIX,
    SkillDocument,
    SkillInfo,
    build_skill_request,
    decode_skill_request,
)
from cicada.ui.conversation_view import ConversationView
from cicada.ui.main_screen import visible_width
from cicada.ui.transcript import (
    InformationEntry,
    SkillRequestEntry,
    Transcript,
    UserEntry,
)

STOP = [StreamDone("stop")]


# --- 夹具 --------------------------------------------------------------------------------------


def write_skill(root: Path, name: str, *, body: str = "Body text.",
                manual: bool = False) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    flag = "disable-model-invocation: true\n" if manual else ""
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: The {name} skill.\n{flag}---\n\n{body}\n",
        encoding="utf-8",
    )


def make_snapshot(root: Path, skills_dir: Path):
    return load_skill_catalog(str(skills_dir), root)


def make_config(root: Path, model, snapshot) -> ApplicationConfig:
    prompt, prompt_info = compose_system_prompt(
        root, tools=("read", "load_skill"), skills=snapshot
    )
    return ApplicationConfig(
        workspace=root,
        system_prompt=prompt,
        prompt_info=prompt_info,
        model_plugin=fake_model_plugin(model),
        max_turns=6,
        model_label="fake",
        skills=snapshot,
    )


def identity_for(prompt_info) -> LaunchIdentity:
    return LaunchIdentity(
        provider="fake", model="fake", base_url=None, think=None, options={},
        max_turns=6, prompt_version=prompt_info.version,
        prompt_sha256=prompt_info.prompt_sha256,
        instructions_sha256=prompt_info.instructions_sha256,
        checks_signature=None, sources={},
    )


class HeldModel:
    """stream 挂起直到 release(): 制造 executing/reservation 忙碌窗口。"""

    def __init__(self) -> None:
        self.requests: list = []
        self._release = asyncio.Event()

    def release(self) -> None:
        self._release.set()

    async def stream(self, request, cancel: CancelToken):
        self.requests.append(request)
        await self._release.wait()
        if cancel.cancelled:
            yield StreamDone("aborted")
            return
        yield StreamDone("stop")


class Harness:
    """ConversationView 直接驱动; session 由用例 open 后注入。"""

    def __init__(self, session) -> None:
        self.session = session
        self.view = ConversationView(session, on_update=lambda: None)

    def submit(self, text: str) -> bool:
        return self.view.on_submit(text)

    def entries(self) -> list:
        return self.view.transcript.entries()

    def lines(self, width: int = 80) -> list[str]:
        from cicada.ui.main_screen import strip_terminal_sequences

        return [strip_terminal_sequences(line) for line in self.view.document_lines(width)]

    def model(self):
        return self.session._app.agent._model

    def user_entries(self) -> list[UserEntry]:
        return [e for e in self.entries() if isinstance(e, UserEntry)]

    async def wait_idle(self) -> None:
        for _ in range(1000):
            if self.session.state == "idle" and not self.view._gate.busy():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("session 未回到空闲")


# --- /skill 命令: 目录、用法与有界裁剪 ----------------------------------------------------------


async def test_skill_task_suffix_survives_frontend_model_and_archive(tmp_path):
    from dataclasses import replace

    from cicada.plugins.fake_model import FakeModel
    from cicada.plugins.session_store import open_session_store
    from cicada.ui.frontend import MainScreenFrontend
    from cicada.ui.terminal_io import FakeTerminal

    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha")
    model = FakeModel([STOP])
    config = make_config(root, model, make_snapshot(root, skills_dir))
    config = replace(config, persistence=PersistenceConfig(
        session_id=None, identity=identity_for(config.prompt_info)))
    session = await open_application(config)
    h = Harness(session)
    try:
        session_id = session.persistence.session_id
        frontend = MainScreenFrontend(FakeTerminal(60, 20), editor=h.view.editor,
                                      document_lines=h.view.document_lines, on_submit=h.submit)
        h.view.editor.insert_text("/skill alpha task \n第二行\n\n  ")
        frontend._handle_enter()
        await h.wait_idle()
        text = model.requests[0].messages[0].text
        assert decode_skill_request(text).task == "task \n第二行\n\n  "
        assert session.history[0].text == text
    finally:
        h.view.close()
        await session.aclose()
    store = open_session_store(directory=root / ".cicada" / "sessions",
                               session_id=session_id, workspace=root)
    try:
        assert store.recovery_snapshot().canonical_history[0].text == text
    finally:
        await store.aclose()


def test_surrogate_envelope_displays_as_original_user_text():
    import json

    payload = {"version": 1, "task": "task", "skill": {
        "name": "alpha", "source_path": "\ud800", "base_dir": "base",
        "source_sha256": "a" * 64, "body": "body", "body_sha256": "b" * 64,
    }}
    text = SKILL_REQUEST_PREFIX + "\n" + json.dumps(payload, ensure_ascii=True)
    transcript = Transcript()
    transcript.append_user(text)
    assert type(transcript.entries()[0]) is UserEntry
    assert transcript.entries()[0].text == text
    assert transcript.render(40)


async def test_bare_skill_shows_catalog_usage_and_manual_marker(tmp_path):
    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha")
    write_skill(skills_dir, "secret", manual=True)
    session = await open_application(make_config(root, HeldModel(), make_snapshot(root, skills_dir)))
    h = Harness(session)
    try:
        assert h.submit("/skill") is True
        info = [e for e in h.entries() if isinstance(e, InformationEntry)]
        assert len(info) == 1
        text = "\n".join(h.lines())
        assert "/skill alpha" in text
        assert "/skill secret" in text and "手动专用" in text
        assert "用法: /skill <名称> [任务]" in text
        assert "manifest sha256:" in text
        assert h.user_entries() == []
    finally:
        h.view.close()
        await session.aclose()


async def test_bare_skill_catalog_is_bounded_with_omission_note(tmp_path):
    root = tmp_path / "ws"
    skills_dir = root / "skills"
    for index in range(24):
        write_skill(skills_dir, f"skill-{index:02d}")
    session = await open_application(make_config(root, HeldModel(), make_snapshot(root, skills_dir)))
    h = Harness(session)
    try:
        assert h.submit("/skill") is True
        text = "\n".join(h.lines())
        assert "共 24 个 Skill" in text
        assert "已截断: 显示 20/" in text, "超 20 行须有省略计数"
        # 裁剪不解释成不存在: 计数行仍声明总数
        assert h.user_entries() == []
    finally:
        h.view.close()
        await session.aclose()


async def test_skill_without_skills_enabled_keeps_draft(tmp_path):
    root = tmp_path / "ws"
    prompt, prompt_info = compose_system_prompt(root)
    config = ApplicationConfig(
        workspace=root, system_prompt=prompt, prompt_info=prompt_info,
        model_plugin=fake_model_plugin(HeldModel()), max_turns=6, model_label="fake",
    )
    session = await open_application(config)
    h = Harness(session)
    try:
        assert h.submit("/skill") is False  # 拒绝且草稿保留 (A07 无目录)
        assert h.user_entries() == []
        text = "\n".join(h.lines())
        assert "未启用 Skills" in text
        assert h.model().requests == []
    finally:
        h.view.close()
        await session.aclose()


async def test_unknown_name_and_over_limit_keep_draft_without_entries(tmp_path):
    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha")
    session = await open_application(make_config(root, HeldModel(), make_snapshot(root, skills_dir)))
    h = Harness(session)
    try:
        assert h.submit("/skill nope") is False
        assert h.submit("/skill alpha " + "x" * 70_000) is False
        assert h.user_entries() == []
        assert h.model().requests == []
        text = "\n".join(h.lines())
        assert "unknown skill" in text
        assert "byte limit" in text
        # 名称带参数形状但不存在的命令仍是未知命令
        assert h.submit("/skills") is False
    finally:
        h.view.close()
        await session.aclose()


# --- /skill <name> [任务]: 经原准入发送封装 -------------------------------------------------------


async def test_skill_submit_decodes_entry_and_sends_envelope(tmp_path):
    from cicada.__main__ import parse_script

    from cicada.plugins.fake_model import FakeModel

    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha", body="Alpha explicit body.")
    session = await open_application(make_config(
        root, FakeModel(parse_script('[{"text": "完成", "stop": true}]')),
        make_snapshot(root, skills_dir),
    ))
    h = Harness(session)
    try:
        assert h.submit("/skill alpha 修复登录 bug") is True
        users = h.user_entries()
        assert len(users) == 1
        assert isinstance(users[0], SkillRequestEntry)
        assert isinstance(users[0], UserEntry), "显示实现保持 UserEntry 契约"
        assert users[0].text.startswith(SKILL_REQUEST_PREFIX + "\n")
        assert users[0].invocation.task == "修复登录 bug"
        await h.wait_idle()
        request = h.model().requests[0]
        assert request.messages[0].text == users[0].text
        # 模型实际收到与显示条目同一份封装数据 (正文含 write_skill 追加的尾随换行)
        assert decode_skill_request(request.messages[0].text) == users[0].invocation
    finally:
        h.view.close()
        await session.aclose()


async def test_skill_manual_only_and_multiline_task_verbatim(tmp_path):
    from cicada.__main__ import parse_script

    from cicada.plugins.fake_model import FakeModel

    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "manual", body="Manual body.", manual=True)
    session = await open_application(make_config(
        root, FakeModel(parse_script('[{"text": "完成", "stop": true}]')),
        make_snapshot(root, skills_dir),
    ))
    h = Harness(session)
    try:
        draft = "/skill manual\n任务第一行\n第二行 保持原样"
        assert h.submit(draft) is True
        entry = h.user_entries()[0]
        assert isinstance(entry, SkillRequestEntry)
        assert entry.invocation.task == "任务第一行\n第二行 保持原样"
        await h.wait_idle()
    finally:
        h.view.close()
        await session.aclose()


async def test_skill_submit_while_busy_and_double_submit_are_silent(tmp_path):
    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha")
    model = HeldModel()
    session = await open_application(make_config(root, model, make_snapshot(root, skills_dir)))
    h = Harness(session)
    try:
        assert h.submit("先占住执行窗") is True
        await asyncio.sleep(0.05)  # 任务起跑进入 executing
        assert session.state == "executing"
        before = len(h.entries())
        assert h.submit("/skill alpha 任务") is False  # 屏障内静默拒绝
        assert len(h.entries()) == before
        # reservation 窗口 (任务尚未终结) 的同批第二次提交同样被拒
        assert h.submit("/skill alpha 任务") is False
        assert len(model.requests) == 1
        model.release()
        await h.wait_idle()
    finally:
        h.view.close()
        await session.aclose()


async def test_skill_silent_during_clear_and_compact_barriers(tmp_path):
    """审查处置 F4 (D8): 清空/压缩提交屏障内 /skill 静默无动作, 落定后恢复."""
    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha")
    session = await open_application(make_config(root, HeldModel(), make_snapshot(root, skills_dir)))
    h = Harness(session)
    try:
        h.view._gate.note_clear_started()
        before = len(h.entries())
        assert h.submit("/skill alpha 任务") is False
        assert h.submit("/skill") is False
        assert len(h.entries()) == before
        h.view._gate.note_clear_settled()
        h.view._gate.note_compact_started()
        assert h.submit("/skill alpha 任务") is False
        assert h.submit("/skill") is False
        assert len(h.entries()) == before
        h.view._gate.note_compact_settled()
        assert h.submit("/skill") is True  # 落定后恢复可用
        assert h.user_entries() == []
    finally:
        h.view.close()
        await session.aclose()


# --- 显示条目: 收起/展开、窄窗、控制字符、坏封装回退 ----------------------------------------------


async def test_ctrl_o_toggles_skill_body_and_shares_tool_flag(tmp_path):
    from cicada.__main__ import parse_script

    from cicada.plugins.fake_model import FakeModel

    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha", body="Alpha body line one.\nAlpha body line two.")
    session = await open_application(make_config(
        root, FakeModel(parse_script('[{"text": "完成", "stop": true}]')),
        make_snapshot(root, skills_dir),
    ))
    h = Harness(session)
    try:
        assert h.submit("/skill alpha 任务") is True
        await h.wait_idle()
        collapsed = h.lines()
        # 标题按列宽换行: 名称与路径可能分块, 逐段断言
        assert "Skill 文档" in "\n".join(collapsed)
        assert "SKILL.md" in "\n".join(collapsed)
        assert "正文已收起" in "\n".join(collapsed)
        assert "Alpha body line one." not in "\n".join(collapsed)
        h.view.handle_ctrl_o()  # 与工具输出共享的全局展开标志
        assert h.view.tools.expanded is True
        expanded = "\n".join(h.lines())
        assert "Alpha body line one." in expanded
        assert "Alpha body line two." in expanded
        assert "sha256:" in expanded
        h.view.handle_ctrl_o()
        assert "Alpha body line one." not in "\n".join(h.lines())
    finally:
        h.view.close()
        await session.aclose()


def test_skill_entry_narrow_window_and_control_chars_sanitized():
    body = "col1\x1b[31mRED\x1b[0mtail\r\nCJK 正文与宽字 ✓\nthird"
    info = SkillInfo(
        name="alpha", description="d", source_path="C:/s/alpha/SKILL.md",
        base_dir="C:/s/alpha", source_sha256="a" * 64,
    )
    import hashlib as _hashlib

    document = SkillDocument(
        info=info, body=body,
        body_sha256=_hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )
    text = build_skill_request(document, "任务")
    entry = Transcript().append_user(text)
    assert isinstance(entry, SkillRequestEntry)
    # 无 getter 注入: 完整显示正文 (契约: 未注入的调用者不丢内容)
    for width in (20, 40, 80):
        for line in entry.render(width):
            assert visible_width(line) <= width, (width, line)
    expanded = "\n".join(entry.render(80))
    assert "RED" in expanded and "\x1b[31m" not in expanded
    assert "CJK 正文与宽字 ✓" in expanded
    assert expanded.count("third") == 1


def test_bad_envelope_falls_back_to_raw_user_entry():
    transcript = Transcript()
    broken_variants = [
        "CICADA_SKILL_REQUEST_V1\n{not json",
        "CICADA_SKILL_REQUEST_V1\n{\"version\": 999}",
        "CICADA_SKILL_REQUEST_V1",  # 无换行
    ]
    for variant in broken_variants:
        entry = transcript.append_user(variant)
        assert type(entry) is UserEntry, variant
        assert entry.text == variant
    rendered = "\n".join(transcript.render(80))
    assert "{not json" in rendered


def test_deep_nested_envelope_falls_back_instead_of_crashing():
    """审查处置 F1: 深嵌套封装在显示入口同样按原始文本回退, 不抛 RecursionError."""
    transcript = Transcript()
    entry = transcript.append_user(SKILL_REQUEST_PREFIX + "\n" + "[" * 60_000)
    assert type(entry) is UserEntry
    assert entry.text.startswith(SKILL_REQUEST_PREFIX)
    assert transcript.render(80)  # 渲染有界, 不抛异常


def test_skill_header_source_path_control_chars_sanitized():
    """审查处置 F2: 标题行整体显示清理——记录文本中的 ANSI 不得进入渲染输出."""
    import hashlib as _hashlib

    info = SkillInfo(
        name="alpha", description="d",
        source_path="C:/s/\x1b[31malpha\x1b[0m/SKILL.md\x1b[48;2;255;0;0m",
        base_dir="C:/s/alpha", source_sha256="a" * 64,
    )
    body = "clean body"
    document = SkillDocument(
        info=info, body=body,
        body_sha256=_hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )
    entry = Transcript().append_user(build_skill_request(document, "任务"))
    assert isinstance(entry, SkillRequestEntry)
    for expanded in (False, True):
        if expanded:
            entry._expand_getter = lambda: True  # 展开态标题含 sha256 段, 同样受清理约束
        rendered = "\n".join(entry.render(80))
        assert "\x1b[31m" not in rendered
        assert "\x1b[48;2;255;0;0m" not in rendered
        assert "SKILL.md" in rendered


# --- 恢复: 解码来自记录, 不读当前目录 -------------------------------------------------------------


async def test_restored_skill_request_decodes_from_record_without_current_dir(tmp_path):
    from dataclasses import replace

    from cicada.__main__ import parse_script

    from cicada.plugins.fake_model import FakeModel

    root = tmp_path / "ws"
    skills_dir = root / "skills"
    write_skill(skills_dir, "alpha", body="Archived body.")
    snapshot = make_snapshot(root, skills_dir)
    config = make_config(root, FakeModel(parse_script('[{"text": "完成", "stop": true}]')), snapshot)
    persisted = replace(
        config,
        persistence=PersistenceConfig(
            session_id=None, identity=identity_for(config.prompt_info)
        ),
    )
    session = await open_application(persisted)
    session_id = session.persistence.session_id
    prepared = session.prepare_skill_request("alpha", "任务")
    saved_invocation = decode_skill_request(prepared)
    await session.execute(prepared)
    await session.aclose()

    resumed = await open_application(replace(
        config,
        persistence=PersistenceConfig(
            session_id=session_id, identity=identity_for(config.prompt_info)
        ),
    ))
    # 恢复准入后删除当前目录: 显示投影只解码记录, 不访问磁盘
    shutil.rmtree(skills_dir)
    h = Harness(resumed)
    try:
        assert resumed.persistence.restored_request_count == 1
        users = [e for e in h.entries() if isinstance(e, SkillRequestEntry)]
        assert len(users) == 1
        assert users[0].invocation == saved_invocation
        h.view.handle_ctrl_o()
        assert "Archived body." in "\n".join(h.lines())
    finally:
        h.view.close()
        await resumed.aclose()
