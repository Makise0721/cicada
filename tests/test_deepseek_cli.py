"""main() 的真实配置组合在 mock HTTP 下保存恢复；离线入口不读取 key。"""

import json
from pathlib import Path

import httpx
import pytest

from cicada import __main__ as cli
from cicada.plugins.deepseek import deepseek_plugin
from cicada.plugins.session_store import read_session_identity
from test_cli import _CaptureStream
from test_deepseek_application import response


def invoke(monkeypatch, args):
    out, err = _CaptureStream(), _CaptureStream()
    monkeypatch.setattr(cli.sys, "argv", ["cicada", *args])
    monkeypatch.setattr(cli.sys, "stdout", out)
    monkeypatch.setattr(cli.sys, "stderr", err)
    return cli.main(), out.value, err.value


def mock_provider(monkeypatch):
    bodies = []
    def handle(req):
        bodies.append(json.loads(req.content))
        assert req.url == "https://api.deepseek.com/chat/completions"
        return httpx.Response(200, content=response("思考", "离线答复"))
    monkeypatch.setattr(cli, "deepseek_plugin", lambda config: deepseek_plugin(
        config, api_key="cli-mock-key", transport=httpx.MockTransport(handle)))
    async def forbidden(*args):
        pytest.fail("DeepSeek must not probe Ollama")
    monkeypatch.setattr(cli, "_ollama_reachable", forbidden)
    return bodies


def test_default_flash_and_pro_explicit_limits_persist_and_restore(monkeypatch, tmp_path):
    bodies = mock_provider(monkeypatch)
    code, out, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "--save-session",
        "--model", "deepseek-v4-pro", "--num-ctx", "65536", "--max-output-tokens", "4096",
        "--reasoning-effort", "low", "第一问"])
    assert code == 0, err
    session_id = out.split("[session id=")[1].split(" ")[0]
    identity = read_session_identity(directory=tmp_path / ".cicada" / "sessions",
                                    session_id=session_id, workspace=tmp_path).launch
    assert identity.provider == "deepseek" and identity.think is True
    assert identity.options == {"num_ctx": 65536, "max_tokens": 4096, "reasoning_effort": "low"}
    code, out, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "--resume", session_id, "续问"])
    assert code == 0, err
    assert bodies[-1]["max_tokens"] == 4096 and bodies[-1]["model"] == "deepseek-v4-pro"
    assert bodies[-1]["messages"][2]["reasoning_content"] == "思考"
    before = len(bodies)
    code, _, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "--resume", session_id,
                                      "--no-think", "冲突"])
    assert code == 2 and "think" in err and len(bodies) == before
    code, _, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "默认新建"])
    assert code == 0, err
    assert bodies[-1]["model"] == "deepseek-flash" and bodies[-1]["reasoning_effort"] == "high"
    assert bodies[-1]["max_tokens"] == 32768


@pytest.mark.parametrize("args", [
    ["--ollama-url", "http://localhost:1"], ["--model", "deepseek-chat"],
    ["--max-output-tokens", "0"], ["--max-output-tokens", "393217"],
    ["--num-ctx", "1048577"], ["--num-ctx", "4096"],
    ["--provider", "ollama", "--reasoning-effort", "low"],
])
def test_bad_provider_flags_zero_network(monkeypatch, tmp_path, args):
    bodies = mock_provider(monkeypatch)
    code, _, _ = invoke(monkeypatch, ["--workspace", str(tmp_path), *args, "任务"])
    assert code == 2 and bodies == []


def test_list_and_fake_do_not_read_credentials_or_use_provider_network(monkeypatch, tmp_path):
    # 让读取指定环境变量直接失败，而非仅删除真实环境值。
    original_get = cli.os.environ.get
    def guarded_get(name, *args):
        if name == "DEEPSEEK_API_KEY":
            pytest.fail("offline entry read provider credentials")
        return original_get(name, *args)
    monkeypatch.setattr(cli.os.environ, "get", guarded_get)
    monkeypatch.setattr(httpx.AsyncClient, "send", lambda *args, **kwargs: pytest.fail("offline network"))
    script = tmp_path / "script.json"
    script.write_text('[{"text":"done","stop":true}]', encoding="utf-8")
    code, _, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "--script", str(script), "任务"])
    assert code == 0, err
    code, _, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "--list-sessions"])
    assert code == 0, err
    for flag in (["--provider", "deepseek"], ["--no-think"], ["--max-output-tokens", "10"], ["--reasoning-effort", "high"]):
        code, _, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "--script", str(script), *flag, "任务"])
        assert code == 2 and "互斥" in err


def test_missing_environment_key_is_bounded_startup_error(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "send", lambda *args, **kwargs: pytest.fail("missing key network"))
    code, _, err = invoke(monkeypatch, ["--workspace", str(tmp_path), "任务"])
    assert code == 2 and "DEEPSEEK_API_KEY is missing" in err
