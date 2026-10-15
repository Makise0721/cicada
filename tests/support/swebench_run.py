"""私有单题 launcher：真实 main/解析/应用出口，只注入 transport 和只读观察。"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

from tests.e2e.p4_observer import serialize
from tests.support.swebench_cost import observe_model_stream


class EphemeralStore:
    def __init__(self, key):
        self.key = key

    def load(self, environment_key=None):
        from cicada.user_config import SecretValue, UserConfigSnapshot, UserDefaults
        return UserConfigSnapshot(UserDefaults(), frozenset(), SecretValue(self.key),
                                  "environment", "ephemeral-benchmark")


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def cli_arguments(run_root, prompt):
    runner = Path(__file__).resolve().parents[1] / "e2e/swebench_runner.py"
    command = (f"& {ps_quote(sys.executable)} {ps_quote(runner)} check "
               f"--run-root {ps_quote(run_root)} --workspace {ps_quote(run_root / 'workspace')}")
    return ["--workspace", str(run_root / "workspace"), "--provider", "deepseek",
            "--model", "deepseek-flash", "--think", "--reasoning-effort", "low",
            "--max-output-tokens", "8192", "--num-ctx", "262144",
            "--check-command", command, "--check-timeout", "300", prompt]


def run_cli(run_root, transport, key, prompt, *, timeout_s=900):
    import cicada.__main__ as cli
    from cicada.plugins.deepseek.model import DeepSeekModel

    evidence = run_root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    facts = {"main_exit_code": None, "application_closed": False, "delivery": None,
             "recording_errors": [], "run_timeout_s": timeout_s, "timed_out": False}
    originals = (sys.argv, cli.deepseek_plugin, cli._run, cli.open_application, DeepSeekModel.stream)
    session_restorations = []
    unsubscribers = []

    def record(filename, value, append=False):
        try:
            with (evidence / filename).open("a" if append else "w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(serialize(value), ensure_ascii=False, allow_nan=False) + "\n")
        except Exception as exc:
            facts["recording_errors"].append(type(exc).__name__)
            transport.stop("observer-write-failure")

    def factory(config, *, api_key=None):
        if (config.model, config.think, config.reasoning_effort, config.num_ctx, config.max_tokens) != (
                "deepseek-flash", True, "low", 262144, 8192):
            raise ValueError("production configuration differs from frozen profile")
        record("profile.json", {"model": config.model, "think": config.think,
            "reasoning_effort": config.reasoning_effort, "num_ctx": config.num_ctx,
            "max_tokens": config.max_tokens, "max_request_bytes": config.max_request_bytes})
        return originals[1](config, api_key=api_key, transport=transport)

    async def opened(*args, **kwargs):
        session = await originals[3](*args, **kwargs)
        unsubscribers.append(session.subscribe(lambda update: record("events.jsonl", update, True)))
        execute, close = session.execute, session.aclose
        session_restorations.append((session, execute, close))

        async def executed(*args, **kwargs):
            outcome = await execute(*args, **kwargs)
            facts["delivery"] = serialize({"result": outcome.result, "view": outcome.view,
                "evidence": outcome.evidence, "decision": outcome.decision,
                "exit_code": outcome.exit_code})
            record("delivery.json", facts["delivery"])
            return outcome

        async def closed():
            await close()
            facts["application_closed"] = True

        session.execute, session.aclose = executed, closed
        return session

    async def supervised(args, **kwargs):
        transport.deadline = time.monotonic() + timeout_s
        task = asyncio.create_task(originals[2](args, **kwargs))
        loop = asyncio.get_running_loop()
        transport.on_stop = lambda reason: loop.call_soon(task.cancel)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_s)
            if not done:
                facts["timed_out"] = True
                transport.stop("run-timeout")  # 封网/持久封存先于实际生产任务取消。
                task.cancel()
            try:
                return await task
            except asyncio.CancelledError:
                return 130
        except asyncio.CancelledError:
            transport.stop("production-cancelled")
            task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            # 消费子任务的最终异常；不遗弃原生产 finally/aclose。
            if task.done() and not task.cancelled():
                task.exception()
            raise
        finally:
            transport.on_stop = None

    started = time.monotonic()
    arguments = cli_arguments(run_root, prompt)
    try:
        cli.deepseek_plugin, cli._run, cli.open_application = factory, supervised, opened
        DeepSeekModel.stream = observe_model_stream(originals[4], transport)
        sys.argv = [str(Path(cli.__file__).resolve()), *arguments]
        with (evidence / "cli-stdout.txt").open("w", encoding="utf-8") as stdout, (
                evidence / "cli-stderr.txt").open("w", encoding="utf-8") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                facts["main_exit_code"] = cli.main(user_config_store=EphemeralStore(key))
    except BaseException as exc:
        facts["main_exception"] = type(exc).__name__
        facts["main_exit_code"] = 130 if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else 2
        transport.stop("production-exception")
    finally:
        sys.argv, cli.deepseek_plugin, cli._run, cli.open_application, DeepSeekModel.stream = originals
        for unsubscribe in unsubscribers:
            unsubscribe()
        for session, execute, close in session_restorations:
            session.execute, session.aclose = execute, close
        facts["elapsed_s"] = time.monotonic() - started
        facts["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        facts["cli_arguments"] = arguments
        record("production.json", facts)
    return facts
