"""P4 验收留证 launcher：调用真实 CLI，仅观察返回对象与事件。

用法：python tests/e2e/p4_observer.py --evidence-dir <已有项目内目录> -- <CLI参数>
记录失败不改变生产退出码，observer-result.evidence_complete=false 必须阻断验收。
生产异常原样传播；此入口不改变模型请求、工具结果、usage 或工作区内容。
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]


def serialize(value: Any) -> Any:
    """保留完整 dataclass 字段，以 type 记录具体协议类型。"""
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__, **{
            field.name: serialize(getattr(value, field.name)) for field in fields(value)
        }}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("evidence dictionaries must have string keys")
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


class Observer:
    """只写指定证据目录；观察写入故障独立记录，不能冒充生产成功。"""

    def __init__(self, evidence_dir: Path):
        self.evidence_dir = evidence_dir
        self.recording_errors: list[dict[str, str]] = []
        self._created: set[str] = set()

    def note_error(self, operation: str, exc: BaseException) -> None:
        self.recording_errors.append({"operation": operation, **_exception(exc)})

    def record(self, filename: str, value: Any, *, append: bool = False) -> None:
        try:
            encoded = json.dumps(serialize(value), ensure_ascii=False, allow_nan=False) + "\n"
            # 首次写入用 exclusive 模式，避免覆盖已存在的历史证据。
            mode = "a" if append and filename in self._created else "x"
            with (self.evidence_dir / filename).open(mode, encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
            self._created.add(filename)
        except Exception as exc:
            self.note_error(filename, exc)

    def observe_async(self, original, operation: str, filename: str, *, append: bool = True):
        """先 await 原调用，仅序列化其原返回值，异常记录后原样抛出。"""
        @functools.wraps(original)
        async def observed(*args, **kwargs):
            try:
                value = await original(*args, **kwargs)
            except BaseException as exc:
                self.record("observed-exceptions.jsonl", {
                    "operation": operation, "exception": _exception(exc),
                }, append=True)
                raise
            self.record(filename, {"operation": operation, "value": value} if append else value,
                        append=append)
            return value
        return observed


def run_observed_cli(evidence_dir: Path, cli_args: list[str]) -> int:
    """执行同一生产 main，返回实际退出码，保存完整观察身份与故障。"""
    evidence_dir = evidence_dir.resolve(strict=True)
    if not evidence_dir.is_dir() or not evidence_dir.is_relative_to(PROJECT):
        raise ValueError("evidence-dir must be an existing directory inside the project")
    import cicada
    import cicada.__main__ as cli
    from cicada.plugins.coding.verification_contracts import VERIFICATION_CAPABILITY

    observer = Observer(evidence_dir)
    original_bootstrap, original_decide = cli.bootstrap, cli.decide_delivery
    invocation_argv = list(sys.argv)
    original_argv = sys.argv
    restorations: list[tuple[Any, str, Any]] = []
    unsubscribers = []

    def replace(instance, attribute: str, replacement) -> None:
        restorations.append((instance, attribute, getattr(instance, attribute)))
        setattr(instance, attribute, replacement)

    async def observed_bootstrap(*args, **kwargs):
        app = await original_bootstrap(*args, **kwargs)
        try:
            unsubscribers.append(app.agent.subscribe(
                lambda event: observer.record("events.jsonl", event, append=True)))
            replace(app.agent, "run", observer.observe_async(
                app.agent.run, "agent.run", "run-result.json", append=False))
            if kwargs.get("model_policy") is not None:
                service = app.runtime.capability(VERIFICATION_CAPABILITY)
                observer.record("verification.jsonl", {"operation": "plan", "value": service.plan},
                                append=True)
                for method in ("initialize", "run_check", "refresh", "finalize"):
                    replace(service, method, observer.observe_async(
                        getattr(service, method), method, "verification.jsonl"))
                replace(service.snapshotter, "capture", observer.observe_async(
                    service.snapshotter.capture, "capture", "snapshots.jsonl"))
        except Exception as exc:
            # 安装观察器的失败也不能改变已组装的生产 app。
            observer.note_error("attach", exc)
        return app

    @functools.wraps(original_decide)
    def observed_decide(result, view, evidence, artifact_problems=()):
        decision = original_decide(result, view, evidence, artifact_problems)
        observer.record("delivery.json", {
            "result": result, "view": view, "evidence": evidence,
            "artifact_problems": artifact_problems, "decision": decision,
        })
        return decision

    main_exit_code: int | None = None
    main_exception: dict[str, str] | None = None
    started = time.monotonic()
    try:
        cli.bootstrap, cli.decide_delivery = observed_bootstrap, observed_decide
        sys.argv = [str(Path(cli.__file__).resolve()), *cli_args]
        main_exit_code = cli.main()
        return main_exit_code
    except BaseException as exc:
        main_exception = _exception(exc)
        if isinstance(exc, SystemExit) and (isinstance(exc.code, int) or exc.code is None):
            main_exit_code = exc.code or 0
        raise
    finally:
        elapsed = time.monotonic() - started
        cli.bootstrap, cli.decide_delivery = original_bootstrap, original_decide
        sys.argv = original_argv
        for instance, attribute, original in reversed(restorations):
            setattr(instance, attribute, original)
        for unsubscribe in reversed(unsubscribers):
            unsubscribe()
        result = {
            "main_exit_code": main_exit_code, "main_exception": main_exception,
            "wall_time_s": elapsed, "recording_errors": observer.recording_errors,
            "evidence_complete": not observer.recording_errors,
            "invocation_argv": invocation_argv, "cli_argv": list(cli_args),
            "sys_executable": sys.executable,
            "module_paths": {
                "cicada": str(Path(cicada.__file__).resolve()),
                "cicada.__main__": str(Path(cli.__file__).resolve()),
                "observer": str(Path(__file__).resolve()),
            },
        }
        # 最终回执自身写入失败时 stdout 保持不变，在 stderr 明确报告观察失败。
        before = len(observer.recording_errors)
        observer.record("observer-result.json", result)
        if len(observer.recording_errors) != before:
            print("P4 observer failed to record observer-result.json: "
                  + json.dumps(observer.recording_errors, ensure_ascii=False), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe the real Cicada CLI for P4 evidence")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cli_args = args.cli_args
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]
    return run_observed_cli(args.evidence_dir, cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
