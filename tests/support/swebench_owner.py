"""主 runner 持有评测服务；checker 仅发送请求，不拥有 Docker 生命周期。"""
from __future__ import annotations

import ctypes
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


def pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel.CloseHandle(handle)


class EvaluationOwner:
    """串行执行评测；请求者消失时由 owner 查询持久 intent 并终止资源。"""

    def __init__(self, run_root: Path, evaluate, cleanup):
        self.root = run_root
        self.evaluate = evaluate
        self.cleanup = cleanup
        self.token = uuid.uuid4().hex
        self.requester: int | None = None
        self.stopping = threading.Event()
        self.watch_done = threading.Event()
        self.active = threading.Event()
        self.gate = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.cleanup_errors: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.headers.get("Authorization") != owner.token:
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(400)
                    return
                if not 0 < length <= 1024:
                    self.send_error(400)
                    return
                if not owner.gate.acquire(blocking=False):
                    self.send_error(409)
                    return
                if owner.stopping.is_set():
                    owner.gate.release()
                    self.send_error(503)
                    return
                owner.active.set()
                try:
                    payload = json.loads(self.rfile.read(length))
                    pid = payload["pid"]
                    if type(pid) is not int or pid == os.getpid() or not pid_alive(pid):
                        raise ValueError("requester is not alive")
                    owner.requester = pid
                    code, summary = owner.evaluate(owner.stopping)
                except Exception as exc:
                    code, summary = 2, {"result": "anomaly", "reason": type(exc).__name__}
                finally:
                    owner.requester = None
                    owner.active.clear()
                    owner.gate.release()
                data = json.dumps({"exit": code, "summary": summary}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.watch = threading.Thread(target=self._watch, daemon=True)

    def _watch(self):
        while not self.watch_done.wait(0.2):
            pid = self.requester
            if self.stopping.is_set() or (pid is not None and not pid_alive(pid)):
                try:
                    if pid is not None and not pid_alive(pid):
                        self.stopping.set()
                    results = self._cleanup()
                    if any(r.get("outcome") == "uncertain" for r in results):
                        self.cleanup_errors.append("orphan cleanup uncertain")
                except Exception as exc:
                    self.cleanup_errors.append(type(exc).__name__)

    def _cleanup(self):
        with self.cleanup_lock:
            return self.cleanup()

    def __enter__(self):
        path = self.root / "evidence" / "supervisor" / "window.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": os.getpid(), "purpose": "run",
                                   "port": self.server.server_port, "token": self.token}),
                        encoding="utf-8")
        self.thread.start()
        self.watch.start()
        return self

    def __exit__(self, *args):
        self.stopping.set()
        self.server.shutdown()
        deadline = time.monotonic() + 30
        while self.active.is_set() and time.monotonic() < deadline:
            self._cleanup()
            time.sleep(0.1)
        results = self._cleanup()
        self.watch_done.set()
        self.thread.join(timeout=5)
        self.watch.join(timeout=5)
        self.server.server_close()
        if self.active.is_set() or self.thread.is_alive() or self.watch.is_alive() or self.cleanup_errors or any(
                r.get("outcome") == "uncertain" for r in results):
            raise RuntimeError("evaluation owner cleanup remains uncertain")


def request_check(run_root: Path) -> tuple[int, dict]:
    window = json.loads((run_root / "evidence" / "supervisor" / "window.json").read_text())
    if not pid_alive(window["pid"]) or window["pid"] == os.getpid():
        raise RuntimeError("check requires a healthy external evaluation owner")
    request = Request(f"http://127.0.0.1:{window['port']}/check",
                      data=json.dumps({"pid": os.getpid()}).encode(),
                      headers={"Authorization": window["token"]}, method="POST")
    with urlopen(request, timeout=290) as response:
        result = json.load(response)
    return result["exit"], result["summary"]
