#!/usr/bin/env python3
"""Least-outstanding-request reverse proxy for multiple local vLLM OpenAI servers."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class Backend:
    def __init__(self, url: str) -> None:
        p = urlparse(url if "://" in url else f"http://{url}")
        self.host = p.hostname or "127.0.0.1"
        self.port = int(p.port or 80)
        self.label = f"{self.host}:{self.port}"
        self.in_flight = 0
        self.fail_until = 0.0
        self.lock = threading.Lock()

    def healthy(self) -> bool:
        return time.time() >= self.fail_until


class ProxyState:
    def __init__(self, backends: list[Backend]) -> None:
        self.backends = backends
        self.lock = threading.Lock()

    def pick(self) -> Backend | None:
        with self.lock:
            ready = [b for b in self.backends if b.healthy()]
            if not ready:
                ready = list(self.backends)
            if not ready:
                return None
            b = min(ready, key=lambda x: x.in_flight)
            b.in_flight += 1
            return b

    def done(self, b: Backend, ok: bool) -> None:
        with b.lock:
            b.in_flight = max(0, b.in_flight - 1)
            if not ok:
                b.fail_until = time.time() + 2.0


STATE: ProxyState | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def _handle(self) -> None:
        assert STATE is not None
        if self.path in ("/health", "/lb/health"):
            payload = {
                "status": "ok",
                "backends": [
                    {"label": b.label, "in_flight": b.in_flight, "healthy": b.healthy()}
                    for b in STATE.backends
                ],
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        length = int(self.headers.get("Content-Length") or 0)
        req_body = self.rfile.read(length) if length else b""
        backend = STATE.pick()
        if backend is None:
            self.send_error(503, "no backends")
            return
        ok = False
        try:
            conn = HTTPConnection(backend.host, backend.port, timeout=1800)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length"}}
            headers["Host"] = backend.label
            conn.request(self.command, self.path, body=req_body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            hop = {"connection", "transfer-encoding", "keep-alive", "proxy-connection"}
            for k, v in resp.getheaders():
                if k.lower() not in hop:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            ok = 200 <= resp.status < 500
            conn.close()
        except Exception:
            self.send_error(502, f"backend {backend.label} failed")
        finally:
            STATE.done(backend, ok)

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()


def main() -> int:
    global STATE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--backends", required=True, help="comma-separated host:port")
    args = p.parse_args()
    backends = []
    for raw in args.backends.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "://" not in raw:
            raw = "http://" + raw
        backends.append(Backend(raw))
    if not backends:
        raise SystemExit("no backends")
    STATE = ProxyState(backends)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[lb] listening {args.host}:{args.port} -> {[b.label for b in backends]}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
