#!/usr/bin/env python3
"""Fail closed if isolated Ray (or reward/judge) listens on a non-loopback address."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CONTROL_ROLES = ("gcs", "dashboard", "client")


def _run_ss() -> str:
    try:
        return subprocess.check_output(["ss", "-ltn"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            return subprocess.check_output(["netstat", "-ltn"], text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(f"cannot list listeners: {exc}") from exc


def parse_listeners(raw: str) -> List[Tuple[str, int]]:
    """Return (bind_ip, port) from ss/netstat -ltn output."""
    rows: List[Tuple[str, int]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("state") or line.lower().startswith("proto"):
            continue
        parts = line.split()
        addrs = [tok for tok in parts if ":" in tok]
        if not addrs:
            continue
        local = addrs[0]
        # ss format: 127.0.0.1:26379 or [::]:80 or *:80
        if local.startswith("[") and "]:" in local:
            host, _, port_s = local[1:].partition("]:")
        else:
            host, _, port_s = local.rpartition(":")
        if not port_s.isdigit():
            continue
        host = host.strip() or "*"
        if host in {"*", "::", "[::]"}:
            host = "0.0.0.0"
        if host.startswith("::ffff:"):
            host = host.split("::ffff:")[-1]
        rows.append((host, int(port_s)))
    return rows


def _is_loopback(host: str) -> bool:
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        return bool(ip_address(host).is_loopback)
    except ValueError:
        return False


def _is_unspecified(host: str) -> bool:
    return host in {"0.0.0.0", "::", "*"}


def classify_bind(host: str) -> str:
    if _is_loopback(host):
        return "loopback"
    if _is_unspecified(host):
        return "wildcard"
    try:
        ip = ip_address(host)
        if ip.is_private or ip.is_global or ip.is_link_local:
            return "network"
    except ValueError:
        return "network"
    return "network"


def audit_listeners(
    listeners: List[Tuple[str, int]],
    *,
    gcs_port: int,
    dashboard_port: int,
    client_port: Optional[int],
    worker_min: int,
    worker_max: int,
    extra_ports: Optional[Dict[str, int]] = None,
    allow_wildcard_workers: bool = True,
) -> Dict[str, Any]:
    role_ports = {
        "gcs": gcs_port,
        "dashboard": dashboard_port,
    }
    if client_port:
        role_ports["client"] = client_port
    extra_ports = extra_ports or {}
    role_ports.update(extra_ports)

    findings: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    by_port: Dict[int, List[str]] = {}
    for host, port in listeners:
        by_port.setdefault(port, []).append(host)

    for role, port in role_ports.items():
        hosts = by_port.get(int(port), [])
        if not hosts:
            # Port not up yet is not a bind violation.
            findings.append({"role": role, "port": port, "hosts": [], "status": "missing"})
            continue
        for host in hosts:
            kind = classify_bind(host)
            rec = {"role": role, "port": port, "host": host, "bind": kind}
            findings.append(rec)
            if kind != "loopback":
                msg = f"{role} :{port} bound to {host} ({kind})"
                failures.append(msg)

    for port, hosts in sorted(by_port.items()):
        if port < worker_min or port > worker_max:
            continue
        if port in role_ports.values():
            continue
        for host in hosts:
            kind = classify_bind(host)
            rec = {"role": "worker", "port": port, "host": host, "bind": kind}
            findings.append(rec)
            if kind == "network":
                failures.append(f"worker :{port} bound to network address {host}")
            elif kind == "wildcard":
                if allow_wildcard_workers:
                    warnings.append(f"worker :{port} wildcard bind {host} (GCS is loopback-only)")
                else:
                    failures.append(f"worker :{port} bound to {host}")

    return {
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "findings": findings,
        "allow_wildcard_workers": allow_wildcard_workers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs-port", type=int, default=int(os.environ.get("RAY_GCS_PORT", "26379")))
    parser.add_argument("--dashboard-port", type=int, default=int(os.environ.get("RAY_DASHBOARD_PORT", "28265")))
    parser.add_argument("--client-port", type=int, default=int(os.environ.get("RAY_CLIENT_PORT", "0") or 0))
    parser.add_argument("--min-worker-port", type=int, default=int(os.environ.get("RAY_MIN_WORKER_PORT", "26381")))
    parser.add_argument("--max-worker-port", type=int, default=int(os.environ.get("RAY_MAX_WORKER_PORT", "27380")))
    parser.add_argument("--reward-port", type=int, default=0)
    parser.add_argument("--judge-port", type=int, default=0)
    parser.add_argument("--ss-file", default="")
    parser.add_argument("--ss-text", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--strict-workers", action="store_true")
    args = parser.parse_args()

    raw = args.ss_text
    if args.ss_file:
        raw = Path(args.ss_file).read_text(encoding="utf-8")
    if not raw:
        raw = _run_ss()
    listeners = parse_listeners(raw)
    extra = {}
    if args.reward_port:
        extra["reward"] = args.reward_port
    if args.judge_port:
        extra["judge"] = args.judge_port
    report = audit_listeners(
        listeners,
        gcs_port=args.gcs_port,
        dashboard_port=args.dashboard_port,
        client_port=args.client_port or None,
        worker_min=args.min_worker_port,
        worker_max=args.max_worker_port,
        extra_ports=extra,
        allow_wildcard_workers=not args.strict_workers,
    )
    report["hostname"] = socket.gethostname()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps({k: report[k] for k in ("ok", "failures", "warnings")}, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
