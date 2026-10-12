"""Bench-specific failure classification and persistent whole-batch backoff."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from scripts.experiment_manifest import sha256_file
from scripts.run_retrieval_comparison import retry_after_deadline, upstream_overloaded, write_json

POLICY = {
    "version": "bench_recovery_v2",
    "max_stream_attempts_per_task": 6,
    "attempt_cap_excludes_proven_pre_http_connection_failures": True,
    "task_cooldown_seconds": 600,
    "unknown_request_wait_seconds": 1800,
    "global_backoff_seconds": [300, 600, 1200, 1800],
    "generation_policy": "unchanged; completed content is never rerolled",
}


def pre_http_connection_failure(record):
    return (record.get("error_type") in {"ConnectError", "ConnectTimeout"}
            and record.get("http_status") is None and not record.get("response")
            and not record.get("stream_response_id") and not record.get("stream_events"))


def provider_stream_attempts(records):
    return sum(bool(rec["request"].get("stream")) and not pre_http_connection_failure(rec)
               for _, rec in records)


def execution_limits(out, config, records, *, create=False):
    """Credit historical pre-HTTP failures once, without clearing their ledger.

    This restores the original provider-request allowance consumed by TLS
    connection attempts. It is not an extra model-compute or monetary budget.
    New failures never enlarge this fixed recovery ceiling on later restarts.
    """
    path = Path(out) / "runtime_budget.json"
    if not path.exists() and create:
        credits = [{"path": str(p.relative_to(out)), "sha256": sha256_file(p)}
                   for p, rec in records if pre_http_connection_failure(rec)]
        write_json(path, {"policy": "one_time_pre_http_connection_credit_v1",
                          "manifest_sha256": sha256_file(Path(out)/"manifest.json"),
                          "original_request_limit": config["max_requests"],
                          "effective_request_limit": config["max_requests"]+len(credits),
                          "credit_records": credits, "created_unix": time.time(),
                          "reason": "2026-10-12 mainline engineering repair: historical TLS/connect failures consumed the provider-call allowance; all attempts and conservative token reservations retained",
                          "monetary_and_generation_limits": "unchanged key, quota floor, model, reasoning, prompts, per-response and aggregate token caps"})
    ceiling = config["max_requests"]
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt.get("manifest_sha256") != sha256_file(Path(out)/"manifest.json") or receipt.get("original_request_limit") != ceiling:
            raise ValueError("runtime budget manifest mismatch")
        credits = receipt["credit_records"]
        if len({x["path"] for x in credits}) != len(credits):
            raise ValueError("duplicate connection credit")
        for item in credits:
            p = (Path(out)/item["path"]).resolve()
            if not p.is_relative_to(Path(out).resolve()) or sha256_file(p) != item["sha256"] or not pre_http_connection_failure(json.loads(p.read_text())):
                raise ValueError("connection-credit evidence mismatch")
        ceiling += len(credits)
        if receipt.get("effective_request_limit") != ceiling:
            raise ValueError("runtime request cap changed")
    return {"requests": ceiling, "input_tokens": config["max_input_tokens"], "output_tokens": config["max_output_tokens"]}


def response_error(record, folder):
    response = record.get("response") or {}
    error = record.get("stream_error") or response.get("error")
    if isinstance(error, dict):
        return error
    name = record.get("stream_events")
    path = folder / name if name else None
    if path and path.is_file():
        for line in reversed(path.read_text().splitlines()):
            event = json.loads(line)
            if event.get("type") == "error":
                return event.get("error") or event
            if event.get("type") == "response.failed":
                return (event.get("response") or {}).get("error") or {}
    return {}


def failure_kind(record, folder):
    """Only positive transport/overload evidence permits a retry."""
    response = record.get("response") or {}
    if response.get("status") in {"completed", "incomplete"}:
        return None  # Completed/length-limited content needs review, never a reroll.
    if any(s in str(record.get("error", "")) for s in (
        "identity mismatch", "did not confirm", "Malformed SSE")):
        return None
    error = response_error(record, folder)
    code, error_type = error.get("code"), error.get("type")
    if error_type in {"invalid_request_error", "authentication_error", "permission_error",
                      "rate_limit_error", "insufficient_quota"} or code in {
        "invalid_api_key", "insufficient_quota", "rate_limit_exceeded", "invalid_request_error"}:
        return None
    status = record.get("http_status")
    if status in {400, 401, 402, 403}:
        return None
    if status == 429 and not upstream_overloaded(record):
        return None
    if code in {"server_is_overloaded", "upstream_overloaded"} or upstream_overloaded(record):
        return "upstream_overload"
    if code == "ticket_wait_timeout" and error_type == "server_error" and error.get("status_code") == 503:
        return "upstream_queue_timeout"
    if code in {"request_timeout", "server_error"}:
        return "upstream_timeout" if code == "request_timeout" else "upstream_error"
    if "Missing usage" in str(record.get("error", "")):
        return None
    if status in {408, 500, 502, 503, 504}:
        return "upstream_http_" + str(status)
    if record.get("error_type") in {"ConnectError", "ConnectTimeout"}:
        return "connection_failed_before_response"
    if record.get("retryable_transport_error"):
        return "transport_disconnect_unknown_completion"
    if not record.get("elapsed_seconds") and not response:
        return "interrupted_request_unknown_completion"
    if record.get("error") == "stream ended without terminal response; no automatic retry" and not error:
        return "stream_eof_unknown_completion"
    return None


def completed_at(path, record):
    return record.get("started_unix", path.stat().st_mtime) + record.get("elapsed_seconds", 0)


def task_ready_at(path, record):
    return max(record.get("started_unix", 0) + POLICY["unknown_request_wait_seconds"],
               completed_at(path, record) + POLICY["task_cooldown_seconds"],
               retry_after_deadline(record))


class Circuit:
    """A task switch or process restart cannot bypass upstream cooldown."""
    def __init__(self, out):
        self.path = Path(out) / "scheduler.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "policy": POLICY, "consecutive_technical_failures": 0, "next_request_unix": 0,
            "last_request": None, "last_request_sha256": None,
        }
        if self.state.get("policy") != POLICY:
            raise ValueError("scheduler policy mismatch")

    def observe(self, path, record):
        sha = sha256_file(path)
        kind = failure_kind(record, path.parent)
        same_record = self.state.get("last_request_sha256") == sha
        if same_record and self.state.get("last_failure_kind") == kind:
            return
        if (record.get("response") or {}).get("status") == "completed" and not record.get("error"):
            count, deadline = 0, 0
        elif kind:
            count = self.state["consecutive_technical_failures"] + (0 if same_record and self.state.get("last_failure_kind") else 1)
            delays = POLICY["global_backoff_seconds"]
            deadline = max(completed_at(path, record) + delays[min(count, len(delays))-1],
                           retry_after_deadline(record))
        else:
            # Caller stops on nontechnical errors. Do not erase prior cooldown.
            count, deadline = self.state["consecutive_technical_failures"], self.state["next_request_unix"]
        self.state.update(consecutive_technical_failures=count, next_request_unix=deadline,
                          last_request=str(path), last_request_sha256=sha,
                          last_failure_kind=kind, updated_unix=time.time())
        write_json(self.path, self.state)

    def initialize_from_history(self, records):
        ordered = sorted(records, key=lambda x: x[1].get("started_unix", 0))
        if self.state.get("last_request"):
            matches = [i for i, (path, _) in enumerate(ordered)
                       if str(path) == self.state["last_request"] and sha256_file(path) == self.state["last_request_sha256"]]
            if len(matches) != 1:
                raise ValueError("scheduler request evidence missing or changed")
            # New explicit provider error codes may have been unclassified in
            # an earlier implementation. Reclassify from raw evidence once.
            self.observe(*ordered[matches[0]])
            for path, rec in ordered[matches[0]+1:]:
                self.observe(path, rec)
            return
        # Replay only the latest consecutive failure suffix, not hundreds of old
        # connection errors before a later successful response.
        suffix = []
        for path, rec in reversed(ordered):
            suffix.append((path, rec))
            if (rec.get("response") or {}).get("status") == "completed":
                break
        for path, rec in reversed(suffix):
            self.observe(path, rec)


def request_statistics(records, completed_tasks, config):
    types = Counter()
    for path, rec in records:
        if (rec.get("response") or {}).get("status") == "completed" and not rec.get("error"):
            kind = "completed_response"
        elif rec.get("http_status") == 429 and not upstream_overloaded(rec):
            kind = "unclassified_http_429"
        else:
            kind = failure_kind(rec, path.parent) or "nontechnical_or_unclassified"
        types[kind] += 1
    available = max(0, config["max_requests"] - len(records))
    missing = config["formal_tasks"] - completed_tasks
    return {"recorded_attempts": len(records), "attempt_types": dict(types),
            "remaining_tasks": missing, "remaining_request_slots": available,
            "minimum_request_shortfall": max(0, missing - available),
            "note": "Connection attempts are not evidence of billed generation; unknown usage remains reserved."}
