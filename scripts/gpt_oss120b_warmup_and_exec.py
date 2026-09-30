"""Verify the isolated final-channel adapter, then exec the frozen 150-task run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_gpt_oss_selfcheck import EXPECTED_MODEL, chat_response, parse_json, write_json, request_body, sha256_value
from scripts.start_gpt_oss120b_adapter import build_patch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--health-check", type=Path)
    parser.add_argument("--warmup-only", action="store_true", help="Load the model before engineering controls; never launch formal samples")
    parser.add_argument("--warmup-timeout", type=float, default=10800)
    args = parser.parse_args()
    if not args.warmup_only and any(x is None for x in (args.input, args.catalog, args.health_check)):
        parser.error("Formal execution requires --input, --catalog and --health-check")

    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    warmup = output / "warmup_preflight"
    suffix = 1
    while warmup.exists():
        warmup = output / f"warmup_preflight_{suffix:03d}"
        suffix += 1
    warmup.mkdir(parents=True, exist_ok=False)
    adapter = repo / "scripts/start_gpt_oss120b_adapter.py"
    patched, patch_sha = build_patch()
    launcher_sha = sha256_file(adapter)
    started = time.time()
    messages = [
        {"role": "system", "content": 'Return exactly one raw JSON object: {"diagnostics":[]}. No other text.'},
        {"role": "user", "content": 'Return exactly {"diagnostics":[]}.'},
    ]
    request = request_body(EXPECTED_MODEL, messages, 256)
    write_json(warmup / "preflight.json", {"status": "running", "request_sha256": sha256_value(request), "model": EXPECTED_MODEL,
        "adapter_sha256": patch_sha, "launcher_sha256": launcher_sha, "started_unix": started})
    try:
        client = httpx.Client(trust_env=False)
        response = chat_response(client, endpoint=args.api_base.rstrip("/") + "/chat/completions",
                                model=EXPECTED_MODEL, messages=messages, max_tokens=256,
                                timeout=args.warmup_timeout, raw_path=warmup / "response.sse",
                                response_path=warmup / "response.json", request_path=warmup / "request.json")
        parsed = parse_json(response["content"])
        if parsed != {"diagnostics": []}:
            raise ValueError("Warmup response did not return the required empty-diagnostics JSON")
        if response.get("finish_reasons") != ["stop"]:
            raise ValueError("Warmup response did not finish normally")
        write_json(warmup / "preflight.json", {"status": "passed", "model": EXPECTED_MODEL,
                   "adapter_sha256": patch_sha, "launcher_sha256": launcher_sha,
                   "response_sha256": sha256_file(warmup / "response.json"),
                   "elapsed_seconds": round(time.time() - started, 3), "counted_in_formal_150": False})
    except Exception as exc:
        write_json(warmup / "preflight.json", {"status": "failed", "model": EXPECTED_MODEL,
                   "adapter_sha256": patch_sha, "launcher_sha256": launcher_sha,
                   "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                   "elapsed_seconds": round(time.time() - started, 3), "formal_runner_started": False})
        raise

    if args.warmup_only:
        print(json.dumps({"preflight": "passed", "formal_runner_started": False, "next": "Run six engineering controls"}), flush=True)
        return

    command = [sys.executable, str(repo / "scripts/run_gpt_oss_selfcheck.py"),
               "--input", str(args.input.resolve()), "--catalog", str(args.catalog.resolve()),
               "--model-path", EXPECTED_MODEL, "--api-base", args.api_base,
               "--server-adapter-sha256", patch_sha,
               "--adapter-launcher-sha256", launcher_sha, "--output", str(output),
               "--health-check", str(args.health_check.resolve())]
    print(json.dumps({"preflight": "passed", "formal_runner_exec": command,
                      "output": str(output)}, ensure_ascii=False), flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
