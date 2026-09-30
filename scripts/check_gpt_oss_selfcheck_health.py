"""Require positive/negative controls in all arms before a paid/formal batch.

This is an engineering check on synthetic problems, not a physics benchmark.
Uses the actual runner, frozen catalog, tools, parser and evidence locator.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from scripts.run_gpt_oss_selfcheck import (
    EXPECTED_MODEL, run_sample, sha256_file, sha256_value, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--server-adapter-sha256", required=True)
    args = parser.parse_args()
    if args.backend == "vllm":
        with httpx.Client(trust_env=False, timeout=5) as client:
            client.get(args.api_base.rstrip("/").removesuffix("/v1") + "/health").raise_for_status()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EXPECTED_MODEL, local_files_only=True)
    catalog = json.loads(args.catalog.read_text())
    rows = [
        {"id": "health_wrong_newton", "question": "A 2 kg body experiences a constant net force of 6 N. What is its acceleration?", "prediction": "Using Newton second law F=ma, the acceleration is a=F*m=12 m/s^2.\n<|endoftext|>"},
        {"id": "health_correct_newton", "question": "A 2 kg body experiences a constant net force of 6 N. What is its acceleration?", "prediction": "Using Newton second law F=ma, the acceleration is a=F/m=3 m/s^2.\n<|endoftext|>"},
    ]
    limits = {"max_requests_per_sample": 8, "max_tokens_per_response": 2048,
              "output_tokens_per_sample": 16384, "input_tokens_per_sample": 600000,
              "context_window": 131072, "wall_seconds_per_sample": 1800,
              "request_timeout_seconds": 1200, "tool_returned_chars": 500000}
    limits["backend"] = args.backend
    if args.backend == "vllm":
        limits["max_tokens_per_response"] = limits["output_tokens_per_sample"]
    config = {"backend": args.backend, "purpose": "synthetic engineering acceptance, not benchmark scores",
              "model": EXPECTED_MODEL, "rows": rows, "limits": limits,
              "api_base": args.api_base.rstrip("/"), "server_adapter_sha256": args.server_adapter_sha256,
              "catalog_sha256": sha256_file(args.catalog),
              "runner_sha256": sha256_file(Path(__file__).with_name("run_gpt_oss_selfcheck.py"))}
    signature = sha256_value(config)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("configuration_sha256") != signature:
        raise ValueError("Existing engineering evidence has a different configuration; preserve it and use a new run directory")
    write_json(manifest_path, {"configuration": config, "configuration_sha256": signature})
    checks = []
    with httpx.Client(trust_env=False) as client:
        for row in rows:
            for arm in ("model_only", "leaf_self_check", "tree_self_check"):
                result = run_sample(row, arm=arm, catalog=catalog, tokenizer=tokenizer,
                                    client=client, endpoint=args.api_base.rstrip("/") + "/chat/completions",
                                    model=EXPECTED_MODEL, root=args.output, signature=signature, limits=limits)
                diagnostics = result.get("diagnostics", [])
                valid = result.get("status") == "completed"
                quotes_grounded = all(d["evidence"]["quote"] in row["prediction"] for d in diagnostics)
                expected_found = bool(diagnostics) if row["id"] == "health_wrong_newton" else not diagnostics
                check = {"id": row["id"], "arm": arm, "status": result.get("status"),
                         "diagnostics": len(diagnostics), "quotes_grounded": quotes_grounded,
                         "passed": valid and quotes_grounded and expected_found}
                checks.append(check)
                write_json(args.output / "acceptance.json", {"checks": checks, "passed": len(checks) == 6 and all(c["passed"] for c in checks)})
                print(json.dumps(check), flush=True)
                if not valid:
                    raise SystemExit("Technical failure in engineering controls; inspect service before any further calls.")
    if not all(c["passed"] for c in checks):
        raise SystemExit("Engineering acceptance failed; do not launch the formal batch.")


if __name__ == "__main__":
    main()
