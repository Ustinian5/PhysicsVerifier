"""Continue independent frozen samples after a reviewed, preserved sample failure.

This controller imports the exact runner named by the original manifest. It does
not retry any task with a saved result or alter prompts, limits, model identity,
input, catalog, or the original configuration signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gpt_oss_selfcheck as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    args = parser.parse_args()
    input_path, catalog_path, output = args.input.resolve(), args.catalog.resolve(), args.output.resolve()
    manifest_path, progress_path = output / "manifest.json", output / "progress.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["configuration"]
    signature = manifest["configuration_sha256"]
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("state") != "paused_after_sample_failure":
        raise RuntimeError("Only a paused run with a reviewed failure can be continued")
    if config.get("script_sha256") != runner.sha256_file(Path(runner.__file__)):
        raise RuntimeError("Original runner source no longer matches the prepared run")
    if config.get("input_sha256") != runner.sha256_file(input_path) or config.get("catalog_sha256") != runner.sha256_file(catalog_path):
        raise RuntimeError("Frozen input/catalog changed since the paused run")
    if config.get("local_model_path") != runner.EXPECTED_MODEL or config.get("api_base") != args.api_base:
        raise RuntimeError("Model or service endpoint differs from the original configuration")
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if len(rows) != 50 or len(runner.Corpus(catalog, tree=True).leaves) != 1123:
        raise RuntimeError("Frozen sample/catalog dimensions changed")
    limits = config["per_sample_limits"]
    order = [(str(row["id"]), arm) for row in rows for arm in ("model_only", "leaf_self_check", "tree_self_check")]
    random.Random(config["execution_order_seed"]).shuffle(order)
    if runner.sha256_value(order) != config["execution_order_sha256"]:
        raise RuntimeError("Reconstructed execution order differs from the frozen manifest")

    failed_preserved = []
    existing = 0
    for sid, arm in order:
        result_path = output / "samples" / arm / hashlib.sha256(sid.encode()).hexdigest()[:16] / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("configuration_sha256") != signature or str(result.get("id")) != sid or result.get("verifier") != arm:
            raise RuntimeError(f"Existing sample identity/config mismatch: {sid}/{arm}")
        for item in result.get("trace", {}).get("response_evidence", []):
            for path_key, hash_key in (("request_path", "request_sha256"), ("raw_sse_path", "raw_sse_sha256"), ("response_path", "response_sha256")):
                artifact = Path(item[path_key])
                if not artifact.is_file() or runner.sha256_file(artifact) != item[hash_key]:
                    raise RuntimeError(f"Saved response evidence hash mismatch: {artifact}")
        existing += 1
        if result.get("status") != "completed":
            failed_preserved.append({"sample_id": sid, "arm": arm,
                                     "failure_reason": result.get("failure_reason", "unknown"),
                                     "result_sha256": runner.sha256_file(result_path)})
    if not failed_preserved:
        raise RuntimeError("Paused run has no preserved failed sample")
    if existing != progress.get("processed"):
        raise RuntimeError("Processed count does not match persisted per-sample results")

    continuation = {"status": "running", "started_unix": time.time(),
                    "controller_sha256": runner.sha256_file(Path(__file__)),
                    "original_runner_sha256": config["script_sha256"],
                    "original_configuration_sha256": signature,
                    "failure_policy": "preserve every existing result; never replay a saved failure; continue independent unsaved samples; score only if all 150 outputs conform",
                    "preserved_failures": failed_preserved, "existing_results_reused_without_calls": existing,
                    "api_base": args.api_base}
    runner.write_json(output / "continuation_policy.json", continuation)
    manifest["phase"] = "running_after_reviewed_failure"
    manifest["continuation_controller_sha256"] = continuation["controller_sha256"]
    manifest["previous_pause"] = {"reason": progress.get("failure_reason"), "sample_id": progress.get("last_sample_id"),
                                   "arm": progress.get("last_arm"), "processed": progress.get("processed")}
    runner.write_json(manifest_path, manifest)

    row_by_id = {str(row["id"]): row for row in rows}
    results: dict[tuple[str, str], dict] = {}
    with runner.run_lock(output / "active.lock"), httpx.Client(trust_env=False) as client:
        (output / "runner.pid").write_text(str(__import__("os").getpid()) + "\n", encoding="utf-8")
        for sid, arm in order:
            row = row_by_id[sid]
            result_path = output / "samples" / arm / hashlib.sha256(sid.encode()).hexdigest()[:16] / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = runner.run_sample(row, arm=arm, catalog=catalog, tokenizer=tokenizer,
                    client=client, endpoint=args.api_base.rstrip("/") + "/chat/completions",
                    model=runner.EXPECTED_MODEL, root=output, signature=signature, limits=limits)
            results[(sid, arm)] = result
            completed = sum(x.get("status") == "completed" for x in results.values())
            failed = sum(x.get("status") != "completed" for x in results.values())
            processed = len(results)
            failure_rows = [{"sample_id": key[0], "arm": key[1], "failure_reason": value.get("failure_reason", "unknown")}
                            for key, value in results.items() if value.get("status") != "completed"]
            runner.write_json(progress_path, {"state": "running_after_reviewed_failure", "planned": 150,
                "completed": completed, "failed": failed, "processed": processed,
                "last_sample_id": sid, "last_arm": arm, "failures": failure_rows, "updated_unix": time.time()})
            print(sid, arm, result.get("status"), result.get("failure_reason", ""), flush=True)

        for arm in ("model_only", "leaf_self_check", "tree_self_check"):
            group = [results[(str(row["id"]), arm)] for row in rows]
            runner.write_json(output / arm / "results.json", group)
        completed = sum(x.get("status") == "completed" for x in results.values())
        failed = 150 - completed
        state = "finished" if failed == 0 else "finished_with_failures"
        runner.write_json(progress_path, {"state": state, "planned": 150, "completed": completed,
            "failed": failed, "processed": 150,
            "failures": [{"sample_id": key[0], "arm": key[1], "failure_reason": value.get("failure_reason", "unknown")}
                         for key, value in results.items() if value.get("status") != "completed"],
            "updated_unix": time.time()})
        manifest["phase"] = state
        manifest["finished_unix"] = time.time()
        runner.write_json(manifest_path, manifest)
        continuation["status"] = state
        continuation["finished_unix"] = time.time()
        continuation["final_completed"] = completed
        continuation["final_failed"] = failed
        runner.write_json(output / "continuation_policy.json", continuation)
        if completed == 150:
            metric = runner.score_completed(input_path.parent.parent, output)
            print("scored", {arm: value["summary"] for arm, value in metric["scores_all_50"].items()}, flush=True)
        else:
            print("completed with preserved per-sample failures; no all-50 score written", flush=True)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load a tokenizer only on the GPU-free runner process. Inference stays on
    # the separately managed service bound to the experiment's personal GPU.
    tokenizer = AutoTokenizer.from_pretrained(runner.EXPECTED_MODEL, local_files_only=True, trust_remote_code=False)
    main()
