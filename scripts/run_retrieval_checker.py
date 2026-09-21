"""Replay frozen retrieval candidates through the unchanged Qwen verifier.

Development entry point for the four-arm comparison. --prepare-only validates
inputs and publication gates without contacting a model. Actual HTTP responses
are shared only when the entire request body, endpoint and model are identical.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from core.physics_rule_verifier import PhysicsRuleVerifier
from scripts.experiment_manifest import capture_source_state, sha256_file
from scripts.run_checker_replay import _catalog_index, _checker_status_succeeded, _validate_checker_result
from scripts.run_retrieval_comparison import QWEN, Transport, api_identity, digest, model_input, write_json

SOURCES = {1: "native_semantic_tree", 2: "native_semantic_tree", 3: "autonomous_tree_tools", 4: "autonomous_leaf_tools"}
POLICY = "existing_semantic_0_1_gates_with_truthful_candidate_source"


def job_key(sample_id):
    return hashlib.sha256(str(sample_id).encode()).hexdigest()[:16]


def read_frozen_jobs(runs, input_path, catalog_path, rows, arms):
    """Resolve explicitly supplied, hash-linked retries; never pick the best run."""
    jobs = {}
    manifests = {}
    api_models = set()
    for run in runs:
        run = Path(run).resolve()
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for key, path in (("input_sha256", input_path), ("catalog_sha256", catalog_path)):
            if manifest.get(key) != sha256_file(path):
                raise ValueError("Retrieval manifest mismatch: " + key)
        if manifest.get("phase") not in {"development", "retrieval_batch"}:
            raise ValueError("Only frozen retrieval or development outputs can be replayed")
        retry = manifest.get("retry_of")
        if retry and retry.get("manifest_sha256") not in manifests:
            raise ValueError("Supply the original run before its linked retry")
        manifests[sha256_file(manifest_path)] = manifest
        for row in rows:
            sid = str(row["id"])
            for arm in arms:
                path = run / f"arm{arm}" / job_key(sid) / "result.json"
                if not path.exists():
                    if retry and any((str(item.get("sample_id")), item.get("arm")) == (sid, arm) for item in retry.get("jobs", [])):
                        previous = jobs.get((sid, arm))
                        if previous:
                            jobs[(sid, arm)] = {**previous, "result": {**previous["result"], "status": "retry_pending"}}
                    continue
                value = json.loads(path.read_text())
                if (str(value.get("id")), value.get("arm")) != (sid, arm):
                    raise ValueError("Retrieval result identity mismatch")
                if value.get("configuration_sha256") != manifest.get("configuration_sha256"):
                    raise ValueError("Retrieval result configuration mismatch")
                if value.get("candidate_source") != SOURCES[arm]:
                    raise ValueError("Retrieval candidate source mismatch")
                model, effort = api_identity(manifest)
                if value.get("model") != (QWEN if arm == 1 else model):
                    raise ValueError("Retrieval model mismatch")
                if arm != 1:
                    if value.get("reasoning_effort", effort) != effort:
                        raise ValueError("Retrieval reasoning effort mismatch")
                    api_models.add((model, effort))
                    if len(api_models) > 1:
                        raise ValueError("Cannot mix API models or reasoning efforts in one comparison")
                previous = jobs.get((sid, arm))
                evidence = {"path": str(path), "sha256": sha256_file(path), "status": value.get("status")}
                links = retry.get("jobs", []) if retry else []
                link = next((item for item in links if (str(item.get("sample_id")), item.get("arm")) == (sid, arm)), {})
                if previous:
                    if previous["attempts"][-1]["status"] != "failed" or link.get("result_sha256") != previous["attempts"][-1]["sha256"]:
                        raise ValueError("Only a linked failed attempt can be superseded")
                elif retry and link.get("new_pending") is not True:
                    raise ValueError("Retry has no original result in the supplied runs")
                jobs[(sid, arm)] = {"result": value, "attempts": (previous["attempts"] if previous else []) + [evidence]}
    return jobs


def matcher_result(result, catalog, sample):
    """Restore catalog metadata on the operator side without inventing a tree search."""
    if result.get("status") != "completed":
        raise ValueError("Failed retrieval must not execute checker rules")
    selected = result.get("selected_rules")
    if not isinstance(selected, list) or len(selected) > 6:
        raise ValueError("Invalid frozen selection")
    index = _catalog_index(catalog)
    seen = set()
    for item in selected:
        rid, score = item.get("rule_id"), item.get("score")
        if rid not in index or rid in seen:
            raise ValueError("Unknown or duplicate frozen rule")
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Frozen score must be in [0,1]")
        seen.add(rid)
    if result["candidate_source"] == "native_semantic_tree":
        native = copy.deepcopy(result["native_result"])
        actual = native.get("selected_rules", [])
        if [(i["rule_id"], i["score"]) for i in actual] != [(i["rule_id"], i["score"]) for i in selected]:
            raise ValueError("Native result differs from the frozen selection")
        for item in actual:
            if item.get("rule_obj") != index[item["rule_id"]]["rule"]:
                raise ValueError("Native rule body differs from the frozen catalog")
        return native
    restored = []
    for item in selected:
        rid = item["rule_id"]
        quote = item.get("answer_quote")
        if rid not in result.get("read_rule_ids", []) or not isinstance(quote, str) or not quote.strip() or quote not in sample["prediction"]:
            raise ValueError("Autonomous selection lacks a read rule or exact quote")
        owner = index[rid]
        restored.append({"rule_id": rid, "score": item["score"], "rule_obj": owner["rule"], "domain": owner["domain_name"], "topic": owner["topic_name"], "topic_id": owner["topic_id"]})
    # No fabricated topic choices/scores, navigation or background analysis.
    # Astra's explanations and answer quotes never enter checker prompts.
    return {"selected_rules": restored, "input_policy": "full_raw_question_context_prediction", "terminal_stage": "autonomous_submission", "empty_reason": "" if restored else "autonomous_empty_selection"}


class FrozenVerifier(PhysicsRuleVerifier):
    """Use production verify(), SRD, checker, localization and release functions."""

    def __init__(self, *, result, sample, catalog_path, output, transport=None):
        catalog = json.loads(Path(catalog_path).read_text())
        frozen = matcher_result(result, catalog, sample)
        self.candidate_source = result["candidate_source"]
        self.frozen_sample_hash = digest(model_input(sample))
        matcher = SimpleNamespace(available=True, select_tree_semantically=lambda *_: copy.deepcopy(frozen))
        super().__init__(llm_model=None, log_dir=str(output), results_dir=str(output), unified_rules_path=str(catalog_path), enable_symbolic_check=False, enable_llm_cache=False, experience_code_manifest_path=str(Path(output) / "symbolic_disabled.json"), unified_retrieval_mode="semantic", unified_rule_top_n=6, semantic_min_publish_score=0.0, semantic_matcher=matcher, precision_mode="strict", max_diagnostics_per_sample=0, max_diagnostics_per_paragraph=0, quote_required_symbol_ratio=0.0, checker_gate_mode="legacy", checker_json_attempts=1, require_provider_identity=True, expected_provider_model=QWEN)
        self.semantic_checker.llm_model = QWEN
        self.semantic_checker._llm = transport
        self.semantic_checker.llm_temperature = 0.1
        self.semantic_checker.llm_max_output_tokens = 2048
        self.semantic_checker.llm_timeout_sec = 900
        self.semantic_checker.llm_trace_path = ""  # Raw HTTP records are sufficient.

    def _retrieve_unified_v2_semantic_tree(self, sample):
        if digest(model_input(sample)) != self.frozen_sample_hash:
            raise ValueError("Checker sample differs from frozen retrieval input")
        trace = super()._retrieve_unified_v2_semantic_tree(sample)
        if trace.get("semantic_selection_error"):
            raise ValueError(trace["semantic_selection_error"])
        if self.candidate_source != "native_semantic_tree":
            trace["selection_strategy"] = self.candidate_source
            for record in trace["selected_rule_records"]:
                record["retrieval_strategy"] = self.candidate_source
                record["publish_gate"]["selection_strategy"] = self.candidate_source
            for record in trace["retrieved_rules"]:
                record["publish_gate"]["selection_strategy"] = self.candidate_source
        return trace

    def _filter_low_confidence_unified_diagnostics(self, diagnostics, records):
        # Production dispatches the normalized-score policy by the old tree
        # strategy label. Translate only this function's internal argument;
        # persistent traces/release records retain the actual retrieval source.
        policy_records = []
        for record in records:
            if record.get("score_kind") != "semantic_0_1":
                raise ValueError("Only normalized semantic scores use this adapter")
            policy_records.append({**record, "retrieval_strategy": "semantic_tree_selection"})
        return super()._filter_low_confidence_unified_diagnostics(diagnostics, policy_records)


class SharedResponses(Transport):
    """Reuse the first identified HTTP response for an identical full request."""

    def __init__(self, *, response_index, **kwargs):
        super().__init__(**kwargs)
        self.index_path = Path(response_index)
        self.reused = []

    def post(self, endpoint, body, *, request_timeout=None):
        key = digest({"base": self.base, "endpoint": endpoint, "body": body})
        index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        if key in index:
            entry = index[key]
            source = self.index_path.parent / entry["path"]
            if sha256_file(source) != entry["sha256"]:
                raise ValueError("Frozen checker response hash mismatch")
            record = json.loads(source.read_text())
            data = record.get("response", {})
            if record.get("request") != body or record.get("endpoint") != endpoint or record.get("http_status") != 200 or data.get("model") != self.model or not data.get("id"):
                raise ValueError("Frozen checker response identity mismatch")
            self.reused.append({"request_sha256": key, **entry})
            write_json(self.output / "reused_responses.json", self.reused)
            return copy.deepcopy(data)
        data = super().post(endpoint, body, request_timeout=request_timeout)
        source = self.output / f"request_{self.requests:03d}.json"
        index[key] = {"path": str(source.relative_to(self.index_path.parent)), "sha256": sha256_file(source)}
        write_json(self.index_path, index)
        return data


def run_job(sample, result, *, catalog_path, output, transport):
    verifier = FrozenVerifier(result=result, sample=sample, catalog_path=catalog_path, output=output, transport=transport)
    trace = verifier.verify(model_input(sample))
    rule_ids = [r["rule_id"] for r in trace["retrieved_rules"] if r["publish_gate"]["publishable"] is True]
    checker_fields = {**trace, "checker_mode": trace["checker_gate_mode"], "checker_suppressed": trace["checker_suppressed_diagnostics"]}
    _, _, failures, _, status = _validate_checker_result(checker_fields, mode="legacy", rule_ids=rule_ids)
    return {"status": "completed" if _checker_status_succeeded(status, failures) else "failed", "checker_executed": bool(rule_ids), "trace": trace, "counts": {"selected_rules": len(result["selected_rules"]), "rules_passing_gate": len(rule_ids), "logical_rule_checks": len(trace["checker_decisions"]), "checker_diagnostics": len(trace["candidate_diagnostics"]), "released_diagnostics": len(trace["diagnostics"])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=ROOT / "catalogs/rules_unified_3000.json")
    parser.add_argument("--retrieval-run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--arms", type=int, choices=[1, 2, 3, 4], nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not os.getenv("CONDA_PREFIX"):
        parser.error("Run in the project conda environment")
    load_dotenv(args.env_file, override=False)
    rows = json.loads(args.input.read_text())
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    arms = list(dict.fromkeys(args.arms))
    jobs = read_frozen_jobs(args.retrieval_run, args.input, args.catalog, rows, arms)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for row in rows:
        for arm in arms:
            job = jobs.get((str(row["id"]), arm))
            entry = {"id": row["id"], "arm": arm, "retrieval_status": job["result"]["status"] if job else "missing", "attempts": job["attempts"] if job else []}
            if job and job["result"]["status"] == "completed":
                verifier = FrozenVerifier(result=job["result"], sample=model_input(row), catalog_path=args.catalog, output=output)
                trace = verifier._retrieve_unified_v2_semantic_tree(model_input(row))
                entry["selected_rule_ids"] = [r["rule_id"] for r in trace["retrieved_rules"]]
                entry["executable_rule_ids"] = [r["rule_id"] for r in trace["retrieved_rules"] if r["publish_gate"]["publishable"] is True]
                entry["rejected_rules"] = [{"rule_id": r["rule_id"], "gate": r["publish_gate"]} for r in trace["retrieved_rules"] if r["publish_gate"]["publishable"] is not True]
            prepared.append(entry)
    phases = {json.loads((run / "manifest.json").read_text())["phase"] for run in args.retrieval_run}
    if len(phases) != 1:
        raise ValueError("Do not mix development and batch retrieval evidence")
    source = capture_source_state(ROOT)
    config = {"phase": phases.pop(), "input_sha256": sha256_file(args.input), "catalog_sha256": sha256_file(args.catalog), "script_sha256": sha256_file(Path(__file__)), "transport_sha256": sha256_file(ROOT / "scripts/run_retrieval_comparison.py"), "source": source, "jobs": prepared, "model": QWEN, "candidate_policy": POLICY, "checker_mode": "legacy", "temperature": 0.1, "max_output_tokens": 2048, "request_timeout_seconds": 900, "disable_thinking_flag": os.getenv("OPENAI_DISABLE_THINKING", ""), "json_attempts": 1, "symbolic": False, "semantic_min_publish_score": 0.0, "max_diagnostics_per_sample": 0, "max_diagnostics_per_paragraph": 0, "quote_required_symbol_ratio": 0.0, "response_reuse": "first_response_for_identical_entire_http_request", "gt_reviewed": False}
    write_json(output / "preparation.json", {**config, "api_calls": 0, "note": "Input/gate preparation only; not an end-to-end Qwen result or physics score."})
    if args.prepare_only:
        print(f"Prepared {len(prepared)} jobs; no API calls. {sum(len(p.get('executable_rule_ids', [])) for p in prepared)} logical rule checks ready.")
        return
    if any(p["retrieval_status"] in {"missing", "retry_pending"} for p in prepared):
        parser.error("Requested retrieval jobs are missing; wait for completion or explicitly select available arms")
    base = os.getenv("QWEN_BASE_URL", "")
    if not base:
        parser.error("QWEN_BASE_URL is required; never substitute the Astra/group endpoint")
    config["endpoint_sha256"] = digest(base.rstrip("/"))
    signature = digest(config)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["configuration_sha256"] != signature:
        raise ValueError("Existing checker configuration differs")
    if not manifest.exists():
        write_json(manifest, {"configuration_sha256": signature, **config})
    for row in rows:
        for arm in arms:
            frozen = jobs[(str(row["id"]), arm)]
            directory = output / f"arm{arm}" / job_key(row["id"])
            target = directory / "result.json"
            if target.exists():
                continue
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError("Reconcile interrupted checker requests before resuming: " + str(directory))
            result = {"status": "not_run_retrieval_failed", "checker_executed": False}
            if frozen["result"]["status"] == "completed":
                transport = SharedResponses(base=base, key=os.getenv("QWEN_API_KEY", "EMPTY"), model=QWEN, output=directory, response_index=output / "response_index.json", limits={"requests": 6, "input_tokens": 600000, "output_tokens": 12288, "seconds": 1800, "request_timeout_seconds": 900})
                try:
                    result = run_job(model_input(row), frozen["result"], catalog_path=args.catalog, output=directory, transport=transport)
                except Exception as exc:
                    result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
                result.setdefault("checker_executed", transport.requests > 0 or bool(transport.reused))
                result["metrics"] = {**transport.summary(), "reused_responses": len(transport.reused)}
            result.update(id=row["id"], arm=arm, model=QWEN, candidate_source=SOURCES[arm], candidate_policy=POLICY, configuration_sha256=signature, retrieval_attempts=frozen["attempts"])
            write_json(target, result)
            print(f"{row['id']} arm{arm}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
