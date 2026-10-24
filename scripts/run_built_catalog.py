"""Same-context C0p/C4 comparison using a newly built, source-traced catalog.

Reuses the validated continuous executor and its native tool/quote protocol.
No independent checker, rule-selection handoff, embedding call or teacher at
runtime. Results use the frozen new reference and remain location proxies.
"""
from contextlib import ExitStack, contextmanager
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from scripts import run_continuous_context_v3 as execution
from scripts import resume_continuous_context_v3 as serialization
from scripts import rescore_frozen_bench as evaluation
from scripts import build_120b_rules as construction

runner = execution.runner
r = runner.r


def verified_catalog(build):
    release = json.loads((build / "release_manifest.json").read_text())
    if not release.get("usable_for_runtime") or release.get("human_gold") is not False:
        raise ValueError("Need a completed nonempty candidate catalog, never fabricated human gold")
    if not construction.RELEASE_FILES <= set(release.get("files", {})):
        raise ValueError("Build release omits required source/isolation artifacts")
    if release.get("status") != "candidate_catalog_complete_not_default_release":
        raise ValueError("Build release is not completed")
    for name, sha in release["files"].items():
        path = (build / name).resolve()
        if not path.is_relative_to(build.resolve()) or r.sha256_file(path) != sha:
            raise ValueError("Construction evidence hash/path mismatch")
    # A checksum alone cannot establish separation. Replay the exclusion and
    # seeded selection using the mandatory current benchmark cohort union.
    plan = construction.verify(build)
    if plan["configuration_sha256"] != release["configuration_sha256"]:
        raise ValueError("Build release/plan mismatch")
    config = plan["configuration"]
    isolation = release.get("data_isolation", {})
    if (isolation.get("passed") is not True or isolation.get("selected_overlap_count") != 0 or
            isolation.get("bench_questions_sha256") != config["bench_questions_sha256"] or
            isolation.get("isolation_sha256") != config["isolation_sha256"] or
            isolation.get("policy_version") != construction.p.ISOLATION_POLICY["version"] or
            release.get("online_model") != r.EXPECTED_MODEL or
            release.get("model_roles") != construction.model_roles(config) or
            "external_teacher" not in release or release["external_teacher"] is not None or
            release.get("source_prediction_model") != config["source_prediction_model"] or
            release.get("source_model_evidence") != config["source_model_evidence"]):
        raise ValueError("Build isolation or model role contract is missing/inconsistent")
    if config.get('corpus_release') and not {'source_release.json', 'source_audits.json'} <= set(release['files']):
        raise ValueError('Build release omits the fresh corpus handoff')
    if any((build / name).exists() and name not in release['files']
           for name, _, _ in construction.RECOVERIES.values()):
        raise ValueError('Build release omits its structural normalization recovery')
    rows = json.loads((build / "inputs.json").read_text())
    formal = json.loads(runner.INPUT.read_text())
    _, overlap = construction.p.isolate_sources(rows, formal)
    if overlap:
        raise ValueError("Build sources overlap actual evaluation questions")
    return build / "catalog.json", release


def trace_summary(items):
    """Report actual lookup behavior and measured usage, not just equal caps."""
    counts = {"records_with_trace": 0, "model_requests": 0, "usage_known_responses": 0,
              "usage_unknown_responses": 0, "known_input_tokens": 0, "known_output_tokens": 0,
              "samples_with_search": 0, "samples_with_active_tree_navigation": 0,
              "samples_with_full_rule_read": 0}
    for item in items:
        trace = item.get("trace")
        if trace is None:
            continue
        counts["records_with_trace"] += 1
        requests = trace["model_requests"]
        counts["model_requests"] += requests
        known = 0
        for response in trace.get("response_evidence", []):
            usage = response.get("usage") or {}
            incoming = usage.get("prompt_tokens", usage.get("input_tokens"))
            outgoing = usage.get("completion_tokens", usage.get("output_tokens"))
            if type(incoming) is int and type(outgoing) is int and incoming >= 0 and outgoing >= 0:
                known += 1
                counts["known_input_tokens"] += incoming
                counts["known_output_tokens"] += outgoing
        if known > requests:
            raise ValueError("Duplicate response evidence exceeds request count")
        counts["usage_known_responses"] += known
        counts["usage_unknown_responses"] += requests - known
        events = trace.get("tool_events", [])
        counts["samples_with_search"] += any(e.get("tool") == "search_rules" for e in events)
        counts["samples_with_active_tree_navigation"] += any(
            e.get("tool") == "read_tree" and e.get("arguments", {}).get("node_id") not in {None, "root"}
            for e in events)
        counts["samples_with_full_rule_read"] += bool(trace.get("read_rule_ids"))
    return dict(counts, records_missing_trace=len(items) - counts["records_with_trace"],
                output_tokens_include_reasoning=True, equal_budget_caps_not_equal_actual_compute=True)


def score(output):
    plan = runner.validate_plan(output)
    rows = runner.frozen_rows()
    groups = runner.collect(output / "formal", rows, plan["configuration_sha256"])
    _, dataset, unknown = evaluation.reference_data(runner.ROOT)
    scored = {}
    for arm, items in groups.items():
        if len(items) != 50:
            continue  # Report unfinished, never turn missing calls into empty predictions.
        predictions = [dict(x, execution_failure=not runner.healthy_result(x), source_status=x["status"],
                            protocol_noncompliance=not x.get("protocol_audit", {}).get("passed", False)) for x in items]
        scored[arm] = evaluation.score(dataset, predictions, unknown)
    report = {"metric": "frozen_new_reference_location_proxy_not_semantic_accuracy", "human_gold": False,
              "independent_test": False, "reference_version": evaluation.RELEASE_VERSION,
              "configuration_sha256": plan["configuration_sha256"], "scores_all_50": scored,
              "completed_records": {a: len(v) for a, v in groups.items()},
              "resource_and_library_use": {a: trace_summary(v) for a, v in groups.items()},
              "unscored_arms": [a for a in runner.protocol.ARMS if a not in scored]}
    if set(scored) == {"C0p", "C4"}:
        report["rule_minus_pure_paired_bootstrap"] = evaluation.bootstrap(scored["C4"], scored["C0p"])
    r.write_json(output / "metrics_new_reference_location_proxy.json", report)
    return report


@contextmanager
def configured(build):
    catalog, release = verified_catalog(build)
    original = runner.configuration
    # Reuse historical generation code without modifying its frozen defaults.
    def configuration():
        # Missing/drifted labels must stop BEFORE generation, not after 100 jobs.
        evaluation.reference_data(runner.ROOT)
        config = original()
        config.update(build_release_sha256=r.sha256_file(build / "release_manifest.json"),
            build_configuration_sha256=release["configuration_sha256"],
            reference_version=evaluation.RELEASE_VERSION,
            scoring_source_sha256={n: r.sha256_file(runner.ROOT / n) for n in (evaluation.RELEASE, evaluation.REFERENCE)},
            score="new reference location proxy; physical correctness not established by overlap")
        return config
    with execution.configured(), ExitStack() as stack:
        stack.enter_context(patch.object(runner, "CATALOG", catalog))
        stack.enter_context(patch.object(r, "EXPECTED_CATALOG_SHA256", r.sha256_file(catalog)))
        stack.enter_context(patch.object(runner, "configuration", configuration))
        stack.enter_context(patch.object(runner, "score", score))
        stack.enter_context(patch.object(execution.transport, "decode_response", serialization.decode_response))
        stack.enter_context(patch.object(runner, "SOURCE_FILES", [*runner.SOURCE_FILES,
            "scripts/run_built_catalog.py", "scripts/resume_continuous_context_v3.py",
            "scripts/rescore_frozen_bench.py", "scripts/evaluate_direct_baseline_comparison.py",
            "scripts/build_120b_rules.py", "rule_framework/oss120b_construction.py"]))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "confirm-detached", "score"))
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with configured(args.build.resolve()):
        {"prepare": runner.prepare, "run": runner.run_batch,
         "confirm-detached": runner.confirm_detached, "score": score}[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
