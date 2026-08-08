import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# Ensure project root is on sys.path so `import core.*` works when this file is executed
# as a script from arbitrary working directories.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier


def _write_json_checkpoint(path: Path, payload: Any) -> None:
    """Atomically preserve completed samples during long retrieval runs."""
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _strip_symbolic_fields_from_diagnostic(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    out.pop("symbolic_cross_checks", None)
    out.pop("symbolic_reconciliation", None)
    return out


def _build_main_result(sample_result: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = []
    for d in sample_result.get("diagnostics", []) or []:
        if isinstance(d, dict):
            diagnostics.append(_strip_symbolic_fields_from_diagnostic(d))

    return {
        "id": sample_result.get("id"),
        "topic": sample_result.get("topic"),
        "verifier": sample_result.get("verifier"),
        "unified_retrieval_mode": sample_result.get("unified_retrieval_mode"),
        "semantic_min_publish_score": sample_result.get("semantic_min_publish_score"),
        "selection_strategy": sample_result.get("selection_strategy"),
        "retrieval_score_kind": sample_result.get("retrieval_score_kind"),
        "semantic_selection_error": sample_result.get("semantic_selection_error"),
        "semantic_failed_stage": sample_result.get("semantic_failed_stage"),
        "terminal_stage": sample_result.get("terminal_stage"),
        "empty_reason": sample_result.get("empty_reason"),
        "checker_gate_mode": sample_result.get("checker_gate_mode"),
        "checker_min_confidence": sample_result.get("checker_min_confidence"),
        "checker_status": sample_result.get("checker_status"),
        "checker_failure_count": len(sample_result.get("checker_failures") or []),
        "diagnostics": diagnostics,
        "score": sample_result.get("score"),
    }


def _print_semantic_retrieval_summary(raw_results: List[Dict[str, Any]]) -> None:
    semantic_results = [
        item
        for item in (raw_results or [])
        if isinstance(item, dict)
        and str(item.get("selection_strategy") or "").startswith("semantic_")
    ]
    if not semantic_results:
        return

    rule_hit_samples = len(
        [
            item
            for item in semantic_results
            if str(item.get("selection_strategy") or "") == "semantic_tree_selection"
            and bool(item.get("retrieved_rules"))
        ]
    )
    empty_without_rules = len(
        [
            item
            for item in semantic_results
            if str(item.get("selection_strategy") or "") == "semantic_tree_empty"
        ]
    )
    errors = len(
        [
            item
            for item in semantic_results
            if str(item.get("selection_strategy") or "") in {"semantic_error", "semantic_unavailable"}
        ]
    )
    print(
        "[PhysicsVerifier] semantic retrieval summary: "
        f"processed={len(semantic_results)}, rule_hit_samples={rule_hit_samples}, "
        f"empty_without_rules={empty_without_rules}, errors={errors}",
        flush=True,
    )
    if empty_without_rules:
        print(
            "[PhysicsVerifier] semantic_tree_empty is not a successful rule hit; "
            "inspect terminal_stage and empty_reason in the saved trace.",
            flush=True,
        )


def _build_symbolic_audit(sample_result: Dict[str, Any]) -> Dict[str, Any]:
    checked: List[Dict[str, Any]] = []
    for d in sample_result.get("diagnostics", []) or []:
        if not isinstance(d, dict):
            continue
        if d.get("symbolic_cross_checks") or d.get("symbolic_reconciliation"):
            recon = d.get("symbolic_reconciliation") or {}
            checked.append(
                {
                    "severity": d.get("severity"),
                    "rule": d.get("rule"),
                    "symbol": d.get("symbol"),
                    "message": d.get("message"),
                    "evidence": d.get("evidence"),
                    "rule_match": d.get("rule_match"),
                    "release_gate": d.get("release_gate"),
                    "symbolic_cross_checks": d.get("symbolic_cross_checks") or [],
                    "symbolic_reconciliation": d.get("symbolic_reconciliation"),
                    "symbolic_status": recon.get("status"),
                }
            )

    agentic = sample_result.get("agentic") if isinstance(sample_result.get("agentic"), dict) else {}

    symbolic_checks = []
    for sd in sample_result.get("symbolic_post_diagnostics", []) or []:
        if not isinstance(sd, dict):
            continue
        symbolic_checks.append(
            {
                "spec_id": sd.get("spec_id"),
                "primitive": sd.get("primitive"),
                "title": sd.get("title"),
                "result": sd.get("symbolic_result"),
                "rule": sd.get("rule"),
                "symbol": sd.get("symbol"),
                "message": sd.get("message"),
                "evidence": sd.get("evidence"),
                "details": sd.get("details"),
            }
        )

    summary = {
        "total": len(symbolic_checks),
        "pass": len([c for c in symbolic_checks if c.get("result") == "pass"]),
        "fail": len([c for c in symbolic_checks if c.get("result") == "fail"]),
        "inconclusive": len([c for c in symbolic_checks if c.get("result") == "inconclusive"]),
    }

    experience_checks = []
    for ed in sample_result.get("experience_post_diagnostics", []) or []:
        if not isinstance(ed, dict):
            continue
        experience_checks.append(
            {
                "rule": ed.get("rule"),
                "message": ed.get("message"),
                "evidence": ed.get("evidence"),
                "symbolic_cross_checks": ed.get("experience_symbolic_cross_checks") or [],
                "symbolic_reconciliation": ed.get("experience_symbolic_reconciliation"),
            }
        )

    experience_symbolic_checks = []
    for sd in sample_result.get("experience_symbolic_post_diagnostics", []) or []:
        if not isinstance(sd, dict):
            continue
        experience_symbolic_checks.append(
            {
                "spec_id": sd.get("spec_id"),
                "primitive": sd.get("primitive"),
                "title": sd.get("title"),
                "result": sd.get("symbolic_result"),
                "rule": sd.get("rule"),
                "symbol": sd.get("symbol"),
                "message": sd.get("message"),
                "evidence": sd.get("evidence"),
            }
        )

    experience_code_checks = []
    for cd in sample_result.get("experience_code_post_diagnostics", []) or []:
        if not isinstance(cd, dict):
            continue
        experience_code_checks.append(
            {
                "rule": cd.get("rule"),
                "rule_id": cd.get("rule_id"),
                "result": cd.get("result"),
                "symbolic_result": cd.get("symbolic_result"),
                "source": cd.get("source"),
                "bridge_for_rule_id": cd.get("bridge_for_rule_id"),
                "publish_skipped": cd.get("publish_skipped"),
                "message": cd.get("message"),
                "evidence": cd.get("evidence"),
            }
        )

    return {
        "id": sample_result.get("id"),
        "topic": sample_result.get("topic"),
        "checker_gate_mode": sample_result.get("checker_gate_mode"),
        "checker_status": sample_result.get("checker_status"),
        "candidate_diagnostic_count": len(sample_result.get("candidate_diagnostics") or []),
        "checked_diagnostics": checked,
        "symbolic_summary": summary,
        "symbolic_checks": symbolic_checks,
        "experience_checks": experience_checks,
        "experience_symbolic_checks": experience_symbolic_checks,
        "experience_code_checks": experience_code_checks,
        "suppressed_diagnostics": (agentic.get("suppressed_diagnostics") or []),
    }


def _write_batch_outputs(
    raw_results: List[Dict[str, Any]],
    *,
    output_path: Path,
    symbolic_output_path: Path,
    full_output_path: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbolic_output_path.parent.mkdir(parents=True, exist_ok=True)
    if full_output_path is not None:
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
    results = [_build_main_result(item) for item in raw_results if isinstance(item, dict)]
    symbolic_audit_all = [_build_symbolic_audit(item) for item in raw_results if isinstance(item, dict)]
    symbolic_audit = [
        item
        for item in symbolic_audit_all
        if (
            item.get("checked_diagnostics")
            or item.get("symbolic_checks")
            or item.get("experience_code_checks")
            or item.get("suppressed_diagnostics")
        )
    ]
    _write_json_checkpoint(output_path, results)
    _write_json_checkpoint(symbolic_output_path, symbolic_audit)
    if full_output_path is not None:
        _write_json_checkpoint(full_output_path, raw_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhysicsVerifier top-down checking.")
    parser.add_argument("--input", "-i", type=str, default="data/evaluation_sample_30.json")
    parser.add_argument("--output", "-o", type=str, default="results/top_down_results.json")
    parser.add_argument(
        "--symbolic-output",
        type=str,
        default="results/symbolic_audit.json",
        help="Write symbolic cross-check audit (checked diagnostics + symbolic results) to this file.",
    )
    parser.add_argument(
        "--full-output",
        type=str,
        default="",
        help="Optional path for full traces, including selected domains/topics/clusters/rules and gate details.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Run only the canonical unified-v2 semantic Domain/Topic/Cluster/Rule tree. "
            "Write the full retrieval trace to --output without invoking the Semantic Checker or symbolic checks."
        ),
    )
    parser.add_argument(
        "--continue-on-semantic-error",
        action="store_true",
        help=(
            "Continue processing remaining samples after semantic API/JSON failures. "
            "All partial traces are saved, but the command still exits non-zero when any failure occurred."
        ),
    )
    parser.add_argument("--catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b")
    parser.add_argument(
        "--no-symbolic-check",
        action="store_true",
        help="Disable the deterministic experience-code symbolic verification (on by default).",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--llm-cache",
        dest="enable_llm_cache",
        action="store_true",
        help="Enable the cross-run Semantic Checker disk cache (legacy default).",
    )
    cache_group.add_argument(
        "--no-llm-cache",
        dest="enable_llm_cache",
        action="store_false",
        help="Disable cross-run LLM cache reuse; required for controlled repeated experiments.",
    )
    parser.set_defaults(enable_llm_cache=True)
    parser.add_argument(
        "--unified-catalog",
        type=str,
        default=None,
        help=(
            "Path to a unified rules catalog JSON. An explicit missing path is fatal; "
            "the formal evaluation pipeline additionally requires unified_rules_v2."
        ),
    )
    parser.add_argument(
        "--experience-code-manifest",
        type=str,
        default="results/experience_symbolic_program_manifest_v2_unified.json",
        help="Manifest produced by scripts/generate_symbolic_checks.py mapping rule_id -> Python check function.",
    )
    parser.add_argument(
        "--experience-code-module",
        type=str,
        default="symbolic.generated_experience_checks_v2_unified",
        help="Python module path containing the generated experience-code check functions.",
    )
    parser.add_argument(
        "--symbolic-topic-check-limit",
        type=int,
        default=40,
        help="Max bottom-up experience-code checks to run per retrieved topic (<=0 disables the cap).",
    )
    # Legacy flags (accepted for backward compatibility, ignored).
    parser.add_argument("--no-agentic", action="store_true", help="(deprecated, no-op)")
    parser.add_argument("--agentic-max", type=int, default=2, help="(deprecated, no-op)")
    parser.add_argument(
        "--experience",
        action="store_true",
        help="(deprecated, no-op; experience-code symbolic check now runs by default)",
    )
    parser.add_argument(
        "--experience-rules",
        type=str,
        default=None,
        help="(deprecated, no-op; rule library is taken from --unified-catalog)",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default="strict",
        choices=["strict", "balanced", "score_only"],
        help="Diagnostic publication policy for unified v2 rules.",
    )
    parser.add_argument(
        "--min-diagnostic-rule-score",
        type=float,
        default=None,
        help="Lexical diagnostics only: override the legacy lexical rule-score threshold.",
    )
    parser.add_argument(
        "--semantic-min-publish-score",
        type=float,
        default=None,
        help="Semantic mode: minimum 0-1 API score for checker injection/publication; trace keeps lower scores.",
    )
    parser.add_argument(
        "--unified-rule-top-n",
        type=int,
        default=None,
        help="Unified v2: maximum rules to retrieve per sample for SRD injection (default 6).",
    )
    parser.add_argument(
        "--unified-retrieval-mode",
        choices=["semantic", "lexical"],
        default="semantic",
        help="Unified v2 retrieval path. Semantic API tree is the production default; lexical is diagnostics only.",
    )
    parser.add_argument(
        "--semantic-json-attempts",
        type=int,
        default=None,
        metavar="N",
        help="Total JSON-response attempts per semantic selection stage (initial request included; N >= 1).",
    )
    parser.add_argument(
        "--semantic-output-adapter",
        choices=[
            "openai_json_schema",
            "vllm_structured_outputs",
            "vllm_guided_json",
            "forced_tool_call",
        ],
        default=None,
        help=(
            "Structured-output transport for semantic retrieval. "
            "Defaults to UNIFIED_SEMANTIC_OUTPUT_ADAPTER or openai_json_schema."
        ),
    )
    parser.add_argument(
        "--checker-gate-mode",
        choices=["legacy", "dual_evidence", "dual_evidence_consistency"],
        default="legacy",
        help=(
            "Semantic Checker publication protocol. legacy preserves the historical single-quote arm; "
            "the two dual modes require source-isolated problem and prediction evidence."
        ),
    )
    parser.add_argument(
        "--checker-json-attempts",
        type=int,
        default=1,
        metavar="N",
        help="Total structured Checker JSON attempts per selected rule (initial request included; N >= 1).",
    )
    parser.add_argument(
        "--topic-skip-prediction",
        action="store_true",
        help="Lexical diagnostics only: exclude prediction text from topic scoring.",
    )
    parser.add_argument(
        "--max-per-sample",
        type=int,
        default=None,
        help="Maximum diagnostics published per sample (precision cap). Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-per-paragraph",
        type=int,
        default=None,
        help="Maximum diagnostics published per paragraph within a sample. Set <=0 to disable.",
    )
    parser.add_argument(
        "--quote-symbol-ratio",
        type=float,
        default=None,
        help="Minimum required-symbol overlap ratio in a diagnostic quote (strict mode only).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        metavar="N",
        help="Print a throughput summary every N completed samples; 0 disables.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="Atomically persist full-run outputs every N completed samples; 0 writes only at the end.",
    )
    parser.add_argument(
        "--verbose-per-sample",
        action="store_true",
        help="Print one line before each sample (very noisy; use for deep debugging).",
    )

    args = parser.parse_args()

    if args.semantic_json_attempts is not None and args.semantic_json_attempts < 1:
        parser.error("--semantic-json-attempts must be at least 1")
    if not 1 <= args.checker_json_attempts <= 5:
        parser.error("--checker-json-attempts must be between 1 and 5")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.unified_catalog and not Path(args.unified_catalog).is_file():
        parser.error(f"Unified rules catalog does not exist: {args.unified_catalog}")

    if args.topic_skip_prediction:
        os.environ["PHYSICSVERIFIER_TOPIC_SKIP_PREDICTION"] = "1"
        if args.unified_retrieval_mode == "semantic":
            print(
                "[PhysicsVerifier] --topic-skip-prediction is ignored in semantic mode; "
                "Domain/Topic/Cluster already use question+context only.",
                flush=True,
            )

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    n_samples = len(samples) if isinstance(samples, list) else 0
    print(
        f"[PhysicsVerifier] loaded {n_samples} samples from {args.input!r} | "
        f"progress every {max(0, int(args.progress_interval))} (0=off)",
        flush=True,
    )

    verifier = PhysicsRuleVerifier(
        rules_catalog_path=args.catalog,
        llm_model=args.model,
        enable_symbolic_check=not args.no_symbolic_check,
        enable_llm_cache=bool(args.enable_llm_cache),
        unified_rules_path=args.unified_catalog,
        experience_code_manifest_path=args.experience_code_manifest,
        experience_code_module=args.experience_code_module,
        symbolic_topic_check_limit=args.symbolic_topic_check_limit,
        precision_mode=args.precision_mode,
        min_diagnostic_rule_score=args.min_diagnostic_rule_score,
        max_diagnostics_per_sample=args.max_per_sample,
        max_diagnostics_per_paragraph=args.max_per_paragraph,
        quote_required_symbol_ratio=args.quote_symbol_ratio,
        unified_rule_top_n=args.unified_rule_top_n,
        unified_retrieval_mode=args.unified_retrieval_mode,
        semantic_min_publish_score=args.semantic_min_publish_score,
        semantic_json_attempts=args.semantic_json_attempts,
        semantic_output_adapter=args.semantic_output_adapter,
        checker_gate_mode=args.checker_gate_mode,
        checker_json_attempts=args.checker_json_attempts,
    )

    if args.retrieval_only:
        if not verifier._unified_v2_mode:
            parser.error("--retrieval-only requires a unified_rules_v2 catalog via --unified-catalog")
        if args.unified_retrieval_mode != "semantic":
            parser.error("--retrieval-only requires --unified-retrieval-mode semantic")

    if (
        verifier._unified_v2_mode
        and args.unified_retrieval_mode == "semantic"
        and (
            verifier.semantic_matcher is None
            or not bool(getattr(verifier.semantic_matcher, "available", False))
        )
    ):
        print(
            "Semantic unified retrieval is required but unavailable. Configure OPENAI_API_KEY, "
            "the model, and optionally OPENAI_BASE_URL/OPENAI_API_BASE.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.retrieval_only:
        raw_results = []
        progress_interval = max(0, int(args.progress_interval))
        for index, sample in enumerate(samples, start=1):
            if args.verbose_per_sample:
                print(f"Retrieving semantic tree for sample {sample.get('id')}...", flush=True)
            result = verifier.retrieve_unified_semantic_tree(sample)
            raw_results.append(result)
            _write_json_checkpoint(out_path, raw_results)
            if progress_interval > 0 and (index % progress_interval == 0 or index == n_samples):
                print(
                    f"[PhysicsVerifier] retrieval progress {index}/{n_samples} samples | "
                    f"last_id={sample.get('id')!r}",
                    flush=True,
                )
            if (
                not args.continue_on_semantic_error
                and str(result.get("selection_strategy") or "")
                in {"semantic_error", "semantic_unavailable"}
            ):
                break

        _write_json_checkpoint(out_path, raw_results)
        print(f"Semantic retrieval traces saved to {out_path}")
    else:
        sym_path = Path(args.symbolic_output)
        full_path = Path(args.full_output) if args.full_output else None
        checkpoint_every = max(0, int(args.checkpoint_every))

        def _checkpoint(rows: List[Dict[str, Any]]) -> None:
            if checkpoint_every <= 0 or len(rows) % checkpoint_every != 0:
                return
            _write_batch_outputs(
                rows,
                output_path=out_path,
                symbolic_output_path=sym_path,
                full_output_path=full_path,
            )

        raw_results = verifier.run_batch(
            samples,
            progress_interval=max(0, int(args.progress_interval)),
            verbose_per_sample=bool(args.verbose_per_sample),
            fail_fast_on_semantic_error=(
                args.unified_retrieval_mode == "semantic"
                and not args.continue_on_semantic_error
            ),
            on_result=_checkpoint,
        )
        _write_batch_outputs(
            raw_results or [],
            output_path=out_path,
            symbolic_output_path=sym_path,
            full_output_path=full_path,
        )

        print(f"Done. Results saved to {out_path}")
        print(f"Done. Symbolic audit saved to {sym_path}")
        if args.full_output:
            print(f"Done. Full raw results saved to {args.full_output}")

    _print_semantic_retrieval_summary(raw_results or [])

    semantic_failures = [
        item
        for item in (raw_results or [])
        if str(item.get("selection_strategy") or "") in {"semantic_error", "semantic_unavailable"}
    ]
    if semantic_failures:
        failed = semantic_failures[0]
        print(
            "Semantic retrieval failed; available results and traces were saved. "
            f"sample={failed.get('id')!r}, stage={failed.get('semantic_failed_stage')!r}, "
            f"error={failed.get('semantic_selection_error')!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    checker_failures = [
        item
        for item in (raw_results or [])
        if isinstance(item, dict) and bool(item.get("checker_failures"))
    ]
    if checker_failures:
        failed = checker_failures[0]
        print(
            "Semantic Checker had transport/parse/schema failures; outputs were saved and "
            "must be excluded from TN/FN scoring. "
            f"sample={failed.get('id')!r}, status={failed.get('checker_status')!r}",
            file=sys.stderr,
        )
        raise SystemExit(4)

    if args.retrieval_only and raw_results and not any(
        str(item.get("selection_strategy") or "") == "semantic_tree_selection"
        and bool(item.get("retrieved_rules"))
        for item in raw_results
        if isinstance(item, dict)
    ):
        print(
            "Semantic retrieval completed without any rule hit; traces were saved for diagnosis.",
            file=sys.stderr,
        )
        raise SystemExit(3)


if __name__ == "__main__":
    main()
