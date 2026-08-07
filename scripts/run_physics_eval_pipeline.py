from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.semantic_rule_checker import SEMANTIC_RULE_CHECKER_PROMPT_VERSION
from core.unified_semantic_matcher import UNIFIED_SEMANTIC_MATCHER_PROMPT_VERSION
from scripts.experiment_manifest import (
    ExperimentManifest,
    capture_source_state,
    fingerprint_file,
    probe_python_runtime,
    resolve_python_executable,
    utc_now,
    validate_unified_catalog,
)


_ACTIVE_MANIFEST: Optional[ExperimentManifest] = None


def _run(cmd: str, *, allowed_returncodes: Sequence[int] = (0,)) -> int:
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode not in set(allowed_returncodes):
        raise SystemExit(f"Command failed ({res.returncode}): {cmd}")
    return int(res.returncode)


def _resolve_existing_dataset(outdir: Path, preferred: Path, pattern: str, label: str) -> Path:
    if preferred.exists():
        return preferred
    candidates: List[Path] = sorted(outdir.glob(pattern))
    if len(candidates) == 1:
        print(f"[INFO] {label} dataset auto-resolved to {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(str(x.name) for x in candidates[:8])
        raise SystemExit(
            f"{label} dataset not found at expected path {preferred}. Multiple candidates found: {names}. "
            f"Please specify matching recall/precision sizes or run without --skip-build."
        )
    raise SystemExit(
        f"{label} dataset not found at expected path {preferred}, and no files match pattern {pattern} in {outdir}."
    )


def _json_array_size(path: Path, *, label: str) -> int:
    if not path.exists():
        raise SystemExit(f"{label} dataset was not created: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{label} dataset must be a JSON array: {path}")
    return len(data)


def _require_output_files(paths: Sequence[Path], *, label: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{label} did not create required outputs: {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end independent error-level and question-level evaluation pipeline.")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Conda Python interpreter used for all child commands (defaults to the current interpreter).",
    )
    parser.add_argument("--input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--recall-input", type=str, default="data/evaluation_sample_1000_expansion.json")
    parser.add_argument("--precision-input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--recall-size", type=int, default=20)
    parser.add_argument("--precision-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--strong-model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--check-model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--max-errors", type=int, default=0, help="0 means exhaustive GT extraction mode.")
    parser.add_argument("--max-recall-scan", type=int, default=500)
    parser.add_argument("--min-valid-gt-per-sample", type=int, default=1)
    parser.add_argument(
        "--unified-catalog",
        type=str,
        required=True,
        help="Required unified_rules_v2 catalog JSON. No legacy fallback is allowed in this pipeline.",
    )
    parser.add_argument(
        "--no-symbolic-check",
        action="store_true",
        help="Disable the experience-code symbolic verification (on by default).",
    )
    parser.add_argument(
        "--experience-code-manifest",
        type=str,
        default="results/experience_symbolic_program_manifest_v2_unified.json",
        help="Forwarded to run_verifier.py.",
    )
    parser.add_argument(
        "--experience-code-module",
        type=str,
        default="symbolic.generated_experience_checks_v2_unified",
        help="Forwarded to run_verifier.py.",
    )
    parser.add_argument(
        "--symbolic-topic-check-limit",
        type=int,
        default=40,
        help="Forwarded to run_verifier.py.",
    )
    parser.add_argument(
        "--unified-rule-top-n",
        type=int,
        default=None,
        help="Forwarded to run_verifier.py (unified v2 rule pool width per sample).",
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
        help="Forwarded to run_verifier.py. Omit to use the environment/default adapter.",
    )
    parser.add_argument(
        "--semantic-json-attempts",
        type=int,
        default=None,
        metavar="N",
        help="Forwarded to run_verifier.py as total structured JSON attempts per stage.",
    )
    parser.add_argument(
        "--min-diagnostic-rule-score",
        type=float,
        default=None,
        help="Forwarded to run_verifier.py.",
    )
    # Legacy flag, kept for backward compatibility (no-op).
    parser.add_argument("--disable-agentic", action="store_true", help="(deprecated, no-op)")
    parser.add_argument(
        "--max-per-sample",
        type=int,
        default=12,
        help="Forwarded to run_verifier.py: cap published diagnostics per sample (<=0 disables). Default 12 improves precision on long rollouts.",
    )
    parser.add_argument(
        "--max-per-paragraph",
        type=int,
        default=2,
        help="Forwarded to run_verifier.py: cap diagnostics per paragraph (<=0 disables). Default 2 reduces redundant alarms.",
    )
    parser.add_argument("--run-quality-audit", action="store_true")
    parser.add_argument("--require-quality-pass", action="store_true")
    parser.add_argument("--min-locatable-ratio", type=float, default=0.70)
    parser.add_argument("--min-avg-errors", type=float, default=2.0)
    parser.add_argument("--max-generic-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default="results/eval_pipeline")
    parser.add_argument(
        "--run-kind",
        choices=["development", "validation", "final"],
        default="development",
        help="Validation/final runs require a clean Git worktree; development runs record and allow dirty state.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Portable experiment identifier stored in run_config.json.")
    parser.add_argument(
        "--manifest-output",
        type=str,
        default="",
        help="Manifest JSON path. Defaults to <output-dir>/run_config.json; frozen runs may use experiments/manifests/.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-error-eval", action="store_true")
    parser.add_argument("--skip-question-eval", action="store_true")
    parser.add_argument(
        "--verifier-progress-interval",
        type=int,
        default=10,
        metavar="N",
        help="Forwarded to run_verifier.py --progress-interval for each verifier pass (0 disables).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    global _ACTIVE_MANIFEST
    _ACTIVE_MANIFEST = None

    parser = build_parser()
    command_argv = list(sys.argv) if argv is None else [str(Path(__file__)), *list(argv)]
    args = parser.parse_args(argv)

    try:
        args.python = resolve_python_executable(args.python)
        runtime = probe_python_runtime(args.python, require_conda=True)
        catalog_fingerprint = validate_unified_catalog(args.unified_catalog, project_root=PROJECT_ROOT)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    error_dataset = outdir / f"error_eval_dataset_{args.recall_size}.json"
    question_dataset = outdir / f"question_eval_dataset_{args.recall_size}_{args.precision_size}.json"
    question_right_only_dataset = outdir / f"question_right_only_{args.precision_size}.json"

    if args.skip_build:
        error_dataset = _resolve_existing_dataset(outdir, error_dataset, "error_eval_dataset_*.json", "Error-level")
        question_dataset = _resolve_existing_dataset(outdir, question_dataset, "question_eval_dataset_*.json", "Question-level")

    error_results = outdir / "error_verifier_results.json"
    error_audit = outdir / "error_symbolic_audit.json"
    error_traces = outdir / "error_verifier_traces.json"
    question_results = outdir / "question_verifier_results.json"
    question_audit = outdir / "question_symbolic_audit.json"
    question_traces = outdir / "question_verifier_traces.json"
    error_metrics_output = outdir / "error_metrics.json"
    question_metrics_output = outdir / "question_metrics.json"
    quality_output = outdir / "error_quality_audit.json"

    py = shlex.quote(args.python)
    source_state = capture_source_state(PROJECT_ROOT)
    created_at = utc_now()
    run_stamp = "".join(character for character in created_at if character.isdigit())[:20]
    run_id = args.run_id.strip() or f"{args.run_kind}-{run_stamp}Z"
    manifest_path = Path(args.manifest_output) if args.manifest_output else outdir / "run_config.json"
    dataset_paths = {
        "error_dataset": error_dataset,
        "question_dataset": question_dataset,
        "question_right_only_dataset": question_right_only_dataset,
    }
    artifact_paths = {
        "error_results": error_results,
        "error_audit": error_audit,
        "error_traces": error_traces,
        "question_results": question_results,
        "question_audit": question_audit,
        "question_traces": question_traces,
        "error_metrics": error_metrics_output,
        "question_metrics": question_metrics_output,
        "quality_audit": quality_output,
    }
    run_config = {
        "run_id": run_id,
        "run_kind": args.run_kind,
        "created_at_utc": created_at,
        "argv": command_argv,
        "cwd": "." if Path.cwd().resolve() == PROJECT_ROOT else str(Path.cwd().resolve()),
        "python": args.python,
        "runtime": runtime,
        "source_state": source_state,
        "prompt_versions": {
            "semantic_rule_checker": SEMANTIC_RULE_CHECKER_PROMPT_VERSION,
            "unified_semantic_matcher": UNIFIED_SEMANTIC_MATCHER_PROMPT_VERSION,
        },
        "api_environment": {
            "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
            "OPENAI_BASE_URL_present": bool(os.getenv("OPENAI_BASE_URL")),
            "OPENAI_API_BASE_present": bool(os.getenv("OPENAI_API_BASE")),
        },
        "effective_config": dict(vars(args)),
        "seed": args.seed,
        "recall_size": args.recall_size,
        "precision_size": args.precision_size,
        "strong_model": args.strong_model,
        "check_model": args.check_model,
        "inputs": {
            "input": fingerprint_file(
                args.input,
                project_root=PROJECT_ROOT,
                required=not args.skip_build,
            ),
            "recall_input": fingerprint_file(
                args.recall_input,
                project_root=PROJECT_ROOT,
                required=not args.skip_build,
            ),
            "precision_input": fingerprint_file(
                args.precision_input,
                project_root=PROJECT_ROOT,
                required=not args.skip_build,
            ),
        },
        "unified_catalog": catalog_fingerprint,
        "no_symbolic_check": bool(args.no_symbolic_check),
        "llm_cache": False,
        "continue_on_semantic_error": True,
        "experience_code_manifest": fingerprint_file(
            args.experience_code_manifest,
            project_root=PROJECT_ROOT,
            required=not args.no_symbolic_check,
        ),
        "experience_code_module": args.experience_code_module,
        "symbolic_topic_check_limit": args.symbolic_topic_check_limit,
        "unified_rule_top_n": args.unified_rule_top_n,
        "semantic_output_adapter": args.semantic_output_adapter,
        "semantic_json_attempts": args.semantic_json_attempts,
        "min_diagnostic_rule_score": args.min_diagnostic_rule_score,
        "max_per_sample": args.max_per_sample,
        "max_per_paragraph": args.max_per_paragraph,
        "skip_build": bool(args.skip_build),
        "skip_run": bool(args.skip_run),
        "outputs": {name: str(path) for name, path in artifact_paths.items()},
    }
    _ACTIVE_MANIFEST = ExperimentManifest(
        manifest_path,
        run_config,
        project_root=PROJECT_ROOT,
        artifact_groups={"datasets": dataset_paths, "artifacts": artifact_paths},
    )
    _ACTIVE_MANIFEST.refresh_artifacts()
    _ACTIVE_MANIFEST.set_value("prepared_at_utc", utc_now())

    git_state = source_state.get("git") if isinstance(source_state, dict) else {}
    if args.run_kind in {"validation", "final"}:
        if not isinstance(git_state, dict) or not git_state.get("available"):
            raise RuntimeError(f"{args.run_kind} runs require an available Git repository state")
        if bool(git_state.get("dirty")):
            raise RuntimeError(
                f"{args.run_kind} runs require a clean Git worktree; use --run-kind development while iterating"
            )

    if not args.skip_build:
        _ACTIVE_MANIFEST.start_stage("build_datasets")
        _run(
            f"{py} scripts/build_physics_eval_sets.py "
            f"--input {shlex.quote(args.input)} "
            f"--recall-input {shlex.quote(args.recall_input)} "
            f"--precision-input {shlex.quote(args.precision_input)} "
            f"--error-output {shlex.quote(str(error_dataset))} "
            f"--question-output {shlex.quote(str(question_dataset))} "
            f"--precision-output {shlex.quote(str(question_right_only_dataset))} "
            f"--recall-size {args.recall_size} "
            f"--precision-size {args.precision_size} "
            f"--seed {args.seed} "
            f"--strong-model {shlex.quote(args.strong_model)} "
            f"--max-errors {args.max_errors} "
            f"--max-recall-scan {args.max_recall_scan} "
            f"--min-valid-gt-per-sample {args.min_valid_gt_per_sample}"
        )

        error_size = _json_array_size(error_dataset, label="Error-level")
        question_size = _json_array_size(question_dataset, label="Question-level")
        expected_question_size = int(args.recall_size) + int(args.precision_size)
        if error_size < int(args.recall_size):
            raise SystemExit(
                f"Evaluation dataset build shortfall: error rows {error_size}/{args.recall_size}. "
                "Inspect the annotation output before running the verifier."
            )
        if question_size < expected_question_size:
            raise SystemExit(
                f"Evaluation dataset build shortfall: question rows "
                f"{question_size}/{expected_question_size}. "
                "Inspect the annotation output before running the verifier."
            )
        _ACTIVE_MANIFEST.complete_stage("build_datasets")
    else:
        _ACTIVE_MANIFEST.complete_stage("reuse_frozen_datasets")

    if args.run_quality_audit:
        _ACTIVE_MANIFEST.start_stage("quality_audit")
        _run(
            f"{py} scripts/audit_eval_set_quality.py "
            f"--recall-dataset {shlex.quote(str(error_dataset))} "
            f"--output {shlex.quote(str(quality_output))} "
            f"--min-locatable-ratio {args.min_locatable_ratio} "
            f"--min-avg-errors {args.min_avg_errors} "
            f"--max-generic-ratio {args.max_generic_ratio}"
        )
        if args.require_quality_pass:
            quality = json.loads(quality_output.read_text(encoding="utf-8"))
            if not bool(quality.get("quality_gate_passed")):
                raise SystemExit(f"Quality gate failed: {quality.get('quality_gate_issues')}")
        _ACTIVE_MANIFEST.complete_stage("quality_audit")

    cap_parts: List[str] = []
    if int(args.max_per_sample) > 0:
        cap_parts.append(f"--max-per-sample {int(args.max_per_sample)}")
    else:
        cap_parts.append("--max-per-sample 0")
    if int(args.max_per_paragraph) > 0:
        cap_parts.append(f"--max-per-paragraph {int(args.max_per_paragraph)}")
    else:
        cap_parts.append("--max-per-paragraph 0")
    cap_flag = " ".join(cap_parts) + " "

    vf_extra = ""
    if args.unified_rule_top_n is not None:
        vf_extra += f" --unified-rule-top-n {int(args.unified_rule_top_n)}"
    if args.semantic_output_adapter is not None:
        vf_extra += f" --semantic-output-adapter {shlex.quote(args.semantic_output_adapter)}"
    if args.semantic_json_attempts is not None:
        vf_extra += f" --semantic-json-attempts {int(args.semantic_json_attempts)}"
    if args.min_diagnostic_rule_score is not None:
        vf_extra += f" --min-diagnostic-rule-score {float(args.min_diagnostic_rule_score)}"
    vf_extra += f" --progress-interval {max(0, int(args.verifier_progress_interval))}"
    vf_extra += " --continue-on-semantic-error --no-llm-cache"

    if not args.skip_run:
        catalog_flag = f"--unified-catalog {shlex.quote(args.unified_catalog)}"
        symbolic_flag_parts: List[str] = [
            f"--experience-code-manifest {shlex.quote(args.experience_code_manifest)}",
            f"--experience-code-module {shlex.quote(args.experience_code_module)}",
            f"--symbolic-topic-check-limit {int(args.symbolic_topic_check_limit)}",
        ]
        if args.no_symbolic_check:
            symbolic_flag_parts.append("--no-symbolic-check")
        symbolic_flag = " ".join(symbolic_flag_parts) + " "
        _ACTIVE_MANIFEST.start_stage("verify_error_dataset")
        error_verifier_exit = _run(
            f"{py} scripts/run_verifier.py "
            f"--input {shlex.quote(str(error_dataset))} "
            f"--output {shlex.quote(str(error_results))} "
            f"--symbolic-output {shlex.quote(str(error_audit))} "
            f"--full-output {shlex.quote(str(error_traces))} "
            f"--model {shlex.quote(args.check_model)} "
            + cap_flag
            + symbolic_flag
            + (catalog_flag + " " if catalog_flag else "")
            + vf_extra,
            allowed_returncodes=(0, 2),
        )
        _require_output_files(
            [error_results, error_audit, error_traces],
            label="Error-dataset verifier",
        )
        verifier_exit_codes = {"error_dataset": error_verifier_exit}
        _ACTIVE_MANIFEST.set_value("verifier_exit_codes", verifier_exit_codes)
        _ACTIVE_MANIFEST.complete_stage("verify_error_dataset")
        _ACTIVE_MANIFEST.start_stage("verify_question_dataset")
        question_verifier_exit = _run(
            f"{py} scripts/run_verifier.py "
            f"--input {shlex.quote(str(question_dataset))} "
            f"--output {shlex.quote(str(question_results))} "
            f"--symbolic-output {shlex.quote(str(question_audit))} "
            f"--full-output {shlex.quote(str(question_traces))} "
            f"--model {shlex.quote(args.check_model)} "
            + cap_flag
            + symbolic_flag
            + (catalog_flag + " " if catalog_flag else "")
            + vf_extra,
            allowed_returncodes=(0, 2),
        )
        _require_output_files(
            [question_results, question_audit, question_traces],
            label="Question-dataset verifier",
        )
        verifier_exit_codes["question_dataset"] = question_verifier_exit
        _ACTIVE_MANIFEST.set_value("verifier_exit_codes", verifier_exit_codes)
        _ACTIVE_MANIFEST.complete_stage("verify_question_dataset")
    if not args.skip_error_eval:
        _ACTIVE_MANIFEST.start_stage("evaluate_error_location")
        _run(
            f"{py} scripts/evaluate_physics_eval_sets.py "
            f"--dataset {shlex.quote(str(error_dataset))} "
            f"--results {shlex.quote(str(error_results))} "
            f"--audit {shlex.quote(str(error_audit))} "
            f"--output {shlex.quote(str(error_metrics_output))} "
            f"--match-mode location"
        )
        _require_output_files([error_metrics_output], label="Error-location evaluation")
        _ACTIVE_MANIFEST.complete_stage("evaluate_error_location")

    if not args.skip_question_eval:
        _ACTIVE_MANIFEST.start_stage("evaluate_question_level")
        _run(
            f"{py} scripts/evaluate_question_level_sets.py "
            f"--dataset {shlex.quote(str(question_dataset))} "
            f"--results {shlex.quote(str(question_results))} "
            f"--audit {shlex.quote(str(question_audit))} "
            f"--output {shlex.quote(str(question_metrics_output))}"
        )
        _require_output_files([question_metrics_output], label="Question-level evaluation")
        _ACTIVE_MANIFEST.complete_stage("evaluate_question_level")

    final_catalog = validate_unified_catalog(args.unified_catalog, project_root=PROJECT_ROOT)
    final_source_state = capture_source_state(PROJECT_ROOT)
    _ACTIVE_MANIFEST.set_value("source_state_at_completion", final_source_state)
    if final_catalog.get("sha256") != catalog_fingerprint.get("sha256"):
        raise RuntimeError("Unified catalog changed while the experiment was running")
    if (
        final_source_state.get("source_tree", {}).get("sha256")
        != source_state.get("source_tree", {}).get("sha256")
    ):
        raise RuntimeError("Runtime source files changed while the experiment was running")
    _ACTIVE_MANIFEST.complete()

    print("Done.")
    print(f"  Error dataset:          {error_dataset}")
    print(f"  Error verifier output:  {error_results}")
    print(f"  Error verifier traces:  {error_traces}")
    print(f"  Error metrics:          {error_metrics_output}")
    print(f"  Question dataset:       {question_dataset}")
    print(f"  Question verifier out:  {question_results}")
    print(f"  Question verifier trace:{question_traces}")
    print(f"  Question metrics:       {question_metrics_output}")
    if args.run_quality_audit:
        print(f"Quality audit: {quality_output}")
    print(f"Run manifest: {manifest_path}")


def cli(argv: Optional[Sequence[str]] = None) -> None:
    try:
        main(argv)
    except BaseException as exc:
        if _ACTIVE_MANIFEST is not None:
            _ACTIVE_MANIFEST.fail(exc)
        raise


if __name__ == "__main__":
    cli()
