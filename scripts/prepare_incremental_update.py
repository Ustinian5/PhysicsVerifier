from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unified_catalog import _resolve_distilled_topic
from scripts.prepare_incremental_candidates import prepare_incremental_candidates
from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster
from rule_framework.incremental_validation import (
    incremental_manifest_configuration_sha256,
    validate_change_policy,
)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _input_record(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required incremental input not found: {path}")
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _command(parts: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part).replace("\\", "/")) for part in parts)


def _canonicalize_candidate_affected_topics(
    merge_report: Dict[str, Any],
    base_catalog: Dict[str, Any],
) -> Dict[str, Any]:
    catalog_topics = {
        (
            str(domain.get("name") or domain.get("id") or "").strip().casefold(),
            str(topic.get("name") or topic.get("id") or "").strip().casefold(),
        ): {
            "domain": str(
                domain.get("name") or domain.get("id") or ""
            ).strip(),
            "topic": str(topic.get("name") or topic.get("id") or "").strip(),
        }
        for domain in base_catalog.get("domains", []) or []
        if isinstance(domain, dict)
        for topic in domain.get("topics", []) or []
        if isinstance(topic, dict)
    }
    canonical: Dict[tuple[str, str], Dict[str, str]] = {}
    for item in merge_report.get("affected_topics", []) or []:
        if not isinstance(item, dict):
            continue
        domain, topic = _resolve_distilled_topic(
            str(item.get("domain") or ""),
            str(item.get("topic") or ""),
        )
        key = (domain.casefold(), topic.casefold())
        matched = catalog_topics.get(key)
        if matched is None:
            raise ValueError(
                "Candidate affected Topic does not map to the frozen base catalog: "
                f"{item.get('domain')}::{item.get('topic')} -> {domain}::{topic}"
            )
        canonical[key] = matched
    merge_report["affected_topics"] = [
        canonical[key] for key in sorted(canonical)
    ]
    summary = dict(merge_report.get("summary") or {})
    summary["affected_topic_count"] = len(canonical)
    merge_report["summary"] = summary
    return merge_report


def prepare_incremental_update(
    *,
    new_candidates_path: Path,
    workspace: Path,
    current_candidates_path: Path,
    current_generalized_path: Path,
    current_formal_path: Path,
    current_cluster_proposals_path: Path,
    current_catalog_path: Path,
    knowledge_path: Path,
    tagged_path: Path,
    candidate_embedding_cache_path: Path,
    formal_embedding_cache_path: Path,
    reset_seed_outputs: bool = False,
    change_mode: str = "additive",
    max_added_formal_rules: int | None = None,
    max_existing_rule_cluster_moves: int = 0,
    max_existing_rule_group_moves: int = 0,
    max_added_clusters: int | None = None,
    max_removed_clusters: int = 0,
    max_changed_cluster_definitions: int = 0,
    max_changed_topics: int | None = None,
    max_added_generalized_rules: int | None = None,
    max_removed_generalized_rules: int = 0,
    max_modified_generalized_rules: int = 0,
    allowed_recluster_topics: Sequence[Dict[str, str]] = (),
) -> Dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    current_catalog = _load_json(current_catalog_path)
    merged_path = workspace / "semantic_experience_distilled.json"
    merge_report_path = workspace / "incremental_merge_report.json"
    candidate_for_cluster_path = (
        workspace / "semantic_experience_distilled_for_cluster.json"
    )
    candidate_catalog_path = workspace / "candidate_catalog.json"
    candidate_report_path = workspace / "precluster_report.json"
    candidate_embedding_input_path = workspace / "rule_embedding_input.json"
    generalized_path = workspace / "semantic_experience_generalized.json"
    cluster_proposals_path = workspace / "cluster_proposals.json"

    merged, merge_report = prepare_incremental_candidates(
        current_payload=_load_json(current_candidates_path),
        new_payload=_load_json(new_candidates_path),
        formal_payload=_load_json(current_formal_path),
    )
    _canonicalize_candidate_affected_topics(merge_report, current_catalog)
    _write_json(merged_path, merged)
    _write_json(merge_report_path, merge_report)

    prepare_rules_for_cluster(
        distilled_input=merged_path,
        knowledge_path=knowledge_path,
        tagged_path=tagged_path,
        baseline_catalog_path=None,
        distilled_output=candidate_for_cluster_path,
        catalog_output=candidate_catalog_path,
        report_output=candidate_report_path,
        embedding_input_output=candidate_embedding_input_path,
        scenario_cluster_blueprints_paths=[],
    )

    for source, target in (
        (current_generalized_path, generalized_path),
        (current_cluster_proposals_path, cluster_proposals_path),
    ):
        if reset_seed_outputs or not target.exists():
            _write_json(target, _load_json(source))

    candidate_clusters_path = workspace / "rule_embedding_clusters.json"
    formal_path = workspace / "semantic_experience_generalized_for_cluster.json"
    precluster_catalog_path = workspace / "catalog_precluster.json"
    formal_report_path = workspace / "formal_precluster_report.json"
    formal_embedding_input_path = workspace / "formal_rule_embedding_input.json"
    formal_clusters_path = workspace / "formal_rule_embedding_clusters.json"
    incremental_manifest_path = workspace / "incremental_manifest.json"
    formal_seed_catalog_path = current_catalog_path

    commands: List[Dict[str, Any]] = [
        {
            "step": "candidate_embedding",
            "calls_api": True,
            "command": _command(
                [
                    "conda",
                    "run",
                    "-n",
                    "physicsverifier",
                    "python",
                    "scripts/run_rule_embedding_clustering.py",
                    "--input",
                    candidate_embedding_input_path,
                    "--output",
                    candidate_clusters_path,
                    "--cache",
                    candidate_embedding_cache_path,
                    "--embedding-model",
                    "text-embedding-3-large",
                    "--similarity-threshold",
                    "0.74",
                    "--min-cluster-size",
                    "4",
                    "--resume",
                    "--incremental-manifest",
                    incremental_manifest_path,
                    "--incremental-stage",
                    "candidate_embedding",
                ]
            ),
        },
        {
            "step": "candidate_generalization",
            "calls_api": True,
            "command": _command(
                [
                    "conda",
                    "run",
                    "-n",
                    "physicsverifier",
                    "python",
                    "scripts/generalize_experience_candidates.py",
                    "--candidates",
                    candidate_for_cluster_path,
                    "--clusters",
                    candidate_clusters_path,
                    "--output",
                    generalized_path,
                    "--model",
                    "deepseek-v4-flash-nothinking",
                    "--fallback-model",
                    "gemini-2.5-flash-nothinking",
                    "--max-clusters",
                    "0",
                    "--max-candidates-per-batch",
                    "12",
                    "--request-timeout",
                    "120",
                    "--attempts",
                    "2",
                    "--min-source-candidates",
                    "2",
                    "--min-source-samples",
                    "2",
                    "--max-tokens",
                    "4000",
                    "--incremental-manifest",
                    incremental_manifest_path,
                    "--resume",
                    "--continue-on-error",
                ]
            ),
        },
        {
            "step": "prepare_formal_rules",
            "calls_api": False,
            "command": _command(
                [
                    "conda",
                    "run",
                    "-n",
                    "physicsverifier",
                    "python",
                    "scripts/prepare_rules_for_cluster.py",
                    "--distilled-input",
                    generalized_path,
                    "--knowledge",
                    knowledge_path,
                    "--tagged",
                    tagged_path,
                    "--baseline-catalog",
                    formal_seed_catalog_path,
                    "--preserve-baseline-rule-ids",
                    "--distilled-output",
                    formal_path,
                    "--catalog-output",
                    precluster_catalog_path,
                    "--report-output",
                    formal_report_path,
                    "--embedding-input-output",
                    formal_embedding_input_path,
                    "--no-scenario-cluster-blueprints",
                    "--incremental-manifest",
                    incremental_manifest_path,
                ]
            ),
        },
        {
            "step": "formal_embedding",
            "calls_api": True,
            "command": _command(
                [
                    "conda",
                    "run",
                    "-n",
                    "physicsverifier",
                    "python",
                    "scripts/run_rule_embedding_clustering.py",
                    "--input",
                    formal_embedding_input_path,
                    "--output",
                    formal_clusters_path,
                    "--cache",
                    formal_embedding_cache_path,
                    "--embedding-model",
                    "text-embedding-3-large",
                    "--similarity-threshold",
                    "0.72",
                    "--min-cluster-size",
                    "4",
                    "--resume",
                    "--incremental-manifest",
                    incremental_manifest_path,
                    "--incremental-stage",
                    "formal_embedding",
                ]
            ),
        },
        {
            "step": "cluster_labeling",
            "calls_api": True,
            "command": _command(
                [
                    "conda",
                    "run",
                    "-n",
                    "physicsverifier",
                    "python",
                    "scripts/generate_cluster_proposals.py",
                    "--catalog",
                    precluster_catalog_path,
                    "--embedding-clusters",
                    formal_clusters_path,
                    "--rule-input",
                    formal_embedding_input_path,
                    "--output",
                    cluster_proposals_path,
                    "--model",
                    "deepseek-v4-flash-nothinking",
                    "--max-topics",
                    "0",
                    "--min-rule-count",
                    "1",
                    "--request-timeout",
                    "180",
                    "--temperature",
                    "0",
                    "--max-rules-per-cluster",
                    "8",
                    "--max-output-tokens",
                    "8192",
                    "--incremental-manifest",
                    incremental_manifest_path,
                    "--resume",
                    "--continue-on-error",
                ]
            ),
        },
        {
            "step": "finalize_and_validate",
            "calls_api": False,
            "command": _command(
                [
                    "conda",
                    "run",
                    "-n",
                    "physicsverifier",
                    "python",
                    "scripts/finalize_incremental_update.py",
                    "--workspace",
                    workspace,
                    "--base-catalog",
                    current_catalog_path,
                    "--knowledge",
                    knowledge_path,
                    "--tagged",
                    tagged_path,
                ]
            ),
        },
    ]
    affected_topics = merge_report.get("affected_topics") or []
    added_candidate_ids = list(merge_report.get("added_candidate_ids") or [])
    support_updated_candidate_ids = list(
        merge_report.get("support_updated_candidate_ids") or []
    )
    changed_candidate_ids = list(
        dict.fromkeys([*added_candidate_ids, *support_updated_candidate_ids])
    )
    changed_candidate_count = len(changed_candidate_ids)
    if change_mode not in {"additive", "scoped_recluster"}:
        raise ValueError("change_mode must be additive or scoped_recluster")
    executable_catalog_topics = {
        (
            str(domain.get("name") or "").strip().casefold(),
            str(topic.get("name") or "").strip().casefold(),
        ): {
            "domain": str(domain.get("name") or "").strip(),
            "topic": str(topic.get("name") or "").strip(),
        }
        for domain in current_catalog.get("domains", []) or []
        if isinstance(domain, dict)
        for topic in domain.get("topics", []) or []
        if isinstance(topic, dict)
        and any(
            isinstance(rule, dict)
            and str(rule.get("rule_id") or rule.get("id") or "").strip()
            for rule in topic.get("rules", []) or []
        )
    }
    normalized_allowed_recluster_topics: List[Dict[str, str]] = []
    seen_allowed_topics: set[tuple[str, str]] = set()
    for item in allowed_recluster_topics:
        raw_key = (
            str(item.get("domain") or "").strip().casefold(),
            str(item.get("topic") or "").strip().casefold(),
        )
        if not all(raw_key) or raw_key in seen_allowed_topics:
            continue
        canonical = executable_catalog_topics.get(raw_key)
        if canonical is None:
            raise ValueError(
                "allowed_recluster_topics must reference an executable Topic in "
                f"the frozen base catalog: {item.get('domain')}::{item.get('topic')}"
            )
        seen_allowed_topics.add(raw_key)
        normalized_allowed_recluster_topics.append(canonical)
    if change_mode == "additive" and normalized_allowed_recluster_topics:
        raise ValueError(
            "allowed_recluster_topics requires change_mode='scoped_recluster'"
        )
    declared_change_topics: List[Dict[str, str]] = []
    seen_declared_topics: set[tuple[str, str]] = set()
    for item in [*affected_topics, *normalized_allowed_recluster_topics]:
        topic_key = (str(item["domain"]), str(item["topic"]))
        if topic_key in seen_declared_topics:
            continue
        seen_declared_topics.add(topic_key)
        declared_change_topics.append(
            {"domain": topic_key[0], "topic": topic_key[1]}
        )
    change_policy = {
        "mode": change_mode,
        "max_added_formal_rules": (
            changed_candidate_count
            if max_added_formal_rules is None
            else int(max_added_formal_rules)
        ),
        "max_removed_formal_rules": 0,
        "max_modified_existing_rules": 0,
        "max_existing_rule_topic_moves": 0,
        "max_existing_rule_cluster_moves": int(max_existing_rule_cluster_moves),
        "max_existing_rule_group_moves": int(max_existing_rule_group_moves),
        "max_added_clusters": (
            changed_candidate_count
            if max_added_clusters is None
            else int(max_added_clusters)
        ),
        "max_removed_clusters": int(max_removed_clusters),
        "max_changed_cluster_definitions": int(max_changed_cluster_definitions),
        "max_changed_topics": (
            len(declared_change_topics)
            if max_changed_topics is None
            else int(max_changed_topics)
        ),
        "max_unexpected_changed_topics": 0,
        "max_added_generalized_rules": (
            changed_candidate_count
            if max_added_generalized_rules is None
            else int(max_added_generalized_rules)
        ),
        "max_removed_generalized_rules": int(max_removed_generalized_rules),
        "max_modified_generalized_rules": int(max_modified_generalized_rules),
        "min_added_rule_incremental_source_ratio": 1.0,
        "allowed_recluster_topics": normalized_allowed_recluster_topics,
    }
    change_policy = validate_change_policy(change_policy)
    inputs = {
        "base_catalog": _input_record(current_catalog_path),
        "base_generalized": _input_record(current_generalized_path),
        "base_formal": _input_record(current_formal_path),
        "base_cluster_proposals": _input_record(current_cluster_proposals_path),
        "current_candidates": _input_record(current_candidates_path),
        "new_candidates": _input_record(new_candidates_path),
        "knowledge": _input_record(knowledge_path),
        "tagged": _input_record(tagged_path),
        "merged_candidates": _input_record(merged_path),
        "candidate_rules_for_cluster": _input_record(candidate_for_cluster_path),
        "candidate_embedding_input": _input_record(
            candidate_embedding_input_path
        ),
    }
    run_configuration = {
        "conda_environment": "physicsverifier",
        "candidate_embedding": {
            "embedding_model": "text-embedding-3-large",
            "similarity_threshold": 0.74,
            "min_cluster_size": 4,
            "batch_size": 64,
            "cache_path": _portable_path(candidate_embedding_cache_path),
            "resume": True,
        },
        "candidate_generalization": {
            "model_chain": [
                "deepseek-v4-flash-nothinking",
                "gemini-2.5-flash-nothinking",
            ],
            "temperature": 0.0,
            "max_clusters": 0,
            "max_candidates_per_batch": 12,
            "min_source_candidates": 2,
            "min_source_samples": 2,
            "max_tokens": 4000,
            "request_timeout_seconds": 120.0,
            "attempts": 2,
            "thinking_enabled": False,
            "resume": True,
            "continue_on_error": True,
        },
        "formal_preparation": {
            "baseline_catalog": _portable_path(formal_seed_catalog_path),
            "preserve_baseline_rule_ids": True,
        },
        "formal_embedding": {
            "embedding_model": "text-embedding-3-large",
            "similarity_threshold": 0.72,
            "min_cluster_size": 4,
            "batch_size": 64,
            "cache_path": _portable_path(formal_embedding_cache_path),
            "resume": True,
        },
        "cluster_labeling": {
            "model": "deepseek-v4-flash-nothinking",
            "temperature": 0.0,
            "max_topics": 0,
            "min_rule_count": 1,
            "max_rules_per_cluster": 8,
            "max_output_tokens": 8192,
            "request_timeout_seconds": 180.0,
            "resume": True,
            "continue_on_error": True,
        },
        "finalize": {
            "base_catalog": _portable_path(current_catalog_path),
            "knowledge": _portable_path(knowledge_path),
            "tagged": _portable_path(tagged_path),
        },
    }
    manifest = {
        "schema_version": 2,
        "status": "prepared" if declared_change_topics else "no_rebuild_needed",
        "inputs": inputs,
        "base_catalog": _portable_path(current_catalog_path),
        "base_catalog_sha256": inputs["base_catalog"]["sha256"],
        "base_generalized": _portable_path(current_generalized_path),
        "base_generalized_sha256": inputs["base_generalized"]["sha256"],
        "formal_seed_catalog": _portable_path(formal_seed_catalog_path),
        "new_candidates": _portable_path(new_candidates_path),
        "workspace": _portable_path(workspace),
        "affected_topics": declared_change_topics,
        "declared_change_topics": declared_change_topics,
        "candidate_affected_topics": affected_topics,
        "candidate_delta": {
            "added_candidate_ids": added_candidate_ids,
            "support_updated_candidate_ids": support_updated_candidate_ids,
            "changed_candidate_ids": changed_candidate_ids,
        },
        "change_policy": change_policy,
        "run_configuration": run_configuration,
        "merge_summary": merge_report.get("summary") or {},
        "commands": commands if declared_change_topics else [],
        "promotion_policy": (
            "Never overwrite the current catalog automatically. Existing formal rules "
            "must be preserved; finalize, review added rules, run full verifier regression, "
            "then promote manually."
        ),
    }
    manifest["configuration_sha256"] = incremental_manifest_configuration_sha256(
        manifest
    )
    _write_json(incremental_manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated incremental unified-rule update and its runbook."
    )
    parser.add_argument("--new-candidates", required=True)
    parser.add_argument(
        "--workspace",
        default="results/unified_rules_incremental",
    )
    parser.add_argument(
        "--current-candidates",
        default="results/unified_rules_3000/semantic_experience_distilled_for_cluster.json",
    )
    parser.add_argument(
        "--current-generalized",
        default="results/unified_rules_3000/semantic_experience_generalized.json",
    )
    parser.add_argument(
        "--current-formal",
        default="results/unified_rules_3000/semantic_experience_generalized_for_cluster.json",
    )
    parser.add_argument(
        "--current-cluster-proposals",
        default="results/unified_rules_3000/cluster_proposals.json",
    )
    parser.add_argument("--current-catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    parser.add_argument(
        "--candidate-embedding-cache",
        default="results/unified_rules_3000/rule_embedding_cache.json",
    )
    parser.add_argument(
        "--formal-embedding-cache",
        default="results/unified_rules_3000/formal_rule_embedding_cache.json",
    )
    parser.add_argument("--reset-seed-outputs", action="store_true")
    parser.add_argument(
        "--change-mode",
        choices=("additive", "scoped_recluster"),
        default="additive",
    )
    parser.add_argument("--max-added-formal-rules", type=int, default=None)
    parser.add_argument("--max-existing-rule-cluster-moves", type=int, default=0)
    parser.add_argument("--max-existing-rule-group-moves", type=int, default=0)
    parser.add_argument("--max-added-clusters", type=int, default=None)
    parser.add_argument("--max-removed-clusters", type=int, default=0)
    parser.add_argument("--max-changed-cluster-definitions", type=int, default=0)
    parser.add_argument("--max-changed-topics", type=int, default=None)
    parser.add_argument("--max-added-generalized-rules", type=int, default=None)
    parser.add_argument("--max-removed-generalized-rules", type=int, default=0)
    parser.add_argument("--max-modified-generalized-rules", type=int, default=0)
    parser.add_argument(
        "--allow-recluster-topic",
        action="append",
        default=[],
        metavar="DOMAIN::TOPIC",
    )
    args = parser.parse_args()

    allowed_recluster_topics = []
    for raw in args.allow_recluster_topic:
        domain, separator, topic = str(raw).partition("::")
        if not separator or not domain.strip() or not topic.strip():
            parser.error("--allow-recluster-topic must use DOMAIN::TOPIC")
        allowed_recluster_topics.append(
            {"domain": domain.strip(), "topic": topic.strip()}
        )

    manifest = prepare_incremental_update(
        new_candidates_path=Path(args.new_candidates),
        workspace=Path(args.workspace),
        current_candidates_path=Path(args.current_candidates),
        current_generalized_path=Path(args.current_generalized),
        current_formal_path=Path(args.current_formal),
        current_cluster_proposals_path=Path(args.current_cluster_proposals),
        current_catalog_path=Path(args.current_catalog),
        knowledge_path=Path(args.knowledge),
        tagged_path=Path(args.tagged),
        candidate_embedding_cache_path=Path(args.candidate_embedding_cache),
        formal_embedding_cache_path=Path(args.formal_embedding_cache),
        reset_seed_outputs=bool(args.reset_seed_outputs),
        change_mode=args.change_mode,
        max_added_formal_rules=args.max_added_formal_rules,
        max_existing_rule_cluster_moves=args.max_existing_rule_cluster_moves,
        max_existing_rule_group_moves=args.max_existing_rule_group_moves,
        max_added_clusters=args.max_added_clusters,
        max_removed_clusters=args.max_removed_clusters,
        max_changed_cluster_definitions=args.max_changed_cluster_definitions,
        max_changed_topics=args.max_changed_topics,
        max_added_generalized_rules=args.max_added_generalized_rules,
        max_removed_generalized_rules=args.max_removed_generalized_rules,
        max_modified_generalized_rules=args.max_modified_generalized_rules,
        allowed_recluster_topics=allowed_recluster_topics,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
