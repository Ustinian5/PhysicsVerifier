from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_rule_coarsening import audit_rule_coarsening
from scripts.build_unified_catalog import build_unified_catalog
from scripts.generate_cluster_proposals import (
    CLUSTER_LABEL_SYSTEM_PROMPT,
    _build_rule_index,
    _cluster_label_resume_fingerprint,
    add_catalog_fallback_proposals,
)
from scripts.generalize_experience_candidates import generalize_candidates
from scripts.prepare_incremental_candidates import prepare_incremental_candidates
from scripts.prepare_incremental_update import (
    _canonicalize_candidate_affected_topics,
)
from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster
from scripts.refine_cluster_blueprints import (
    build_generated_blueprints_from_refined_proposals,
)
from scripts.validate_unified_catalog_structure import validate_catalog_structure
from rule_framework.incremental_validation import (
    audit_added_rule_lineage,
    build_catalog_snapshot,
    build_generalized_lineage,
    compare_catalog_snapshots,
    compare_generalized_outputs,
    evaluate_change_policy,
    incremental_manifest_configuration_sha256,
    sha256_file,
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
    return sha256_file(path)


def _portable_path(value: Any) -> Path:
    return Path(str(value or "").replace("\\", "/"))


def _validate_manifest_inputs(
    manifest: Dict[str, Any],
    *,
    overrides: Mapping[str, Path],
) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    schema_version = int(manifest.get("schema_version") or 0)
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    if schema_version != 2:
        mismatches.append(
            {
                "input": "manifest",
                "reason": "unsupported_schema_version",
                "expected": 2,
                "actual": schema_version,
            }
        )
    expected_configuration_sha256 = str(
        manifest.get("configuration_sha256") or ""
    ).strip()
    actual_configuration_sha256 = incremental_manifest_configuration_sha256(
        manifest
    )
    if not expected_configuration_sha256:
        mismatches.append(
            {
                "input": "manifest",
                "reason": "missing_configuration_sha256",
            }
        )
    elif expected_configuration_sha256 != actual_configuration_sha256:
        mismatches.append(
            {
                "input": "manifest",
                "reason": "configuration_sha256_mismatch",
                "expected_sha256": expected_configuration_sha256,
                "actual_sha256": actual_configuration_sha256,
            }
        )
    required_inputs = (
        "base_catalog",
        "base_generalized",
        "base_formal",
        "base_cluster_proposals",
        "current_candidates",
        "new_candidates",
        "knowledge",
        "tagged",
        "merged_candidates",
        "candidate_rules_for_cluster",
        "candidate_embedding_input",
    )
    for name in required_inputs:
        record = inputs.get(name) if isinstance(inputs.get(name), dict) else {}
        expected_sha256 = str(record.get("sha256") or "").strip()
        path = overrides.get(name) or _portable_path(record.get("path"))
        if not expected_sha256:
            mismatches.append(
                {"input": name, "reason": "missing_sha256", "path": str(path)}
            )
            continue
        if not path.is_file():
            mismatches.append(
                {"input": name, "reason": "missing_file", "path": str(path)}
            )
            continue
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            mismatches.append(
                {
                    "input": name,
                    "reason": "sha256_mismatch",
                    "path": str(path),
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                }
            )
    run_configuration = (
        manifest.get("run_configuration")
        if isinstance(manifest.get("run_configuration"), dict)
        else {}
    )
    required_run_fields = {
        "candidate_embedding": {
            "embedding_model",
            "similarity_threshold",
            "min_cluster_size",
            "batch_size",
        },
        "candidate_generalization": {
            "model_chain",
            "temperature",
            "max_clusters",
            "max_candidates_per_batch",
            "min_source_candidates",
            "min_source_samples",
            "max_tokens",
            "request_timeout_seconds",
            "attempts",
            "thinking_enabled",
        },
        "formal_embedding": {
            "embedding_model",
            "similarity_threshold",
            "min_cluster_size",
            "batch_size",
        },
        "cluster_labeling": {
            "model",
            "temperature",
            "max_topics",
            "min_rule_count",
            "max_rules_per_cluster",
            "max_output_tokens",
        },
    }
    for stage, fields in required_run_fields.items():
        configuration = run_configuration.get(stage)
        if not isinstance(configuration, dict):
            mismatches.append(
                {
                    "input": "run_configuration",
                    "reason": "missing_stage",
                    "stage": stage,
                }
            )
            continue
        missing = sorted(fields - set(configuration))
        if missing:
            mismatches.append(
                {
                    "input": "run_configuration",
                    "reason": "missing_fields",
                    "stage": stage,
                    "fields": missing,
                }
            )
    cluster_labeling = run_configuration.get("cluster_labeling")
    formal_embedding = run_configuration.get("formal_embedding")
    if isinstance(cluster_labeling, dict):
        if cluster_labeling.get("max_topics") != 0:
            mismatches.append(
                {
                    "input": "run_configuration",
                    "reason": "partial_cluster_labeling_scope_not_allowed",
                    "stage": "cluster_labeling",
                    "field": "max_topics",
                    "expected": 0,
                    "actual": cluster_labeling.get("max_topics"),
                }
            )
        min_rule_count = cluster_labeling.get("min_rule_count")
        min_cluster_size = (
            formal_embedding.get("min_cluster_size")
            if isinstance(formal_embedding, dict)
            else None
        )
        if (
            not isinstance(min_rule_count, int)
            or isinstance(min_rule_count, bool)
            or not isinstance(min_cluster_size, int)
            or isinstance(min_cluster_size, bool)
            or not 0 <= min_rule_count <= min_cluster_size
        ):
            mismatches.append(
                {
                    "input": "run_configuration",
                    "reason": "cluster_labeling_threshold_can_skip_clusters",
                    "stage": "cluster_labeling",
                    "field": "min_rule_count",
                    "expected": "0 <= min_rule_count <= formal_embedding.min_cluster_size",
                    "actual": min_rule_count,
                    "formal_min_cluster_size": min_cluster_size,
                }
            )
    normalized_policy: Dict[str, Any] | None = None
    try:
        normalized_policy = validate_change_policy(
            manifest.get("change_policy")
            if isinstance(manifest.get("change_policy"), dict)
            else {}
        )
    except (TypeError, ValueError) as exc:
        mismatches.append(
            {"input": "change_policy", "reason": "invalid", "error": str(exc)}
        )
    if normalized_policy is not None:
        def topic_set(value: Any) -> set[tuple[str, str]]:
            return {
                (
                    str(item.get("domain") or "").strip().casefold(),
                    str(item.get("topic") or "").strip().casefold(),
                )
                for item in (value or [])
                if isinstance(item, dict)
                and str(item.get("domain") or "").strip()
                and str(item.get("topic") or "").strip()
            }

        expected_scope = topic_set(manifest.get("candidate_affected_topics"))
        expected_scope.update(
            topic_set(normalized_policy.get("allowed_recluster_topics"))
        )
        declared_scope = topic_set(manifest.get("declared_change_topics"))
        legacy_scope = topic_set(manifest.get("affected_topics"))
        if declared_scope != expected_scope or legacy_scope != expected_scope:
            mismatches.append(
                {
                    "input": "manifest",
                    "reason": "declared_change_scope_mismatch",
                    "expected": sorted(expected_scope),
                    "declared": sorted(declared_scope),
                    "affected_topics": sorted(legacy_scope),
                }
            )
        expected_status = "prepared" if expected_scope else "no_rebuild_needed"
        if str(manifest.get("status") or "") != expected_status:
            mismatches.append(
                {
                    "input": "manifest",
                    "reason": "status_scope_mismatch",
                    "expected": expected_status,
                    "actual": manifest.get("status"),
                }
            )
        commands = manifest.get("commands")
        expected_command_count = 6 if expected_scope else 0
        actual_command_count = len(commands) if isinstance(commands, list) else -1
        if actual_command_count != expected_command_count:
            mismatches.append(
                {
                    "input": "manifest",
                    "reason": "runbook_scope_mismatch",
                    "expected_command_count": expected_command_count,
                    "actual_command_count": actual_command_count,
                }
            )
    return {
        "schema_version": schema_version,
        "strict_v2": schema_version == 2,
        "input_hashes_match": not mismatches,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def _validate_candidate_delta(
    manifest: Dict[str, Any], workspace: Path
) -> Dict[str, Any]:
    inputs = manifest["inputs"]
    merged, merge_report = prepare_incremental_candidates(
        current_payload=_load_json(
            _portable_path(inputs["current_candidates"]["path"])
        ),
        new_payload=_load_json(_portable_path(inputs["new_candidates"]["path"])),
        formal_payload=_load_json(_portable_path(inputs["base_formal"]["path"])),
    )
    _canonicalize_candidate_affected_topics(
        merge_report,
        _load_json(_portable_path(inputs["base_catalog"]["path"])),
    )
    expected_delta = {
        "added_candidate_ids": list(merge_report.get("added_candidate_ids") or []),
        "support_updated_candidate_ids": list(
            merge_report.get("support_updated_candidate_ids") or []
        ),
    }
    expected_delta["changed_candidate_ids"] = list(
        dict.fromkeys(
            [
                *expected_delta["added_candidate_ids"],
                *expected_delta["support_updated_candidate_ids"],
            ]
        )
    )
    mismatches: List[Dict[str, Any]] = []
    if manifest.get("candidate_delta") != expected_delta:
        mismatches.append(
            {
                "field": "candidate_delta",
                "expected": expected_delta,
                "actual": manifest.get("candidate_delta"),
            }
        )
    expected_topics = list(merge_report.get("affected_topics") or [])
    if manifest.get("candidate_affected_topics") != expected_topics:
        mismatches.append(
            {
                "field": "candidate_affected_topics",
                "expected": expected_topics,
                "actual": manifest.get("candidate_affected_topics"),
            }
        )
    if manifest.get("merge_summary") != merge_report.get("summary"):
        mismatches.append(
            {
                "field": "merge_summary",
                "expected": merge_report.get("summary"),
                "actual": manifest.get("merge_summary"),
            }
        )
    workspace_merge_report_path = workspace / "incremental_merge_report.json"
    if not workspace_merge_report_path.is_file():
        mismatches.append(
            {"field": "workspace_merge_report", "reason": "missing_file"}
        )
    else:
        workspace_merge_report = _load_json(workspace_merge_report_path)
        for field in (
            "summary",
            "added_candidate_ids",
            "support_updated_candidate_ids",
            "covered_by_formal",
            "unchanged_duplicate_candidate_ids",
            "affected_topics",
        ):
            if workspace_merge_report.get(field) != merge_report.get(field):
                mismatches.append(
                    {"field": f"workspace_merge_report.{field}", "reason": "mismatch"}
                )
    merged_path = workspace / "semantic_experience_distilled.json"
    if not merged_path.is_file():
        mismatches.append({"field": "merged_candidate_bank", "reason": "missing_file"})
    elif _load_json(merged_path) != merged:
        mismatches.append({"field": "merged_candidate_bank", "reason": "mismatch"})
    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "recomputed_candidate_delta": expected_delta,
        "recomputed_affected_topics": expected_topics,
    }


def _validate_run_output_bindings(
    manifest: Dict[str, Any],
    *,
    generalized: Dict[str, Any],
    proposals: Dict[str, Any],
) -> Dict[str, Any]:
    expected = str(manifest.get("configuration_sha256") or "")
    observed = {
        "semantic_experience_generalized": str(
            (generalized.get("metadata") or {}).get(
                "incremental_configuration_sha256"
            )
            or ""
        ),
        "cluster_proposals": str(
            (proposals.get("metadata") or {}).get(
                "incremental_configuration_sha256"
            )
            or ""
        ),
    }
    mismatches = [
        {
            "output": name,
            "expected_configuration_sha256": expected,
            "actual_configuration_sha256": actual,
        }
        for name, actual in observed.items()
        if actual != expected
    ]
    generalized_metadata = (
        generalized.get("metadata")
        if isinstance(generalized.get("metadata"), dict)
        else {}
    )
    proposal_metadata = (
        proposals.get("metadata")
        if isinstance(proposals.get("metadata"), dict)
        else {}
    )
    generalized_requirements = {
        "generator": "experience_candidate_generalizer_v1",
        "scope_mode": "full",
        "complete": True,
        "failed_batch_count": 0,
        "missing_candidate_count": 0,
    }
    for field, required in generalized_requirements.items():
        if generalized_metadata.get(field) != required:
            mismatches.append(
                {
                    "output": "semantic_experience_generalized",
                    "field": f"metadata.{field}",
                    "expected": required,
                    "actual": generalized_metadata.get(field),
                }
            )
    proposal_requirements = {
        "generator": "embedding_cluster_labeling_v1",
        "failure_count": 0,
        "fallback_label_count": 0,
    }
    for field, required in proposal_requirements.items():
        if proposal_metadata.get(field) != required:
            mismatches.append(
                {
                    "output": "cluster_proposals",
                    "field": f"metadata.{field}",
                    "expected": required,
                    "actual": proposal_metadata.get(field),
                }
            )
    non_model_proposals = [
        {
            "topic_key": str(item.get("topic_key") or ""),
            "label_source": str(item.get("label_source") or ""),
        }
        for item in (proposals.get("proposals") or [])
        if isinstance(item, dict) and item.get("label_source") != "model"
    ]
    if non_model_proposals:
        mismatches.append(
            {
                "output": "cluster_proposals",
                "field": "proposals.label_source",
                "expected": "model",
                "actual": non_model_proposals,
            }
        )
    proposal_topic_count = int(proposal_metadata.get("topic_count") or 0)
    proposal_target_count = int(proposal_metadata.get("target_topic_count") or 0)
    if proposal_topic_count != proposal_target_count:
        mismatches.append(
            {
                "output": "cluster_proposals",
                "field": "metadata.topic_count",
                "expected": proposal_target_count,
                "actual": proposal_topic_count,
            }
        )
    return {"passed": not mismatches, "observed": observed, "mismatches": mismatches}


def _expected_lineage(
    *, stage: str, input_sha256: Mapping[str, str]
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "input_sha256": dict(sorted(input_sha256.items())),
    }


def _validate_artifact_chain(
    *,
    manifest: Dict[str, Any],
    workspace: Path,
    base_catalog_path: Path,
    knowledge_path: Path,
    tagged_path: Path,
    generalized: Dict[str, Any],
    formal: Dict[str, Any],
    precluster_catalog: Dict[str, Any],
    formal_rule_input: Dict[str, Any],
    formal_clusters: Dict[str, Any],
    proposals: Dict[str, Any],
) -> Dict[str, Any]:
    configuration_sha256 = str(manifest.get("configuration_sha256") or "")
    inputs = manifest["inputs"]
    paths = {
        "candidate_clusters": workspace / "rule_embedding_clusters.json",
        "generalized": workspace / "semantic_experience_generalized.json",
        "formal_rules": workspace
        / "semantic_experience_generalized_for_cluster.json",
        "formal_rule_input": workspace / "formal_rule_embedding_input.json",
        "formal_clusters": workspace / "formal_rule_embedding_clusters.json",
        "precluster_catalog": workspace / "catalog_precluster.json",
        "cluster_proposals": workspace / "cluster_proposals.json",
    }
    payloads = {
        "candidate_clusters": _load_json(paths["candidate_clusters"]),
        "generalized": generalized,
        "formal_rules": formal,
        "formal_rule_input": formal_rule_input,
        "formal_clusters": formal_clusters,
        "precluster_catalog": precluster_catalog,
        "cluster_proposals": proposals,
    }
    expected_lineages = {
        "candidate_clusters": _expected_lineage(
            stage="candidate_embedding",
            input_sha256={
                "rule_input": str(inputs["candidate_embedding_input"]["sha256"])
            },
        ),
        "generalized": _expected_lineage(
            stage="candidate_generalization",
            input_sha256={
                "base_generalized": str(inputs["base_generalized"]["sha256"]),
                "candidate_clusters": _sha256(paths["candidate_clusters"]),
                "candidate_rules": str(
                    inputs["candidate_rules_for_cluster"]["sha256"]
                ),
            },
        ),
        "formal_rules": _expected_lineage(
            stage="formal_preparation",
            input_sha256={
                "base_catalog": _sha256(base_catalog_path),
                "generalized": _sha256(paths["generalized"]),
            },
        ),
        "formal_rule_input": _expected_lineage(
            stage="formal_embedding_input",
            input_sha256={"formal_rules": _sha256(paths["formal_rules"])},
        ),
        "formal_clusters": _expected_lineage(
            stage="formal_embedding",
            input_sha256={"rule_input": _sha256(paths["formal_rule_input"])},
        ),
        "precluster_catalog": _expected_lineage(
            stage="precluster_catalog",
            input_sha256={
                "base_catalog": _sha256(base_catalog_path),
                "formal_rules": _sha256(paths["formal_rules"]),
                "generalized": _sha256(paths["generalized"]),
                "knowledge": _sha256(knowledge_path),
                "tagged": _sha256(tagged_path),
            },
        ),
        "cluster_proposals": _expected_lineage(
            stage="cluster_labeling",
            input_sha256={
                "base_cluster_proposals": str(
                    inputs["base_cluster_proposals"]["sha256"]
                ),
                "formal_clusters": _sha256(paths["formal_clusters"]),
                "formal_rule_input": _sha256(paths["formal_rule_input"]),
                "precluster_catalog": _sha256(paths["precluster_catalog"]),
            },
        ),
    }
    mismatches: List[Dict[str, Any]] = []
    nodes: Dict[str, Any] = {}
    for name, path in paths.items():
        metadata = (
            payloads[name].get("metadata")
            if isinstance(payloads[name].get("metadata"), dict)
            else {}
        )
        actual_configuration = str(
            metadata.get("incremental_configuration_sha256") or ""
        )
        actual_lineage = metadata.get("incremental_lineage")
        expected_lineage = expected_lineages[name]
        nodes[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "expected_lineage": expected_lineage,
            "actual_lineage": actual_lineage,
        }
        if actual_configuration != configuration_sha256:
            mismatches.append(
                {
                    "artifact": name,
                    "reason": "configuration_sha256_mismatch",
                    "expected": configuration_sha256,
                    "actual": actual_configuration,
                }
            )
        if actual_lineage != expected_lineage:
            mismatches.append(
                {
                    "artifact": name,
                    "reason": "lineage_mismatch",
                    "expected": expected_lineage,
                    "actual": actual_lineage,
                }
            )
    run_configuration = manifest.get("run_configuration") or {}
    candidate_embedding_configuration = (
        run_configuration.get("candidate_embedding") or {}
    )
    formal_embedding_configuration = (
        run_configuration.get("formal_embedding") or {}
    )
    generalization_configuration = (
        run_configuration.get("candidate_generalization") or {}
    )
    proposal_configuration = run_configuration.get("cluster_labeling") or {}
    expected_behavior = {
        "candidate_clusters": {
            field: candidate_embedding_configuration.get(field)
            for field in (
                "embedding_model",
                "similarity_threshold",
                "min_cluster_size",
                "batch_size",
            )
        },
        "formal_clusters": {
            field: formal_embedding_configuration.get(field)
            for field in (
                "embedding_model",
                "similarity_threshold",
                "min_cluster_size",
                "batch_size",
            )
        },
        "generalized": {
            field: generalization_configuration.get(field)
            for field in (
                "model_chain",
                "temperature",
                "max_clusters",
                "max_candidates_per_batch",
                "min_source_candidates",
                "min_source_samples",
                "max_tokens",
                "request_timeout_seconds",
                "attempts",
                "thinking_enabled",
            )
        },
        "cluster_proposals": {
            field: proposal_configuration.get(field)
            for field in (
                "model",
                "temperature",
                "max_topics",
                "min_rule_count",
                "max_rules_per_cluster",
                "max_output_tokens",
            )
        },
    }
    actual_behavior = {
        "candidate_clusters": {
            field: (payloads["candidate_clusters"].get("metadata") or {}).get(
                field
            )
            for field in expected_behavior["candidate_clusters"]
        },
        "formal_clusters": {
            field: (payloads["formal_clusters"].get("metadata") or {}).get(
                field
            )
            for field in expected_behavior["formal_clusters"]
        },
        "generalized": (payloads["generalized"].get("metadata") or {}).get(
            "incremental_behavior_configuration"
        ),
        "cluster_proposals": {
            field: (payloads["cluster_proposals"].get("metadata") or {}).get(
                field
            )
            for field in expected_behavior["cluster_proposals"]
        },
    }
    for name, expected in expected_behavior.items():
        nodes[name]["expected_behavior_configuration"] = expected
        nodes[name]["actual_behavior_configuration"] = actual_behavior[name]
        if actual_behavior[name] != expected:
            mismatches.append(
                {
                    "artifact": name,
                    "reason": "behavior_configuration_mismatch",
                    "expected": expected,
                    "actual": actual_behavior[name],
                }
            )
    return {"passed": not mismatches, "nodes": nodes, "mismatches": mismatches}


def _catalog_without_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(payload)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("generated_at", None)
    return normalized


def _validate_deterministic_formal_bundle(
    *,
    workspace: Path,
    base_catalog_path: Path,
    knowledge_path: Path,
    tagged_path: Path,
    formal: Dict[str, Any],
    precluster_catalog: Dict[str, Any],
    formal_rule_input: Dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        replay_root = Path(temp)
        expected_formal_path = replay_root / "formal.json"
        expected_catalog_path = replay_root / "catalog.json"
        expected_report_path = replay_root / "report.json"
        expected_rule_input_path = replay_root / "rule_input.json"
        prepare_rules_for_cluster(
            distilled_input=workspace / "semantic_experience_generalized.json",
            knowledge_path=knowledge_path,
            tagged_path=tagged_path,
            baseline_catalog_path=base_catalog_path,
            distilled_output=expected_formal_path,
            catalog_output=expected_catalog_path,
            report_output=expected_report_path,
            embedding_input_output=expected_rule_input_path,
            scenario_cluster_blueprints_paths=[],
            preserve_baseline_rule_ids=True,
            incremental_manifest_path=workspace / "incremental_manifest.json",
        )
        expected = {
            "formal_rules": _load_json(expected_formal_path),
            "formal_rule_input": _load_json(expected_rule_input_path),
            "precluster_catalog": _catalog_without_timestamp(
                _load_json(expected_catalog_path)
            ),
        }
    actual = {
        "formal_rules": formal,
        "formal_rule_input": formal_rule_input,
        "precluster_catalog": _catalog_without_timestamp(precluster_catalog),
    }
    mismatches = [
        {"artifact": name, "reason": "deterministic_replay_mismatch"}
        for name in expected
        if actual[name] != expected[name]
    ]
    return {"passed": not mismatches, "mismatches": mismatches}


def _validate_formal_cluster_coverage(
    formal_rule_input: Dict[str, Any], formal_clusters: Dict[str, Any]
) -> Dict[str, Any]:
    expected_by_topic: Dict[str, List[str]] = {}
    for rule in formal_rule_input.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        topic_key = str(rule.get("topic_key") or "")
        rule_id = str(rule.get("rule_id") or "")
        if topic_key and rule_id:
            expected_by_topic.setdefault(topic_key, []).append(rule_id)
    assigned_by_topic: Dict[str, List[str]] = {}
    duplicate_rule_ids: List[str] = []
    duplicate_topic_keys: List[str] = []
    rule_count_mismatches: List[Dict[str, Any]] = []
    cluster_count_mismatches: List[Dict[str, Any]] = []
    cluster_size_mismatches: List[Dict[str, Any]] = []
    for topic in formal_clusters.get("topics", []) or []:
        if not isinstance(topic, dict):
            continue
        topic_key = str(topic.get("topic_key") or "")
        if topic_key in assigned_by_topic:
            duplicate_topic_keys.append(topic_key)
        clusters = [
            cluster
            for cluster in (topic.get("clusters") or [])
            if isinstance(cluster, dict)
        ]
        assigned = [
            str(rule_id)
            for cluster in clusters
            for rule_id in (cluster.get("rule_ids") or [])
        ]
        assigned.extend(str(item) for item in topic.get("residual_rule_ids") or [])
        if topic.get("rule_count") != len(assigned):
            rule_count_mismatches.append(
                {
                    "topic_key": topic_key,
                    "expected": len(assigned),
                    "actual": topic.get("rule_count"),
                }
            )
        if topic.get("cluster_count") != len(clusters):
            cluster_count_mismatches.append(
                {
                    "topic_key": topic_key,
                    "expected": len(clusters),
                    "actual": topic.get("cluster_count"),
                }
            )
        for cluster in clusters:
            rule_ids = list(cluster.get("rule_ids") or [])
            if cluster.get("size") != len(rule_ids):
                cluster_size_mismatches.append(
                    {
                        "topic_key": topic_key,
                        "cluster_id": str(
                            cluster.get("cluster_id") or cluster.get("id") or ""
                        ),
                        "expected": len(rule_ids),
                        "actual": cluster.get("size"),
                    }
                )
        if len(assigned) != len(set(assigned)):
            duplicate_rule_ids.extend(
                rule_id for rule_id in assigned if assigned.count(rule_id) > 1
            )
        assigned_by_topic[topic_key] = assigned
    expected_topic_keys = set(expected_by_topic)
    assigned_topic_keys = set(assigned_by_topic)
    missing_topic_keys = sorted(expected_topic_keys - assigned_topic_keys)
    unexpected_topic_keys = sorted(assigned_topic_keys - expected_topic_keys)
    expected_pairs = {
        (topic_key, rule_id)
        for topic_key, rule_ids in expected_by_topic.items()
        for rule_id in rule_ids
    }
    assigned_pairs = {
        (topic_key, rule_id)
        for topic_key, rule_ids in assigned_by_topic.items()
        for rule_id in rule_ids
    }
    missing = sorted(expected_pairs - assigned_pairs)
    foreign = sorted(assigned_pairs - expected_pairs)
    duplicates = sorted(set(duplicate_rule_ids))
    return {
        "passed": (
            not missing
            and not foreign
            and not duplicates
            and not duplicate_topic_keys
            and not missing_topic_keys
            and not unexpected_topic_keys
            and not rule_count_mismatches
            and not cluster_count_mismatches
            and not cluster_size_mismatches
        ),
        "expected_rule_count": len(expected_pairs),
        "assigned_rule_count": len(assigned_pairs),
        "missing": [list(item) for item in missing],
        "foreign": [list(item) for item in foreign],
        "duplicate_rule_ids": duplicates,
        "duplicate_topic_keys": sorted(set(duplicate_topic_keys)),
        "missing_topic_keys": missing_topic_keys,
        "unexpected_topic_keys": unexpected_topic_keys,
        "rule_count_mismatches": rule_count_mismatches,
        "cluster_count_mismatches": cluster_count_mismatches,
        "cluster_size_mismatches": cluster_size_mismatches,
    }


def _validate_generalization_fingerprints(
    *, manifest: Dict[str, Any], workspace: Path, generalized: Dict[str, Any]
) -> Dict[str, Any]:
    configuration = manifest["run_configuration"]["candidate_generalization"]
    api_calls: List[Dict[str, Any]] = []

    def reject_api(
        domain: str, topic: str, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        api_calls.append(
            {
                "domain": domain,
                "topic": topic,
                "candidate_ids": [
                    str(item.get("rule_id") or "") for item in candidates
                ],
            }
        )
        raise RuntimeError("stale generalization batch requires an API call")

    try:
        replayed = generalize_candidates(
            candidate_payload=_load_json(
                workspace / "semantic_experience_distilled_for_cluster.json"
            ),
            cluster_payload=_load_json(workspace / "rule_embedding_clusters.json"),
            generate=reject_api,
            max_clusters=int(configuration["max_clusters"]),
            min_source_candidates=int(configuration["min_source_candidates"]),
            min_source_samples=int(configuration["min_source_samples"]),
            max_candidates_per_batch=int(
                configuration["max_candidates_per_batch"]
            ),
            existing_payload=generalized,
            resume_context={
                "model_chain": list(configuration["model_chain"]),
                "temperature": float(configuration["temperature"]),
                "max_tokens": int(configuration["max_tokens"]),
                "attempts": int(configuration["attempts"]),
                "request_timeout_seconds": float(
                    configuration["request_timeout_seconds"]
                ),
                "thinking_enabled": bool(configuration["thinking_enabled"]),
            },
        )
    except (RuntimeError, ValueError) as exc:
        return {
            "passed": False,
            "reason": str(exc),
            "api_calls_required": api_calls,
        }
    replayed_results = [
        {
            key: value
            for key, value in item.items()
            if key != "reused"
        }
        for item in (replayed.get("cluster_results") or [])
        if isinstance(item, dict)
    ]
    actual_results = [
        {
            key: value
            for key, value in item.items()
            if key != "reused"
        }
        for item in (generalized.get("cluster_results") or [])
        if isinstance(item, dict)
    ]
    matched = replayed_results == actual_results
    return {
        "passed": matched and not api_calls,
        "cluster_result_count": len(actual_results),
        "api_calls_required": api_calls,
        "cluster_results_match": matched,
    }
def _validate_proposal_fingerprints(
    *,
    manifest: Dict[str, Any],
    workspace: Path,
    formal_clusters: Dict[str, Any],
    formal_rule_input: Dict[str, Any],
    proposals: Dict[str, Any],
) -> Dict[str, Any]:
    configuration = manifest.get("run_configuration", {}).get(
        "cluster_labeling", {}
    )
    rule_index = _build_rule_index(formal_rule_input)
    topics = [
        item
        for item in (formal_clusters.get("topics") or [])
        if isinstance(item, dict)
        and int(item.get("rule_count") or 0)
        >= int(configuration.get("min_rule_count") or 1)
        and item.get("clusters")
    ]
    topics.sort(
        key=lambda item: (
            -int(item.get("rule_count") or 0),
            str(item.get("topic_key") or ""),
        )
    )
    max_topics = int(configuration.get("max_topics") or 0)
    if max_topics > 0:
        topics = topics[:max_topics]
    proposal_lineage = _expected_lineage(
        stage="cluster_labeling",
        input_sha256={
            "base_cluster_proposals": str(
                manifest["inputs"]["base_cluster_proposals"]["sha256"]
            ),
            "formal_clusters": _sha256(
                workspace / "formal_rule_embedding_clusters.json"
            ),
            "formal_rule_input": _sha256(
                workspace / "formal_rule_embedding_input.json"
            ),
            "precluster_catalog": _sha256(workspace / "catalog_precluster.json"),
        },
    )
    expected = {
        str(item.get("topic_key") or "").casefold(): _cluster_label_resume_fingerprint(
            item,
            rule_index=rule_index,
            system_prompt=CLUSTER_LABEL_SYSTEM_PROMPT,
            model=str(configuration.get("model") or ""),
            temperature=float(configuration.get("temperature") or 0.0),
            max_topics=max_topics,
            min_rule_count=int(configuration.get("min_rule_count") or 1),
            max_rules_per_cluster=int(
                configuration.get("max_rules_per_cluster") or 8
            ),
            max_output_tokens=int(
                configuration.get("max_output_tokens") or 8192
            ),
            incremental_configuration_sha256=str(
                manifest.get("configuration_sha256") or ""
            ),
            incremental_lineage=proposal_lineage,
        )
        for item in topics
    }
    actual_items = [
        item for item in (proposals.get("proposals") or []) if isinstance(item, dict)
    ]
    actual: Dict[str, str] = {}
    duplicate_topic_keys: List[str] = []
    for item in actual_items:
        topic_key = str(item.get("topic_key") or "").casefold()
        if topic_key in actual:
            duplicate_topic_keys.append(topic_key)
        actual[topic_key] = str(item.get("source_fingerprint") or "")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    stale = sorted(
        topic_key
        for topic_key in set(expected).intersection(actual)
        if expected[topic_key] != actual[topic_key]
    )
    return {
        "passed": not missing and not extra and not stale and not duplicate_topic_keys,
        "expected_topic_count": len(expected),
        "actual_topic_count": len(actual_items),
        "missing_topic_keys": missing,
        "extra_topic_keys": extra,
        "stale_topic_keys": stale,
        "duplicate_topic_keys": sorted(set(duplicate_topic_keys)),
    }


def _catalog_blueprints(catalog: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    blueprints: Dict[str, List[Dict[str, Any]]] = {}
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = str(domain.get("name") or "").strip()
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = str(topic.get("name") or "").strip()
            topic_key = f"{domain_name}::{topic_name}".casefold()
            clusters: List[Dict[str, Any]] = []
            for cluster in topic.get("scenario_clusters", []) or []:
                if not isinstance(cluster, dict):
                    continue
                groups = []
                for group in cluster.get("rule_groups", []) or []:
                    if not isinstance(group, dict):
                        continue
                    groups.append(
                        {
                            "group_id": str(
                                group.get("id") or group.get("group_id") or ""
                            ).strip(),
                            "name": str(group.get("name") or "").strip(),
                            "summary": str(group.get("summary") or "").strip(),
                            "activation_condition": str(
                                group.get("activation_condition") or ""
                            ).strip(),
                            "rule_ids": list(group.get("rule_ids") or []),
                        }
                    )
                cluster_rule_ids = list(cluster.get("rule_ids") or [])
                if not groups and cluster_rule_ids:
                    cluster_id = str(
                        cluster.get("id") or cluster.get("cluster_id") or ""
                    ).strip()
                    summary = str(cluster.get("summary") or "").strip()
                    groups.append(
                        {
                            "group_id": f"{cluster_id}_rules",
                            "name": str(cluster.get("name") or "").strip(),
                            "summary": summary,
                            "activation_condition": summary,
                            "rule_ids": cluster_rule_ids,
                        }
                    )
                clusters.append(
                    {
                        "cluster_id": str(
                            cluster.get("id") or cluster.get("cluster_id") or ""
                        ).strip(),
                        "name": str(cluster.get("name") or "").strip(),
                        "description": str(cluster.get("summary") or "").strip(),
                        "includes": [],
                        "excludes": [],
                        "entry_cues": [],
                        "related_clusters": [],
                        "rule_groups": groups,
                    }
                )
            if clusters:
                blueprints[topic_key] = clusters
    return blueprints


def _filter_blueprints_to_rules(
    blueprints: Dict[str, List[Dict[str, Any]]],
    allowed_rule_ids: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    allowed = set(allowed_rule_ids)
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for topic_key, clusters in blueprints.items():
        kept_clusters: List[Dict[str, Any]] = []
        for cluster in clusters:
            kept_groups = []
            for group in cluster.get("rule_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                rule_ids = [
                    str(rule_id)
                    for rule_id in (group.get("rule_ids") or [])
                    if str(rule_id) in allowed
                ]
                if rule_ids:
                    kept_groups.append({**group, "rule_ids": rule_ids})
            if kept_groups:
                kept_clusters.append({**cluster, "rule_groups": kept_groups})
        if kept_clusters:
            filtered[topic_key] = kept_clusters
    return filtered


def _merge_incremental_blueprints(
    base: Dict[str, List[Dict[str, Any]]],
    incremental: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Append new-rule blueprints without changing an existing Cluster."""
    merged = copy.deepcopy(base)
    for topic_key, clusters in incremental.items():
        target_clusters = merged.setdefault(topic_key, [])
        by_id = {
            str(cluster.get("cluster_id") or ""): cluster
            for cluster in target_clusters
        }
        for cluster in clusters:
            cluster_id = str(cluster.get("cluster_id") or "")
            existing = by_id.get(cluster_id)
            if existing is None:
                copied = copy.deepcopy(cluster)
                target_clusters.append(copied)
                by_id[cluster_id] = copied
                continue
            copied = copy.deepcopy(cluster)
            suffix = 1
            unique_cluster_id = f"{cluster_id}__incremental_{suffix:02d}"
            while unique_cluster_id in by_id:
                suffix += 1
                unique_cluster_id = f"{cluster_id}__incremental_{suffix:02d}"
            copied["cluster_id"] = unique_cluster_id
            target_clusters.append(copied)
            by_id[unique_cluster_id] = copied
    return merged


def _catalog_topic_rule_ids(catalog: Dict[str, Any]) -> Dict[str, List[str]]:
    by_topic: Dict[str, List[str]] = {}
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = str(domain.get("name") or "").strip()
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = str(topic.get("name") or "").strip()
            topic_key = f"{domain_name}::{topic_name}".casefold()
            by_topic[topic_key] = list(
                dict.fromkeys(
                    str(rule.get("rule_id") or rule.get("id") or "").strip()
                    for rule in topic.get("rules", []) or []
                    if isinstance(rule, dict)
                    and str(rule.get("rule_id") or rule.get("id") or "").strip()
                )
            )
    return by_topic


def _audit_blueprint_rule_cover(
    *,
    topic_key: str,
    clusters: Iterable[Dict[str, Any]],
    expected_rule_ids: Iterable[str],
    kind: str,
    base_cluster_count: int,
) -> Dict[str, Any]:
    cluster_list = list(clusters)
    expected = list(dict.fromkeys(str(item) for item in expected_rule_ids if str(item)))
    expected_set = set(expected)
    assigned: List[str] = []
    invalid_cluster_indexes: List[int] = []
    empty_cluster_ids: List[int] = []
    duplicate_cluster_ids: List[str] = []
    empty_group_ids: List[Dict[str, Any]] = []
    duplicate_group_ids: List[Dict[str, str]] = []
    empty_group_rule_ids: List[Dict[str, str]] = []
    seen_cluster_ids: set[str] = set()

    for cluster_index, cluster in enumerate(cluster_list):
        if not isinstance(cluster, dict):
            invalid_cluster_indexes.append(cluster_index)
            continue
        cluster_id = str(cluster.get("cluster_id") or cluster.get("id") or "").strip()
        if not cluster_id:
            empty_cluster_ids.append(cluster_index)
        elif cluster_id in seen_cluster_ids:
            duplicate_cluster_ids.append(cluster_id)
        else:
            seen_cluster_ids.add(cluster_id)
        seen_group_ids: set[str] = set()
        groups = cluster.get("rule_groups") or []
        if not isinstance(groups, list) or not groups:
            empty_group_rule_ids.append(
                {"cluster_id": cluster_id, "group_id": ""}
            )
            continue
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                empty_group_ids.append(
                    {"cluster_id": cluster_id, "group_index": group_index}
                )
                continue
            group_id = str(group.get("group_id") or group.get("id") or "").strip()
            if not group_id:
                empty_group_ids.append(
                    {"cluster_id": cluster_id, "group_index": group_index}
                )
            elif group_id in seen_group_ids:
                duplicate_group_ids.append(
                    {"cluster_id": cluster_id, "group_id": group_id}
                )
            else:
                seen_group_ids.add(group_id)
            raw_rule_ids = [
                str(item).strip()
                for item in (group.get("rule_ids") or [])
                if str(item).strip()
            ]
            if not raw_rule_ids:
                empty_group_rule_ids.append(
                    {"cluster_id": cluster_id, "group_id": group_id}
                )
            assigned.extend(raw_rule_ids)

    counts: Dict[str, int] = {}
    for rule_id in assigned:
        counts[rule_id] = counts.get(rule_id, 0) + 1
    assigned_unique = list(dict.fromkeys(assigned))
    assigned_set = set(assigned_unique)
    missing = sorted(expected_set - assigned_set)
    foreign = sorted(assigned_set - expected_set)
    duplicates = sorted(rule_id for rule_id, count in counts.items() if count > 1)
    passed = not any(
        (
            not expected,
            invalid_cluster_indexes,
            empty_cluster_ids,
            duplicate_cluster_ids,
            empty_group_ids,
            duplicate_group_ids,
            empty_group_rule_ids,
            missing,
            foreign,
            duplicates,
        )
    )
    return {
        "kind": kind,
        "topic_key": topic_key,
        "passed": passed,
        "expected_rule_count": len(expected),
        "assigned_rule_count": len(assigned_unique),
        "expected_rule_ids": expected,
        "assigned_rule_ids": assigned_unique,
        "missing_rule_ids": missing,
        "foreign_rule_ids": foreign,
        "duplicate_rule_ids": duplicates,
        "invalid_cluster_indexes": invalid_cluster_indexes,
        "empty_cluster_id_indexes": empty_cluster_ids,
        "duplicate_cluster_ids": sorted(set(duplicate_cluster_ids)),
        "empty_group_ids": empty_group_ids,
        "duplicate_group_ids": duplicate_group_ids,
        "empty_group_rule_ids": empty_group_rule_ids,
        "base_cluster_count": int(base_cluster_count),
        "generated_cluster_count": len(cluster_list),
    }


def _compose_incremental_blueprints(
    *,
    base_catalog: Dict[str, Any],
    precluster_catalog: Dict[str, Any],
    generated_blueprints: Dict[str, List[Dict[str, Any]]],
    policy: Mapping[str, Any],
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Any],
]:
    base_blueprints = _catalog_blueprints(base_catalog)
    base_snapshot = build_catalog_snapshot(base_catalog)
    precluster_snapshot = build_catalog_snapshot(precluster_catalog)
    incremental_rule_ids = sorted(
        set(precluster_snapshot["rules"]) - set(base_snapshot["rules"])
    )
    allowed_topic_keys = {
        f"{str(item.get('domain') or '').strip()}::{str(item.get('topic') or '').strip()}".casefold()
        for item in policy.get("allowed_recluster_topics", []) or []
        if isinstance(item, dict)
        and str(item.get("domain") or "").strip()
        and str(item.get("topic") or "").strip()
    }
    frozen_generated = {
        key: clusters
        for key, clusters in generated_blueprints.items()
        if key.casefold() not in allowed_topic_keys
    }
    additive_blueprints = _filter_blueprints_to_rules(
        frozen_generated,
        incremental_rule_ids,
    )
    combined = _merge_incremental_blueprints(base_blueprints, additive_blueprints)
    changed_blueprints = copy.deepcopy(additive_blueprints)

    precluster_rules_by_topic = _catalog_topic_rule_ids(precluster_catalog)
    incremental_rule_set = set(incremental_rule_ids)
    additive_expected_by_topic = {
        topic_key: [
            rule_id for rule_id in rule_ids if rule_id in incremental_rule_set
        ]
        for topic_key, rule_ids in precluster_rules_by_topic.items()
        if topic_key not in allowed_topic_keys
        and any(rule_id in incremental_rule_set for rule_id in rule_ids)
    }
    additive_audit_keys = sorted(
        set(additive_expected_by_topic).union(additive_blueprints)
    )
    additive_audits = [
        _audit_blueprint_rule_cover(
            topic_key=topic_key,
            clusters=additive_blueprints.get(topic_key, []),
            expected_rule_ids=additive_expected_by_topic.get(topic_key, []),
            kind="additive_delta",
            base_cluster_count=len(base_blueprints.get(topic_key, [])),
        )
        for topic_key in additive_audit_keys
    ]

    scoped_audits: List[Dict[str, Any]] = []
    for topic_key in sorted(allowed_topic_keys):
        expected_rule_ids = precluster_rules_by_topic.get(topic_key, [])
        scoped_clusters = copy.deepcopy(generated_blueprints.get(topic_key, []))
        audit = _audit_blueprint_rule_cover(
            topic_key=topic_key,
            clusters=scoped_clusters,
            expected_rule_ids=expected_rule_ids,
            kind="scoped_recluster",
            base_cluster_count=len(base_blueprints.get(topic_key, [])),
        )
        scoped_audits.append(audit)
        if audit["passed"]:
            combined[topic_key] = scoped_clusters
            changed_blueprints[topic_key] = scoped_clusters

    audits = [*additive_audits, *scoped_audits]
    topology_validation = {
        "passed": all(item["passed"] for item in audits),
        "mode": str(policy.get("mode") or ""),
        "incremental_rule_ids": incremental_rule_ids,
        "allowed_recluster_topic_keys": sorted(allowed_topic_keys),
        "additive_topic_audits": additive_audits,
        "scoped_topic_audits": scoped_audits,
    }
    return combined, changed_blueprints, topology_validation


def _stable_existing_then_new(
    base_items: Iterable[Dict[str, Any]],
    candidate_items: Iterable[Dict[str, Any]],
    *,
    id_fields: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    def item_id(item: Dict[str, Any]) -> str:
        return next(
            (
                str(item.get(field) or "").strip()
                for field in id_fields
                if str(item.get(field) or "").strip()
            ),
            "",
        )

    candidate_list = [item for item in candidate_items if isinstance(item, dict)]
    candidate_by_id = {item_id(item): item for item in candidate_list if item_id(item)}
    base_ids = [
        item_id(item) for item in base_items if isinstance(item, dict) and item_id(item)
    ]
    base_id_set = set(base_ids)
    return [
        *[candidate_by_id[item_id] for item_id in base_ids if item_id in candidate_by_id],
        *[item for item in candidate_list if item_id(item) not in base_id_set],
    ]


def _stabilize_catalog_order(
    base_catalog: Dict[str, Any],
    candidate_catalog: Dict[str, Any],
    *,
    recluster_topic_keys: Iterable[str] = (),
) -> None:
    reclustered = {str(item).casefold() for item in recluster_topic_keys}
    base_domains = [
        domain
        for domain in base_catalog.get("domains", []) or []
        if isinstance(domain, dict)
    ]
    candidate_domains = [
        domain
        for domain in candidate_catalog.get("domains", []) or []
        if isinstance(domain, dict)
    ]
    candidate_catalog["domains"] = _stable_existing_then_new(
        base_domains,
        candidate_domains,
        id_fields=("name", "id"),
    )
    base_domains_by_name = {
        str(domain.get("name") or domain.get("id") or ""): domain
        for domain in base_domains
    }
    for domain in candidate_catalog["domains"]:
        domain_name = str(domain.get("name") or domain.get("id") or "")
        base_domain = base_domains_by_name.get(domain_name)
        if not isinstance(base_domain, dict):
            continue
        domain["topics"] = _stable_existing_then_new(
            base_domain.get("topics", []) or [],
            domain.get("topics", []) or [],
            id_fields=("name", "id"),
        )
    base_topics = {
        (str(domain.get("name") or ""), str(topic.get("name") or "")): topic
        for domain in base_catalog.get("domains", []) or []
        if isinstance(domain, dict)
        for topic in domain.get("topics", []) or []
        if isinstance(topic, dict)
    }
    for domain in candidate_catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = str(domain.get("name") or "")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            base_topic = base_topics.get((domain_name, str(topic.get("name") or "")))
            if not isinstance(base_topic, dict):
                continue
            topic["rules"] = _stable_existing_then_new(
                base_topic.get("rules", []) or [],
                topic.get("rules", []) or [],
                id_fields=("rule_id", "id"),
            )
            topic_key = f"{domain_name}::{str(topic.get('name') or '')}".casefold()
            if topic_key in reclustered:
                continue
            topic["scenario_clusters"] = _stable_existing_then_new(
                base_topic.get("scenario_clusters", []) or [],
                topic.get("scenario_clusters", []) or [],
                id_fields=("id", "cluster_id"),
            )
            base_clusters = {
                str(cluster.get("id") or cluster.get("cluster_id") or ""): cluster
                for cluster in base_topic.get("scenario_clusters", []) or []
                if isinstance(cluster, dict)
            }
            for cluster in topic.get("scenario_clusters", []) or []:
                if not isinstance(cluster, dict):
                    continue
                cluster_id = str(
                    cluster.get("id") or cluster.get("cluster_id") or ""
                )
                base_cluster = base_clusters.get(cluster_id)
                if not isinstance(base_cluster, dict):
                    continue
                base_rule_ids = list(base_cluster.get("rule_ids") or [])
                candidate_rule_ids = list(cluster.get("rule_ids") or [])
                base_rule_set = set(base_rule_ids)
                cluster["rule_ids"] = [
                    *[rule_id for rule_id in base_rule_ids if rule_id in candidate_rule_ids],
                    *[rule_id for rule_id in candidate_rule_ids if rule_id not in base_rule_set],
                ]
                if base_cluster.get("rule_groups") or cluster.get("rule_groups"):
                    cluster["rule_groups"] = _stable_existing_then_new(
                        base_cluster.get("rule_groups", []) or [],
                        cluster.get("rule_groups", []) or [],
                        id_fields=("id", "group_id"),
                    )
                else:
                    cluster.pop("rule_groups", None)
                base_groups = {
                    str(group.get("id") or group.get("group_id") or ""): group
                    for group in base_cluster.get("rule_groups", []) or []
                    if isinstance(group, dict)
                }
                for group in cluster.get("rule_groups", []) or []:
                    if not isinstance(group, dict):
                        continue
                    group_id = str(group.get("id") or group.get("group_id") or "")
                    base_group = base_groups.get(group_id)
                    if not isinstance(base_group, dict):
                        continue
                    base_group_rule_ids = list(base_group.get("rule_ids") or [])
                    candidate_group_rule_ids = list(group.get("rule_ids") or [])
                    base_group_rule_set = set(base_group_rule_ids)
                    group["rule_ids"] = [
                        *[
                            rule_id
                            for rule_id in base_group_rule_ids
                            if rule_id in candidate_group_rule_ids
                        ],
                        *[
                            rule_id
                            for rule_id in candidate_group_rule_ids
                            if rule_id not in base_group_rule_set
                        ],
                    ]


def finalize_incremental_update(
    *,
    workspace: Path,
    base_catalog_path: Path,
    knowledge_path: Path = Path("catalogs/rules_catalog_top_down.json"),
    tagged_path: Path = Path("catalogs/rules_300_tagged.json"),
) -> Dict[str, Any]:
    manifest = _load_json(workspace / "incremental_manifest.json")
    manifest_validation = _validate_manifest_inputs(
        manifest,
        overrides={
            "base_catalog": base_catalog_path,
            "knowledge": knowledge_path,
            "tagged": tagged_path,
        },
    )
    if not manifest_validation["passed"]:
        report = {
            "ready_for_retrieval_evaluation": False,
            "promotion_ready": False,
            "promotion_blocker": (
                "Incremental manifest identity validation failed. Re-run the prepare "
                "step against the intended frozen baseline."
            ),
            "manifest_validation": manifest_validation,
        }
        _write_json(workspace / "incremental_validation.json", report)
        details = "; ".join(
            f"{item.get('input')}: {item.get('reason')}"
            for item in manifest_validation["mismatches"]
        )
        raise ValueError(f"Incremental manifest validation failed: {details}")

    candidate_delta_validation = _validate_candidate_delta(manifest, workspace)
    if not candidate_delta_validation["passed"]:
        report = {
            "ready_for_retrieval_evaluation": False,
            "promotion_ready": False,
            "promotion_blocker": (
                "Incremental candidate delta validation failed. Re-run the prepare "
                "step instead of editing the workspace outputs."
            ),
            "manifest_validation": manifest_validation,
            "candidate_delta_validation": candidate_delta_validation,
        }
        _write_json(workspace / "incremental_validation.json", report)
        details = "; ".join(
            str(item.get("field") or item.get("reason") or "mismatch")
            for item in candidate_delta_validation["mismatches"]
        )
        raise ValueError(f"Incremental candidate delta validation failed: {details}")

    inputs = manifest["inputs"]
    base_catalog = _load_json(base_catalog_path)
    base_generalized = _load_json(
        _portable_path(inputs["base_generalized"]["path"])
    )
    generalized = _load_json(workspace / "semantic_experience_generalized.json")
    raw_proposals = _load_json(workspace / "cluster_proposals.json")
    output_binding_validation = _validate_run_output_bindings(
        manifest,
        generalized=generalized,
        proposals=raw_proposals,
    )
    if not output_binding_validation["passed"]:
        report = {
            "ready_for_retrieval_evaluation": False,
            "promotion_ready": False,
            "promotion_blocker": (
                "Incremental run output binding failed. Re-run generalization and "
                "cluster labeling with this manifest."
            ),
            "manifest_validation": manifest_validation,
            "candidate_delta_validation": candidate_delta_validation,
            "output_binding_validation": output_binding_validation,
        }
        _write_json(workspace / "incremental_validation.json", report)
        details = "; ".join(
            str(item.get("output") or "output")
            for item in output_binding_validation["mismatches"]
        )
        raise ValueError(f"Incremental run output binding failed: {details}")

    candidate_bank = _load_json(workspace / "semantic_experience_distilled.json")
    formal = _load_json(
        workspace / "semantic_experience_generalized_for_cluster.json"
    )
    formal_rule_input = _load_json(workspace / "formal_rule_embedding_input.json")
    formal_clusters = _load_json(
        workspace / "formal_rule_embedding_clusters.json"
    )
    precluster_catalog = _load_json(workspace / "catalog_precluster.json")
    artifact_chain_validation = _validate_artifact_chain(
        manifest=manifest,
        workspace=workspace,
        base_catalog_path=base_catalog_path,
        knowledge_path=knowledge_path,
        tagged_path=tagged_path,
        generalized=generalized,
        formal=formal,
        precluster_catalog=precluster_catalog,
        formal_rule_input=formal_rule_input,
        formal_clusters=formal_clusters,
        proposals=raw_proposals,
    )
    deterministic_artifact_validation = _validate_deterministic_formal_bundle(
        workspace=workspace,
        base_catalog_path=base_catalog_path,
        knowledge_path=knowledge_path,
        tagged_path=tagged_path,
        formal=formal,
        precluster_catalog=precluster_catalog,
        formal_rule_input=formal_rule_input,
    )
    formal_cluster_coverage = _validate_formal_cluster_coverage(
        formal_rule_input,
        formal_clusters,
    )
    generalization_fingerprint_validation = (
        _validate_generalization_fingerprints(
            manifest=manifest,
            workspace=workspace,
            generalized=generalized,
        )
    )
    proposal_fingerprint_validation = _validate_proposal_fingerprints(
        manifest=manifest,
        workspace=workspace,
        formal_clusters=formal_clusters,
        formal_rule_input=formal_rule_input,
        proposals=raw_proposals,
    )
    artifact_bundle_validation = {
        "passed": bool(
            artifact_chain_validation["passed"]
            and deterministic_artifact_validation["passed"]
            and generalization_fingerprint_validation["passed"]
            and formal_cluster_coverage["passed"]
            and proposal_fingerprint_validation["passed"]
        ),
        "artifact_chain": artifact_chain_validation,
        "deterministic_replay": deterministic_artifact_validation,
        "generalization_fingerprints": generalization_fingerprint_validation,
        "formal_cluster_coverage": formal_cluster_coverage,
        "proposal_fingerprints": proposal_fingerprint_validation,
    }
    if not artifact_bundle_validation["passed"]:
        report = {
            "ready_for_retrieval_evaluation": False,
            "promotion_ready": False,
            "promotion_blocker": (
                "Incremental artifact lineage validation failed. Re-run the full "
                "manifest runbook without mixing intermediate outputs."
            ),
            "manifest_validation": manifest_validation,
            "candidate_delta_validation": candidate_delta_validation,
            "output_binding_validation": output_binding_validation,
            "artifact_bundle_validation": artifact_bundle_validation,
        }
        _write_json(workspace / "incremental_validation.json", report)
        raise ValueError("Incremental artifact lineage validation failed")

    proposals = add_catalog_fallback_proposals(
        raw_proposals,
        precluster_catalog,
    )
    rule_index = {
        str(rule.get("rule_id") or ""): rule
        for domain in precluster_catalog.get("domains", []) or []
        for topic in domain.get("topics", []) or []
        for rule in topic.get("rules", []) or []
        if str(rule.get("rule_id") or "")
    }
    generated_blueprints = build_generated_blueprints_from_refined_proposals(
        proposals,
        rule_index=rule_index,
    )
    (
        combined_blueprints,
        incremental_blueprints,
        topology_validation,
    ) = _compose_incremental_blueprints(
        base_catalog=base_catalog,
        precluster_catalog=precluster_catalog,
        generated_blueprints=generated_blueprints,
        policy=manifest["change_policy"],
    )
    if not topology_validation["passed"]:
        report = {
            "ready_for_retrieval_evaluation": False,
            "promotion_ready": False,
            "promotion_blocker": (
                "Generated blueprints do not provide exact, topic-local Rule coverage. "
                "Re-run embedding clustering and cluster labeling with this manifest."
            ),
            "manifest_validation": manifest_validation,
            "candidate_delta_validation": candidate_delta_validation,
            "output_binding_validation": output_binding_validation,
            "artifact_bundle_validation": artifact_bundle_validation,
            "topology_validation": topology_validation,
        }
        _write_json(workspace / "incremental_validation.json", report)
        failed_topics = [
            item["topic_key"]
            for item in [
                *topology_validation["additive_topic_audits"],
                *topology_validation["scoped_topic_audits"],
            ]
            if not item["passed"]
        ]
        raise ValueError(
            "Incremental blueprint topology validation failed: "
            + ", ".join(failed_topics)
        )
    incremental_blueprints_path = workspace / "cluster_blueprints_incremental.json"
    generated_blueprints_path = workspace / "cluster_blueprints_generated.json"
    _write_json(incremental_blueprints_path, incremental_blueprints)
    _write_json(generated_blueprints_path, combined_blueprints)

    final_catalog = build_unified_catalog(
        knowledge_path=knowledge_path,
        distilled_path=workspace
        / "semantic_experience_generalized_for_cluster.json",
        tagged_path=tagged_path,
        scenario_cluster_blueprints_paths=[generated_blueprints_path],
    )
    _stabilize_catalog_order(
        base_catalog,
        final_catalog,
        recluster_topic_keys=topology_validation["allowed_recluster_topic_keys"],
    )
    final_catalog.setdefault("metadata", {}).update(
        {
            "incremental_configuration_sha256": str(
                manifest.get("configuration_sha256") or ""
            ),
            "incremental_lineage": _expected_lineage(
                stage="final_catalog",
                input_sha256={
                    "base_catalog": _sha256(base_catalog_path),
                    "cluster_proposals": _sha256(
                        workspace / "cluster_proposals.json"
                    ),
                    "formal_rules": _sha256(
                        workspace
                        / "semantic_experience_generalized_for_cluster.json"
                    ),
                    "generalized": _sha256(
                        workspace / "semantic_experience_generalized.json"
                    ),
                    "knowledge": _sha256(knowledge_path),
                    "precluster_catalog": _sha256(
                        workspace / "catalog_precluster.json"
                    ),
                    "tagged": _sha256(tagged_path),
                },
            ),
        }
    )
    final_catalog_path = workspace / "rules_unified_incremental.json"
    _write_json(final_catalog_path, final_catalog)

    structure = validate_catalog_structure(final_catalog)
    coarsening = audit_rule_coarsening(
        candidates=_load_json(
            workspace / "semantic_experience_distilled_for_cluster.json"
        ),
        generalized=_load_json(
            workspace / "semantic_experience_generalized.json"
        ),
        formal=_load_json(
            workspace / "semantic_experience_generalized_for_cluster.json"
        ),
        catalog=final_catalog,
    )
    affected_topics = {
        (str(item.get("domain") or ""), str(item.get("topic") or ""))
        for item in manifest.get("declared_change_topics", []) or []
        if isinstance(item, dict)
    }
    catalog_diff = compare_catalog_snapshots(
        build_catalog_snapshot(base_catalog),
        build_catalog_snapshot(final_catalog),
        affected_topics=affected_topics,
    )
    generalized_diff = compare_generalized_outputs(base_generalized, generalized)
    raw_lineage = build_generalized_lineage(generalized, candidate_bank)
    changed_candidate_ids = (
        manifest.get("candidate_delta", {}).get("changed_candidate_ids", [])
        if isinstance(manifest.get("candidate_delta"), dict)
        else []
    )
    lineage = audit_added_rule_lineage(
        added_rule_ids=catalog_diff["added_rule_ids"],
        lineage=raw_lineage,
        changed_candidate_ids=changed_candidate_ids,
        generalized=generalized,
    )
    change_policy = evaluate_change_policy(
        catalog_diff=catalog_diff,
        generalized_diff=generalized_diff,
        lineage=lineage,
        policy=manifest["change_policy"],
    )
    identity_stable = not catalog_diff["removed_rule_ids"]
    ready_for_retrieval = bool(
        structure.get("valid")
        and coarsening.get("complete")
        and manifest_validation["passed"]
        and candidate_delta_validation["passed"]
        and output_binding_validation["passed"]
        and artifact_bundle_validation["passed"]
        and topology_validation["passed"]
        and change_policy["passed"]
    )

    if not structure.get("valid"):
        promotion_blocker = "The incremental catalog failed structural validation."
    elif not coarsening.get("complete"):
        promotion_blocker = "Candidate-to-formal coarsening coverage is incomplete."
    elif not change_policy["passed"]:
        promotion_blocker = (
            "Incremental change policy failed: "
            + ", ".join(change_policy["violations"])
        )
    else:
        promotion_blocker = (
            "Pass an independent full-verifier regression and manual review before "
            "replacing the current catalog."
        )

    report = {
        "ready_for_retrieval_evaluation": ready_for_retrieval,
        "promotion_ready": False,
        "promotion_blocker": promotion_blocker,
        "outputs": {
            "catalog": str(final_catalog_path),
            "catalog_sha256": _sha256(final_catalog_path),
            "generated_blueprints": str(generated_blueprints_path),
            "generated_blueprints_sha256": _sha256(generated_blueprints_path),
            "incremental_blueprints": str(incremental_blueprints_path),
            "incremental_blueprints_sha256": _sha256(
                incremental_blueprints_path
            ),
        },
        "manifest_validation": manifest_validation,
        "candidate_delta_validation": candidate_delta_validation,
        "output_binding_validation": output_binding_validation,
        "artifact_bundle_validation": artifact_bundle_validation,
        "topology_validation": topology_validation,
        "structure": structure,
        "coarsening": coarsening,
        "generalized_diff": generalized_diff,
        "lineage": lineage,
        "catalog_diff": catalog_diff,
        "change_policy": change_policy,
        "change_scope": {
            "declared_affected_topics": [
                {"domain": domain, "topic": topic}
                for domain, topic in sorted(affected_topics)
            ],
            "changed_topics": catalog_diff["topics"]["changed"],
            "unexpected_changed_topics": catalog_diff["topics"][
                "unexpected_changed"
            ],
            "identity_stable": identity_stable,
            "base_rule_ids_preserved": identity_stable,
            "added_rule_ids": catalog_diff["added_rule_ids"],
            "removed_rule_ids": catalog_diff["removed_rule_ids"],
            "modified_existing_rule_ids": catalog_diff[
                "modified_existing_rule_ids"
            ],
            "cluster_moved_existing_rule_ids": [
                item["rule_id"]
                for item in catalog_diff["rule_cluster_mapping"]["changes"]
            ],
            "group_moved_existing_rule_ids": [
                item["rule_id"]
                for item in catalog_diff["rule_group_mapping"]["changes"]
            ],
        },
    }
    _write_json(workspace / "incremental_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate an isolated incremental unified-rule catalog."
    )
    parser.add_argument(
        "--workspace",
        default="results/unified_rules_incremental",
    )
    parser.add_argument("--base-catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    args = parser.parse_args()

    report = finalize_incremental_update(
        workspace=Path(args.workspace),
        base_catalog_path=Path(args.base_catalog),
        knowledge_path=Path(args.knowledge),
        tagged_path=Path(args.tagged),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready_for_retrieval_evaluation"] else 1)


if __name__ == "__main__":
    main()
