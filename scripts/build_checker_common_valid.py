from __future__ import annotations

"""Freeze the common-valid subset of three controlled Checker replay arms.

The existing evaluators can consume the emitted dataset unchanged.  This tool
only establishes the paired sample population and its audit manifest; it does
not inspect labels or calculate effect metrics.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CHECKER_GATE_MODES: Tuple[str, ...] = (
    "legacy",
    "dual_evidence",
    "dual_evidence_consistency",
)
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_TYPE = "checker_common_valid_v1"
REPLAY_REPORT_TYPES = {
    "semantic_retrieval": "checker_only_frozen_retrieval_replay",
    "frozen_target_binding": "checker_only_frozen_target_binding_replay",
}
TERMINAL_REPORT_STATUSES = {
    "complete",
    "complete_with_failures",
    "incomplete_failures",
}
SUCCESS_CHECKER_STATUSES = {
    "complete",
    "complete_no_rules",
    "ok",
    "success",
    "valid_empty",
    "valid_with_diagnostics",
}

# These fields identify arm-local artifact locations, not a semantic or runtime
# difference.  Every other configuration field must compare exactly.
ALLOWED_ARM_CONFIGURATION_DIFFERENCES: Tuple[str, ...] = (
    "system_arm",
    "llm_trace_path",
    "output_path",
    "report_path",
    "checkpoint_path",
    "results_path",
)


class CommonValidError(ValueError):
    """Raised when paired replay artifacts are incomplete or incomparable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _contains_prompt_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in {"system_prompt", "user_prompt"}:
                return True
            if _contains_prompt_field(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_prompt_field(item) for item in value)
    return False


def _audit_llm_trace(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise CommonValidError(f"LLM trace does not exist: {path}")
    digest = hashlib.sha256()
    size_bytes = 0
    record_count = 0
    raw_response_record_count = 0
    parse_status_counts: Counter[str] = Counter()
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                size_bytes += len(raw_line)
                if not raw_line.endswith(b"\n"):
                    raise CommonValidError(
                        f"LLM trace ends with an incomplete JSONL record at line {line_number}: {path}"
                    )
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CommonValidError(
                        f"LLM trace contains invalid JSONL at line {line_number}: {path}"
                    ) from exc
                if not isinstance(record, dict):
                    raise CommonValidError(
                        f"LLM trace record {line_number} must be a JSON object: {path}"
                    )
                parse_status = str(record.get("parse_status") or "").strip()
                if not parse_status:
                    raise CommonValidError(
                        f"LLM trace record {line_number} is missing parse_status: {path}"
                    )
                if _contains_prompt_field(record):
                    raise CommonValidError(
                        f"LLM trace must not contain prompt fields: {path}"
                    )
                record_count += 1
                parse_status_counts[parse_status] += 1
                if "raw_response" in record:
                    raw_response_record_count += 1
    except OSError as exc:
        raise CommonValidError(f"cannot read LLM trace {path}: {exc}") from exc
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "record_count": record_count,
        "raw_response_record_count": raw_response_record_count,
        "prompt_record_count": 0,
        "prompts_included": False,
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
    }


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise CommonValidError(f"{label} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommonValidError(f"{label} is not valid JSON: {path}: {exc}") from exc


def _load_json_array(path: Path, *, label: str) -> List[Dict[str, Any]]:
    payload = _load_json(path, label=label)
    if not isinstance(payload, list):
        raise CommonValidError(f"{label} must be a JSON array")
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise CommonValidError(f"{label}[{index}] must be a JSON object")
        rows.append(item)
    return rows


def _typed_id(value: Any, *, label: str) -> Tuple[str, Dict[str, Any]]:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise CommonValidError(
            f"{label} must be a non-empty string or an integer (bool is invalid)"
        )
    if isinstance(value, str) and not value.strip():
        raise CommonValidError(f"{label} must not be an empty string")
    type_name = "str" if isinstance(value, str) else "int"
    descriptor = {"type": type_name, "value": value}
    return f"{type_name}:{_canonical_json(value)}", descriptor


def _ordered_typed_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    keys: List[str] = []
    descriptors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if "id" not in row:
            raise CommonValidError(f"{label}[{index}] is missing id")
        key, descriptor = _typed_id(row.get("id"), label=f"{label}[{index}].id")
        if key in seen:
            raise CommonValidError(
                f"duplicate typed ID in {label}: {descriptor!r}"
            )
        seen.add(key)
        keys.append(key)
        descriptors.append(descriptor)
    return keys, descriptors


def _fingerprint(
    path: Path,
    *,
    record_count: Optional[int] = None,
    ordered_typed_ids: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if record_count is not None:
        result["record_count"] = int(record_count)
    if ordered_typed_ids is not None:
        typed_ids = [dict(item) for item in ordered_typed_ids]
        result["ordered_typed_ids"] = typed_ids
        result["ordered_typed_ids_sha256"] = _object_sha256(typed_ids)
    return result


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CommonValidError(f"{label} must be an integer >= {minimum}")
    return value


def _require_fixed_number(value: Any, *, expected: float, label: str) -> None:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise CommonValidError(
            f"{label} must be a JSON number fixed at {expected:g}"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != expected:
        raise CommonValidError(f"{label} must be fixed at {expected:g}")


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommonValidError(f"{label} must be a JSON object")
    return value


def _validate_configuration_identity(
    configuration: Mapping[str, Any],
    *,
    mode: str,
    dataset_sha256: str,
) -> None:
    if configuration.get("system_arm") != mode:
        raise CommonValidError(
            f"{mode} report configuration.system_arm does not match the arm"
        )
    if configuration.get("checker_cache_enabled") is not False:
        raise CommonValidError(
            f"{mode} controlled replay must disable the Checker cache"
        )
    if configuration.get("llm_trace_include_prompts") is not False:
        raise CommonValidError(
            f"{mode} controlled replay must exclude prompts from the LLM trace"
        )
    if configuration.get("retrieval_execution") is not False:
        raise CommonValidError(
            f"{mode} controlled replay must set retrieval_execution=false"
        )
    if configuration.get("bottom_up_enabled") is not False:
        raise CommonValidError(
            f"{mode} controlled replay must set bottom_up_enabled=false"
        )
    candidate_source = str(
        configuration.get("candidate_source") or "semantic_retrieval"
    )
    if candidate_source not in REPLAY_REPORT_TYPES:
        raise CommonValidError(
            f"{mode} configuration.candidate_source is unsupported"
        )
    expected_retrieval_source = (
        "frozen_target_binding"
        if candidate_source == "frozen_target_binding"
        else "frozen_semantic_trace"
    )
    if configuration.get("retrieval_source") not in {
        None,
        expected_retrieval_source,
    }:
        raise CommonValidError(
            f"{mode} configuration.retrieval_source does not match candidate_source"
        )
    if candidate_source == "frozen_target_binding" and not _valid_sha256(
        configuration.get("target_mapping_sha256")
    ):
        raise CommonValidError(
            f"{mode} target-binding configuration must bind target_mapping_sha256"
        )
    llm_trace_path = configuration.get("llm_trace_path")
    if not isinstance(llm_trace_path, str) or not llm_trace_path.strip():
        raise CommonValidError(f"{mode} configuration.llm_trace_path is missing")

    required_hashes = (
        "dataset_sha256",
        "retrieval_trace_sha256",
        "unified_catalog_sha256",
        "selection_projection_sha256",
        "frozen_manifest_sha256",
    )
    for key in required_hashes:
        if not _valid_sha256(configuration.get(key)):
            raise CommonValidError(f"{mode} configuration.{key} is not a SHA-256")
    if configuration.get("dataset_sha256") != dataset_sha256:
        raise CommonValidError(
            f"{mode} configuration.dataset_sha256 does not match the supplied dataset"
        )

    source = _require_mapping(
        configuration.get("source_identity"),
        label=f"{mode} configuration.source_identity",
    )
    if not _valid_sha256(source.get("source_tree_sha256")):
        raise CommonValidError(
            f"{mode} configuration.source_identity.source_tree_sha256 is invalid"
        )
    runtime = _require_mapping(
        configuration.get("runtime_identity"),
        label=f"{mode} configuration.runtime_identity",
    )
    if runtime.get("is_conda") is not True or not _valid_sha256(
        runtime.get("package_set_sha256")
    ):
        raise CommonValidError(
            f"{mode} configuration.runtime_identity must bind a conda package set"
        )
    api = _require_mapping(
        configuration.get("api_transport_identity"),
        label=f"{mode} configuration.api_transport_identity",
    )
    if not _valid_sha256(api.get("endpoint_sha256")):
        raise CommonValidError(
            f"{mode} configuration.api_transport_identity.endpoint_sha256 is invalid"
        )


def _normalized_configuration(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(configuration))
    for key in ALLOWED_ARM_CONFIGURATION_DIFFERENCES:
        normalized.pop(key, None)
    return normalized


def _configuration_differences(
    left: Any,
    right: Any,
    *,
    prefix: str = "configuration",
    limit: int = 8,
) -> List[str]:
    if limit <= 0:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        differences: List[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}"
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(
                    _configuration_differences(
                        left[key],
                        right[key],
                        prefix=path,
                        limit=limit - len(differences),
                    )
                )
            if len(differences) >= limit:
                break
        return differences
    if left != right:
        return [prefix]
    return []


def _row_failure_reason(row: Mapping[str, Any]) -> str:
    selection = str(row.get("selection_strategy") or "").strip().lower()
    if selection not in {
        "semantic_tree_selection",
        "semantic_tree_empty",
        "semantic_error",
        "semantic_unavailable",
        "target_rule_binding",
    }:
        raise CommonValidError(
            f"result row has unsupported selection_strategy={selection!r}"
        )
    failures = row.get("checker_failures")
    if not isinstance(failures, list):
        raise CommonValidError("result row checker_failures must be an array")
    failure_count = _require_int(
        row.get("checker_failure_count"),
        label="result row checker_failure_count",
    )
    if failure_count != len(failures):
        raise CommonValidError(
            "result row checker_failure_count does not match checker_failures"
        )
    checker_status = str(row.get("checker_status") or "").strip().lower()
    if not checker_status:
        raise CommonValidError("result row checker_status is missing")
    if not isinstance(row.get("replay_completed"), bool):
        raise CommonValidError("result row replay_completed must be an explicit boolean")
    if selection in {"semantic_error", "semantic_unavailable"} or str(
        row.get("semantic_selection_error") or ""
    ).strip():
        return "semantic_retrieval_failure"
    if failure_count > 0 or checker_status not in SUCCESS_CHECKER_STATUSES:
        return "checker_failure"
    if row.get("replay_completed") is not True:
        return "replay_incomplete"
    return ""


def _reported_artifact_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CommonValidError(f"{label}.path is missing")
    path = Path(value)
    if not path.is_file():
        raise CommonValidError(f"{label} does not exist: {path}")
    return path


def _validate_reported_artifact(
    entry: Mapping[str, Any],
    *,
    label: str,
    expected_path: Optional[Path] = None,
) -> Path:
    path = _reported_artifact_path(entry.get("path"), label=label)
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise CommonValidError(f"{label}.path does not match the supplied artifact")
    expected_sha256 = entry.get("sha256")
    if not _valid_sha256(expected_sha256):
        raise CommonValidError(f"{label}.sha256 is invalid")
    if _sha256_file(path) != expected_sha256:
        raise CommonValidError(f"{label} SHA256 does not match the actual file")
    return path


def _validate_arm(
    *,
    mode: str,
    dataset_path: Path,
    result_path: Path,
    report_path: Path,
    dataset_keys: Sequence[str],
    dataset_typed_ids: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_sha256: str,
) -> Dict[str, Any]:
    rows = _load_json_array(result_path, label=f"{mode} results")
    result_keys, result_typed_ids = _ordered_typed_ids(
        rows,
        label=f"{mode} results",
    )
    if list(result_keys) != list(dataset_keys):
        raise CommonValidError(
            f"{mode} results must contain exactly the dataset typed IDs in the same order"
        )
    if list(result_typed_ids) != [dict(item) for item in dataset_typed_ids]:
        raise CommonValidError(f"{mode} typed ID projection does not match the dataset")

    report = _load_json(report_path, label=f"{mode} report")
    report = _require_mapping(report, label=f"{mode} report")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1:
        raise CommonValidError(f"{mode} report has an unsupported schema_version")
    if report.get("report_type") not in set(REPLAY_REPORT_TYPES.values()):
        raise CommonValidError(f"{mode} report has the wrong report_type")
    if report.get("status") not in TERMINAL_REPORT_STATUSES:
        raise CommonValidError(f"{mode} report is not terminal: {report.get('status')!r}")

    configuration = _require_mapping(
        report.get("configuration"),
        label=f"{mode} report.configuration",
    )
    _validate_configuration_identity(
        configuration,
        mode=mode,
        dataset_sha256=dataset_sha256,
    )
    candidate_source = str(
        configuration.get("candidate_source") or "semantic_retrieval"
    )
    if report.get("report_type") != REPLAY_REPORT_TYPES[candidate_source]:
        raise CommonValidError(
            f"{mode} report_type does not match configuration.candidate_source"
        )
    if report.get("candidate_source") not in {None, candidate_source}:
        raise CommonValidError(
            f"{mode} report candidate_source does not match configuration"
        )
    if candidate_source == "frozen_target_binding" and report.get(
        "target_mapping_sha256"
    ) != configuration.get("target_mapping_sha256"):
        raise CommonValidError(
            f"{mode} report target_mapping_sha256 does not match configuration"
        )
    configuration_sha256 = str(report.get("configuration_sha256") or "")
    if configuration_sha256 != _object_sha256(configuration):
        raise CommonValidError(f"{mode} report configuration_sha256 does not match")

    output = _require_mapping(report.get("output"), label=f"{mode} report.output")
    output_path_value = output.get("path")
    if (
        not isinstance(output_path_value, str)
        or not output_path_value.strip()
        or Path(output_path_value).resolve() != result_path.resolve()
    ):
        raise CommonValidError(
            f"{mode} report output.path does not match the supplied results"
        )
    actual_result_sha256 = _sha256_file(result_path)
    if output.get("sha256") != actual_result_sha256:
        raise CommonValidError(
            f"{mode} report output SHA256 does not match the supplied results"
        )
    if _require_int(output.get("record_count"), label=f"{mode} output.record_count") != len(
        rows
    ):
        raise CommonValidError(f"{mode} report output.record_count does not match")

    inputs = _require_mapping(report.get("inputs"), label=f"{mode} report.inputs")
    for input_name, configuration_key in (
        ("dataset", "dataset_sha256"),
        ("retrieval_trace", "retrieval_trace_sha256"),
        ("unified_catalog", "unified_catalog_sha256"),
    ):
        entry = _require_mapping(
            inputs.get(input_name),
            label=f"{mode} report.inputs.{input_name}",
        )
        if entry.get("sha256") != configuration.get(configuration_key):
            raise CommonValidError(
                f"{mode} report input {input_name} SHA256 disagrees with configuration"
            )
        _validate_reported_artifact(
            entry,
            label=f"{mode} report.inputs.{input_name}",
            expected_path=dataset_path if input_name == "dataset" else None,
        )
    if inputs["dataset"].get("sha256") != dataset_sha256:
        raise CommonValidError(f"{mode} report dataset SHA256 does not match input")
    frozen_manifest = _require_mapping(
        report.get("frozen_manifest"),
        label=f"{mode} report.frozen_manifest",
    )
    if frozen_manifest.get("sha256") != configuration.get("frozen_manifest_sha256"):
        raise CommonValidError(
            f"{mode} frozen manifest SHA256 disagrees with configuration"
        )
    if frozen_manifest.get("selection_projection_sha256") != configuration.get(
        "selection_projection_sha256"
    ):
        raise CommonValidError(
            f"{mode} selection projection SHA256 disagrees with configuration"
        )
    _validate_reported_artifact(
        frozen_manifest,
        label=f"{mode} report.frozen_manifest",
    )
    llm_trace = _require_mapping(
        report.get("llm_trace"),
        label=f"{mode} report.llm_trace",
    )
    if llm_trace.get("path") != configuration.get("llm_trace_path"):
        raise CommonValidError(
            f"{mode} report LLM trace path disagrees with configuration"
        )
    if not _valid_sha256(llm_trace.get("sha256")):
        raise CommonValidError(f"{mode} report LLM trace SHA256 is invalid")
    actual_llm_trace = _audit_llm_trace(Path(str(llm_trace.get("path"))))
    for key in (
        "sha256",
        "size_bytes",
        "record_count",
        "raw_response_record_count",
        "prompt_record_count",
        "prompts_included",
        "parse_status_counts",
    ):
        if llm_trace.get(key) != actual_llm_trace[key]:
            raise CommonValidError(
                f"{mode} report LLM trace {key} does not match the actual trace"
            )

    statistics = _require_mapping(
        report.get("statistics"),
        label=f"{mode} report.statistics",
    )
    total_samples = len(dataset_keys)
    if _require_int(
        statistics.get("total_samples"),
        label=f"{mode} statistics.total_samples",
    ) != total_samples:
        raise CommonValidError(f"{mode} report total_samples does not match dataset")
    if _require_int(
        statistics.get("processed_samples"),
        label=f"{mode} statistics.processed_samples",
    ) != total_samples:
        raise CommonValidError(f"{mode} report is not fully processed")
    if _require_int(
        statistics.get("pending_samples"),
        label=f"{mode} statistics.pending_samples",
    ) != 0:
        raise CommonValidError(f"{mode} report still has pending samples")

    valid_keys: List[str] = []
    valid_typed_ids: List[Dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for index, (row, key, descriptor) in enumerate(
        zip(rows, result_keys, result_typed_ids)
    ):
        if row.get("checker_gate_mode") != mode:
            raise CommonValidError(
                f"{mode} results[{index}].checker_gate_mode does not match"
            )
        if row.get("replay_config_sha256") != configuration_sha256:
            raise CommonValidError(
                f"{mode} results[{index}].replay_config_sha256 does not match report"
            )
        row_source = str(row.get("candidate_source") or "semantic_retrieval")
        if row_source != candidate_source:
            raise CommonValidError(
                f"{mode} results[{index}].candidate_source does not match report"
            )
        row_strategy = str(row.get("selection_strategy") or "").strip()
        if candidate_source == "frozen_target_binding":
            if row_strategy != "target_rule_binding":
                raise CommonValidError(
                    f"{mode} results[{index}] target binding has the wrong strategy"
                )
            target_rule_id = row.get("target_rule_id")
            if not isinstance(target_rule_id, str) or not target_rule_id.strip():
                raise CommonValidError(
                    f"{mode} results[{index}].target_rule_id is missing"
                )
            dataset_target_rule_id = dataset_rows[index].get("target_rule_id")
            if (
                not isinstance(dataset_target_rule_id, str)
                or not dataset_target_rule_id.strip()
                or dataset_target_rule_id != target_rule_id
            ):
                raise CommonValidError(
                    f"{mode} results[{index}].target_rule_id does not match dataset"
                )
            dataset_target = dataset_rows[index].get("target_rule")
            if not isinstance(dataset_target, Mapping) or dataset_target.get(
                "rule_id"
            ) != dataset_target_rule_id:
                raise CommonValidError(
                    f"dataset[{index}].target_rule.rule_id does not match target_rule_id"
                )
            retrieved_rules = row.get("retrieved_rules")
            if not isinstance(retrieved_rules, list) or len(retrieved_rules) != 1:
                raise CommonValidError(
                    f"{mode} results[{index}] target binding must contain one retrieved rule"
                )
            selected_rule = retrieved_rules[0]
            if not isinstance(selected_rule, Mapping) or selected_rule.get(
                "rule_id"
            ) != target_rule_id:
                raise CommonValidError(
                    f"{mode} results[{index}] selected rule does not match target_rule_id"
                )
            if row.get("unified_retrieval_mode") != "target_binding":
                raise CommonValidError(
                    f"{mode} results[{index}] target binding has the wrong retrieval mode"
                )
            if row.get("retrieval_score_kind") != "fixed_control_0_1":
                raise CommonValidError(
                    f"{mode} results[{index}] target binding has the wrong score kind"
                )
            if selected_rule.get("score_kind") != "fixed_control_0_1":
                raise CommonValidError(
                    f"{mode} results[{index}] selected rule has the wrong score kind"
                )
            _require_fixed_number(
                selected_rule.get("score"),
                expected=1.0,
                label=f"{mode} results[{index}] selected rule score",
            )
            publish_gate = selected_rule.get("publish_gate")
            if not isinstance(publish_gate, Mapping):
                raise CommonValidError(
                    f"{mode} results[{index}] selected rule publish_gate is missing"
                )
            if (
                publish_gate.get("publishable") is not True
                or publish_gate.get("reasons") != []
                or publish_gate.get("score_kind") != "fixed_control_0_1"
                or publish_gate.get("selection_strategy") != "target_rule_binding"
            ):
                raise CommonValidError(
                    f"{mode} results[{index}] selected rule publish_gate is invalid"
                )
            _require_fixed_number(
                publish_gate.get("score"),
                expected=1.0,
                label=f"{mode} results[{index}] selected rule publish gate score",
            )
            _require_fixed_number(
                publish_gate.get("min_publish_score"),
                expected=0.0,
                label=f"{mode} results[{index}] selected rule minimum publish score",
            )
            if selected_rule.get("partial") is not False or selected_rule.get(
                "executable"
            ) is not True:
                raise CommonValidError(
                    f"{mode} results[{index}] selected target rule is not executable"
                )
            for ownership_key in ("domain", "topic_id", "topic"):
                if not isinstance(selected_rule.get(ownership_key), str) or not str(
                    selected_rule.get(ownership_key)
                ).strip():
                    raise CommonValidError(
                        f"{mode} results[{index}] selected rule lacks catalog ownership"
                    )
            if row.get("used_rules") != [target_rule_id]:
                raise CommonValidError(
                    f"{mode} results[{index}].used_rules does not match target binding"
                )
        elif row_strategy == "target_rule_binding":
            raise CommonValidError(
                f"{mode} results[{index}] semantic replay cannot claim target binding"
            )
        try:
            failure_reason = _row_failure_reason(row)
        except CommonValidError as exc:
            raise CommonValidError(f"invalid {mode} results[{index}]: {exc}") from exc
        if failure_reason:
            failure_counts[failure_reason] += 1
        else:
            valid_keys.append(key)
            valid_typed_ids.append(descriptor)

    invalid_count = total_samples - len(valid_keys)
    reported_failed = _require_int(
        statistics.get("failed_samples"),
        label=f"{mode} statistics.failed_samples",
    )
    if reported_failed != invalid_count:
        raise CommonValidError(
            f"{mode} report failed_samples={reported_failed} but validated invalid_count={invalid_count}"
        )
    retryable_count = sum(row.get("replay_completed") is not True for row in rows)
    expected_report_status = "complete"
    if invalid_count:
        expected_report_status = (
            "incomplete_failures" if retryable_count else "complete_with_failures"
        )
    if report.get("status") != expected_report_status:
        raise CommonValidError(
            f"{mode} report status={report.get('status')!r} does not match "
            f"validated status={expected_report_status!r}"
        )

    return {
        "mode": mode,
        "dataset_path": str(dataset_path),
        "result_path": str(result_path),
        "report_path": str(report_path),
        "rows": rows,
        "configuration": dict(configuration),
        "normalized_configuration": _normalized_configuration(configuration),
        "configuration_sha256": configuration_sha256,
        "valid_keys": valid_keys,
        "valid_typed_ids": valid_typed_ids,
        "failure_by_stage": dict(sorted(failure_counts.items())),
        "result_fingerprint": _fingerprint(
            result_path,
            record_count=len(rows),
            ordered_typed_ids=result_typed_ids,
        ),
        "report_fingerprint": _fingerprint(report_path),
        "report_status": report.get("status"),
        "reported_statistics": dict(statistics),
        "reported_llm_trace": copy.deepcopy(llm_trace),
        "actual_llm_trace": actual_llm_trace,
    }


def audit_replay_artifacts(
    *,
    dataset_path: Path,
    mode: str,
    result_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Strictly audit one frozen Checker replay arm.

    The returned state contains the validated result rows, full configuration,
    common-valid ID projection, failure breakdown, and independently audited
    raw-response trace.  It is intentionally metric-free so P3 and future
    evaluators can share exactly the same artifact integrity boundary.
    """

    if mode not in CHECKER_GATE_MODES:
        raise CommonValidError(f"unsupported Checker mode: {mode!r}")
    dataset = _load_json_array(dataset_path, label="dataset")
    dataset_keys, dataset_typed_ids = _ordered_typed_ids(dataset, label="dataset")
    dataset_sha256 = _sha256_file(dataset_path)
    state = _validate_arm(
        mode=mode,
        result_path=result_path,
        report_path=report_path,
        dataset_keys=dataset_keys,
        dataset_typed_ids=dataset_typed_ids,
        dataset_rows=dataset,
        dataset_sha256=dataset_sha256,
        dataset_path=dataset_path,
    )
    state["dataset_fingerprint"] = _fingerprint(
        dataset_path,
        record_count=len(dataset),
        ordered_typed_ids=dataset_typed_ids,
    )
    return state


def _write_json_new(path: Path, payload: Any) -> None:
    """Atomically create a JSON file, refusing to replace an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CommonValidError(f"refusing to overwrite existing artifact: {path}")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CommonValidError(
                f"refusing to overwrite existing artifact: {path}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_common_valid(
    *,
    dataset_path: Path,
    arms: Mapping[str, Tuple[Path, Path]],
    output_dataset_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    if set(arms) != set(CHECKER_GATE_MODES) or len(arms) != len(CHECKER_GATE_MODES):
        raise CommonValidError(
            "exactly one result/report pair is required for each Checker mode: "
            + ", ".join(CHECKER_GATE_MODES)
        )

    input_paths = [dataset_path]
    for mode in CHECKER_GATE_MODES:
        result_path, report_path = arms[mode]
        input_paths.extend([result_path, report_path])
    resolved_inputs = [path.resolve() for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise CommonValidError("dataset/result/report input paths must all be distinct")
    output_paths = {output_dataset_path.resolve(), manifest_path.resolve()}
    if len(output_paths) != 2 or output_paths.intersection(resolved_inputs):
        raise CommonValidError(
            "common dataset and manifest paths must be distinct from every input"
        )
    for path in (output_dataset_path, manifest_path):
        if path.exists():
            raise CommonValidError(f"refusing to overwrite existing artifact: {path}")

    dataset = _load_json_array(dataset_path, label="dataset")
    dataset_keys, dataset_typed_ids = _ordered_typed_ids(dataset, label="dataset")
    dataset_fingerprint = _fingerprint(
        dataset_path,
        record_count=len(dataset),
        ordered_typed_ids=dataset_typed_ids,
    )

    arm_states: Dict[str, Dict[str, Any]] = {}
    for mode in CHECKER_GATE_MODES:
        result_path, report_path = arms[mode]
        arm_states[mode] = audit_replay_artifacts(
            dataset_path=dataset_path,
            mode=mode,
            result_path=result_path,
            report_path=report_path,
        )

    trace_paths = [
        Path(str(arm_states[mode]["actual_llm_trace"]["path"])).resolve()
        for mode in CHECKER_GATE_MODES
    ]
    if len(set(trace_paths)) != len(CHECKER_GATE_MODES):
        raise CommonValidError("each Checker arm must use a distinct LLM trace path")
    if set(trace_paths).intersection(resolved_inputs) or set(trace_paths).intersection(
        output_paths
    ):
        raise CommonValidError(
            "LLM trace paths must be distinct from dataset/result/report/output paths"
        )

    reference_mode = CHECKER_GATE_MODES[0]
    shared_configuration = arm_states[reference_mode]["normalized_configuration"]
    for mode in CHECKER_GATE_MODES[1:]:
        candidate = arm_states[mode]["normalized_configuration"]
        if candidate != shared_configuration:
            differences = _configuration_differences(shared_configuration, candidate)
            raise CommonValidError(
                f"{mode} configuration differs outside the arm allowlist: "
                + ", ".join(differences or ["configuration"])
            )

    valid_sets = {
        mode: set(state["valid_keys"]) for mode, state in arm_states.items()
    }
    common_keys = set(dataset_keys)
    for mode in CHECKER_GATE_MODES:
        common_keys.intersection_update(valid_sets[mode])
    common_dataset: List[Dict[str, Any]] = []
    common_typed_ids: List[Dict[str, Any]] = []
    for row, key, descriptor in zip(dataset, dataset_keys, dataset_typed_ids):
        if key in common_keys:
            common_dataset.append(row)
            common_typed_ids.append(descriptor)

    _write_json_new(output_dataset_path, common_dataset)
    common_dataset_fingerprint = _fingerprint(
        output_dataset_path,
        record_count=len(common_dataset),
        ordered_typed_ids=common_typed_ids,
    )

    total = len(dataset)
    arm_manifest: Dict[str, Any] = {}
    for mode in CHECKER_GATE_MODES:
        state = arm_states[mode]
        valid_count = len(state["valid_keys"])
        arm_manifest[mode] = {
            "results": state["result_fingerprint"],
            "report": state["report_fingerprint"],
            "report_status": state["report_status"],
            "configuration_sha256": state["configuration_sha256"],
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "coverage": (valid_count / total) if total else 1.0,
            "valid_ordered_typed_ids": state["valid_typed_ids"],
            "valid_ordered_typed_ids_sha256": _object_sha256(
                state["valid_typed_ids"]
            ),
            "failure_by_stage": state["failure_by_stage"],
            "reported_statistics": state["reported_statistics"],
            "reported_llm_trace": state["reported_llm_trace"],
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "immutable": True,
        "created_at_utc": _utc_now(),
        "mode_order": list(CHECKER_GATE_MODES),
        "allowed_arm_configuration_differences": list(
            ALLOWED_ARM_CONFIGURATION_DIFFERENCES
        ),
        "shared_configuration": shared_configuration,
        "shared_configuration_sha256": _object_sha256(shared_configuration),
        "inputs": {
            "dataset": dataset_fingerprint,
            "arms": arm_manifest,
        },
        "common_valid": {
            "dataset": common_dataset_fingerprint,
            "total_input_count": total,
            "record_count": len(common_dataset),
            "excluded_count": total - len(common_dataset),
            "coverage": (len(common_dataset) / total) if total else 1.0,
            "ordered_typed_ids": common_typed_ids,
            "ordered_typed_ids_sha256": _object_sha256(common_typed_ids),
        },
        "evaluation_instruction": (
            "Run each arm's existing evaluator against common_valid.dataset.path; "
            "retain full-dataset coverage from the three original arm reports."
        ),
    }
    _write_json_new(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate three controlled Checker replay arms and freeze their "
            "common-valid dataset for paired evaluation."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=3,
        metavar=("MODE", "RESULTS", "REPORT"),
        required=True,
        help="Repeat once for legacy, dual_evidence, and dual_evidence_consistency.",
    )
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms: Dict[str, Tuple[Path, Path]] = {}
    for mode, result_value, report_value in args.arm:
        if mode not in CHECKER_GATE_MODES:
            parser.error(f"unsupported Checker mode: {mode}")
        if mode in arms:
            parser.error(f"duplicate --arm entry for {mode}")
        arms[mode] = (Path(result_value), Path(report_value))
    try:
        manifest = build_common_valid(
            dataset_path=Path(args.dataset),
            arms=arms,
            output_dataset_path=Path(args.output_dataset),
            manifest_path=Path(args.manifest),
        )
    except CommonValidError as exc:
        parser.error(str(exc))
    common = manifest["common_valid"]
    print(
        "Checker common-valid dataset frozen: "
        f"{common['record_count']}/{common['total_input_count']} "
        f"(coverage={common['coverage']:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
