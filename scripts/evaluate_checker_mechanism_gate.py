from __future__ import annotations

"""Evaluate the preregistered P3 target-rule mechanism gate.

Execution failures are coverage failures, never negative predictions.  Formal
evaluation requires three Checker system arms and three independent repetitions.
"""

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_checker_common_valid import (
    CommonValidError,
    _audit_llm_trace,
    _object_sha256,
    _sha256_file,
    audit_replay_artifacts,
)
from core.semantic_rule_checker import SEMANTIC_RULE_CHECKER_PROMPT_VERSION
from scripts.build_checker_mechanism_dataset import (
    MechanismDatasetError,
    _source_identity as _capture_source_identity,
    audit_generation_artifacts,
)


ARMS: Tuple[str, ...] = (
    "legacy",
    "dual_evidence",
    "dual_evidence_consistency",
)
REPETITIONS: Tuple[int, ...] = (1, 2, 3)
CANDIDATE_ARM = "dual_evidence_consistency"
CHECKER_MODEL = "qwen3-30b-a3b-instruct-2507"
MECHANISMS: Tuple[str, ...] = (
    "true_violation",
    "applicable_correct",
    "symbol_overlap_inapplicable",
    "equivalent_alternative",
    "insufficient_or_self_corrected",
)
NEGATIVE_MECHANISMS: Tuple[str, ...] = MECHANISMS[1:]
VALID_CHECKER_STATUSES = {"valid_empty", "valid_with_diagnostics"}
TERMINAL_REPORT_STATUSES = {
    "complete",
    "complete_with_failures",
    "incomplete_failures",
}
REPORT_TYPE = "checker_only_frozen_target_binding_replay"
DATASET_SCHEMA_VERSION = "p3_checker_mechanism_case_v1"
OUTPUT_SCHEMA_VERSION = 1
EXPECTED_BY_CASE: Dict[Tuple[str, str], Tuple[bool, bool, bool, Optional[str]]] = {
    ("true_violation", "none"): (True, True, True, "confirmed_violation"),
    ("applicable_correct", "none"): (True, False, False, None),
    ("symbol_overlap_inapplicable", "none"): (False, False, False, None),
    ("equivalent_alternative", "none"): (
        True,
        False,
        False,
        "equivalent_or_alternative",
    ),
    ("insufficient_or_self_corrected", "insufficient_information"): (
        True,
        False,
        False,
        "uncertain",
    ),
    ("insufficient_or_self_corrected", "self_corrected"): (
        True,
        False,
        False,
        "self_corrected",
    ),
}
ALLOWED_RUN_CONFIGURATION_DIFFERENCES = {
    "system_arm",
    "llm_trace_path",
    "output_path",
    "report_path",
    "checkpoint_path",
    "results_path",
}


class MechanismEvaluationError(ValueError):
    pass


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise MechanismEvaluationError(f"{label} does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismEvaluationError(f"{label} is not valid JSON: {path}: {exc}") from exc


def _load_array(path: Path, *, label: str) -> List[Dict[str, Any]]:
    payload = _load_json(path, label=label)
    if not isinstance(payload, list):
        raise MechanismEvaluationError(f"{label} must be a JSON array")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise MechanismEvaluationError(f"{label}[{index}] must be an object")
        rows.append(row)
    return rows


def _typed_id(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise MechanismEvaluationError(f"{label} must be a string or integer, not bool")
    if isinstance(value, str) and not value.strip():
        raise MechanismEvaluationError(f"{label} must not be empty")
    kind = "str" if isinstance(value, str) else "int"
    return f"{kind}:{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def _typed_descriptor(value: Any, *, label: str) -> Dict[str, Any]:
    _typed_id(value, label=label)
    return {"type": "str" if isinstance(value, str) else "int", "value": value}


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MechanismEvaluationError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise MechanismEvaluationError(f"{label} must be a boolean")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MechanismEvaluationError(f"{label} must be a non-empty string")
    return value.strip()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_span(
    span: Any,
    *,
    row: Mapping[str, Any],
    allowed_sources: Set[str],
    label: str,
) -> Dict[str, Any]:
    if not isinstance(span, dict):
        raise MechanismEvaluationError(f"{label} must be an object")
    if set(span) != {"source", "quote", "start_char", "end_char"}:
        raise MechanismEvaluationError(f"{label} has the wrong span schema")
    source = _nonempty_string(span.get("source"), label=f"{label}.source")
    if source not in allowed_sources:
        raise MechanismEvaluationError(f"{label}.source is not allowed")
    quote = _nonempty_string(span.get("quote"), label=f"{label}.quote")
    start = _strict_int(span.get("start_char"), label=f"{label}.start_char")
    end = _strict_int(span.get("end_char"), label=f"{label}.end_char", minimum=1)
    text = str(row.get(source) or "")
    if not 0 <= start < end <= len(text) or text[start:end] != quote:
        raise MechanismEvaluationError(f"{label} does not match its exact source span")
    return {"source": source, "quote": quote, "start_char": start, "end_char": end}


def _validate_dataset_row(row: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    label = f"dataset[{index}]"
    required_row = {
        "schema_version",
        "id",
        "question",
        "context",
        "prediction",
        "target_rule_id",
        "target_rule",
        "mechanism",
        "mechanism_subtype",
        "expected",
        "gt_evidence",
        "case_content_sha256",
        "gt_provenance",
    }
    if set(row) != required_row:
        raise MechanismEvaluationError(f"{label} has the wrong schema")
    if row.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise MechanismEvaluationError(f"{label}.schema_version is invalid")
    sample_id = _nonempty_string(row.get("id"), label=f"{label}.id")
    for field in ("question", "context", "prediction"):
        _nonempty_string(row.get(field), label=f"{label}.{field}")

    target = row.get("target_rule")
    if not isinstance(target, dict):
        raise MechanismEvaluationError(f"{label}.target_rule must be an object")
    required_target = {
        "rule_id",
        "domain_id",
        "domain",
        "topic_id",
        "topic",
        "cluster_id",
        "origin",
        "symbolic_primitive",
        "has_symbolic_primitive",
        "trigger_scope_proxy",
    }
    if set(target) != required_target:
        raise MechanismEvaluationError(f"{label}.target_rule has the wrong schema")
    for field in (
        "rule_id",
        "domain_id",
        "domain",
        "topic_id",
        "topic",
        "cluster_id",
        "origin",
        "symbolic_primitive",
        "trigger_scope_proxy",
    ):
        _nonempty_string(target.get(field), label=f"{label}.target_rule.{field}")
    if target.get("origin") not in {"gen", "exp"}:
        raise MechanismEvaluationError(f"{label}.target_rule.origin is invalid")
    if target.get("trigger_scope_proxy") not in {"broad_proxy", "narrow_proxy"}:
        raise MechanismEvaluationError(f"{label}.target_rule.trigger_scope_proxy is invalid")
    _strict_bool(
        target.get("has_symbolic_primitive"),
        label=f"{label}.target_rule.has_symbolic_primitive",
    )
    if target["has_symbolic_primitive"] != (target["symbolic_primitive"] != "none"):
        raise MechanismEvaluationError(
            f"{label}.target_rule symbolic primitive fields contradict"
        )

    target_rule_id = _nonempty_string(
        row.get("target_rule_id"), label=f"{label}.target_rule_id"
    )
    if target_rule_id != target["rule_id"]:
        raise MechanismEvaluationError(f"{label}.target_rule_id does not match target_rule")

    mechanism = _nonempty_string(row.get("mechanism"), label=f"{label}.mechanism")
    if mechanism not in MECHANISMS:
        raise MechanismEvaluationError(f"{label}.mechanism is invalid")
    subtype = _nonempty_string(
        row.get("mechanism_subtype"), label=f"{label}.mechanism_subtype"
    )
    if mechanism == "insufficient_or_self_corrected":
        if subtype not in {"insufficient_information", "self_corrected"}:
            raise MechanismEvaluationError(f"{label}.mechanism_subtype is invalid")
    elif subtype != "none":
        raise MechanismEvaluationError(f"{label}.mechanism_subtype must be 'none'")
    if sample_id != f"p3::{target_rule_id}::{mechanism}":
        raise MechanismEvaluationError(f"{label}.id does not match target/mechanism")

    expected = row.get("expected")
    if not isinstance(expected, dict) or set(expected) != {
        "applicability",
        "current_violation",
        "publish",
        "consistency_status",
    }:
        raise MechanismEvaluationError(f"{label}.expected has the wrong schema")
    for field in ("applicability", "current_violation", "publish"):
        _strict_bool(expected.get(field), label=f"{label}.expected.{field}")
    consistency = expected.get("consistency_status")
    if consistency not in {
        None,
        "confirmed_violation",
        "equivalent_or_alternative",
        "self_corrected",
        "uncertain",
    }:
        raise MechanismEvaluationError(f"{label}.expected.consistency_status is invalid")
    actual_expected = (
        expected["applicability"],
        expected["current_violation"],
        expected["publish"],
        consistency,
    )
    required_expected = EXPECTED_BY_CASE[(mechanism, subtype)]
    if actual_expected != required_expected:
        raise MechanismEvaluationError(f"{label}.expected contradicts mechanism/subtype")

    evidence = row.get("gt_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "applicability_spans",
        "violation_spans",
        "superseded_claim_spans",
        "correction_spans",
    }:
        raise MechanismEvaluationError(f"{label}.gt_evidence has the wrong schema")
    normalized_evidence: Dict[str, List[Dict[str, Any]]] = {}
    for field, sources in (
        ("applicability_spans", {"question", "context"}),
        ("violation_spans", {"prediction"}),
        ("superseded_claim_spans", {"prediction"}),
        ("correction_spans", {"prediction"}),
    ):
        spans = evidence.get(field)
        if not isinstance(spans, list):
            raise MechanismEvaluationError(f"{label}.gt_evidence.{field} must be an array")
        normalized_evidence[field] = [
            _validate_span(
                span,
                row=row,
                allowed_sources=sources,
                label=f"{label}.gt_evidence.{field}[{pos}]",
            )
            for pos, span in enumerate(spans)
        ]
    if expected["applicability"] is True and not normalized_evidence["applicability_spans"]:
        raise MechanismEvaluationError(f"{label} lacks applicability GT evidence")
    if expected["applicability"] is False and normalized_evidence["applicability_spans"]:
        raise MechanismEvaluationError(f"{label} asserts contradictory applicability GT evidence")
    if expected["current_violation"] is True and not normalized_evidence["violation_spans"]:
        raise MechanismEvaluationError(f"{label} lacks violation GT evidence")
    if expected["current_violation"] is False and normalized_evidence["violation_spans"]:
        raise MechanismEvaluationError(f"{label} asserts contradictory violation GT evidence")
    if subtype == "self_corrected" and (
        not normalized_evidence["superseded_claim_spans"]
        or not normalized_evidence["correction_spans"]
    ):
        raise MechanismEvaluationError(f"{label} lacks self-correction GT spans")
    if subtype != "self_corrected" and (
        normalized_evidence["superseded_claim_spans"]
        or normalized_evidence["correction_spans"]
    ):
        raise MechanismEvaluationError(f"{label} has unexpected self-correction GT spans")
    if subtype == "self_corrected" and max(
        span["end_char"] for span in normalized_evidence["superseded_claim_spans"]
    ) > min(span["start_char"] for span in normalized_evidence["correction_spans"]):
        raise MechanismEvaluationError(f"{label} correction must follow superseded claim")

    content_sha256 = row.get("case_content_sha256")
    expected_content_sha256 = _object_sha256(
        [row["question"].strip(), row["context"].strip(), row["prediction"].strip()]
    )
    if content_sha256 != expected_content_sha256:
        raise MechanismEvaluationError(f"{label}.case_content_sha256 is invalid")

    provenance = row.get("gt_provenance")
    required_provenance = {
        "generator_model",
        "prompt_version",
        "plan_sha256",
        "raw_response_sha256",
        "validation_status",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_provenance:
        raise MechanismEvaluationError(f"{label}.gt_provenance has the wrong schema")
    for field in required_provenance:
        _nonempty_string(provenance.get(field), label=f"{label}.gt_provenance.{field}")
    for field in ("plan_sha256", "raw_response_sha256"):
        if not _valid_sha256(provenance.get(field)):
            raise MechanismEvaluationError(f"{label}.gt_provenance.{field} is invalid")
    if not str(provenance.get("generator_model") or "").lower().startswith(
        "gemini-3-flash"
    ):
        raise MechanismEvaluationError(f"{label}.gt_provenance.generator_model is not Gemini 3 Flash")
    if provenance.get("validation_status") != "schema_validated":
        raise MechanismEvaluationError(f"{label}.gt_provenance is not schema validated")

    normalized = copy.deepcopy(dict(row))
    normalized["gt_evidence"] = normalized_evidence
    return normalized


def _validate_dataset(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized: List[Dict[str, Any]] = []
    keys: List[str] = []
    seen: Set[str] = set()
    for index, row in enumerate(rows):
        normalized_row = _validate_dataset_row(row, index=index)
        key = _typed_id(normalized_row.get("id"), label=f"dataset[{index}].id")
        if key in seen:
            raise MechanismEvaluationError(f"duplicate typed ID in dataset: {row.get('id')!r}")
        seen.add(key)
        normalized.append(normalized_row)
        keys.append(key)
    return normalized, keys


def _normalized_configuration(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(configuration))
    for key in ALLOWED_RUN_CONFIGURATION_DIFFERENCES:
        normalized.pop(key, None)
    return normalized


def _trace_matches_report(trace: Mapping[str, Any], *, label: str) -> None:
    path_value = trace.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise MechanismEvaluationError(f"{label}.path is missing")
    try:
        actual = _audit_llm_trace(Path(path_value))
    except Exception as exc:
        raise MechanismEvaluationError(f"{label} is invalid: {exc}") from exc
    for key in (
        "sha256",
        "size_bytes",
        "record_count",
        "raw_response_record_count",
        "prompt_record_count",
        "prompts_included",
        "parse_status_counts",
    ):
        if trace.get(key) != actual.get(key):
            raise MechanismEvaluationError(f"{label}.{key} does not match actual trace")


def _audit_trace_associations(
    path: Path,
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    arm: str,
    model: str,
    label: str,
) -> Dict[str, Any]:
    expected: Dict[str, str] = {
        _typed_id(row.get("id"), label=f"{label} dataset id"): str(
            row.get("target_rule_id") or ""
        )
        for row in dataset_rows
    }
    records_by_key: Dict[str, Dict[int, Dict[str, Any]]] = {
        key: {} for key in expected
    }
    response_ids: Set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MechanismEvaluationError(f"{label} cannot be read: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MechanismEvaluationError(
                f"{label} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise MechanismEvaluationError(f"{label} line {line_number} is not an object")
        meta = record.get("trace_meta")
        if not isinstance(meta, dict):
            raise MechanismEvaluationError(f"{label} line {line_number} lacks trace_meta")
        key = _typed_id(
            meta.get("sample_id"), label=f"{label} line {line_number} sample_id"
        )
        if key not in expected:
            raise MechanismEvaluationError(
                f"{label} line {line_number} references a foreign sample"
            )
        if meta.get("rule_id") != expected[key]:
            raise MechanismEvaluationError(
                f"{label} line {line_number} references the wrong target rule"
            )
        if record.get("checker_mode") != arm or record.get("model") != model:
            raise MechanismEvaluationError(
                f"{label} line {line_number} mode/model does not match its replay"
            )
        parse_status = str(record.get("parse_status") or "")
        actual_model = record.get("actual_model")
        response_id = record.get("response_id")
        if not isinstance(actual_model, str) or not isinstance(response_id, str):
            raise MechanismEvaluationError(
                f"{label} line {line_number} lacks provider response identity"
            )
        is_transport_failure = parse_status in {"exception", "transport_failure"}
        if is_transport_failure:
            if actual_model or response_id:
                raise MechanismEvaluationError(
                    f"{label} line {line_number} transport failure carries stale response identity"
                )
        else:
            if actual_model != model:
                raise MechanismEvaluationError(
                    f"{label} line {line_number} actual response model does not match its replay"
                )
            if not response_id.strip():
                raise MechanismEvaluationError(
                    f"{label} line {line_number} lacks a provider response ID"
                )
            if response_id in response_ids:
                raise MechanismEvaluationError(
                    f"{label} contains a duplicate provider response ID"
                )
            response_ids.add(response_id)
        attempt = meta.get("attempt", 1 if arm == "legacy" else None)
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise MechanismEvaluationError(
                f"{label} line {line_number} has an invalid attempt index"
            )
        if attempt in records_by_key[key]:
            raise MechanismEvaluationError(
                f"{label} has duplicate attempt {attempt} for sample {key}"
            )
        records_by_key[key][attempt] = record

    result_by_key = {
        _typed_id(row.get("id"), label=f"{label} result id"): row
        for row in result_rows
    }
    for key, row in result_by_key.items():
        decisions = row.get("checker_decisions")
        if not isinstance(decisions, list) or len(decisions) != 1:
            continue
        decision = decisions[0]
        if not isinstance(decision, dict):
            continue
        attempt_count = decision.get("attempt_count")
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count < 0
        ):
            continue
        attempts = decision.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != attempt_count:
            raise MechanismEvaluationError(
                f"{label} target decision attempt list is inconsistent"
            )
        expected_attempts = list(range(1, attempt_count + 1))
        actual_attempts = sorted(records_by_key[key])
        if actual_attempts != expected_attempts:
            raise MechanismEvaluationError(
                f"{label} trace attempts do not exactly match the target decision"
            )
        for expected_attempt, decision_attempt in zip(expected_attempts, attempts):
            if not isinstance(decision_attempt, dict):
                raise MechanismEvaluationError(
                    f"{label} target decision attempt is not an object"
                )
            if decision_attempt.get("attempt") != expected_attempt:
                raise MechanismEvaluationError(
                    f"{label} target decision attempts are not contiguous"
                )
            trace_record = records_by_key[key][expected_attempt]
            trace_status = str(trace_record.get("parse_status") or "")
            normalized_trace_status = {
                "json.loads_ok": "valid_json",
                "regex_extract_ok": "valid_json",
                "parse_failed": "parse_failure",
                "exception": "transport_failure",
            }.get(trace_status, trace_status)
            if normalized_trace_status != str(decision_attempt.get("status") or ""):
                raise MechanismEvaluationError(
                    f"{label} trace status contradicts the target decision attempt"
                )
            if normalized_trace_status != "transport_failure" and "raw_response" not in trace_record:
                raise MechanismEvaluationError(
                    f"{label} non-transport attempt lacks a raw response"
                )
        if str(decision.get("status") or "") in VALID_CHECKER_STATUSES:
            expected_success = "valid_json" if arm == "legacy" else "valid_object"
            if not attempts or attempts[-1].get("status") != expected_success:
                raise MechanismEvaluationError(
                    f"{label} successful target decision lacks a successful final attempt"
                )
            if "raw_response" not in records_by_key[key][attempt_count]:
                raise MechanismEvaluationError(
                    f"{label} successful target decision lacks a raw response"
                )
    counts: Counter[str] = Counter(
        {key: len(records) for key, records in records_by_key.items()}
    )
    raw_counts: Counter[str] = Counter(
        {
            key: sum("raw_response" in record for record in records.values())
            for key, records in records_by_key.items()
        }
    )
    return {
        "record_count": sum(counts.values()),
        "sample_count": sum(value > 0 for value in counts.values()),
        "raw_response_sample_count": sum(value > 0 for value in raw_counts.values()),
        "provider_response_id_count": len(response_ids),
        "provider_response_ids_sha256": _object_sha256(sorted(response_ids)),
        "per_sample_record_count_sha256": _object_sha256(
            [counts.get(key, 0) for key in expected]
        ),
    }


def _validate_formal_evaluator_source_identity(
    current: Mapping[str, Any],
    generation: Any,
) -> Dict[str, Any]:
    if not isinstance(generation, Mapping):
        raise MechanismEvaluationError("formal generation lacks source identity")
    if current.get("git_available") is not True or current.get("git_dirty") is not False:
        raise MechanismEvaluationError(
            "formal evaluation requires a clean Git source identity"
        )
    if generation.get("git_available") is not True or generation.get("git_dirty") is not False:
        raise MechanismEvaluationError(
            "formal generation source identity is not clean Git"
        )
    for field in ("git_head", "source_tree_sha256"):
        if not isinstance(current.get(field), str) or not current.get(field):
            raise MechanismEvaluationError(
                f"formal evaluator source identity lacks {field}"
            )
        if current.get(field) != generation.get(field):
            raise MechanismEvaluationError(
                "formal evaluator source identity differs from GT generation"
            )
    return copy.deepcopy(dict(current))


def _failure_kind(item: Any) -> str:
    if not isinstance(item, dict):
        return "checker_failure"
    for field in ("kind", "status", "failure_kind", "error_type", "stage"):
        value = str(item.get(field) or "").strip()
        if value:
            return value[:120]
    return "checker_failure"


def _span_overlap(start: int, end: int, gt_spans: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        max(start, int(span.get("start_char") or 0))
        < min(end, int(span.get("end_char") or 0))
        for span in gt_spans
    )


def _diagnostic_hits_superseded_claim(
    diagnostic: Mapping[str, Any],
    *,
    prediction: str,
    gt_spans: Sequence[Mapping[str, Any]],
) -> bool:
    if not gt_spans:
        return False
    evidence = diagnostic.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    location = evidence.get("location")
    location = location if isinstance(location, dict) else {}
    start = location.get("start_char")
    end = location.get("end_char")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(prediction)
    ):
        return _span_overlap(start, end, gt_spans)
    quote = str(evidence.get("quote") or "")
    if not quote:
        return False
    cursor = 0
    while True:
        found = prediction.find(quote, cursor)
        if found < 0:
            return False
        if _span_overlap(found, found + len(quote), gt_spans):
            return True
        cursor = found + 1


def _protocol_contradiction_count(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> Optional[int]:
    if arm == "legacy":
        return None
    count = 0
    for diagnostic in diagnostics:
        gate = diagnostic.get("checker_evidence_gate")
        gate = gate if isinstance(gate, dict) else {}
        applicability = diagnostic.get("applicability")
        applicability = applicability if isinstance(applicability, dict) else {}
        violation = diagnostic.get("violation")
        violation = violation if isinstance(violation, dict) else {}
        contradictory = (
            gate.get("passed") is not True
            or applicability.get("applies") is not True
            or violation.get("present") is not True
        )
        if arm == "dual_evidence_consistency":
            consistency = diagnostic.get("consistency")
            consistency = consistency if isinstance(consistency, dict) else {}
            contradictory = contradictory or consistency.get("status") != "confirmed_violation"
        if contradictory:
            count += 1
    return count


def _evaluate_result_row(
    row: Any,
    *,
    dataset_row: Mapping[str, Any],
    arm: str,
    configuration_sha256: str,
) -> Dict[str, Any]:
    target_rule_id = str((dataset_row.get("target_rule") or {}).get("rule_id") or "")
    base = {
        "id": dataset_row.get("id"),
        "target_rule_id": target_rule_id,
        "mechanism": dataset_row.get("mechanism"),
        "mechanism_subtype": dataset_row.get("mechanism_subtype"),
        "expected_publish": bool((dataset_row.get("expected") or {}).get("publish")),
        "structured_valid": False,
        "first_attempt_valid": False,
        "replay_completed": None,
        "failure_stage": None,
        "publish_any_rule": None,
        "publish_target_rule": None,
        "published_diagnostic_count": None,
        "published_target_diagnostic_count": None,
        "protocol_contradiction_count": None,
        "self_corrected_probe_diagnostic_count": None,
        "self_corrected_superseded_overlap_count": None,
    }
    if not isinstance(row, dict):
        base["failure_stage"] = "missing_result"
        return base
    base["replay_completed"] = row.get("replay_completed")
    if row.get("replay_config_sha256") != configuration_sha256:
        base["failure_stage"] = "configuration_mismatch"
        return base
    if row.get("checker_gate_mode") != arm:
        base["failure_stage"] = "checker_mode_mismatch"
        return base
    if row.get("candidate_source") != "frozen_target_binding":
        base["failure_stage"] = "candidate_source_mismatch"
        return base
    if row.get("selection_strategy") != "target_rule_binding":
        base["failure_stage"] = "target_binding_invalid"
        return base
    if row.get("unified_retrieval_mode") != "target_binding":
        base["failure_stage"] = "target_binding_invalid"
        return base
    if row.get("retrieval_score_kind") != "fixed_control_0_1":
        base["failure_stage"] = "target_binding_invalid"
        return base
    if row.get("target_rule_id") != target_rule_id:
        base["failure_stage"] = "target_rule_mismatch"
        return base
    if row.get("checker_min_confidence") != 0.8:
        base["failure_stage"] = "checker_threshold_mismatch"
        return base
    retrieved_rules = row.get("retrieved_rules")
    if (
        not isinstance(retrieved_rules, list)
        or len(retrieved_rules) != 1
        or not isinstance(retrieved_rules[0], dict)
        or retrieved_rules[0].get("rule_id") != target_rule_id
        or retrieved_rules[0].get("candidate_source") != "frozen_target_binding"
        or retrieved_rules[0].get("score_kind") != "fixed_control_0_1"
        or retrieved_rules[0].get("score") != 1.0
        or retrieved_rules[0].get("partial") is not False
        or retrieved_rules[0].get("executable") is not True
    ):
        base["failure_stage"] = "target_binding_invalid"
        return base
    if row.get("used_rules") != [target_rule_id]:
        base["failure_stage"] = "target_used_rules_invalid"
        return base
    replay = row.get("replay")
    if (
        not isinstance(replay, dict)
        or replay.get("system_arm") != arm
        or replay.get("bottom_up_enabled") is not False
    ):
        base["failure_stage"] = "replay_schema_invalid"
        return base
    failures = row.get("checker_failures")
    if not isinstance(failures, list):
        base["failure_stage"] = "checker_failure_schema"
        return base
    failure_count = row.get("checker_failure_count")
    if (
        isinstance(failure_count, bool)
        or not isinstance(failure_count, int)
        or failure_count != len(failures)
    ):
        base["failure_stage"] = "checker_failure_schema"
        return base
    if failures:
        base["failure_stage"] = _failure_kind(failures[0])
        return base
    if row.get("replay_completed") is not True:
        base["failure_stage"] = "replay_incomplete"
        return base
    if replay.get("checker_attempted") is not True or replay.get("checker_succeeded") is not True:
        base["failure_stage"] = "replay_schema_invalid"
        return base
    checker_status = str(row.get("checker_status") or "")
    if checker_status not in VALID_CHECKER_STATUSES:
        base["failure_stage"] = (
            "target_rule_missing" if checker_status == "complete_no_rules" else "checker_status_invalid"
        )
        return base
    decisions = row.get("checker_decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        base["failure_stage"] = "target_decision_count_invalid"
        return base
    decision = decisions[0]
    if not isinstance(decision, dict) or decision.get("rule_id") != target_rule_id:
        base["failure_stage"] = "target_decision_mismatch"
        return base
    if str(decision.get("status") or "") not in VALID_CHECKER_STATUSES:
        base["failure_stage"] = "target_decision_invalid"
        return base
    if decision.get("checker_gate_mode") != arm or decision.get("cache_hit") is not False:
        base["failure_stage"] = "target_decision_invalid"
        return base
    attempt_count = decision.get("attempt_count")
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count < 1:
        base["failure_stage"] = "target_decision_attempt_invalid"
        return base
    attempts = decision.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != attempt_count
        or not all(isinstance(item, dict) for item in attempts)
    ):
        base["failure_stage"] = "target_decision_attempt_invalid"
        return base
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, list) or not all(isinstance(item, dict) for item in diagnostics):
        base["failure_stage"] = "diagnostic_schema_invalid"
        return base
    foreign = [item for item in diagnostics if item.get("rule") != target_rule_id]
    if foreign:
        base["failure_stage"] = "foreign_rule_diagnostic"
        return base
    candidate_diagnostics = row.get("candidate_diagnostics")
    if not isinstance(candidate_diagnostics, list) or not all(
        isinstance(item, dict) for item in candidate_diagnostics
    ):
        base["failure_stage"] = "diagnostic_schema_invalid"
        return base
    if any(item.get("rule") != target_rule_id for item in candidate_diagnostics):
        base["failure_stage"] = "foreign_rule_diagnostic"
        return base
    decision_status = str(decision.get("status") or "")
    reported_candidate_count = decision.get("published_diagnostic_count")
    if (
        isinstance(reported_candidate_count, bool)
        or not isinstance(reported_candidate_count, int)
        or reported_candidate_count != len(candidate_diagnostics)
        or (decision_status == "valid_empty" and candidate_diagnostics)
        or (decision_status == "valid_with_diagnostics" and not candidate_diagnostics)
        or checker_status != decision_status
    ):
        base["failure_stage"] = "checker_decision_diagnostic_mismatch"
        return base
    candidate_signatures = {
        _object_sha256(
            {
                key: item.get(key)
                for key in ("rule", "severity", "message", "symbol", "evidence")
            }
        )
        for item in candidate_diagnostics
    }
    for item in diagnostics:
        signature = _object_sha256(
            {
                key: item.get(key)
                for key in ("rule", "severity", "message", "symbol", "evidence")
            }
        )
        if signature not in candidate_signatures:
            base["failure_stage"] = "released_diagnostic_not_from_checker"
            return base

    target_diagnostics = [item for item in diagnostics if item.get("rule") == target_rule_id]
    protocol_count = _protocol_contradiction_count(target_diagnostics, arm=arm)
    superseded = ((dataset_row.get("gt_evidence") or {}).get("superseded_claim_spans") or [])
    self_corrected_count = sum(
        _diagnostic_hits_superseded_claim(
            item,
            prediction=str(dataset_row.get("prediction") or ""),
            gt_spans=superseded,
        )
        for item in target_diagnostics
    )
    base.update(
        {
            "structured_valid": True,
            "first_attempt_valid": attempt_count == 1,
            "publish_any_rule": bool(diagnostics),
            "publish_target_rule": bool(target_diagnostics),
            "published_diagnostic_count": len(diagnostics),
            "published_target_diagnostic_count": len(target_diagnostics),
            "protocol_contradiction_count": protocol_count,
            "self_corrected_probe_diagnostic_count": len(target_diagnostics),
            "self_corrected_superseded_overlap_count": self_corrected_count,
        }
    )
    return base


def _validate_run(
    *,
    arm: str,
    repetition: int,
    result_path: Path,
    report_path: Path,
    dataset_path: Path,
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_keys: Sequence[str],
) -> Dict[str, Any]:
    label = f"{arm}/repetition-{repetition}"
    try:
        audited = audit_replay_artifacts(
            dataset_path=dataset_path,
            mode=arm,
            result_path=result_path,
            report_path=report_path,
        )
    except CommonValidError as exc:
        raise MechanismEvaluationError(f"{label} strict replay audit failed: {exc}") from exc
    results = list(audited["rows"])
    result_keys: List[str] = []
    result_index: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(results):
        key = _typed_id(row.get("id"), label=f"{label} results[{index}].id")
        if key in result_index:
            raise MechanismEvaluationError(f"duplicate typed ID in {label} results")
        result_keys.append(key)
        result_index[key] = row
    if result_keys != list(dataset_keys):
        raise MechanismEvaluationError(
            f"{label} results must contain dataset typed IDs in exact order"
        )

    report = _load_json(report_path, label=f"{label} report")
    if not isinstance(report, dict):
        raise MechanismEvaluationError(f"{label} report must be an object")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1:
        raise MechanismEvaluationError(f"{label} report schema_version is invalid")
    if report.get("report_type") != REPORT_TYPE:
        raise MechanismEvaluationError(f"{label} report_type is invalid")
    if report.get("status") not in TERMINAL_REPORT_STATUSES:
        raise MechanismEvaluationError(f"{label} report is not terminal")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise MechanismEvaluationError(f"{label} report.configuration must be an object")
    if configuration.get("system_arm") != arm:
        raise MechanismEvaluationError(f"{label} configuration.system_arm mismatch")
    if configuration.get("checker_cache_enabled") is not False:
        raise MechanismEvaluationError(f"{label} must disable Checker cache")
    if configuration.get("candidate_source") != "frozen_target_binding":
        raise MechanismEvaluationError(f"{label} is not a target-binding replay")
    if configuration.get("checker_prompt_version") != SEMANTIC_RULE_CHECKER_PROMPT_VERSION:
        raise MechanismEvaluationError(f"{label} Checker prompt version drifted")
    model_name = str(configuration.get("checker_model") or "")
    if model_name != CHECKER_MODEL:
        raise MechanismEvaluationError(f"{label} Checker model is not Qwen30B")
    if configuration.get("dataset_sha256") != _sha256_file(dataset_path):
        raise MechanismEvaluationError(f"{label} dataset SHA mismatch")
    configuration_sha256 = str(report.get("configuration_sha256") or "")
    if configuration_sha256 != _object_sha256(configuration):
        raise MechanismEvaluationError(f"{label} configuration SHA mismatch")
    output = report.get("output")
    if not isinstance(output, dict):
        raise MechanismEvaluationError(f"{label} report.output must be an object")
    if output.get("sha256") != _sha256_file(result_path):
        raise MechanismEvaluationError(f"{label} result SHA mismatch")
    if _strict_int(output.get("record_count"), label=f"{label} output.record_count") != len(results):
        raise MechanismEvaluationError(f"{label} output record_count mismatch")
    llm_trace = report.get("llm_trace")
    if not isinstance(llm_trace, dict):
        raise MechanismEvaluationError(f"{label} report.llm_trace must be an object")
    if llm_trace.get("path") != configuration.get("llm_trace_path"):
        raise MechanismEvaluationError(f"{label} LLM trace path mismatch")
    _trace_matches_report(llm_trace, label=f"{label} LLM trace")
    trace_association = _audit_trace_associations(
        Path(str(llm_trace["path"])),
        dataset_rows=dataset_rows,
        result_rows=results,
        arm=arm,
        model=str(configuration.get("checker_model") or ""),
        label=f"{label} LLM trace",
    )

    records: List[Dict[str, Any]] = []
    valid_keys: Set[str] = set()
    failures: Counter[str] = Counter()
    for dataset_row, key in zip(dataset_rows, dataset_keys):
        record = _evaluate_result_row(
            result_index.get(key),
            dataset_row=dataset_row,
            arm=arm,
            configuration_sha256=configuration_sha256,
        )
        records.append(record)
        if record["structured_valid"]:
            valid_keys.add(key)
        else:
            failures[str(record["failure_stage"] or "unknown_failure")] += 1
    reported_statistics = report.get("statistics")
    if not isinstance(reported_statistics, dict):
        raise MechanismEvaluationError(f"{label} report.statistics must be an object")
    if reported_statistics.get("total_samples") != len(dataset_rows):
        raise MechanismEvaluationError(f"{label} report total_samples mismatch")
    return {
        "arm": arm,
        "repetition": repetition,
        "configuration": configuration,
        "normalized_configuration": _normalized_configuration(configuration),
        "configuration_sha256": configuration_sha256,
        "records": records,
        "records_by_key": dict(zip(dataset_keys, records)),
        "valid_keys": valid_keys,
        "failure_by_stage": dict(sorted(failures.items())),
        "result": {
            "path": str(result_path),
            "sha256": _sha256_file(result_path),
        },
        "report": {
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
            "status": report.get("status"),
        },
        "llm_trace": copy.deepcopy(llm_trace),
        "llm_trace_association": trace_association,
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return (numerator / denominator) if denominator else None


def _cell_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_keys: Optional[Set[str]] = None,
    dataset_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if selected_keys is not None:
        if dataset_keys is None:
            raise AssertionError("dataset_keys are required with selected_keys")
        records = [
            record
            for key, record in zip(dataset_keys, records)
            if key in selected_keys
        ]
    total = len(records)
    valid = [record for record in records if record.get("structured_valid") is True]
    first_attempt = sum(record.get("first_attempt_valid") is True for record in valid)
    positives = [record for record in valid if record.get("mechanism") == "true_violation"]
    true_positives = sum(record.get("publish_target_rule") is True for record in positives)
    negative_metrics: Dict[str, Any] = {}
    for mechanism in NEGATIVE_MECHANISMS:
        items = [record for record in valid if record.get("mechanism") == mechanism]
        false_positives = sum(record.get("publish_any_rule") is True for record in items)
        negative_metrics[mechanism] = {
            "valid_count": len(items),
            "false_positive_count": false_positives,
            "fpr": _ratio(false_positives, len(items)),
        }
    self_corrected = [
        record for record in valid if record.get("mechanism_subtype") == "self_corrected"
    ]
    protocol_values = [
        record.get("protocol_contradiction_count")
        for record in valid
        if record.get("protocol_contradiction_count") is not None
    ]
    metrics = {
        "cohort_size": total,
        "structured_valid_count": len(valid),
        "structured_output_rate": _ratio(len(valid), total),
        "first_attempt_valid_count": first_attempt,
        "first_attempt_structured_rate": _ratio(first_attempt, total),
        "true_violation": {
            "valid_count": len(positives),
            "target_publish_count": true_positives,
            "recall": _ratio(true_positives, len(positives)),
        },
        "negative_mechanisms": negative_metrics,
        "max_negative_fpr": max(
            (item["fpr"] for item in negative_metrics.values() if item["fpr"] is not None),
            default=None,
        ),
        "self_corrected": {
            "valid_count": len(self_corrected),
            "probe_diagnostic_count": sum(
                int(record.get("self_corrected_probe_diagnostic_count") or 0)
                for record in self_corrected
            ),
            "superseded_overlap_diagnostic_count": sum(
                int(record.get("self_corrected_superseded_overlap_count") or 0)
                for record in self_corrected
            ),
        },
        "protocol_contradiction_count": (
            sum(int(value) for value in protocol_values) if protocol_values else None
        ),
    }
    metrics["thresholds"] = {
        "structured_output_rate_ge_0_99": (
            metrics["structured_output_rate"] is not None
            and metrics["structured_output_rate"] >= 0.99
        ),
        "true_violation_recall_ge_0_90": (
            metrics["true_violation"]["recall"] is not None
            and metrics["true_violation"]["recall"] >= 0.90
        ),
        "every_negative_fpr_le_0_05": all(
            item["fpr"] is not None and item["fpr"] <= 0.05
            for item in negative_metrics.values()
        ),
        "self_corrected_probe_diagnostics_eq_0": (
            metrics["self_corrected"]["probe_diagnostic_count"] == 0
        ),
        "protocol_contradictions_eq_0": (
            None
            if metrics["protocol_contradiction_count"] is None
            else metrics["protocol_contradiction_count"] == 0
        ),
    }
    return metrics


def _composition(
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_keys: Sequence[str],
    keys: Set[str],
) -> Dict[str, Any]:
    mechanism_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    proxy_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    rule_ids: Set[str] = set()
    for row, key in zip(dataset_rows, dataset_keys):
        if key not in keys:
            continue
        mechanism_counts[str(row.get("mechanism") or "")] += 1
        subtype_counts[str(row.get("mechanism_subtype") or "")] += 1
        target = row.get("target_rule") or {}
        domain_counts[str(target.get("domain_id") or "")] += 1
        origin_counts[str(target.get("origin") or "")] += 1
        proxy_counts[str(target.get("trigger_scope_proxy") or "")] += 1
        primitive_counts[
            "true" if target.get("has_symbolic_primitive") is True else "false"
        ] += 1
        rule_ids.add(str(target.get("rule_id") or ""))
    return {
        "unique_target_rule_count": len(rule_ids),
        "mechanism": dict(sorted(mechanism_counts.items())),
        "mechanism_subtype": dict(sorted(subtype_counts.items())),
        "domain_id": dict(sorted(domain_counts.items())),
        "origin": dict(sorted(origin_counts.items())),
        "trigger_scope_proxy": dict(sorted(proxy_counts.items())),
        "has_symbolic_primitive": dict(sorted(primitive_counts.items())),
    }


def _intersection_summary(
    keys: Set[str],
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_keys: Sequence[str],
) -> Dict[str, Any]:
    ordered_ids = [
        _typed_descriptor(row.get("id"), label="intersection.id")
        for row, key in zip(dataset_rows, dataset_keys)
        if key in keys
    ]
    return {
        "record_count": len(keys),
        "coverage": _ratio(len(keys), len(dataset_keys)),
        "ordered_typed_ids": ordered_ids,
        "ordered_typed_ids_sha256": _object_sha256(ordered_ids),
        "composition": _composition(dataset_rows, dataset_keys, keys),
    }


def _excluded_records(
    *,
    keys: Set[str],
    dataset_rows: Sequence[Mapping[str, Any]],
    dataset_keys: Sequence[str],
    named_runs: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    excluded: List[Dict[str, Any]] = []
    for row, key in zip(dataset_rows, dataset_keys):
        if key in keys:
            continue
        failures: Dict[str, Any] = {}
        for name, run in named_runs.items():
            record = run["records_by_key"][key]
            if record.get("structured_valid") is not True:
                failures[name] = {
                    "failure_stage": record.get("failure_stage"),
                    "replay_completed": record.get("replay_completed"),
                }
        excluded.append(
            {
                "typed_id": _typed_descriptor(row.get("id"), label="excluded.id"),
                "failures": failures,
            }
        )
    return excluded


def _candidate_acceptance(
    *,
    runs: Mapping[Tuple[str, int], Mapping[str, Any]],
    triplicate_by_arm: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    repetitions: Dict[str, Any] = {}
    for repetition in REPETITIONS:
        run = runs.get((CANDIDATE_ARM, repetition))
        if run is None:
            repetitions[str(repetition)] = {
                "available": False,
                "passed": False,
                "failed_conditions": ["missing_candidate_run"],
            }
            continue
        metrics = _cell_metrics(run["records"])
        thresholds = metrics["thresholds"]
        required = {
            "structured_output_rate_ge_0_99": thresholds[
                "structured_output_rate_ge_0_99"
            ],
            "true_violation_recall_ge_0_90": thresholds[
                "true_violation_recall_ge_0_90"
            ],
            "every_negative_fpr_le_0_05": thresholds[
                "every_negative_fpr_le_0_05"
            ],
            "self_corrected_probe_diagnostics_eq_0": thresholds[
                "self_corrected_probe_diagnostics_eq_0"
            ],
            "protocol_contradictions_eq_0": thresholds[
                "protocol_contradictions_eq_0"
            ],
        }
        failed = [name for name, passed in required.items() if passed is not True]
        repetitions[str(repetition)] = {
            "available": True,
            "passed": not failed,
            "failed_conditions": failed,
            "metrics": metrics,
        }

    triplicate = triplicate_by_arm.get(CANDIDATE_ARM)
    triplicate_pass = bool(
        isinstance(triplicate, Mapping)
        and triplicate.get("agreement_ge_0_95") is True
    )
    all_repetitions_pass = all(
        repetitions[str(repetition)]["passed"] is True
        for repetition in REPETITIONS
    )
    overall_pass = all_repetitions_pass and triplicate_pass
    failed_conditions: List[str] = []
    if not all_repetitions_pass:
        failed_conditions.append("one_or_more_repetitions_failed")
    if not triplicate_pass:
        failed_conditions.append("triplicate_publish_agreement_below_0_95_or_unavailable")
    return {
        "candidate_arm": CANDIDATE_ARM,
        "baseline_arms_are_comparators_not_acceptance_requirements": True,
        "repetitions": repetitions,
        "all_repetitions_pass": all_repetitions_pass,
        "triplicate_agreement": (
            triplicate.get("agreement") if isinstance(triplicate, Mapping) else None
        ),
        "triplicate_agreement_pass": triplicate_pass,
        "overall_pass": overall_pass,
        "failed_conditions": failed_conditions,
    }


def _formal_dataset_design(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    generation_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    if len(dataset_rows) != 300:
        raise MechanismEvaluationError("formal P3 dataset must contain exactly 300 cases")
    generation_configuration = generation_audit.get("configuration")
    if not isinstance(generation_configuration, Mapping):
        raise MechanismEvaluationError("generation audit lacks configuration")
    ordered_targets = generation_configuration.get("target_rule_ids")
    if not isinstance(ordered_targets, list) or len(ordered_targets) != 60:
        raise MechanismEvaluationError("formal generation target order is invalid")

    observed_targets: List[str] = []
    fifth_subtypes: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    proxy_counts: Counter[str] = Counter()
    primitive_counts: Counter[bool] = Counter()
    for rule_index in range(60):
        rows = list(dataset_rows[5 * rule_index : 5 * (rule_index + 1)])
        target_rule_id = str(rows[0].get("target_rule_id") or "")
        if any(row.get("target_rule_id") != target_rule_id for row in rows):
            raise MechanismEvaluationError("formal dataset rule cases are not contiguous")
        if [row.get("mechanism") for row in rows] != list(MECHANISMS):
            raise MechanismEvaluationError(
                f"formal target {target_rule_id} does not contain the five mechanisms in order"
            )
        target_projection = rows[0].get("target_rule")
        if any(row.get("target_rule") != target_projection for row in rows[1:]):
            raise MechanismEvaluationError(
                f"formal target {target_rule_id} has inconsistent ownership projection"
            )
        if not isinstance(target_projection, Mapping):
            raise MechanismEvaluationError("formal target projection is invalid")
        observed_targets.append(target_rule_id)
        fifth_subtypes[str(rows[4].get("mechanism_subtype") or "")] += 1
        domain_counts[str(target_projection.get("domain_id") or "")] += 1
        origin_counts[str(target_projection.get("origin") or "")] += 1
        proxy_counts[str(target_projection.get("trigger_scope_proxy") or "")] += 1
        primitive_counts[target_projection.get("has_symbolic_primitive") is True] += 1
    if observed_targets != ordered_targets or len(set(observed_targets)) != 60:
        raise MechanismEvaluationError("formal dataset target order differs from frozen plan")
    if fifth_subtypes != Counter(
        {"self_corrected": 30, "insufficient_information": 30}
    ):
        raise MechanismEvaluationError("formal fifth-mechanism subtype balance drifted")
    if origin_counts != Counter({"gen": 30, "exp": 30}):
        raise MechanismEvaluationError("formal origin quota drifted")
    if proxy_counts != Counter({"broad_proxy": 30, "narrow_proxy": 30}):
        raise MechanismEvaluationError("formal trigger proxy quota drifted")
    if primitive_counts != Counter({True: 46, False: 14}):
        raise MechanismEvaluationError("formal symbolic primitive quota drifted")
    expected_domains = Counter(
        {
            "mechanics": 11,
            "electromagnetism": 11,
            "thermodynamics_statistical_physics": 10,
            "optics": 10,
            "modern_physics": 10,
            "experimental_physics": 8,
        }
    )
    if domain_counts != expected_domains:
        raise MechanismEvaluationError("formal domain quota drifted")
    return {
        "case_count": 300,
        "target_rule_count": 60,
        "mechanism_counts": {mechanism: 60 for mechanism in MECHANISMS},
        "fifth_subtype_counts": dict(sorted(fifth_subtypes.items())),
        "domain_rule_counts": dict(sorted(domain_counts.items())),
        "origin_rule_counts": dict(sorted(origin_counts.items())),
        "trigger_scope_proxy_rule_counts": dict(sorted(proxy_counts.items())),
        "has_symbolic_primitive_rule_counts": {
            "true": primitive_counts[True],
            "false": primitive_counts[False],
        },
    }


def _validate_generation_replay_binding(
    runs: Mapping[Tuple[str, int], Mapping[str, Any]],
    *,
    generation_audit: Mapping[str, Any],
    formal: bool,
) -> None:
    generation_configuration = generation_audit.get("configuration")
    if not isinstance(generation_configuration, Mapping):
        raise MechanismEvaluationError("generation audit lacks configuration")
    expected_dataset_sha = generation_audit.get("dataset_sha256")
    expected_trace_sha = generation_audit.get("candidate_trace_sha256")
    expected_catalog_sha = generation_configuration.get("catalog_sha256")
    generation_source = generation_configuration.get("source_identity")
    for (arm, repetition), run in runs.items():
        configuration = run.get("configuration")
        if not isinstance(configuration, Mapping):
            raise MechanismEvaluationError("validated replay lacks configuration")
        label = f"{arm}/repetition-{repetition}"
        if configuration.get("dataset_sha256") != expected_dataset_sha:
            raise MechanismEvaluationError(f"{label} is not bound to the generated dataset")
        if configuration.get("retrieval_trace_sha256") != expected_trace_sha:
            raise MechanismEvaluationError(
                f"{label} is not bound to the generated target candidate trace"
            )
        if configuration.get("unified_catalog_sha256") != expected_catalog_sha:
            raise MechanismEvaluationError(f"{label} catalog differs from GT generation")
        if formal:
            if configuration.get("run_kind") not in {"validation", "final"}:
                raise MechanismEvaluationError(f"{label} is not a formal replay run")
            source = configuration.get("source_identity")
            if (
                not isinstance(source, Mapping)
                or source.get("git_available") is not True
                or source.get("git_dirty") is not False
                or not isinstance(generation_source, Mapping)
                or source.get("git_head") != generation_source.get("git_head")
                or source.get("source_tree_sha256")
                != generation_source.get("source_tree_sha256")
            ):
                raise MechanismEvaluationError(
                    f"{label} source identity differs from formal GT generation"
                )
            if configuration.get("checker_json_attempts") != 3:
                raise MechanismEvaluationError(f"{label} Checker attempt policy drifted")
            if configuration.get("llm_temperature") != 0.1:
                raise MechanismEvaluationError(f"{label} Checker temperature drifted")
            if configuration.get("llm_max_output_tokens") != 2048:
                raise MechanismEvaluationError(f"{label} Checker token limit drifted")
            if configuration.get("precision_mode") != "strict":
                raise MechanismEvaluationError(f"{label} precision mode drifted")


def evaluate_mechanism_gate(
    *,
    dataset_path: Path,
    run_specs: Sequence[Tuple[str, int, Path, Path]],
    require_full_matrix: bool = True,
    formal: bool,
    generation_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    raw_dataset = _load_array(dataset_path, label="mechanism dataset")
    dataset_rows, dataset_keys = _validate_dataset(raw_dataset)
    generation_audit: Optional[Dict[str, Any]] = None
    evaluator_source_identity: Optional[Dict[str, Any]] = None
    if generation_manifest_path is not None:
        try:
            generation_audit = audit_generation_artifacts(
                dataset_path=dataset_path,
                manifest_path=generation_manifest_path,
            )
        except MechanismDatasetError as exc:
            raise MechanismEvaluationError(
                f"generation artifact audit failed: {exc}"
            ) from exc
    if formal:
        if not require_full_matrix:
            raise MechanismEvaluationError("formal evaluation cannot relax the 3x3 matrix")
        if generation_audit is None:
            raise MechanismEvaluationError(
                "formal evaluation requires a generation manifest"
            )
        if generation_audit.get("complete") is not True:
            raise MechanismEvaluationError(
                "formal evaluation requires a complete 60-rule/300-case generation audit"
            )
        if (
            generation_audit.get("planned_rule_count") != 60
            or generation_audit.get("completed_rule_count") != 60
            or generation_audit.get("dataset_case_count") != 300
            or generation_audit.get("candidate_trace_count") != 300
        ):
            raise MechanismEvaluationError("formal generation composition is invalid")
        generation_configuration = generation_audit.get("configuration")
        if not isinstance(generation_configuration, Mapping):
            raise MechanismEvaluationError("formal generation audit lacks configuration")
        evaluator_source_identity = _validate_formal_evaluator_source_identity(
            _capture_source_identity(),
            generation_configuration.get("source_identity"),
        )
    formal_design = (
        _formal_dataset_design(dataset_rows, generation_audit=generation_audit)
        if formal and generation_audit is not None
        else None
    )
    expected_cells = {(arm, rep) for arm in ARMS for rep in REPETITIONS}
    supplied_cells = {(arm, rep) for arm, rep, _, _ in run_specs}
    if len(supplied_cells) != len(run_specs):
        raise MechanismEvaluationError("duplicate arm/repetition run cell")
    if require_full_matrix and supplied_cells != expected_cells:
        missing = sorted(expected_cells - supplied_cells)
        extra = sorted(supplied_cells - expected_cells)
        raise MechanismEvaluationError(
            f"formal evaluation requires exactly the 3x3 matrix; missing={missing}, extra={extra}"
        )
    for arm, repetition, _, _ in run_specs:
        if arm not in ARMS or repetition not in REPETITIONS:
            raise MechanismEvaluationError(f"invalid run cell: {(arm, repetition)!r}")
    all_paths = [dataset_path.resolve()]
    for _, _, result_path, report_path in run_specs:
        all_paths.extend([result_path.resolve(), report_path.resolve()])
    if len(all_paths) != len(set(all_paths)):
        raise MechanismEvaluationError("dataset/result/report paths must all be distinct")

    runs: Dict[Tuple[str, int], Dict[str, Any]] = {}
    trace_paths: Set[Path] = set()
    trace_hashes: Set[str] = set()
    shared_configuration: Optional[Dict[str, Any]] = None
    for arm, repetition, result_path, report_path in sorted(
        run_specs, key=lambda item: (ARMS.index(item[0]), item[1])
    ):
        run = _validate_run(
            arm=arm,
            repetition=repetition,
            result_path=result_path,
            report_path=report_path,
            dataset_path=dataset_path,
            dataset_rows=dataset_rows,
            dataset_keys=dataset_keys,
        )
        normalized = run["normalized_configuration"]
        if shared_configuration is None:
            shared_configuration = normalized
        elif normalized != shared_configuration:
            raise MechanismEvaluationError(
                f"{arm}/repetition-{repetition} configuration differs outside the run allowlist"
            )
        trace_path = Path(str(run["llm_trace"]["path"])).resolve()
        if trace_path in trace_paths:
            raise MechanismEvaluationError("every arm/repetition must use a distinct LLM trace")
        trace_paths.add(trace_path)
        trace_sha256 = str(run["llm_trace"].get("sha256") or "")
        trace_record_count = run["llm_trace"].get("record_count")
        if (
            isinstance(trace_record_count, int)
            and not isinstance(trace_record_count, bool)
            and trace_record_count > 0
            and trace_sha256 in trace_hashes
        ):
            raise MechanismEvaluationError(
                "every arm/repetition must use an independently produced LLM trace"
            )
        if isinstance(trace_record_count, int) and trace_record_count > 0:
            trace_hashes.add(trace_sha256)
        runs[(arm, repetition)] = run

    if generation_audit is not None:
        _validate_generation_replay_binding(
            runs,
            generation_audit=generation_audit,
            formal=formal,
        )

    cell_outputs: Dict[str, Any] = {}
    for (arm, repetition), run in runs.items():
        key = f"{arm}::r{repetition}"
        cell_outputs[key] = {
            "arm": arm,
            "repetition": repetition,
            "configuration_sha256": run["configuration_sha256"],
            "result": run["result"],
            "report": run["report"],
            "llm_trace": run["llm_trace"],
            "llm_trace_association": run["llm_trace_association"],
            "coverage": _ratio(len(run["valid_keys"]), len(dataset_keys)),
            "failure_by_stage": run["failure_by_stage"],
            "metrics": _cell_metrics(run["records"]),
        }

    paired_by_repetition: Dict[str, Any] = {}
    for repetition in REPETITIONS:
        available = [runs[(arm, repetition)] for arm in ARMS if (arm, repetition) in runs]
        if len(available) != len(ARMS):
            continue
        keys = set(dataset_keys)
        for run in available:
            keys.intersection_update(run["valid_keys"])
        paired_by_repetition[str(repetition)] = {
            **_intersection_summary(
                keys,
                dataset_rows=dataset_rows,
                dataset_keys=dataset_keys,
            ),
            "arm_metrics": {
                run["arm"]: _cell_metrics(
                    run["records"], selected_keys=keys, dataset_keys=dataset_keys
                )
                for run in available
            },
            "excluded_records": _excluded_records(
                keys=keys,
                dataset_rows=dataset_rows,
                dataset_keys=dataset_keys,
                named_runs={run["arm"]: run for run in available},
            ),
        }

    triplicate_by_arm: Dict[str, Any] = {}
    for arm in ARMS:
        available = [runs[(arm, repetition)] for repetition in REPETITIONS if (arm, repetition) in runs]
        if len(available) != len(REPETITIONS):
            continue
        keys = set(dataset_keys)
        for run in available:
            keys.intersection_update(run["valid_keys"])
        consistent = 0
        per_mechanism: Dict[str, Dict[str, int]] = {
            mechanism: {"eligible": 0, "consistent": 0} for mechanism in MECHANISMS
        }
        for dataset_row, key in zip(dataset_rows, dataset_keys):
            if key not in keys:
                continue
            decisions = [
                bool(run["records_by_key"][key]["publish_any_rule"])
                for run in available
            ]
            is_consistent = len(set(decisions)) == 1
            consistent += int(is_consistent)
            mechanism = str(dataset_row.get("mechanism") or "")
            per_mechanism[mechanism]["eligible"] += 1
            per_mechanism[mechanism]["consistent"] += int(is_consistent)
        triplicate_by_arm[arm] = {
            **_intersection_summary(
                keys,
                dataset_rows=dataset_rows,
                dataset_keys=dataset_keys,
            ),
            "consistent_count": consistent,
            "agreement": _ratio(consistent, len(keys)),
            "agreement_ge_0_95": bool(keys) and consistent / len(keys) >= 0.95,
            "by_mechanism": {
                mechanism: {
                    **counts,
                    "agreement": _ratio(counts["consistent"], counts["eligible"]),
                }
                for mechanism, counts in per_mechanism.items()
            },
            "excluded_records": _excluded_records(
                keys=keys,
                dataset_rows=dataset_rows,
                dataset_keys=dataset_keys,
                named_runs={f"r{run['repetition']}": run for run in available},
            ),
        }

    global_keys: Optional[Set[str]] = None
    if runs:
        global_keys = set(dataset_keys)
        for run in runs.values():
            global_keys.intersection_update(run["valid_keys"])
    global_keys = global_keys or set()
    global_summary = _intersection_summary(
        global_keys,
        dataset_rows=dataset_rows,
        dataset_keys=dataset_keys,
    )
    global_summary["excluded_records"] = _excluded_records(
        keys=global_keys,
        dataset_rows=dataset_rows,
        dataset_keys=dataset_keys,
        named_runs={
            f"{arm}::r{repetition}": run
            for (arm, repetition), run in runs.items()
        },
    )
    global_summary["cell_metrics"] = {
        f"{arm}::r{repetition}": _cell_metrics(
            run["records"], selected_keys=global_keys, dataset_keys=dataset_keys
        )
        for (arm, repetition), run in runs.items()
    }
    candidate_acceptance = _candidate_acceptance(
        runs=runs,
        triplicate_by_arm=triplicate_by_arm,
    )
    formal_gate = {
        "mode": "formal" if formal else "development",
        "authenticity_and_design_pass": True if formal else None,
        "candidate_metrics_pass": (
            candidate_acceptance["overall_pass"] if formal else None
        ),
        "overall_pass": candidate_acceptance["overall_pass"] if formal else None,
    }

    generation_projection: Optional[Dict[str, Any]] = None
    if generation_audit is not None:
        generation_projection = {
            key: copy.deepcopy(generation_audit.get(key))
            for key in (
                "valid",
                "complete",
                "run_complete",
                "run_kind",
                "provider_kind",
                "planned_rule_count",
                "completed_rule_count",
                "dataset_case_count",
                "candidate_trace_count",
                "raw_trace_record_count",
                "manifest_sha256",
                "dataset_sha256",
                "candidate_trace_sha256",
                "raw_response_trace_sha256",
                "artifact_paths",
            )
        }

    if formal:
        assert evaluator_source_identity is not None
        final_evaluator_source = _capture_source_identity()
        for field in (
            "git_available",
            "git_head",
            "git_dirty",
            "git_status_sha256",
            "git_tracked_diff_sha256",
            "source_tree_sha256",
            "source_file_count",
        ):
            if final_evaluator_source.get(field) != evaluator_source_identity.get(field):
                raise MechanismEvaluationError(
                    "formal evaluator source identity changed during evaluation"
                )

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "evaluation_type": "p3_checker_target_rule_mechanism_gate",
        "effect_scope": (
            "Conditional Checker+release performance with the preregistered target rule "
            "provided; retrieval performance is excluded."
        ),
        "evaluation_mode": "formal" if formal else "development",
        "evaluator_source_identity": evaluator_source_identity,
        "generation_audit": generation_projection,
        "formal_dataset_design": formal_design,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "record_count": len(dataset_rows),
            "ordered_typed_ids": [
                _typed_descriptor(row.get("id"), label="dataset.id")
                for row in dataset_rows
            ],
            "ordered_typed_ids_sha256": _object_sha256(
                [
                    _typed_descriptor(row.get("id"), label="dataset.id")
                    for row in dataset_rows
                ]
            ),
            "composition": _composition(dataset_rows, dataset_keys, set(dataset_keys)),
        },
        "shared_configuration": shared_configuration or {},
        "shared_configuration_sha256": _object_sha256(shared_configuration or {}),
        "cells": cell_outputs,
        "common_valid_by_repetition": paired_by_repetition,
        "triplicate_by_arm": triplicate_by_arm,
        "global_common_valid": global_summary,
        "candidate_acceptance": candidate_acceptance,
        "formal_gate": formal_gate,
        "threshold_policy": {
            "candidate_arm": CANDIDATE_ARM,
            "baseline_arms_are_not_acceptance_requirements": True,
            "each_repetition_must_pass": True,
            "repeat_outputs_are_not_independent_samples": True,
            "structured_output_rate_min": 0.99,
            "true_violation_recall_min": 0.90,
            "each_negative_mechanism_fpr_max": 0.05,
            "self_corrected_probe_diagnostic_count_max": 0,
            "protocol_contradiction_count_max": 0,
            "triplicate_agreement_min": 0.95,
        },
    }


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MechanismEvaluationError(f"refusing to overwrite existing output: {path}")
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
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise MechanismEvaluationError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the preregistered target-rule Checker mechanism gate. "
            "Formal runs require 3 arms x 3 repetitions."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--generation-manifest",
        default="",
        help="Required in formal mode; binds the Gemini GT, plan, catalog, and target trace.",
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("ARM", "REPETITION", "RESULTS", "REPORT"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-incomplete-matrix",
        action="store_true",
        help="Development/smoke only; formal evaluation requires all 9 cells.",
    )
    parser.add_argument(
        "--development-smoke",
        action="store_true",
        help="Explicitly disable the formal 60-rule/300-case acceptance decision.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.allow_incomplete_matrix and not args.development_smoke:
        parser.error("--allow-incomplete-matrix requires --development-smoke")
    if not args.development_smoke and not args.generation_manifest:
        parser.error("formal evaluation requires --generation-manifest")
    specs: List[Tuple[str, int, Path, Path]] = []
    for arm, repetition_value, result_value, report_value in args.run:
        try:
            repetition = int(repetition_value)
        except ValueError:
            parser.error(f"invalid repetition: {repetition_value!r}")
        specs.append((arm, repetition, Path(result_value), Path(report_value)))
    try:
        report = evaluate_mechanism_gate(
            dataset_path=Path(args.dataset),
            run_specs=specs,
            require_full_matrix=not args.allow_incomplete_matrix,
            formal=not args.development_smoke,
            generation_manifest_path=(
                Path(args.generation_manifest) if args.generation_manifest else None
            ),
        )
        _write_json_new(Path(args.output), report)
    except MechanismEvaluationError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "dataset_size": report["dataset"]["record_count"],
                "run_cells": len(report["cells"]),
                "global_common_valid": report["global_common_valid"]["record_count"],
                "formal_gate_pass": report["formal_gate"]["overall_pass"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if report["evaluation_mode"] == "formal" and report["formal_gate"]["overall_pass"] is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
