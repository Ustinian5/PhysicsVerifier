from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _typed_id_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("sample IDs must be non-empty strings or integers")
    if isinstance(value, str) and not value.strip():
        raise ValueError("sample IDs must not be empty")
    return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False, sort_keys=True)}"


def _index_by_id(
    items: List[Dict[str, Any]],
    *,
    label: str,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(items):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{index}] must be a JSON object")
        if "id" not in row:
            raise ValueError(f"{label}[{index}] is missing id")
        try:
            sid = _typed_id_key(row.get("id"))
        except ValueError as exc:
            raise ValueError(f"invalid {label}[{index}].id: {exc}") from exc
        if sid in out:
            raise ValueError(f"duplicate typed sample ID in {label}: {row.get('id')!r}")
        out[sid] = row
    return out


def _result_identity(items: List[Dict[str, Any]]) -> Dict[str, str]:
    identity: Dict[str, str] = {}
    for field in ("checker_gate_mode", "replay_config_sha256"):
        present = [str(row.get(field) or "").strip() for row in items]
        nonempty = {value for value in present if value}
        if len(nonempty) > 1:
            raise ValueError(f"results mix multiple {field} values")
        if nonempty and any(not value for value in present):
            raise ValueError(f"results mix missing and populated {field} values")
        identity[field] = next(iter(nonempty), "")
    return identity


def _collect_pred_findings(pred_item: Dict[str, Any], audit_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    diagnostics = pred_item.get("diagnostics") if isinstance(pred_item, dict) else []
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    for d in diagnostics:
        if not isinstance(d, dict):
            continue
        rule = str(d.get("rule") or "").strip()
        message = str(d.get("message") or "").strip()
        evidence = d.get("evidence")
        quote = ""
        if isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "").strip()
        elif isinstance(evidence, str):
            quote = evidence.strip()

        text = " | ".join([x for x in [rule, message, quote] if x])
        if text:
            findings.append(
                {
                    "source": "diagnostic",
                    "rule": rule,
                    "message": message,
                    "quote": quote,
                    "text": text,
                }
            )

    checker_mode = str(pred_item.get("checker_gate_mode") or "legacy").strip().lower()
    checks = audit_item.get("experience_code_checks") if isinstance(audit_item, dict) else []
    if checker_mode != "legacy":
        checks = []
    checks = checks if isinstance(checks, list) else []
    for c in checks:
        if not isinstance(c, dict):
            continue
        result = str(c.get("result") or "").strip().lower()
        if result != "fail" or str(c.get("publish_skipped") or "").strip():
            continue
        rule = str(c.get("rule") or "").strip()
        message = str(c.get("message") or "").strip()
        evidence = str(c.get("evidence") or "").strip()
        text = " | ".join([x for x in [rule, message, evidence] if x])
        if text:
            findings.append(
                {
                    "source": "experience_code",
                    "rule": rule,
                    "message": message,
                    "quote": "",
                    "text": text,
                }
            )

    out: List[Dict[str, Any]] = []
    seen = set()
    for f in findings:
        key = (str(f.get("source") or ""), str(f.get("rule") or ""), str(f.get("message") or ""), str(f.get("quote") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _execution_failure_reason(pred_item: Any) -> str:
    if not isinstance(pred_item, dict):
        return "missing_result"
    selection = str(pred_item.get("selection_strategy") or "").strip().lower()
    if selection in {"semantic_error", "semantic_unavailable"} or str(
        pred_item.get("semantic_selection_error") or ""
    ).strip():
        return "semantic_retrieval_failure"
    checker_status = str(pred_item.get("checker_status") or "").strip().lower()
    try:
        checker_failure_count = int(pred_item.get("checker_failure_count") or 0)
    except (TypeError, ValueError):
        checker_failure_count = 1
    raw_checker_failures = pred_item.get("checker_failures")
    if isinstance(raw_checker_failures, list):
        checker_failure_count = max(checker_failure_count, len(raw_checker_failures))
    if checker_failure_count > 0 or checker_status in {
        "failed",
        "partial_failure",
        "not_run",
    }:
        return "checker_failure"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate question-level metrics on mixed positive/negative dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--audit", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ds = _load_json(args.dataset)
    pred = _load_json(args.results)
    audit = _load_json(args.audit)

    if not isinstance(ds, list):
        raise SystemExit("Dataset file must be a JSON array.")
    if not isinstance(pred, list):
        raise SystemExit("Results file must be a JSON array.")
    if not isinstance(audit, list):
        raise SystemExit("Audit file must be a JSON array.")

    try:
        _index_by_id(ds, label="dataset")
        pred_idx = _index_by_id(pred, label="results")
        audit_idx = _index_by_id(audit, label="audit")
        result_identity = _result_identity(pred)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    details: List[Dict[str, Any]] = []
    failure_by_stage: Dict[str, int] = {}

    for row in ds:
        if not isinstance(row, dict):
            continue
        raw_id = row.get("id")
        sid_key = _typed_id_key(raw_id)
        sid = str(raw_id)

        expected_has_error = bool(row.get("expected_has_physics_error"))
        pred_item = pred_idx.get(sid_key)
        failure_reason = _execution_failure_reason(pred_item)
        if failure_reason:
            failure_by_stage[failure_reason] = failure_by_stage.get(failure_reason, 0) + 1
            details.append(
                {
                    "id": sid,
                    "eval_split": str(row.get("eval_split") or ""),
                    "expected_has_physics_error": expected_has_error,
                    "scored": False,
                    "execution_failure": failure_reason,
                }
            )
            continue
        assert isinstance(pred_item, dict)
        audit_item = audit_idx.get(sid_key, {})
        findings = _collect_pred_findings(pred_item, audit_item)
        pred_has_error = bool(findings)

        if expected_has_error and pred_has_error:
            tp += 1
        elif expected_has_error and (not pred_has_error):
            fn += 1
        elif (not expected_has_error) and pred_has_error:
            fp += 1
        else:
            tn += 1

        details.append(
            {
                "id": sid,
                "eval_split": str(row.get("eval_split") or ""),
                "expected_has_physics_error": expected_has_error,
                "pred_has_error": pred_has_error,
                "pred_finding_count": len(findings),
                "decision_source": "diagnostics_or_audit",
                "scored": True,
                "execution_failure": "",
            }
        )

    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    output = {
        "summary": {
            "level": "question",
            "dataset_size": len(details),
            "scored_size": tp + fp + tn + fn,
            "failed_size": sum(failure_by_stage.values()),
            "coverage": ((tp + fp + tn + fn) / len(details)) if details else 0.0,
            "failure_by_stage": failure_by_stage,
            "checker_gate_mode": result_identity["checker_gate_mode"],
            "replay_config_sha256": result_identity["replay_config_sha256"],
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "precision_proxy": 1.0 - (fp / (fp + tn)) if (fp + tn) else 0.0,
        },
        "details": details,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
