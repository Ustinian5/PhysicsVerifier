"""Freeze an existing historical pool; never regenerate answers or choose by outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from run_retrieval_comparison import model_input, sha256_file, write_json


def question_key(text):
    return hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()


def freeze(source, audit, pilot_dir, output, *, size=50, seed=20260918):
    rows = json.loads(source.read_text())
    historical = json.loads(audit.read_text())
    exposed = []
    for path in sorted(pilot_dir.glob("C0[1-6].json")):
        case = json.loads(path.read_text())
        original = case["original_row"]
        exposed.append({"id": str(original["id"]), "question_key": question_key(original["question"]),
                        "case": case["case_id"], "evidence": str(path), "evidence_sha256": sha256_file(path)})
    if len(exposed) != 6:
        raise ValueError("Expected the six previously used Astra development cases")
    excluded_ids = {r["id"] for r in exposed}
    excluded_questions = {r["question_key"] for r in exposed}
    eligible, excluded, seen = [], [], set()
    for index, row in enumerate(rows):
        if not row.get("question") or not row.get("prediction"):
            raise ValueError("Historical input is incomplete; review instead of silently dropping it")
        key = question_key(row["question"])
        reason = "Astra_development_exposure" if str(row["id"]) in excluded_ids or key in excluded_questions else "duplicate_question" if key in seen else None
        seen.add(key)
        if reason:
            excluded.append({"source_index": index, "id": row["id"], "reason": reason})
        else:
            eligible.append(index)
    if len(eligible) < size:
        raise ValueError("Not enough eligible historical questions")
    indices = sorted(random.Random(seed).sample(eligible, size))
    samples = [model_input(rows[i]) for i in indices]
    if len({str(r["id"]) for r in samples}) != size:
        raise ValueError("Duplicate selected IDs")
    audit_index = {(str(d["id"]), str(e["error_id"])): e for d in historical["details"] for e in d.get("items", [])}
    references, label_counts, location_counts = [], Counter(), Counter()
    for index in indices:
        row = rows[index]
        reference = {"id": row["id"], "source_index": index, "answer": row.get("answer"),
                     "expected_has_physics_error": row.get("expected_has_physics_error"), "errors": []}
        for error in row.get("physics_error_gt", []):
            previous = audit_index.get((str(row["id"]), str(error["error_id"])))
            label_counts[previous["label"] if previous else "not_covered_by_historical_audit"] += 1
            start, end = error.get("start_char"), error.get("end_char")
            quote = error.get("answer_quote", "")
            exact = bool(quote and type(start) is int and type(end) is int and 0 <= start < end <= len(row["prediction"]) and row["prediction"][start:end] == quote)
            grounded = bool(quote and quote in row["prediction"])
            location_counts["exact_offset" if exact else "quote_found_offset_needs_review" if grounded else "quote_not_exact"] += 1
            reference["errors"].append({"source_gt": error, "historical_model_audit": previous,
                                        "exact_offset": exact, "exact_quote_present": grounded,
                                        "current_review_status": "pending"})
        references.append(reference)
    manifest = {"phase": "inputs_frozen_annotations_pending", "seed": seed, "sample_size": size,
                "source_sha256": sha256_file(source), "audit_sha256": sha256_file(audit),
                "source_rows": len(rows), "source_gt_count": sum(len(r.get("physics_error_gt", [])) for r in rows),
                "historical_audit_model": historical["summary"].get("model"),
                "question_deduplication": "SHA256 of full question after removing whitespace; case preserved",
                "eligible_count": len(eligible), "exposed_development_cases": exposed, "excluded": excluded,
                "selected_source_indices": indices, "selected_ids": [r["id"] for r in samples],
                "selected_gt_count": sum(len(r["errors"]) for r in references),
                "historical_audit_label_counts": dict(label_counts), "location_checks": dict(location_counts),
                "selected_expected_error_counts": dict(Counter(str(r.get("expected_has_physics_error")) for r in references)),
                "rule_construction_overlap": "unknown: minimal catalog has no complete sample-to-rule source map",
                "historical_test_exposure": "all selected inputs come from a previously evaluated internal pool",
                "script_sha256": sha256_file(Path(__file__)),
                "input_field_hashes": [{"id": r["id"], **{k: hashlib.sha256(r[k].encode()).hexdigest() for k in ("question", "prediction", "context") if k in r}} for r in samples]}
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already exists; do not overwrite frozen inputs")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inputs.json", samples)
    write_json(output / "operator_only" / "reference_review.json", references)
    manifest["inputs_sha256"] = sha256_file(output / "inputs.json")
    manifest["reference_review_sha256"] = sha256_file(output / "operator_only" / "reference_review.json")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--pilot-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.source, args.audit, args.pilot_cases, args.output)
    print(json.dumps({k: manifest[k] for k in ("source_rows", "source_gt_count", "eligible_count", "sample_size", "selected_gt_count", "historical_audit_label_counts", "location_checks", "inputs_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
