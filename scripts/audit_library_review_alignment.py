"""Replay the two construction reviews behind a released verifier library, offline."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rule_framework import flat_skill_rag as flat
from rule_framework import organization_followup as bounded
from rule_framework import verified_resume as vr


def checked(path, expected_hash):
    data = path.read_bytes()
    flat.require(hashlib.sha256(data).hexdigest() == expected_hash, f"Evidence changed: {path}")
    return json.loads(data)


def model_request(record, model):
    request = record["request"]
    flat.require(request["model"] == model and request["reasoning_effort"] == "medium"
                 and record["response"]["model"] == model, "Unexpected construction model")
    return request["messages"]


def flat_outcome(directory, manifest, row, binding, messages, model):
    key = vr.digest(row["id"])[:24]
    path = directory / "collected" / (key + ".json")
    saved = checked(path, manifest[str(path)])
    outcome = saved["outcome"]
    flat.require(outcome["id"] == row["id"] and saved["source_sha256"] == vr.digest(row)
                 and saved["configuration_sha256"] == binding, f"Wrong source: {path}")
    folder = directory / "arm_extract_skill" / key
    if outcome["status"] == "collected":
        result_file = folder / "result.json"
        result = checked(result_file, manifest[str(result_file)])
        flat.require(result["value"] == outcome["value"], f"Decoded outcome differs: {result_file}")
        raw = folder / result["request_file"]
        flat.require(result["request_sha256"] == manifest[str(raw)], f"Wrong response: {raw}")
    else:
        flat.require(outcome["status"] == "content_failure", f"Unfinished source: {path}")
        raw, = sorted(folder.glob("request_*.json"))
    actual = model_request(checked(raw, manifest[str(raw)]), model)
    flat.require(actual == messages, f"Review input differs: {raw}")
    return saved, raw


def audit(release, example_id):
    plan = vr.read(release / "plan.json")
    cfg = plan["configuration"]
    flat.require(vr.digest(cfg) == plan["configuration_sha256"], "Release plan drift")
    archive = Path(cfg["archive"])
    aligned = Path(cfg["material_parent"]).parent
    construction = Path(cfg["root"]) / "corrected" / "construction"
    prior = vr.read(archive / "plan.json")["configuration"]
    files = vr.read(archive / "source_evidence.json")
    flat.require(vr.digest(files) == prior["evidence_digest"], "Construction manifest drift")
    bundle = checked(archive / "tree_bundle.json", prior["frozen_files"]["tree_bundle.json"])
    inputs = {r["id"]: r for r in checked(construction / "inputs.json", files[str(construction / "inputs.json")])}
    parent = vr.read(aligned / "plan.json")
    binding = parent["configuration_sha256"]
    flat.require(vr.digest(parent["configuration"]) == binding, "Case plan drift")
    manifests = {k: vr.unseal(aligned / k / "request_index.json")["files"] for k in ("experience", "skills")}
    docs = {k: checked(release / (k + ".json"), h) for k, h in cfg["material_hashes"].items()}
    support = defaultdict(set)
    for domain in docs["catalog"]["domains"]:
        for topic in domain["topics"]:
            for leaf in topic["rules"]:
                rid = leaf["rule_id"]
                for entry in bundle["provenance"][rid]:
                    support[entry["sample_id"]].add(rid)
    statuses, skill_outcomes, rows = {}, [], []
    example = {}
    for case in docs["cases"]:
        sid, = case["source_ids"]
        row = flat.source_payload(inputs[sid])
        flat.require(sid not in statuses, f"Duplicate published case: {sid}")
        path = construction / "experiences" / (sid + ".json")
        first = checked(path, files[str(path)])
        audit_record = first["value"]["semantic_audit"]
        flat.require(first["value"]["sample_id"] == sid and audit_record["status"] == "error_found",
                     f"Case outside first review's error pool: {sid}")
        ev = first["evidence"][0]
        raw = construction / ev["file"]
        flat.require(ev["file"].startswith("raw/extract_experience/") and ev["sha256"] == files[str(raw)],
                     f"Wrong first review: {sid}")
        first_messages = model_request(checked(raw, ev["sha256"]), cfg["model"])
        visible = json.loads(first_messages[1]["content"])
        flat.require(visible["sample_id"] == sid and visible["question"] == row["question"]
                     and "".join(line["text"] for line in visible["prediction_lines"]) == row["prediction"],
                     f"30B answer differs across reviews: {sid}")
        note, note_raw = flat_outcome(aligned / "experience", manifests["experience"], row, binding,
                                      flat.build_messages(row), cfg["model"])
        outcome = note["outcome"]
        bound = dict(outcome, configuration_sha256=binding, source_sha256=note["source_sha256"])
        rebuilt, = flat.materialize([row], [bound], binding)["cases"]
        flat.require(rebuilt == case, f"Published case differs: {sid}")
        skill, skill_raw = flat_outcome(aligned / "skills", manifests["skills"], row, binding,
                                        bounded.skill_messages(case), cfg["model"])
        flat.require(skill["case_sha256"] == vr.digest(case), f"Wrong skill parent: {sid}")
        rows.append(row)
        skill_outcomes.append(skill["outcome"])
        statuses[sid] = outcome["value"]["answer_status"] if outcome["status"] == "collected" else "note_unavailable"
        if sid == example_id:
            example = dict(source_id=sid, case_id=case["id"], title=case["title"],
                           first_review=audit_record["summary"], second_status=statuses[sid],
                           second_review=outcome.get("value", {}).get("case_note"), published_leaves=sorted(support.get(sid, ())),
                           published_skills=[s["id"] for s in docs["skills"] if sid in s["source_ids"]],
                           raw_files={str(raw): ev["sha256"], str(note_raw): manifests["experience"][str(note_raw)],
                                      str(skill_raw): manifests["skills"][str(skill_raw)]})
    rebuilt, skill_summary = bounded.materialize_skills(rows, skill_outcomes)
    flat.require(rebuilt == docs["skills"] and support.keys() <= statuses.keys(), "Released material mismatch")
    flat.require(all(len(s["source_ids"]) == 1 for s in rebuilt), "A skill has multiple sources")
    flat.require(not example_id or bool(example), "Example outside the release")
    by_skill = Counter(statuses[s["source_ids"][0]] for s in rebuilt)
    by_leaf = defaultdict(set)
    for sid, leaves in support.items():
        by_leaf[statuses[sid]].update(leaves)
    return dict(release_plan_sha256=vr.sha(release / "plan.json"), material_hashes=cfg["material_hashes"],
                first_pass_input_answers=len(inputs), first_pass_counts=bundle["source_counts"],
                published_case_review_counts=dict(Counter(statuses.values())), published_skill_counts=dict(by_skill),
                distinct_skill_sources=len({s["source_ids"][0] for s in rebuilt}),
                skill_extraction_failures=skill_summary.get("content_failure", 0),
                tree_sources_by_case_status=dict(Counter(statuses[sid] for sid in support)),
                distinct_leaves_by_case_status={s: len(leaves) for s, leaves in by_leaf.items()},
                published_leaves=len(set().union(*support.values())), raw_inputs_compared_per_stage=len(rows),
                example=example, new_model_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--example-id")
    args = parser.parse_args()
    print(json.dumps(audit(args.release, args.example_id), ensure_ascii=False, indent=2, sort_keys=True))
