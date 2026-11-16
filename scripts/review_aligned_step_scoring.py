"""Explicit, zero-API review of frozen step-evaluation evidence.

This opt-in entry never changes the legacy scorer or original experiment.
prepare creates pending review forms; freeze-labels precedes diagnosis review;
freeze-evidence precedes score. Review decisions are supplied by the reviewer,
never inferred from substring overlap or from the resulting group scores.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

ARMS = ("pure", "case_rag", "flat_skills", "rule_tree")
POLICY = {
    "version": "aligned_step_offline_review_v1",
    "human_gold": False, "independent_blind_review": False,
    "api_calls": 0, "generation_changes": False,
    "scope": "all 400 saved results, 234 diagnostics, 180 confirmed + 2 unresolved roots",
    "normalization": "case/sign/number/grouping preserving; display delimiters, whitespace and layout wrappers only",
    "manual_repairs": "verbatim continuous source spans; original diagnostic unchanged; no automatic fuzzy credit",
    "root_links": "explicit assistant-reviewed same-root and same-direction evidence, never overlap alone",
    "self_corrected": "explicit withdrawal/replacement, supported correct repair, no downstream reliance: all required",
    "ambiguity": "retain active unless all correction conditions established; original unresolved retained",
    "iou": 0.5, "coverage": 0.6,
    "assignment": "original GT order, highest overlap, diagnostic order tie break; one-to-one",
    "precision_denominator": "all locatable published diagnostics, including duplicate or self-corrected reports",
    "recall_denominator": "all confirmed roots in that profile, including failed/empty questions",
    "unknown": "unmatched is not adjudicated false; unresolved-overlap sensitivity also reported",
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def occurrences(text, quote):
    if not quote:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(quote), text)]


def normalize(text):
    """Map conservative typography normalization back to original offsets.

    Braces around mathematical groups are preserved. Only the braces belonging
    to an explicitly recognized box wrapper are removed. No case folding,
    punctuation deletion, numeric editing or equation simplification.
    """
    omit = set()
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        opening = m.end() - 1
        depth, closing = 1, None
        for i in range(opening + 1, len(text)):
            if text[i] == "{" and text[i - 1] != "\\":
                depth += 1
            elif text[i] == "}" and text[i - 1] != "\\":
                depth -= 1
                if depth == 0:
                    closing = i
                    break
        if closing is not None:
            omit.update(range(m.start(), opening + 1))
            omit.add(closing)
    pattern = r"\\(?:left|right)(?![A-Za-z])|\\(?:quad|qquad)(?![A-Za-z])|\\[,;! ]|\\[\[\]()]|\$|\*\*"
    for m in re.finditer(pattern, text):
        omit.update(range(m.start(), m.end()))
    chars, offsets = [], []
    for i, c in enumerate(text):
        if i in omit or c.isspace():
            continue
        chars.append(c)
        offsets.append(i)
    return "".join(chars), offsets


def locate_all(text, quote):
    spans = {(s, e): "exact" for s, e in occurrences(text, quote.strip())}
    src, mapping = normalize(text)
    q, _ = normalize(quote)
    if q:
        for s, e in occurrences(src, q):
            pair = mapping[s], mapping[e - 1] + 1
            # Prefer an exact wrapper-inclusive span over its normalized interior.
            if not any(a <= pair[0] and pair[1] <= b for a, b in spans):
                spans[pair] = "format_only"
    return [dict(start=s, end=e, quote=text[s:e], method=method)
            for (s, e), method in sorted(spans.items())]


def source_span(text, start, end):
    if not (0 <= start < end <= len(text)):
        raise ValueError("Invalid source span")
    return dict(start=start, end=end, quote=text[start:end])


def check_span(text, span):
    if source_span(text, span["start"], span["end"])["quote"] != span["quote"]:
        raise ValueError("Review quote no longer equals its source")


def load_sources(root):
    inputs = {r["id"]: r for r in read(root / "inputs.json")}
    reference = {r["id"]: r for r in read(root / "reference.json")}
    annotations = read(root / "reviewed_annotations.json")
    results, files = {}, []
    for sid in inputs:
        key = hashlib.sha256(sid.encode()).hexdigest()[:16]
        for arm in ARMS:
            p = root / "formal" / arm / "samples" / "C0p" / key / "result.json"
            r = read(p)
            if r["id"] != sid or (r.get("organization_arm") is not None and r["organization_arm"] != arm):
                raise ValueError("Result identity mismatch")
            results[sid, arm] = r
            files.append(p)
    return inputs, reference, annotations, results, files


def verify_sources(root):
    out = root / "offline_review"
    manifest = read(out / "protocol.json")
    if manifest["policy"] != POLICY:
        raise ValueError("Policy changed after preparation")
    for name, expected in manifest["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen source drift: {name}")
    return manifest


def prepare(root):
    out = root / "offline_review"
    out.mkdir(exist_ok=True)
    if (out / "protocol.json").exists():
        verify_sources(root)
        print("Existing review retained; nothing overwritten.")
        return
    inputs, ref, ann, results, files = load_sources(root)
    files += [root / f for f in ("inputs.json", "reference.json", "reviewed_annotations.json", "unresolved.json", "metrics.json", "acceptance.json", "plan.json")]
    labels = []
    for sid in sorted(inputs):
        for e in ann[sid]["errors"]:
            root_id = sid + "/" + e["root_id"]
            labels.append(dict(root_id=root_id, id=sid, original_status=e["status"],
                               claim=e["claim"], original_quote=e["quote"], sol_reason=e["reason"],
                               state="pending", rationale="", correction_evidence=[], downstream_evidence=[]))
    ds = []
    for (sid, arm), result in results.items():
        for index, d in enumerate(result["diagnostics"]):
            binding = dict(id=sid, arm=arm, index=index)
            ds.append((value_hash(binding), binding, d))
    ds.sort()
    evidence, operator = [], {}
    for n, (_, binding, d) in enumerate(ds, 1):
        rid = f"D{n:03}"
        sid = binding["id"]
        quote = d.get("evidence", {}).get("quote", "")
        evidence.append(dict(review_id=rid, id=sid, original_diagnostic=d,
                             automatic_spans=locate_all(inputs[sid]["prediction"], quote),
                             approved_spans=[], root_links=[], disposition="pending", rationale=""))
        operator[rid] = binding
    assert len(inputs) == 100 and len(results) == 400 and len(evidence) == 234
    assert Counter(e["original_status"] for e in labels) == {"confirmed": 180, "unresolved": 2}
    write(out / "protocol.json", dict(created_unix=time.time(), policy=POLICY,
                                     source_sha256={str(p.relative_to(root)): digest(p) for p in files}))
    write(out / "labels.json", dict(human_gold=False, records=labels))
    write(out / "evidence.json", dict(human_gold=False, group_names_hidden_in_review=True, records=evidence))
    write(out / "operator_map.json", operator)
    print("Prepared 182 label decisions and 234 pending diagnosis reviews; no score computed.")


def validate_labels(root):
    inputs, _, ann, _, _ = load_sources(root)
    data = read(root / "offline_review/labels.json")
    records = data["records"]
    expected = {sid + "/" + e["root_id"] for sid, a in ann.items() for e in a["errors"]}
    if len(records) != len(expected) or {r["root_id"] for r in records} != expected:
        raise ValueError("Label coverage mismatch")
    for r in records:
        sid = r["id"]
        e = next(e for e in ann[sid]["errors"] if sid + "/" + e["root_id"] == r["root_id"])
        if (r["claim"], r["original_quote"], r["original_status"]) != (e["claim"], e["quote"], e["status"]):
            raise ValueError("Original label edited")
        if r["state"] not in {"active_error", "self_corrected", "unresolved"} or not r["rationale"]:
            raise ValueError(f"Pending label: {r['root_id']}")
        if e["status"] == "unresolved" and r["state"] != "unresolved":
            raise ValueError("Original unresolved label was forced")
        if r["state"] == "self_corrected" and (not r["correction_evidence"] or not r["downstream_evidence"]):
            raise ValueError("Self correction lacks both correction and adoption evidence")
        for s in r["correction_evidence"] + r["downstream_evidence"]:
            check_span(inputs[sid]["prediction"], s)
    return records


def freeze_labels(root):
    verify_sources(root)
    labels = validate_labels(root)
    p = root / "offline_review/labels_frozen.json"
    value = dict(labels_sha256=digest(root / "offline_review/labels.json"),
                 states=dict(Counter(r["state"] for r in labels)), frozen_unix=time.time())
    if p.exists():
        if read(p)["labels_sha256"] != value["labels_sha256"]:
            raise ValueError("Label review already frozen")
    else:
        write(p, value)
    print(json.dumps(value, ensure_ascii=False))


def validate_evidence(root):
    out = root / "offline_review"
    if digest(out / "labels.json") != read(out / "labels_frozen.json")["labels_sha256"]:
        raise ValueError("Label decisions changed after freeze")
    inputs, _, _, results, _ = load_sources(root)
    labels = {r["root_id"]: r for r in validate_labels(root)}
    records, operator = read(out / "evidence.json")["records"], read(out / "operator_map.json")
    if len(records) != 234 or {r["review_id"] for r in records} != set(operator):
        raise ValueError("Diagnosis coverage mismatch")
    seen = set()
    for r in records:
        b = operator[r["review_id"]]
        identity = b["id"], b["arm"], b["index"]
        if identity in seen:
            raise ValueError("Duplicate diagnosis binding")
        seen.add(identity)
        d = results[b["id"], b["arm"]]["diagnostics"][b["index"]]
        if r["original_diagnostic"] != d or r["id"] != b["id"]:
            raise ValueError("Original diagnostic changed")
        if r["disposition"] not in {"same_root", "other_claim", "opposite_claim", "unclear"} or not r["rationale"]:
            raise ValueError(f"Unreviewed diagnosis: {r['review_id']}")
        if bool(r["root_links"]) != (r["disposition"] == "same_root"):
            raise ValueError("Root links inconsistent with disposition")
        t = inputs[r["id"]]["prediction"]
        for s in r["approved_spans"]:
            check_span(t, s)
        for link in r["root_links"]:
            if labels[link["root_id"]]["id"] != r["id"] or not link["rationale"] or not link["same_direction"]:
                raise ValueError("Invalid root association")
            for s in link["acceptable_source_spans"]:
                check_span(t, s)
    return records, operator, labels


def apply_review_decisions(root, decisions):
    """Save explicit reviewer decisions; this does not infer correctness or score.

    A supplied source_quote must be copied from the actual answer. Root aliases
    are accepted only with an explicit rationale affirming the same claim and
    direction. Automatic spans remain proposals until the reviewer approves.
    """
    out = root / "offline_review"
    if (out / "review_frozen.json").exists():
        raise ValueError("Review is frozen")
    if digest(out / "labels.json") != read(out / "labels_frozen.json")["labels_sha256"]:
        raise ValueError("Freeze labels first")
    data = read(out / "evidence.json")
    inputs, _, _, _, _ = load_sources(root)
    indexed = {r["review_id"]: r for r in data["records"]}
    for rid, decision in decisions.items():
        row = indexed[rid]
        if row["disposition"] != "pending":
            raise ValueError(f"Already reviewed: {rid}")
        text = inputs[row["id"]]["prediction"]
        if "source_quote" in decision:
            spans = [dict(**source_span(text, s, e), method="assistant_source_repair")
                     for s, e in occurrences(text, decision["source_quote"])]
            if not spans:
                raise ValueError(f"Nonverbatim source repair: {rid}")
        elif decision.get("unlocatable"):
            spans = []
        else:
            spans = row["automatic_spans"]
        roots = decision.get("roots", [])
        if roots and not spans:
            raise ValueError(f"Root link lacks approved source: {rid}")
        row.update(approved_spans=spans, disposition=decision.get("disposition", "same_root" if roots else "other_claim"),
                   rationale=decision["rationale"], root_links=[
                       dict(root_id=row["id"] + "/" + r, same_direction=True,
                            acceptable_source_spans=spans, rationale=decision["rationale"])
                       for r in roots])
    write(out / "evidence.json", data)


def freeze_evidence(root):
    verify_sources(root)
    validate_evidence(root)
    out = root / "offline_review"
    hashes = {name: digest(out / name) for name in ("labels.json", "evidence.json", "operator_map.json", "protocol.json", "labels_frozen.json")}
    p = out / "review_frozen.json"
    if p.exists():
        if read(p)["sha256"] != hashes:
            raise ValueError("Review already frozen")
    else:
        write(p, dict(sha256=hashes, frozen_unix=time.time(), policy=POLICY))
    print("Review frozen; no score computed.")


def overlap(a, b):
    n = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    la, lb = a["end"] - a["start"], b["end"] - b["start"]
    iou, coverage = n / (la + lb - n), max(n / la, n / lb)
    return iou + .01 * coverage if iou >= .5 or coverage >= .6 else -1.0


def assign(gt, diagnoses, states, active_only):
    used, matches = set(), []
    for g in gt:
        gid = g["error_id"]
        if active_only and states[gid]["state"] != "active_error":
            continue
        original = dict(start=g["start_char"], end=g["end_char"])
        candidates = []
        for i, d in enumerate(diagnoses):
            if i in used:
                continue
            link = next((l for l in d["root_links"] if l["root_id"] == gid), None)
            if link is None:
                continue
            for p in d["approved_spans"]:
                for alias in [original] + link["acceptable_source_spans"]:
                    score = overlap(alias, p)
                    if score >= 0:
                        candidates.append((score, -i, p, alias))
        if candidates:
            score, ni, p, alias = max(candidates, key=lambda v: (v[0], v[1]))
            i = -ni
            used.add(i)
            matches.append(dict(root_id=gid, review_id=diagnoses[i]["review_id"],
                                prediction_span=p, reference_span=alias, overlap_score=score))
    return matches


def rates(tp, pred, gt):
    p, r = tp / pred if pred else 0., tp / gt if gt else 0.
    return dict(precision=p, recall=r, f1=2 * tp / (pred + gt) if pred + gt else 0.)


def verify_legacy(reference, results, original):
    # Only import the standard-library location helpers, not any generation client.
    from scripts import evaluate_physics_eval_sets as legacy
    verified = 0
    for arm in ARMS:
        old = {d["id"]: d for d in original["scores"][arm]["details"]}
        for sid, row in reference.items():
            gt = legacy._extract_gt_entries(row, sid)
            findings = legacy._fill_missing_pred_locations(
                legacy._collect_pred_findings(results[sid, arm], {}), row["prediction"])
            matched, _, used = legacy._match_by_location(
                [g for g in gt if g["locatable_valid"]], findings, .5, .6, preserve_zero_offsets=False)
            observed = dict(gt_error_count=len(gt), pred_finding_count=len(findings),
                            pred_locatable_finding_count=sum(bool(f["locatable_valid"]) for f in findings),
                            matched_claim_ids=sorted(matched), matched_finding_indices=sorted(used))
            for key, value in observed.items():
                if old[sid][key] != value:
                    raise ValueError(f"Legacy parity failed: {arm}/{sid}/{key}")
            verified += 1
    return dict(passed=True, result_count=verified,
                legacy_helper_sha256=digest(legacy.__file__))


def unresolved_overlaps(diagnoses, matches, unresolved):
    used = {m["review_id"] for m in matches}
    unknown = []
    for d in diagnoses:
        if d["review_id"] in used:
            continue
        spans = [dict(start=g["start_char"], end=g["end_char"])
                 for g in unresolved if g.get("span_valid")]
        if any(overlap(g, p) >= 0 for g in spans for p in d["approved_spans"]):
            unknown.append(d["review_id"])
    return unknown


def score(root):
    verify_sources(root)
    out = root / "offline_review"
    freeze = read(out / "review_frozen.json")
    for name, expected in freeze["sha256"].items():
        if digest(out / name) != expected:
            raise ValueError("Frozen review changed")
    evidence, operator, labels = validate_evidence(root)
    inputs, reference, _, results, _ = load_sources(root)
    original = read(root / "metrics.json")
    parity = verify_legacy(reference, results, original)
    unknown_reference = read(root / "unresolved.json")
    indexed = {(b["id"], b["arm"], b["index"]): next(r for r in evidence if r["review_id"] == rid)
               for rid, b in operator.items()}
    profiles = {}
    for profile, active_only in (("evidence_repaired_180gt", False), ("evidence_and_self_correction", True)):
        profiles[profile] = {}
        for arm in ARMS:
            details = []
            for sid in inputs:
                result = results[sid, arm]
                ds = [indexed[sid, arm, i] for i in range(len(result["diagnostics"]))]
                gt = reference[sid]["physics_error_gt"]
                matches = assign(gt, ds, labels, active_only)
                unknown = unresolved_overlaps(ds, matches, unknown_reference[sid])
                details.append(dict(id=sid, published=len(ds), locatable=sum(bool(d["approved_spans"]) for d in ds),
                                    gt=sum(not active_only or labels[g["error_id"]]["state"] == "active_error" for g in gt),
                                    hits=len(matches), matches=matches, unresolved_overlap=unknown,
                                    failed=result["status"] != "completed", source_status=result["status"]))
            sums = {k: sum(d[k] for d in details) for k in ("published", "locatable", "gt", "hits", "failed")}
            unknown_n = sum(len(d["unresolved_overlap"]) for d in details)
            profiles[profile][arm] = dict(summary={**sums, **rates(sums["hits"], sums["locatable"], sums["gt"]),
                "unlocatable": sums["published"] - sums["locatable"], "unresolved_overlap": unknown_n,
                "all_published_sensitivity": rates(sums["hits"], sums["published"], sums["gt"]),
                "unresolved_excluded_sensitivity": rates(sums["hits"], sums["locatable"] - unknown_n, sums["gt"])}, details=details)
    differences = {}
    for arm in ARMS:
        old = {d["id"]: d for d in original["scores"][arm]["details"]}
        rows = []
        for p in profiles["evidence_repaired_180gt"][arm]["details"]:
            previous = old[p["id"]]
            before = set(previous["matched_claim_ids"])
            after = {m["root_id"] for m in p["matches"]}
            if before != after or previous["pred_locatable_finding_count"] != p["locatable"]:
                rows.append(dict(id=p["id"], old_hits=sorted(before), repaired_hits=sorted(after),
                                 added=sorted(after - before), removed=sorted(before - after),
                                 old_locatable=previous["pred_locatable_finding_count"], repaired_locatable=p["locatable"]))
        differences[arm] = rows
    report = dict(policy=POLICY, source_manifest_sha256=digest(out / "protocol.json"),
                  review_sha256=digest(out / "review_frozen.json"), scorer_sha256=digest(__file__),
                  legacy_parity=parity,
                  label_states=dict(Counter(r["state"] for r in labels.values())),
                  original={a: original["scores"][a]["summary"] for a in ARMS}, profiles=profiles, differences=differences)
    verify_sources(root)
    if (out / "metrics.json").exists():
        if read(out / "metrics.json") != report:
            raise ValueError("Existing reviewed metrics differ; refusing overwrite")
    else:
        write(out / "metrics.json", report)
    print(json.dumps({"label_states": report["label_states"], **{p: {a: x["summary"] for a, x in v.items()} for p, v in profiles.items()}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "freeze-labels", "freeze-evidence", "score"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--await-release", action="store_true",
                        help="Server-only reconnect handshake before scoring; timeout after 60 seconds.")
    args = parser.parse_args()
    out = args.root / "offline_review"
    out.mkdir(exist_ok=True)
    with (out / "review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.await_release:
            if args.command != "score":
                raise ValueError("Reconnect handshake is only for score")
            write(out / "runner_status.json", dict(state="awaiting_reconnect", pid=os.getpid(), started_unix=time.time()))
            print("Detached scorer ready; waiting for reconnect verification.", flush=True)
            deadline = time.monotonic() + 60
            while not (out / "ALLOW_SCORE").exists():
                if time.monotonic() >= deadline:
                    write(out / "runner_status.json", dict(state="handshake_timeout", exit_code=1))
                    raise TimeoutError("No reconnect verification within 60 seconds")
                time.sleep(.5)
            (out / "ALLOW_SCORE").unlink()
        try:
            {"prepare": prepare, "freeze-labels": freeze_labels, "freeze-evidence": freeze_evidence, "score": score}[args.command](args.root)
        except Exception:
            if args.command == "score":
                write(out / "runner_status.json", dict(state="failed", exit_code=1, finished_unix=time.time()))
            raise
        if args.command == "score":
            write(out / "runner_status.json", dict(state="completed", exit_code=0, finished_unix=time.time(),
                metrics_sha256=digest(out / "metrics.json"), sources_unchanged=True, api_calls=0))


if __name__ == "__main__":
    main()
