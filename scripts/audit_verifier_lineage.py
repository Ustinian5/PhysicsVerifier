"""Read-only evidence replay for the combined-source 30B verifier release."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

from rule_framework import verified_resume as vr
from scripts import audit_aligned_full_chain as original
from scripts import review_aligned_step_scoring as scoring
from scripts import run_aligned_step_revision as step
from scripts import run_bench_reannotation as teacher
from scripts import run_corrected_step_generation as generation
from scripts import run_final_answer_revision as final

REPO = Path(__file__).resolve().parents[1]
COMBINED_SHA256 = "c1c2401fb8dd8a730dd3777f5d7103bcd4c1a2750f42b5b58911ec2b49da7b24"
SOLVER = "Qwen3-30B-A3B-Instruct-2507"
SOLVER_ROOT = "/slow_share2/huhengyuan/models/" + SOLVER


def check(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, key="id"):
    result = {row[key]: row for row in rows}
    check(len(result) == len(rows), f"Duplicate {key}")
    return result


def source_receipt(root):
    source = REPO / "data/combined_language_only.json"
    inventory = root / "source_question_inventory.json"
    screen = root / "corrected/source_screen"
    references = screen / "references_private.json"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "offline_review").mkdir()
        original.source_audit(work, source, inventory, references)
        raw = vr.read(work / "offline_review/full_chain_audit.json")["raw_source"]
    all_stems = set(raw.pop("question_keys"))
    check(raw["source_sha256"] == COMBINED_SHA256, "Wrong approved combined source")
    by_stem = {original.key(r["question"]): r["id"] for r in vr.read(inventory)["questions"]}
    split = vr.read(screen / "split.json")
    bench_path = root / "corrected/bench_generation/questions.json"
    candidate_path = root / "aligned_old99/final_revision/candidates.json"
    groups = dict(build=split["build"], reserve=split["reserve"], step_bench=vr.read(bench_path))
    check({name: len(rows) for name, rows in groups.items()} ==
          dict(build=2000, reserve=702, step_bench=100), "Question cohort size changed")
    seen = set()
    for name, rows in groups.items():
        stems = {original.key(r["question"]) for r in rows}
        check(len(stems) == len(rows) and stems <= all_stems and not seen.intersection(stems),
              f"Duplicate, shared or out-of-source stem in {name}")
        seen.update(stems)
        if name in ("build", "reserve"):
            check(all(by_stem[original.key(r["question"])] == r["id"] for r in rows),
                  f"Wrong original question ID in {name}")
    candidates = vr.read(candidate_path)
    check(split["reserve"] == [dict(id=r["id"], question=r["question"]) for r in candidates],
          "Final candidates differ from the original reserve or its order")
    ref_index = indexed(vr.read(references)["references"])
    bindings = indexed(vr.read(root / "corrected/final_labels/references_private.json"))
    for row in candidates:
        sid = row["id"]
        ref = bindings[sid]
        variants = ref_index[sid]["variants"]
        check(variants == ref["variants"] and
              row["reference"] == ref["reference"] == variants[ref["selected_variant"]]["value"],
              f"Final reference differs from original for {sid}")
    used = [inventory, references, screen / "split.json", bench_path, candidate_path]
    return dict(raw=raw, groups={name: len(rows) for name, rows in groups.items()},
                final_original_references=len(candidates), shared_normalized_stems=0,
                files={str(path.relative_to(root)): vr.sha(path) for path in used})


def solver_output(row, config, saved, raw_path):
    sid = row["id"]
    check(config["generator"] == SOLVER and saved["id"] == sid, f"Wrong solver or answer ID for {sid}")
    check(vr.sha(raw_path) == saved["evidence"]["sha256"], f"Answer evidence changed for {sid}")
    raw = vr.read(raw_path)
    check(raw["status"] == "received" and raw["request"] == generation.request_body(row, config),
          f"Solver request differs for {sid}")
    response = raw["response"]
    check(response["model"] == saved["model"] == SOLVER and len(response["choices"]) == 1 and
          response["usage"] == saved["usage"], f"Wrong response model, choices or usage for {sid}")
    choice = response["choices"][0]
    check(choice["message"]["content"] == saved["prediction"] and
          choice["finish_reason"] == saved["finish_reason"], f"Export differs from raw answer for {sid}")
    if saved["state"] == "complete":
        generation.validate_import(row, config, saved, raw)
    else:
        check(saved["state"] == "incomplete" and saved["finish_reason"] == "length",
              f"Unsupported generation failure for {sid}")


def generation_replay(root, candidates):
    corrected = root / "corrected"
    folder = corrected / "collection"
    config = vr.read(folder / "plan.json")["configuration"]["generator_config"]
    split = vr.read(corrected / "source_screen/split.json")
    tasks = vr.read(folder / "tasks.json")
    build = {}
    counts = Counter()
    for role, pool in [("build", "build"), ("heldout", "reserve")]:
        rows = [t["row"] for t in tasks if t["role"] == role]
        check(rows == split[pool], f"Task order or contents differ for {role}")
        for row in rows:
            sid = row["id"]
            saved = vr.read(folder / "generated" / (sid + ".json"))
            solver_output(row, config, saved, folder / saved["evidence"]["file"])
            counts[f'{role}/{saved["state"]}'] += 1
            if role == "build" and saved["state"] == "complete":
                build[sid] = dict(row, context="", prediction=saved["prediction"])
            if role == "heldout":
                candidate = candidates[sid]
                check(candidate["prediction"] == saved["prediction"] and candidate["generation_status"] == saved["state"],
                      f"702-pool answer differs from raw response for {sid}")
    check(len(tasks) == sum(counts.values()) == 2702, "Unexpected solver task")
    folder = corrected / "bench_generation"
    plan = vr.read(folder / "plan.json")["configuration"]
    check(plan["configuration"] == config, "Step and final solver configs differ")
    prior = indexed(plan["imports"])
    inputs = indexed(vr.read(folder / "inputs.json"))
    for row in vr.read(folder / "questions.json"):
        sid = row["id"]
        if sid in prior:
            path = folder / prior[sid]["record"]
            raw_path = folder / prior[sid]["evidence"]
            check(vr.sha(path) == prior[sid]["sha256"] and vr.sha(raw_path) == prior[sid]["evidence_sha256"],
                  f"Reused answer evidence changed for {sid}")
            saved = vr.read(path)
        else:
            saved = vr.read(folder / "generated" / (sid + ".json"))
            raw_path = folder / saved["evidence"]["file"]
        solver_output(row, config, saved, raw_path)
        check(saved["state"] == "complete" and inputs[sid] == dict(row, context="", prediction=saved["prediction"]),
              f"Step bench differs from raw response for {sid}")
    check(len(inputs) == 100 and len(prior) == 32, "Wrong step answer inventory")
    admission = vr.read(root / "service/admission.json")
    checkpoint = Path(admission["prior_full_weight_sha_manifest"])
    check(vr.sha(checkpoint) == admission["prior_manifest_sha256"], "Checkpoint manifest changed")
    weights = vr.read(checkpoint)
    check(weights["destination"] == SOLVER_ROOT, "Checkpoint belongs to another solver")
    transition = vr.read(corrected / "collection/concurrency_transition.json")
    for command in ("old_cmd", "new_cmd"):
        args = shlex.split(transition[command])
        check(args[args.index("--model")+1] == SOLVER_ROOT and
              args[args.index("--served-model-name")+1] == SOLVER, "Wrong deployed solver")
    return inputs, build, dict(model=SOLVER, model_path=SOLVER_ROOT, checkpoint_manifest_sha256=vr.sha(checkpoint),
                               collection=dict(counts), step_answers=len(inputs), step_reused=len(prior))


def materials_replay(root, build):
    aligned = root / "aligned_old99"
    release = aligned / "evaluation_revision"
    cases = vr.read(release / "cases.json")
    skills = vr.read(release / "skills.json")
    retained = {sid for doc in cases for sid in doc["source_ids"]}
    source_rows = [r for r in vr.read(aligned / "evaluation/sources.json") if r["id"] in retained]
    check(len(source_rows) == len(retained) == len(cases) == 739 and
          all(build.get(r["id"]) == r for r in source_rows), "RAG source or 30B text differs from raw generation")
    parent_hash = vr.read(aligned / "plan.json")["configuration_sha256"]
    case_outcomes, skill_outcomes = [], []
    case_by_source = {doc["source_ids"][0]: doc for doc in cases}
    for row in source_rows:
        for kind, outcomes in [("experience", case_outcomes), ("skills", skill_outcomes)]:
            saved = vr.read(final.base.old.outcome_path(aligned / kind, row))
            check(saved["source_sha256"] == vr.digest(row) and saved["configuration_sha256"] == parent_hash,
                  f"Wrong {kind} source binding for {row['id']}")
            if kind == "skills":
                check(saved["case_sha256"] == vr.digest(case_by_source[row["id"]]), "Skill refers to different case")
            outcomes.append(dict(saved["outcome"], configuration_sha256=parent_hash, source_sha256=vr.digest(row))
                            if kind == "experience" else saved["outcome"])
    replay_cases = final.base.p.materialize(source_rows, case_outcomes, parent_hash)["cases"]
    replay_skills, _ = final.a.bounded.materialize_skills(source_rows, skill_outcomes)
    check(indexed(cases) == indexed(replay_cases) and indexed(skills) == indexed(replay_skills),
          "Published RAG or skills differ from per-source outcomes")
    bundle = vr.read(root / "corrected/pipeline/tree_bundle.json")
    catalog = vr.read(release / "catalog.json")
    leaves = [r for d in catalog["domains"] for t in d["topics"] for r in t["rules"]]
    for leaf in leaves:
        support = {s["sample_id"] for s in bundle["provenance"][leaf["rule_id"]]}
        check(support and support <= retained, f"Out-of-pool rule support for {leaf['rule_id']}")
    for name in ("cases", "skills", "catalog"):
        check(vr.sha(release / (name + ".json")) == vr.sha(aligned / "final_revision" / (name + ".json")),
              f"New final benchmark uses different {name}")
    check(len(skills) == 1471 and len(indexed(leaves, "rule_id")) == 895, "Published materials inventory drift")
    return dict(sources=len(retained), cases=len(cases), skills=len(skills), leaves=len(leaves))


def labels_replay(root, answers):
    annotations = root / "corrected/annotations"
    parent = root / "aligned_old99/evaluation"
    screen = root / "corrected/source_screen"
    rows = vr.read(parent / "inputs.json")
    check(indexed(rows) == answers, "Formal step inputs differ from solver exports")
    originals = indexed(vr.read(screen / "references_private.json")["references"])
    inventory = {original.key(r["question"]): r["id"] for r in vr.read(root / "source_question_inventory.json")["questions"]}
    refs = indexed(vr.read(annotations / "reference_bindings.json"))
    annotated = vr.read(annotations / "annotation_inputs.json")
    check(set(indexed(annotated)) == set(answers), "Annotation inventory differs")
    finals, hashes = {}, {}
    for row in annotated:
        sid = row["id"]
        binding = refs[sid]
        check(answers[sid] == {k: row[k] for k in answers[sid]} and
              binding["source_id"] == inventory[original.key(row["question"])], f"Wrong labeled answer for {sid}")
        variants = originals[binding["source_id"]]["variants"]
        check(binding["variants"] == variants and json.loads(row["answer"]) == variants[binding["selected_variant"]]["value"],
              f"Wrong reference answer for {sid}")
        for stage in teacher.STAGES:
            previous = teacher.previous_annotations(annotations, stage, sid)
            saved = teacher.validate_saved(annotations, stage, row, previous)
            if stage == "final":
                finals[sid] = saved["annotation"]
                hashes[sid] = vr.sha(annotations / "arm_final" / sid / "result.json")
    reviewed = step.old.apply_quote_review(rows, finals, vr.read(parent / "label_review.json"), hashes)
    dataset, unknown, counts = step.old.reference_rows(rows, reviewed)
    check(dataset == vr.read(parent / "reference.json") and unknown == vr.read(parent / "unresolved.json") and
          reviewed == vr.read(parent / "reviewed_annotations.json"), "Step label release differs from raw annotations")
    for row in dataset:
        for label in row["physics_error_gt"]:
            span = dict(start=label["start_char"], end=label["end_char"], quote=label["answer_quote"])
            scoring.check_span(answers[row["id"]]["prediction"], span)
    return dict(model=teacher.MODEL, rounds={s: len(finals) for s in teacher.STAGES}, **counts)


def publication_replay(root, answers):
    parent = root / "aligned_old99/evaluation"
    release = root / "aligned_old99/evaluation_revision"
    out = release / "offline_review"
    metrics = vr.read(out / "metrics.json")
    labels = indexed(vr.read(parent / "offline_review/labels.json")["records"], "root_id")
    references = indexed(vr.read(parent / "reference.json"))
    diagnostics = indexed(vr.read(out / "evidence.json")["records"], "review_id")
    operators = vr.read(out / "operator_map.json")
    bindings = {(b["id"], b["arm"], b["index"]): rid for rid, b in operators.items()}
    check(len(bindings) == len(operators) and set(operators) == set(diagnostics), "Duplicate or missing diagnosis binding")
    observed = set()
    groups = {}
    tasks = vr.read(release / "tasks.json")
    check(len(tasks) == 400 and len({(t["id"], t["arm"]) for t in tasks}) == 400, "Wrong four-group task inventory")
    for arm, group in metrics["groups"].items():
        details = indexed(group["details"])
        check(set(details) == set(answers), f"Score cohort mismatch for {arm}")
        totals = Counter()
        for task in [t for t in tasks if t["arm"] == arm]:
            sid = task["id"]
            check(task["input_sha256"] == vr.digest(answers[sid]), f"Wrong checked answer for {sid}")
            result, path = step.result_for(release, task)
            check(result["id"] == sid, f"Wrong published question for {sid}")
            ds = []
            for i, item in enumerate(result["diagnostics"]):
                rid = bindings[sid, arm, i]
                d = diagnostics[rid]
                check(d["id"] == sid and d["original_diagnostic"] == item and
                      Path(operators[rid]["result_path"]) == path, f"Wrong publication binding for {rid}")
                for span in d["approved_spans"]:
                    scoring.check_span(answers[sid]["prediction"], span)
                for link in d["root_links"]:
                    check(labels[link["root_id"]]["id"] == sid and link["same_direction"] and link["rationale"],
                          f"Wrong root or diagnosis direction for {rid}")
                    for span in link["acceptable_source_spans"]:
                        scoring.check_span(answers[sid]["prediction"], span)
                ds.append(d)
                observed.add(rid)
            gt = references[sid]["physics_error_gt"]
            matches = scoring.assign(gt, ds, labels, True)
            counts = dict(hits=len(matches), published=len(ds), locatable=sum(bool(d["approved_spans"]) for d in ds),
                          gt=sum(labels[g["error_id"]]["state"] == "active_error" for g in gt),
                          failed=result["status"] != "completed")
            check(matches == details[sid]["matches"] and all(details[sid][k] == v for k, v in counts.items()),
                  f"Step result differs from frozen scoring for {sid}/{arm}")
            totals.update(counts)
        summary = group["summary"]
        check(all(summary[k] == v for k, v in totals.items()) and
              all(summary[k] == v for k, v in scoring.rates(totals["hits"], totals["locatable"], totals["gt"]).items()),
              f"Published step metrics differ for {arm}")
        groups[arm] = dict(totals)
    check(observed == set(diagnostics) and set(groups) == set(final.ARMS), "Uncovered diagnosis or arm")
    return dict(diagnoses=len(observed), groups=groups, label_states=dict(Counter(d["state"] for d in labels.values())))


def evidence_receipt(root, source):
    corrected = root / "corrected"
    aligned = root / "aligned_old99"
    plan = vr.read(aligned / "final_revision/plan.json")
    config = plan["configuration"]
    final.execution_plan(aligned / "final_revision")
    check(source["raw"]["source_sha256"] == COMBINED_SHA256 and vr.digest(config) == plan["configuration_sha256"],
          "Approved source, release configuration or code changed")
    for rel, expected in source["files"].items():
        check(vr.sha(root / rel) == expected, f"Source receipt differs on server for {rel}")
    candidates = indexed(vr.read(aligned / "final_revision/candidates.json"))
    answers, build, solver = generation_replay(root, candidates)
    audit = dict(solver=solver, materials=materials_replay(root, build),
                 annotations=labels_replay(root, answers), step_publication=publication_replay(root, answers))
    archive = corrected / "pipeline/check_prompt_revision_bilingual"
    old = vr.read(archive / "final_metric_audit.json")
    rows = vr.read(archive / "final_inputs.json")
    check(len(indexed(rows)) == 600 and all(final.p.public(candidates[r["id"]]) == final.p.public(r) for r in rows),
          "Archived 600 use different answers")
    acceptance = vr.read(aligned / "evaluation_revision/acceptance.json")
    check(acceptance["passed"] and old["original_formula_parity"], "Missing prior release acceptance")
    manifests = [acceptance["files"], old["files"], config["dependencies"]]
    files = {}
    for manifest in manifests:
        for name, digest in manifest.items():
            check(files.setdefault(name, digest) == digest, f"Conflicting expected hashes for {name}")
    def verify_file(entry):
        path, expected = entry
        check(vr.sha(path) == expected, f"Frozen artifact changed: {path}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify_file, files.items()))
    return dict(audit, source_receipt_sha256=vr.digest(source), configuration_sha256=plan["configuration_sha256"],
                audit_code_sha256=vr.sha(__file__), distinct_frozen_files_checked=len(files),
                archived_600_answers=len(rows), model_calls=0, human_gold=False,
                scope="Engineering provenance and frozen scoring; physical truth and training benefit require separate evidence.")


def repair_receipt(root, scratch):
    """Replay request construction with fixed queries and no network or model weights."""
    from transformers import AutoTokenizer
    out = root / 'aligned_old99/final_revision'
    parent = vr.read(out / 'plan.json')
    plan = final.verify(out)
    c = plan['configuration']
    check({**c, 'source_hashes':parent['configuration']['source_hashes']} == parent['configuration'],
          'An experiment setting changed in the repair')
    old = ModuleType('prepared_final_runner')
    source = subprocess.check_output(['git','cat-file','blob','010247c4691e231a7a67de3146b145f70c046d30'], cwd=REPO)
    check(hashlib.sha256(source).hexdigest() == final.PREPARED_RUNNER, 'Wrong frozen runner blob')
    exec(compile(source, '<frozen-final-runner>', 'exec'), old.__dict__)
    rows = vr.read(out/'pilot_inputs.json')
    memories = final.stores(out)
    tokenizer = AutoTokenizer.from_pretrained(c['model'], local_files_only=True, trust_remote_code=False)
    records = {}
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline repair audit attempted network')):
        for row in rows:
            prefix, path = step.prefix_at(Path(c['material_parent']), row)
            for name, digest in prefix['files'].items():
                check(vr.sha(name) == digest, 'Fixed development query evidence changed')
            original_usage = {k:prefix['evidence']['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')}
            captures = []
            for runner, cfg in ((old,parent),(final,plan)):
                requests = {}
                def capture(output, folder, messages, max_tokens, validator, current, tok, client, **kwargs):
                    key = str(folder.relative_to(output))
                    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='medium')
                    requests[key] = dict(messages_sha256=vr.digest(messages), template_ids_sha256=vr.digest(ids),
                                         max_tokens=max_tokens, model=current['configuration']['model'])
                    query = validator is final.base.p.validate_queries
                    usage = original_usage if query else dict(prompt_tokens=len(ids), completion_tokens=1, total_tokens=len(ids)+1)
                    response = dict(status='complete', reason=None, value=prefix['queries'] if query else
                                    dict(verdict='uncertain',student_final_answer='',reason='Offline fixture'),
                                    usage=usage, files={}, wall_seconds=1)
                    final.immutable(folder/'result.json', response)
                    return response
                with tempfile.TemporaryDirectory(dir=scratch) as tmp, patch.object(runner,'request_once',capture):
                    work = Path(tmp)
                    runner.run_question(work,row,'development',cfg,tokenizer,None,memories)
                    packets = {f.parent.parent.name:vr.sha(f) for f in work.glob('development/*/*/reference_packet.json')}
                check(len(requests)==5 and set(packets)==set(final.ARMS), 'Incomplete six-input parity replay')
                captures.append(dict(requests=requests,packets=packets))
            check(captures[0] == captures[1], 'Model-visible request or retrieval packet changed for ' + row['id'])
            records[row['id']] = dict(captures[1], query_fixture_sha256=vr.sha(path))
    check(len(records)==6, 'Wrong development set')
    # Inspect a fresh interpreter; importing this audit or pytest loads additional modules.
    inventory = """from pathlib import Path
import json, sys
from scripts import run_final_answer_revision
root = Path.cwd().resolve()
paths = [Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)]
print(json.dumps(sorted({str(p.relative_to(root)) for p in paths if p.suffix=='.py' and p.is_relative_to(root)})))
"""
    loaded = json.loads(subprocess.check_output([sys.executable,'-B','-c',inventory], cwd=REPO))
    extra = {name:vr.sha(REPO/name) for name in loaded if name not in c['source_hashes']}
    return dict(parent_configuration_sha256=parent['configuration_sha256'],effective_configuration_sha256=plan['configuration_sha256'],
                repair_sha256=vr.sha(out/'execution_repair.json'), frozen_inputs=len(records), prepare_inputs=6, check_inputs=24,
                records=records, model_calls=0, online_pilot=False, startup_modules=len(loaded),
                configuration_source_files=len(c['source_hashes']), additional_startup_hashes=extra,
                scope='Exact messages, locally rendered tokens, output caps and full BM25 packets; historical development queries fixed for this comparison.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "evidence", "repair"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if args.stage == "repair":
        report = repair_receipt(root, out)
    elif args.stage == "source":
        report = source_receipt(root)
    else:
        report = evidence_receipt(root, vr.read(out / "source.json"))
    path = out / (args.stage + ".json")
    if args.stage == "evidence" and path.exists():
        saved = vr.read(path)
        # Original evidence was issued by the pre-amendment audit in Git 20c048b.
        check(saved["audit_code_sha256"] in {report["audit_code_sha256"],
              "299d34e1a5bd2d5d3bdd70ba742b65f32725bdf4f16d2edd8146039fc2506bc5"} and
              {**saved, "audit_code_sha256": report["audit_code_sha256"]} == report,
              "Replayed release differs from the frozen evidence receipt")
    else:
        final.immutable(path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
