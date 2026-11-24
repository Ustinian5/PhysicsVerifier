"""Publish the November 7 first-review cases, skills and rules from server evidence, offline."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import tempfile

from rule_framework import organization_followup as bounded
from rule_framework import shared_experience as shared
from rule_framework import verified_resume as vr
from scripts import audit_verifier_lineage as lineage
from scripts import run_verifier_library_scale as scale
from scripts.audit_library_review_alignment import checked
from scripts.audit_verifier_library_scale import RecordedResponses

build = scale.construction
extract = build.extraction
require = scale.require
DATE = "2026-11-07"
EXTRA_HOLD = {
    "19ca0c4a6238c9bd0a9766214dc3054f88299d5097a5b50e12e4bc72c18f00f8",
    "4ed1ff9c906aec51d77376697c7db8c598c1633bb24af3220f131ff09842f121",
    "785e6ff9413cd6cd11794fafb316cf40894a27c2dc43c6a5d47df3c3c007d6b0",
    "8457173ce1cf0bc777f93161c8cdc76c631d1050e707a693afedf12ffca9c14c",
    "e6c4156a588f05d56cebca94b4569c48ad49041fe67559378da1e3510028c836",
    "ebe02fb4ba8c76e698549b3f538d342c3dc512cb3d07c1b4cd97c05ca23d96b0",
    "ff1f1ff84264936ad45c4faf3f3b51ee89245a9ebd49061c491afc4f11a616aa",
}
HELD = frozenset(scale.SOURCE_HOLD | EXTRA_HOLD)
TEACUP = "002b71feab653cc1aa3d195264cf23ba7b7928f462e10f1484032e39f9872acb"


def config(plan):
    require(vr.digest(plan["configuration"]) == plan["configuration_sha256"], "Configuration drift")
    return plan["configuration"]


def bound(path, manifest):
    return checked(path, manifest[str(path)])


def source_case(case, sample, audit):
    sid = sample["id"]
    prefix = ("历史题目\n" + sample["question"] + "\n\n原30B回答（可能有错，不是正确范例）\n"
              + sample["prediction"] + "\n\n120B原始核查说明\n")
    require(not sample["answer"] and not sample["context"] and case["source_ids"] == [sid]
            and case["id"] == "case_" + sid and case["text"].startswith(prefix)
            and json.loads(case["text"][len(prefix):]) == audit
            and audit["status"] == "error_found", f"RAG did not use the first review for {sid}")


def inputs(root):
    final = root / "aligned_old99/final_revision_balanced"
    receipt = vr.read(root / "library_review_alignment.json")
    final_cfg = config(checked(final / "plan.json", receipt["release_plan_sha256"]))
    deps = final_cfg["dependencies"]
    step = Path(final_cfg["material_parent"])
    revision = Path(final_cfg["archive"])
    pipe = revision.parent
    ext = pipe.parent / "construction"
    step_cfg = config(bound(step / "plan.json", deps))
    previous = config(bound(revision / "plan.json", step_cfg["dependencies"]))["parent_bindings"]
    build_cfg = config(bound(pipe / "plan.json", previous))
    accepted = bound(pipe / "construction_acceptance.json", previous)
    require(accepted["passed"] and build_cfg["transport"]["fixed_date"] == DATE, "Wrong construction batch")
    original = {k: checked(pipe / "materials" / (k + ".json"), accepted["files"]["materials/" + k + ".json"])
                for k in ("cases", "skills")}
    evidence = vr.read(pipe / "source_evidence.json")
    require(vr.digest(evidence) == build_cfg["evidence_digest"], "First-review manifest drift")
    first = config(bound(ext / "plan.json", evidence))
    require(first["fixed_date"] == DATE and first["model"] == build_cfg["transport"]["model"] == final_cfg["model"]
            and first["reasoning_effort"] == build_cfg["transport"]["reasoning_effort"] == "medium",
            "Construction models differ")
    split = bound(pipe.parent / "source_screen/split.json", deps)
    families = bound(pipe.parent / "source_screen/family_components.json", deps)["components"]
    prior_isolation = bound(step.parent / "evaluation/isolation.json", deps)
    prior_excluded = set(prior_isolation["excluded_source_ids"]) | set(bound(step / "source_quarantine.json", deps)["excluded"])
    previous_cases = checked(final / "cases.json", final_cfg["material_hashes"]["cases"])
    cases = {c["source_ids"][0]: c for c in original["cases"]}
    previous_ids = {c["source_ids"][0] for c in previous_cases}
    require(len(cases) == len(original["cases"]) == accepted["common_error_sources"]
            and set(cases) - prior_excluded == previous_ids and HELD <= previous_ids,
            "Published source cohort or quarantine changed")
    eligible = previous_ids - HELD
    samples = lineage.indexed(bound(ext / "inputs.json", evidence))
    draw = lineage.indexed(split["build"])
    require(all(samples[s]["question"] == draw[s]["question"] for s in eligible), "Build question changed")
    questions = bound(step.parent / "evaluation/bench_questions.json", deps)
    heldout = questions + [dict(id="final:" + r["id"], question=r["question"]) for r in split["reserve"]]
    solver = Path(first["source"])
    gen = config(checked(solver / "plan.json", first["source_plan_sha256"]))["generator_config"]
    return dict(pipe=pipe, ext=ext, build=build_cfg, first=first, solver=solver, generator=gen,
                evidence=evidence, samples=samples,
                taxonomy=bound(ext / "taxonomy.json", evidence), cases=cases, first_review_skills=original["skills"],
                eligible=eligible, excluded=prior_excluded, split=split, families=families, heldout=heldout,
                catalog=checked(final / "catalog.json", final_cfg["material_hashes"]["catalog"]),
                bundle=checked(pipe / "tree_bundle.json", build_cfg["frozen_files"]["tree_bundle.json"]),
                original_release_sha256=vr.sha(pipe / "construction_acceptance.json"))


def replay_source(origin, sid, tokenizer, scratch):
    require(sid in origin["eligible"] and sid not in HELD, "Source outside active pool")
    sample = origin["samples"][sid]
    evidence = origin["evidence"]
    row = dict(id=sid, question=sample["question"])
    solver = origin["solver"]
    answer = bound(solver / "generated" / (sid + ".json"), evidence)
    lineage.solver_output(row, origin["generator"], answer, solver / answer["evidence"]["file"])
    require(answer["state"] == "complete" and answer["prediction"] == sample["prediction"], "Wrong 30B answer")
    ext, first = origin["ext"], origin["first"]
    item = bound(ext / "experiences" / (sid + ".json"), evidence)
    payload = dict(sample=sample, taxonomy=origin["taxonomy"], max_rules_per_sample=first["max_rules_per_sample"])
    require(item["id"] == sid and item["signature"] == extract.digest([payload, first]), "Wrong first review")
    hashes = {ev["file"]: evidence[str(ext / ev["file"])] for ev in item["evidence"]}
    journal = RecordedResponses(ext, hashes, first)
    draft, record = extract.call(scratch, journal, tokenizer, "extract_experience", sid, payload)
    used = [record]
    try:
        value = extract.old.extraction.normalize_120b_experience(draft, **payload)
    except ValueError as exc:
        repair = dict(original=payload, draft=draft, validation_error=str(exc))
        draft, record = extract.call(scratch, journal, tokenizer, "repair_experience", sid, repair)
        used.append(record)
        value = extract.old.extraction.normalize_120b_experience(draft, **payload)
    require(used == item["evidence"] and value == item["value"] and value["sample_id"] == sid,
            f"First review differs from raw responses for {sid}")
    audit = value["semantic_audit"]
    case = origin["cases"][sid]
    source_case(case, sample, audit)
    pipe, cfg = origin["pipe"], origin["build"]
    cid = case["id"]
    saved = vr.read(pipe / "skill_outcomes" / (cid + ".json"))
    require(saved["id"] == cid and saved["signature"] == vr.digest([case, cfg]), "Wrong skill parent")
    ev = saved["evidence"]
    journal = RecordedResponses(pipe, {ev["file"]: ev["sha256"]}, cfg["transport"])
    parsed, actual = build.model_json(scratch, journal, tokenizer, "skills", cid, bounded.skill_messages(case))
    decoded = Path("decoded/skills") / (vr.digest(cid) + ".json")
    require(actual == ev and parsed == saved["value"] and vr.read(scratch / decoded) == vr.read(pipe / decoded),
            f"Skill request, output or actual input tokens changed for {sid}")
    try:
        bounded.validate_skills(parsed)
        outcome = dict(id=sid, status="collected", value=parsed)
    except ValueError as exc:
        outcome = dict(id=sid, status="content_failure", reason=str(exc))
    require(outcome["status"] == saved["status"] and outcome.get("reason") == saved["reason"], "Skill status drift")
    topic = value["topic_guess"]
    rules = {"exp_" + extract.old.extraction._fingerprint_rule(r, topic["domain"], topic["topic"]): r["source_error_ids"]
             for r in value["experience_rules"]}
    ledger = dict(answer_sha256=vr.digest(sample["prediction"]), solver_response=answer["evidence"],
                  audit_sha256=vr.digest(audit), first_responses=used, case_id=cid, case_sha256=vr.digest(case),
                  skill_response=ev, skill_status=outcome["status"], candidate_errors=rules,
                  errors={e["id"]: vr.digest(e) for e in audit["key_errors"]},
                  skill_ids=["procedure_" + vr.digest(s)[:24] for s in parsed.get("skills", [])]
                  if outcome["status"] == "collected" else [])
    return ledger, outcome


def publish_library(output, name, draw, selected, catalog, provenance, origin, ledger, outcomes, common_hash, screen,
                    checked_tree_text, parent_sha):
    folder = output / name
    selected = set(selected)
    require(selected == set(draw) & origin["eligible"], f"Wrong first-review sources in {name}")
    ids = sorted(selected)
    cases = [origin["cases"][sid] for sid in ids]
    skills, counts = bounded.materialize_skills([dict(id=sid) for sid in ids], [outcomes[sid] for sid in ids])
    original_skills = [s for s in origin["first_review_skills"] if selected.intersection(s["source_ids"])]
    require(skills == original_skills and all(set(s["source_ids"]) <= selected for s in skills), "Skill origin drift")
    excluded = set(draw) & HELD
    if excluded:
        catalog, edits = bounded.previous.prune_catalog(catalog, provenance, excluded)
    else:
        edits = {"excluded_source_leaves": [], "reindexed_clusters": [], "removed_clusters": []}
    memory = scale.final.base.p.TreeMemory(catalog)
    leaf_support = {}
    for rid in memory.corpus.leaves:
        links = []
        for entry in provenance[rid]:
            sid, error = entry["sample_id"], entry["source_error"]
            require(sid in selected and error["id"] in ledger[sid]["candidate_errors"][rid]
                    and ledger[sid]["errors"][error["id"]] == vr.digest(error), f"Wrong error for leaf {rid}")
            links.append([sid, error["id"]])
        require(links, f"Missing support for leaf {rid}")
        leaf_support[rid] = links
    for store in (memory.memory, memory.nodes):
        for key in store.ids:
            doc = store.view(key)
            text = doc["title"] + "\n" + doc["text"]
            if text not in checked_tree_text:
                require(not screen.matches(dict(id=key, question="", prediction=text)), f"Held-out match in {name}/{key}")
                checked_tree_text.add(text)
    for filename, value in zip(shared.MATERIALS, (cases, skills, catalog)):
        vr.atomic(folder / "materials" / filename, value)
    vr.atomic(folder / "sources.json", dict(draw=draw, common_reviews={s: ledger[s]["audit_sha256"] for s in ids},
              excluded_prior=sorted(set(draw) & origin["excluded"]), source_hold=sorted(excluded),
              leaf_errors=leaf_support, tree_edits=edits))
    vr.atomic(folder / "material_isolation.json", dict(passed=True, heldout_sha256=vr.digest(origin["heldout"]),
              flat_sources_checked=len(ids), tree_leaves_checked=len(leaf_support), semantic_independence_proven=False))
    counted = dict(drawn=len(draw), cases=len(cases), skills=len(skills), leaves=len(leaf_support),
                   skill_sources=len({s for d in skills for s in d["source_ids"]}),
                   tree_sources=len({s for pairs in leaf_support.values() for s, _ in pairs}),
                   skill_failures=counts.get("content_failure", 0), skill_empty=counts.get("valid_empty_sources", 0),
                   source_holds=len(excluded), pruned_leaves=len(edits["excluded_source_leaves"]))
    hashes = {n: vr.sha(folder / "materials" / n) for n in shared.MATERIALS}
    receipt = dict(passed=True, protocol=shared.PROTOCOL, name=name, counts=counted, source_reviews_sha256=common_hash,
                   parent_catalog_sha256=parent_sha, materials=hashes, sources_sha256=vr.sha(folder / "sources.json"),
                   material_isolation_sha256=vr.sha(folder / "material_isolation.json"), new_model_calls=0)
    vr.atomic(folder / "construction_acceptance.json", receipt)
    return dict(**counted, acceptance_sha256=vr.sha(folder / "construction_acceptance.json"))


def publish(root, out, log):
    from transformers import AutoTokenizer
    origin = inputs(root)
    ids = sorted(origin["eligible"])
    tokenizer = AutoTokenizer.from_pretrained(origin["first"]["model"], local_files_only=True, trust_remote_code=False)
    ledger, outcomes = {}, {}
    with tempfile.TemporaryDirectory(prefix=".decoded-", dir=out) as temp:
        for i, sid in enumerate(ids, 1):
            ledger[sid], outcomes[sid] = replay_source(origin, sid, tokenizer, Path(temp))
            if i == 1 or i % 25 == 0 or i == len(ids):
                vr.atomic(log, dict(stage="replaying_sources", completed=i, total=len(ids)))
    screen = bounded.FamilyScreen(origin["heldout"])
    for i, sid in enumerate(ids, 1):
        row = origin["samples"][sid]
        texts = [origin["cases"][sid]["text"]]
        for skill in outcomes[sid].get("value", {}).get("skills", []):
            texts.extend((skill["title"], skill["text"]))
        matches = screen.matches(dict(id=sid, question=row["question"], prediction="\n".join(texts)))
        require(not matches, f"Held-out match for {sid}: {matches}")
        if i % 100 == 0:
            vr.atomic(log, dict(stage="screening_sources", completed=i, total=len(ids)))
    vr.atomic(out / "source_reviews.json", ledger)
    common_hash = vr.sha(out / "source_reviews.json")
    libraries = {}
    checked_text = set()
    catalog = origin["catalog"]
    libraries["n2000"] = publish_library(out, "n2000", [r["id"] for r in origin["split"]["build"]], ids,
        catalog, origin["bundle"]["provenance"], origin, ledger, outcomes, common_hash, screen, checked_text,
        vr.digest(catalog))
    seeds = (20261122, *scale.RANDOM_SEEDS)
    for seed in seeds:
        for size in scale.BUILD_SIZES:
            name = f"n{size}_seed{seed}"
            vr.atomic(log, dict(stage="checking_scale_library", library=name, completed=len(libraries)))
            parent = root / "library_scale" / name
            plan = scale.verify(parent)["configuration"]
            replay = vr.read(parent / "tree_replay.json")
            require(replay["passed"] and replay["acceptance_sha256"] == vr.sha(parent / "construction_acceptance.json"),
                    f"Missing original tree replay for {name}")
            source = vr.read(parent / "sources.json")
            draw = source["original_draw"]
            registered, _ = scale.source_draw(origin["split"], origin["families"], seed, True, size)
            require(draw == [r["id"] for r in registered] and vr.digest(sorted(draw)) == plan["source_ids_sha256"],
                    f"Registered sample changed: {name}")
            selected = {s for s in draw if source["dispositions"][s] == "error_found"} - HELD
            catalog = vr.read(parent / "materials/catalog.json")
            libraries[name] = publish_library(out, name, draw, selected, catalog,
                vr.read(parent / "tree_bundle.json")["provenance"], origin, ledger, outcomes, common_hash,
                screen, checked_text, vr.digest(catalog))
    index = dict(protocol=shared.PROTOCOL, fixed_construction_date=DATE, model=origin["first"]["model"],
                 original_construction_acceptance_sha256=origin["original_release_sha256"],
                 prior_quarantine=sorted(origin["excluded"]), source_hold=sorted(HELD),
                 source_reviews_sha256=common_hash, libraries=libraries, verified_sources=len(ledger),
                 known_first_review_error=TEACUP, new_model_calls=0, human_gold=False)
    vr.seal(out / "release.json", index)


def ensure_publication(root):
    destination = root / shared.DIRECTORY
    with build.c.corpus.r.run_lock(root / ".shared_experience.lock"):
        if not destination.exists():
            progress = root / ".shared_experience_progress.json"
            with tempfile.TemporaryDirectory(prefix=".shared-build-", dir=root) as temp:
                out = Path(temp) / shared.DIRECTORY
                out.mkdir()
                publish(root, out, progress)
                for name in shared.release_index(out)["libraries"]:
                    shared.library_path(out.parent, name)
                os.rename(out, destination)
            progress.unlink(missing_ok=True)
        else:
            expected = {"n2000"} | {f"n{size}_seed{seed}" for size in scale.BUILD_SIZES
                                    for seed in (20261122, *scale.RANDOM_SEEDS)}
            require(set(shared.release_index(destination)["libraries"]) == expected, "Incomplete publication")
            for name in expected:
                shared.library_path(root, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("publish", "locate", "verify"))
    parser.add_argument("--root", required=True, type=Path, help="qwen30b_step_verifier result root on node14")
    parser.add_argument("--library", default="n2000", help="n2000 or a registered n<size>_seed<seed>")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / shared.DIRECTORY
    if args.mode == "publish":
        ensure_publication(root)
    if args.mode == "verify":
        for name in shared.release_index(destination)["libraries"]:
            print(shared.library_path(root, name))
    else:
        print(shared.library_path(root, args.library) / "materials")


if __name__ == "__main__":
    main()
