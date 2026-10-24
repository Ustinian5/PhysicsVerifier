"""Offline, preregistered rescoring of saved 50-question experiments.

prepare/verify inventory data, reproduce OLD scores and test synthetic known-case
controls. Only run scores saved experiments against the released new reference.
No model client, credentials, or generation is used.
The metric remains the historical location proxy, not semantic error accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_direct_baseline_comparison import reference_dataset
from scripts import evaluate_physics_eval_sets as legacy

BASE = "results/retrieval_comparison_50_20260920"
OLD_METRICS = BASE + "/metrics_automatic_proxy_location_unreviewed.json"
INPUTS = BASE + "/frozen50/inputs.json"
RELEASE = "results/bench_reannotation_99_20261011/assistant_review/release_manifest.json"
REFERENCE = "results/bench_reannotation_99_20261011/assistant_review/reference.json"
RELEASE_VERSION = "sol_high_4pass_99_assistant_review_v2"
DEFAULT_PLAN = "results/bench_rescore_20261018/plan.json"
RECENT_SCOPE = "recent_continuous_v2_v3_four_groups"
FULL_SCOPE = "all_39_frozen_50"
PARENT_BATCH = "results/bench_rescore_20261018"
POLICY = {
    "scope": "recent_continuous_v2_v3_four_groups",
    "metric": "historical_location_proxy_not_physics_accuracy",
    "samples": 50, "old_gt": 187, "old_locatable_gt": 130, "new_confirmed_gt": 201,
    "new_locatable_gt": 198, "new_omissions": 3, "new_unresolved": 3,
    "iou": 0.5, "coverage": 0.6, "gt_paragraph_matching": False,
    "matching": "unchanged legacy greedy one-to-one, original reference order",
    "failures": "full50; retain published final diagnostics; failed/missing stages explicit; no successful-empty imputation",
    "unknown": "unresolved not positive or negative gold; unmatched means location-unmatched, never adjudicated false positive",
    "omissions": "keep in main recall denominator; no invented spans; also report 198-locatable sensitivity",
    "precision": "matched locatable / locatable findings; also report all-published denominator",
    "bootstrap": {"unit": "paired question", "generator": "NumPy PCG64", "seed": 20261018, "replicates": 20000},
    "bench_validation": {
        "known_cases": ["cl_174_96977:missing_density_error", "cl_132_54047:false_time_transform_error"],
        "negative_control": "wrong physical correction with the same quote must not be mistaken for semantic validation",
        "change_sample": "up to 4 gained and 4 lost, question-distinct within each direction, SHA256 order; no group preference",
        "review_causes": ["reference_content_change", "same_error_location_repair", "one_to_one_assignment_change",
                          "overlap_without_same_physical_error", "cannot_determine"],
        "acceptance": "known-issue repair and engineering usability only; no score-increase or rule-win requirement",
    },
    "no_generation": True, "human_gold": False, "independent_test": False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents a second operator silently replacing this result.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def select(value, keys):
    for key in keys:
        value = value[key]
    return value


def registry(scope=RECENT_SCOPE):
    """Latest two completed, budget-matched rounds; selected before new scores."""
    if scope == FULL_SCOPE:
        return full_registry()
    if scope != RECENT_SCOPE:
        raise ValueError("Unknown experiment scope")
    groups = []
    for version, folder, arms in [
        ("v2", "results/continuous_context_v2_20261011/recovery_002", ["C0p", "C3"]),
        ("v3", "results/continuous_context_v3_20261013/recovery_002", ["C0p", "C4"]),
    ]:
        for arm in arms:
            path = folder + (f"/{arm}_results.json" if version == "v2" else f"/formal/{arm}/results.json")
            groups.append(dict(
                id=f"continuous_{version}_{arm}", family="continuous_" + version,
                label="连续会话" + version + " " + arm, paths=[path],
                metrics=folder + "/metrics_automatic_proxy_unreviewed.json",
                score_keys=["scores_all_50", arm], kind="array", variant=None,
                hash_keys=None,
                note="同轮C0p为对照；最终引文修正版；每组完整50题；本轮只重评近期四组"))
    return groups


def full_registry():
    """E01–E39 from the user's existing inventory; no generation or score selection."""
    groups = []

    def add(gid, family, label, paths, metrics, score_keys, kind="array", hash_keys=None, variant=None, note=""):
        groups.append(dict(id=gid, experiment_id=f"E{len(groups)+1:02d}", family=family, label=label,
            paths=paths, metrics=metrics, score_keys=score_keys, kind=kind, hash_keys=hash_keys,
            variant=variant, note=note))

    def checker(gid, family, label, folder, variant="original", note=""):
        add(gid, family, label, [f"{folder}/{variant}/*/result.json"],
            folder+"/metrics_automatic_proxy_unreviewed.json", ["scores_all_50", variant],
            kind="per_sample", hash_keys=["source_results_sha256", variant], variant=variant, note=note)

    for arm, label in [(1, "Qwen原树检索→Qwen检查"), (2, "Terra原树检索→Qwen检查"),
                       (3, "Terra自主树叶检索→Qwen检查"), (4, "Terra自主仅叶检索→Qwen检查")]:
        add(f"retrieval_{arm}", "A_original_pipeline", label,
            [f"{BASE}/formal_checker/arm{arm}/*/result.json", f"{BASE}/formal_checker_retry_001/arm{arm}/*/result.json"],
            OLD_METRICS, ["per_arm_all_available", str(arm)], kind="first_completed",
            note="按原技术补跑顺序取首次成功；E04原指标只计49题，重评保留完整50题和失败。")
    leaf_metrics=BASE+"/terra_leaf_direct/metrics_automatic_proxy.json"
    for gid, label, folder, key in [
        ("terra_pure", "纯Terra", "terra_direct", "pure_terra"),
        ("terra_tree", "Terra自主树叶自检", "terra_tree_direct", "terra_tree_self_check"),
        ("terra_leaf", "Terra自主仅叶自检", "terra_leaf_direct", "terra_leaf_self_check")]:
        short=gid.split("_")[1]
        add(gid, "B_terra_direct", label, [f"{BASE}/{folder}/results.json"], leaf_metrics,
            ["all_50",key], hash_keys=[f"source_{short}_sha256"], note="Terra medium同会话自检，无Qwen。")
    folder="results/gpt_oss_120b_baseline_20260928/formal_vllm_008"
    for arm,label in [("model_only","纯GPT-OSS"),("leaf_self_check","GPT-OSS可选仅叶"),("tree_self_check","GPT-OSS可选树叶")]:
        add("gptoss_"+arm,"C_gptoss_direct",label,[f"{folder}/{arm}/results.json"],
            folder+"/metrics_automatic_proxy_unreviewed.json",["scores_all_50",arm],
            hash_keys=["source_results_sha256",arm],note="修复接口后的正式批次；不纳入故障全零批次。")
    folder="results/gpt_oss_forced_rules_20260930/formal"
    for arm,label in [("leaf_self_check","GPT-OSS强制叶L"),("tree_self_check","GPT-OSS强制树T")]:
        add("gptoss_forced_"+arm,"C_gptoss_direct",label,[f"{folder}/{arm}/results.json"],
            folder+"/metrics_automatic_proxy_unreviewed.json",["scores_all_50_valid_final_outputs",arm],
            hash_keys=["source_results_sha256",arm],note="协议不合规单列；工具/提示/计算量同时变化，非规则内容单变量。")
    folder="results/checker_prompt_iteration_20261005"
    for arm,label in [("original","原提示"),("balanced","平衡提示"),("coverage","全覆盖"),("preconditions","适用前提")]:
        checker("qwen_candidates_"+arm,"D_qwen_candidates", "Qwen候选＋GPT-OSS "+label,folder,arm)
    for arm,label in [("literal_evidence","唯一引文"),("operational_evidence","物理核验＋引文")]:
        checker("qwen_candidates_"+arm,"D_qwen_candidates","Qwen候选＋GPT-OSS "+label,folder+"/round_002",arm)
    folder="results/terra_candidate_checker_20261006"
    for gid,sub,arm,label in [("original","baseline","original","原提示"),
        ("literal_evidence","round_002/literal_evidence","literal_evidence","唯一引文"),
        ("coverage_uncertainty","round_002/coverage_uncertainty","coverage_uncertainty","修正全覆盖")]:
        checker("terra_candidates_"+gid,"E_terra_candidates","Terra候选＋GPT-OSS "+label,folder+"/"+sub,arm)
    jev="results/jev_checker_50_20261005"
    for form,label in [("F","Jev全文判定＋GPT定位"),("P","GPT选段＋Jev判定＋GPT定位")]:
        add("jev_"+form,"F_jev",label,[jev+"/results.json"],jev+"/metrics_automatic_proxy_unreviewed.json",
            ["metrics","jev_"+form],kind="jev",hash_keys=["source_results_sha256"],variant=form,
            note="固定Terra候选；J-F的1题定位技术失败单列。")
    for gid,sub,arm,label in [
        ("anchor_C1","anchor_handoff/revision_002","retrieval_anchor","C1保留Terra检索锚点"),
        ("anchor_C0p","anchor_handoff/matched_no_anchor","matched_no_anchor","C0′不传锚点"),
        ("retrieval_R0","tree_original_runtime_control","original","新R0原树检索提示"),
        ("retrieval_R1","tree_leaf_prompt_study","original","R1末级检索提示改版"),
        ("rule_Kplus","rule_content_ablation/server_local","with_rule_text","K+锚点＋规则正文"),
        ("rule_Kminus","rule_content_ablation/server_local","without_rule_text","K−锚点无规则正文")]:
        checker(gid,"G_attribution",label,folder+"/"+sub,arm,note="仅与对应成对条件归因；失败保留，不筛成功题。")
    joint="results/joint_prompt_iteration_20261008"
    add("joint_pure","H_joint","联合迭代本轮纯GPT-OSS",[joint+"/pure/results.json"],
        joint+"/pure/metrics.json",["score"],hash_keys=["results_sha256"],note="本批独立生成，不能替换为E08。")
    for arm in ["O4","O5"]:
        checker("joint_"+arm,"H_joint","联合提示"+arm,joint+"/"+arm+"/checker",
            note="同时改多个提示；O4/O5分别3/2题检索格式失败，保留全50。")
    folder="results/continuous_context_20261011/budget_supplement"
    for arm,label in [("C0","初始连续会话纯模型"),("C1","初始连续会话规则工具"),("C2","初始连续会话规则工具＋流程参考")]:
        add("continuous_initial_"+arm,"I_continuous_initial",label,[f"{folder}/{arm}/results.json"],
            folder+"/metrics_automatic_proxy_unreviewed.json",["scores_all_50",arm],
            note="C1/C2各1题追加终稿预算；单独标记，不视为严格同预算比较。")
    for group in registry():
        groups.append(dict(group,experiment_id=f"E{len(groups)+1:02d}"))
    return groups


def policy_for_scope(scope):
    if scope not in {RECENT_SCOPE,FULL_SCOPE}:
        raise ValueError("Unknown experiment scope")
    return dict(POLICY,scope=scope)


def comparisons(scope=RECENT_SCOPE):
    if scope == FULL_SCOPE:
        pairs=[]
        def pair(treatment,control,purpose,kind="method_comparison"):
            pairs.append(dict(treatment=treatment,control=control,purpose=purpose,comparison_kind=kind))
        for a,b in [(2,1),(3,2),(4,3)]:
            pair(f"retrieval_{a}",f"retrieval_{b}","原Qwen checker下检索模型/方法对照")
        for arm in ["tree","leaf"]:
            pair("terra_"+arm,"terra_pure","Terra同会话规则辅助与纯模型","rule_effect")
        for arm in ["leaf_self_check","tree_self_check"]:
            pair("gptoss_"+arm,"gptoss_model_only","GPT-OSS可选规则工具与纯模型","rule_effect")
            pair("gptoss_forced_"+arm,"gptoss_model_only","GPT-OSS强制用规则与纯模型，协议和计算量也不同","rule_effect")
        for arm in ["balanced","coverage","preconditions","literal_evidence","operational_evidence"]:
            pair("qwen_candidates_"+arm,"qwen_candidates_original","固定Qwen候选下checker提示改版")
        for arm in ["literal_evidence","coverage_uncertainty"]:
            pair("terra_candidates_"+arm,"terra_candidates_original","固定Terra候选下checker提示改版")
        for form in ["F","P"]:
            pair("jev_"+form,"retrieval_3","同Terra候选下Jev整体checker与Qwen，全50题")
        pair("anchor_C1","anchor_C0p","固定候选下是否传检索锚点")
        pair("retrieval_R1","retrieval_R0","同运行库下末级检索提示改版")
        pair("rule_Kplus","rule_Kminus","固定候选与锚点下规则正文消融","rule_content")
        for arm in ["O4","O5"]:
            pair("joint_"+arm,"joint_pure","联合提示与同批纯模型的整体对照","rule_effect")
        for arm in ["C1","C2"]:
            pair("continuous_initial_"+arm,"continuous_initial_C0","初始连续上下文；规则组各有1题追加预算","rule_effect")
        return pairs+comparisons()
    if scope != RECENT_SCOPE:
        raise ValueError("Unknown experiment scope")
    return [dict(treatment=f"continuous_{version}_{arm}", control=f"continuous_{version}_C0p",
                 purpose="近期连续会话：同轮有无规则辅助", comparison_kind="rule_effect")
            for version, arm in [("v2", "C3"), ("v3", "C4")]] + [dict(
                treatment="continuous_v3_C0p", control="continuous_v2_C0p",
                purpose="相同纯模型提示的跨轮输出波动；不是规则处理效应", comparison_kind="repeat_variation")]


def index_rows(rows, label):
    out = {}
    for row in rows:
        sid = row.get("id")
        if not isinstance(sid, str) or not sid or sid in out:
            raise ValueError(f"Missing/duplicate ID: {label}: {sid}")
        out[sid] = row
    return out


def check_hash(path, expected):
    if not Path(path).is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed frozen source: {path}")


def reference_data(data_root):
    release = read(data_root / RELEASE)
    if release["version"] != RELEASE_VERSION or release["human_gold"] is not False:
        raise ValueError("Unexpected reference release or gold status")
    for name, sha in release["files"].items():
        check_hash(data_root / Path(RELEASE).parent / name, sha)
    source_path = data_root / Path(RELEASE).parent / release["input"]["path"]
    check_hash(source_path, release["input"]["sha256"])
    source = index_rows(read(source_path), "99 inputs")
    refs = index_rows(read(data_root / REFERENCE), "reference")
    if len(refs) != 99 or set(refs) != set(source):
        raise ValueError("Reference must cover the same 99 inputs")
    inputs = read(data_root / INPUTS)
    frozen = index_rows(inputs, "50 inputs")
    if len(frozen) != 50:
        raise ValueError("Expected exactly 50 frozen questions")
    new, unresolved = [], {}
    for row in inputs:
        sid = row["id"]
        for key in ["question", "context", "prediction"]:
            if row.get(key, "") != source[sid].get(key, ""):
                raise ValueError(f"Changed benchmark input {sid}/{key}")
        gt, unknown, seen = [], [], set()
        for error in refs[sid]["errors"]:
            eid = sid + "/" + error["root_id"]
            if eid in seen or error["status"] not in {"confirmed", "unresolved"}:
                raise ValueError(f"Invalid active reference: {eid}")
            seen.add(eid)
            quote = error["quote"]
            if quote:
                if row["prediction"].count(quote) != 1:
                    raise ValueError(f"Nonunique/nonverbatim released quote: {eid}")
                start = row["prediction"].index(quote)
                spans = error["evidence_validation"]["occurrences"]
                if len(spans) != 1 or (spans[0]["start"], spans[0]["end"]) != (start, start + len(quote)):
                    raise ValueError(f"Released quote offsets differ: {eid}")
                end = start + len(quote)
            else:
                if not error.get("omission_evidence"):
                    raise ValueError(f"Missing omission evidence: {eid}")
                start = end = -1
            item = dict(error_id=eid, error_text=error["claim"], answer_quote=quote,
                        start_char=start, end_char=end, span_valid=bool(quote),
                        paragraph_index=-1, paragraph_valid=False)
            (gt if error["status"] == "confirmed" else unknown).append(item)
        new.append(dict(row, physics_error_gt=gt))
        unresolved[sid] = unknown
    if (sum(len(r["physics_error_gt"]) for r in new),
        sum(bool(g["span_valid"]) for r in new for g in r["physics_error_gt"]),
        sum(map(len, unresolved.values()))) != (201, 198, 3):
        raise ValueError("Release counts changed")
    old = reference_dataset(data_root / BASE, read(data_root / OLD_METRICS))
    if sum(len(r["physics_error_gt"]) for r in old) != 187:
        raise ValueError("Old reference must reproduce 187 claims")
    return old, new, unresolved


def group_sources(root, group):
    """Use aggregate hash or ID→hash contract where earlier scores recorded it."""
    files = []
    for pattern in group["paths"]:
        files.extend(sorted(root.glob(pattern)))
    if len(files) != len(set(files)):
        raise ValueError("Repeated source path")
    contract = group.get("source_contract")
    if isinstance(contract, str):
        if not files:
            return [], [dict(path=group["paths"][0], sha256=contract)]
        if len(files) != 1:
            raise ValueError("Expected one aggregate source")
        check_hash(files[0], contract)
    elif isinstance(contract, dict):
        observed = {}
        for file in files:
            row = read(file)
            sid = row.get("id")
            if sid in observed or sid not in contract:
                raise ValueError(f"Duplicate/unexpected raw ID in {group['id']}: {sid}")
            check_hash(file, contract[sid])
            observed[sid] = file
        missing = [dict(sample_id=sid, sha256=sha, pattern=group["paths"][0])
                   for sid, sha in contract.items() if sid not in observed]
        return files, missing
    elif not files:
        return [], [dict(pattern=group["paths"][0], reason="no source contract; cannot hydrate without mainline review")]
    return files, []


def load_predictions(files, group, ids):
    records = []
    for file in files:
        raw = read(file)
        if group["kind"] == "array":
            if not isinstance(raw, list):
                raise ValueError(f"Expected result array: {file}")
            records.extend(raw)
        elif group["kind"] == "jev":
            records.extend(v for v in raw.values() if v["form"] == group["variant"])
        else:
            records.append(raw)
    if group["kind"] == "first_completed":
        chosen = {}
        for row in records:
            sid = row["id"]
            # Frozen recovery order, first completed only; never choose by score.
            if sid not in chosen or chosen[sid].get("status") != "completed":
                chosen[sid] = row
        records = list(chosen.values())
    rows = index_rows(records, group["id"])
    if set(rows) != set(ids):
        raise ValueError(f"Not exactly frozen 50 IDs in {group['id']}")
    out = []
    for sid in ids:
        row = rows[sid]
        trace = row if group["kind"] == "array" else row["trace"]
        # Historical terminal-retrieval failures store identity on the envelope,
        # not the never-run checker trace. Retain the recorded failure and empty
        # diagnostics; do not turn missing or completed traces into empty success.
        envelope_identity = (trace.get("id") is None and group["kind"] == "per_sample"
            and row.get("status") == "failed" and row.get("failure_stage") == "retrieval"
            and row.get("checker_executed") is False
            and trace.get("checker_status") == "not_run_terminal_retrieval_failure"
            and trace.get("diagnostics") == []
            and row.get("retrieval_failure_source", {}).get("status") == "failed")
        if envelope_identity:
            trace = dict(trace, id=sid)
        if trace.get("id") != sid or not isinstance(trace.get("diagnostics"), list):
            raise ValueError(f"Missing final diagnostics/identity: {group['id']}/{sid}")
        if row.get("status", "completed") not in {"completed", "failed", "model_protocol_failure", "partial_failure", "protocol_noncompliance"}:
            raise ValueError(f"Unrecognized execution status: {group['id']}/{sid}: {row.get('status')}")
        failure = legacy._execution_failure_reason(trace)
        if row.get("status", "completed") not in {"completed", "protocol_noncompliance"}:
            failure = failure or str(row["status"])
        stage_failures = [f for f in group.get("stage_failure_evidence", [])
                          if f.get("form") == group.get("variant") and
                          f.get("sample_key") == hashlib.sha256(sid.encode()).hexdigest()[:16]]
        if envelope_identity:
            stage_failures.append(dict(stage="retrieval", status="failed", identity_source="result_envelope",
                                       source=row["retrieval_failure_source"]))
        if stage_failures:
            failure = failure or "recorded_stage_failure"
        out.append(dict(id=sid, diagnostics=trace["diagnostics"], execution_failure=failure,
                        stage_failure_evidence=stage_failures,
                        protocol_noncompliance=row.get("status") == "protocol_noncompliance",
                        source_status=row.get("status", trace.get("checker_status", "completed"))))
    return out


def ratios(tp, pred, gt):
    p = tp / pred if pred else 0.0
    r = tp / gt if gt else 0.0
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.0,
                precision_defined=bool(pred), recall_defined=bool(gt))


def score(dataset, predictions, unresolved=None):
    """Unmodified matching helpers; all 50 remain in the operational denominator."""
    indexed = index_rows(predictions, "predictions")
    details = []
    for row in dataset:
        sid = row["id"]
        pred = indexed[sid]
        gt = legacy._extract_gt_entries(row, sid)
        findings = legacy._fill_missing_pred_locations(legacy._collect_pred_findings(pred, {}), row["prediction"])
        loc_gt = [g for g in gt if g["locatable_valid"]]
        matched, matches, used = legacy._match_by_location(loc_gt, findings, 0.5, 0.6)
        unknown_overlap = []
        if unresolved:
            unknown_gt = legacy._extract_gt_entries(dict(physics_error_gt=unresolved[sid]), sid)
            for i, finding in enumerate(findings):
                if i not in used and legacy._match_by_location(unknown_gt, [finding], 0.5, 0.6)[0]:
                    unknown_overlap.append(i)
        details.append(dict(id=sid, gt_error_count=len(gt), gt_locatable_error_count=len(loc_gt),
            matched_error_count=len(matched), matched_pred_locatable=len(used),
            pred_finding_count=len(findings), published_diagnostic_count=len(pred["diagnostics"]),
            pred_locatable_finding_count=sum(bool(f["locatable_valid"]) for f in findings),
            execution_failure=pred["execution_failure"], source_status=pred["source_status"],
            stage_failure_evidence=pred.get("stage_failure_evidence", []),
            protocol_noncompliance=pred.get("protocol_noncompliance", False),
            successful_empty=not pred["diagnostics"] and not pred["execution_failure"],
            unresolved_overlap_indices=unknown_overlap,
            matched_finding_indices=sorted(used),
            matched_claim_ids=sorted(matched), location_matches=matches))
    return dict(summary=summarize(details), details=details)


def summarize(details):
    def total(key):
        return sum(r[key] for r in details)
    tp, mp, loc, gt = map(total, ["matched_error_count", "matched_pred_locatable", "pred_locatable_finding_count", "gt_error_count"])
    if tp != mp:
        raise ValueError("One-to-one count invariant failed")
    published, all_findings, gt_loc = map(total, ["published_diagnostic_count", "pred_finding_count", "gt_locatable_error_count"])
    unknown = sum(len(r["unresolved_overlap_indices"]) for r in details)
    return dict(dataset_size=len(details), total_gt_errors=gt, total_gt_locatable_errors=gt_loc,
                matched_gt_errors=tp, matched_pred_locatable=mp, pred_locatable_findings=loc,
                total_findings=all_findings, published_diagnostics=published,
                unlocatable_findings=all_findings-loc,
                failed_questions=sum(bool(r["execution_failure"]) for r in details),
                protocol_noncompliance_questions=sum(r.get("protocol_noncompliance", False) for r in details),
                successful_empty_questions=sum(r["successful_empty"] for r in details),
                **ratios(tp, loc, gt),
                all_published_sensitivity=ratios(tp, published, gt),
                locatable_reference_sensitivity=ratios(tp, loc, gt_loc),
                unresolved_location_overlap=unknown,
                unresolved_overlap_excluded_sensitivity=ratios(tp, loc-unknown, gt))


def validate_old_score(actual, published):
    """Reproduce published scored subset (49 for one historical arm), then ITT50."""
    rows = {r["id"]: r for r in actual["details"]}
    scored = []
    for previous in published["details"]:
        if previous.get("scored", True) is False:
            continue
        current = rows[previous["id"]]
        for key in ["gt_error_count", "gt_locatable_error_count", "pred_finding_count",
                    "pred_locatable_finding_count", "matched_error_count"]:
            if previous[key] != current[key]:
                raise ValueError(f"Old score mismatch {previous['id']}/{key}: {previous[key]} != {current[key]}")
        old_ids = sorted(m["gt_error_id"] for m in previous["location_matches"])
        if old_ids != current["matched_claim_ids"]:
            raise ValueError(f"Old matched claims differ: {previous['id']}")
        scored.append(current)
    summary = summarize(scored)
    for key in ["total_gt_errors", "total_gt_locatable_errors", "matched_gt_errors",
                "matched_pred_locatable", "precision", "recall", "f1"]:
        if abs(summary[key] - published["summary"][key]) > 1e-12:
            raise ValueError(f"Published summary mismatch: {key}")
    return dict(reproduced=True, historical_scored_questions=len(scored),
                historical_gt=summary["total_gt_errors"], full50_gt=actual["summary"]["total_gt_errors"])


def benchmark_controls(old, new):
    """Frozen known issues from the earlier assistant review, not new gold labels.

    These are synthetic diagnosis probes against real, unchanged answers. They
    never count as experiment outputs or independent benchmark accuracy evidence.
    """
    a, b = index_rows(old, "old reference"), index_rows(new, "new reference")
    density_id, time_id = "cl_174_96977", "cl_132_54047"
    density = next(g for g in b[density_id]["physics_error_gt"] if g["error_id"] == density_id + "/E1")
    time_claim = next(g for g in a[time_id]["physics_error_gt"] if g["error_id"] == "claim_393743567a5af01f")

    def probe(sid, quote, message):
        pred = dict(id=sid, diagnostics=[dict(message=message, evidence=dict(quote=quote))],
                    execution_failure="", source_status="synthetic_control")
        return {name: score([data[sid]], [pred])["summary"]["matched_gt_errors"]
                for name, data in [("old_hits", a), ("new_hits", b)]}

    recovered = probe(density_id, density["answer_quote"], "The density transformation is inverted; the correct local density is n'=gamma n.")
    removed = probe(time_id, time_claim["answer_quote"], "The coordinate-time relation t_B=gamma t_A is invalid merely because the events are spatially separated.")
    time_locatable = bool(time_claim["span_valid"] or time_claim["paragraph_valid"])
    wrong_reason = probe(density_id, density["answer_quote"], "The correct density is n'=n/gamma^2.")
    checks = [
        dict(id="missing_density_error", sample_id=density_id, expected=dict(old_hits=0, new_hits=1),
             observed=recovered, passed=recovered == dict(old_hits=0, new_hits=1),
             physics_basis="Material volume V'=V/gamma and invariant particle count imply n'=gamma n; the old error list was empty."),
        dict(id="false_time_transform_error", sample_id=time_id, expected=dict(old_hits=0, new_hits=0),
             observed=removed, old_claim_locatable=time_locatable,
             passed=not time_locatable and removed == dict(old_hits=0, new_hits=0),
             reference_review="Assistant checked the released three claims against the prior qualitative review: the unsupported objection to the coordinate-time relation is not retained. This content check is not performed by the location scorer.",
             score_interpretation="The old claim had no valid location and already earned no location credit. Its removal is a content repair, not evidence of a score improvement.",
             physics_basis="In Alice's frame the muon events have zero displacement along the horizontal boost, so Delta t_B=gamma Delta t_A is valid. Visual arrival delay is a separate issue."),
    ]
    return dict(source="pre-existing Codex qualitative review; selected before new group scores",
                synthetic_diagnoses=True, human_gold=False, all_control_checks_pass=all(c["passed"] for c in checks),
                checks=checks, wrong_reason_control=dict(observed=wrong_reason, semantically_valid=False,
                exposes_location_only_limitation=wrong_reason["new_hits"] > 0),
                interpretation="The prior assistant review supports these two content repairs only. Passing the probes confirms expected location behavior, not overall label accuracy. Wrong-reason credit concerns the scorer, not proof that the reference claim is wrong.")


def match_change_audit(groups, loaded, old, new):
    """Unbiased deterministic reading sample of changed location credit, not a judge."""
    old_by_id, new_by_id = index_rows(old, "old"), index_rows(new, "new")
    pool, group_counts = {}, {}
    for gid, group in groups.items():
        previous = index_rows(group["old_full50"]["details"], "old scores")
        current = index_rows(group["new_full50"]["details"], "new scores")
        preds = index_rows(loaded[gid]["predictions"], "predictions")
        counts = dict(gained=0, lost=0)
        for sid, row in previous.items():
            before, after = set(row["matched_finding_indices"]), set(current[sid]["matched_finding_indices"])
            findings = legacy._fill_missing_pred_locations(legacy._collect_pred_findings(preds[sid], {}), old_by_id[sid]["prediction"])
            for direction, indices in [("gained", after-before), ("lost", before-after)]:
                counts[direction] += len(indices)
                for i in sorted(indices):
                    finding = {k: findings[i][k] for k in ["message", "quote", "start_char", "end_char"]}
                    key = object_digest(["bench-change-20261018", sid, direction, finding])
                    if key not in pool:
                        pool[key] = dict(audit_id=key[:16], sample_id=sid, direction=direction,
                            finding=finding, origins=[], physics_review="not_adjudicated",
                            question=old_by_id[sid]["question"],
                            context=old_by_id[sid].get("context", ""),
                            answer=old_by_id[sid]["prediction"],
                            old_reference=old_by_id[sid]["physics_error_gt"],
                            new_reference=new_by_id[sid]["physics_error_gt"])
                    pool[key]["origins"].append(dict(group=gid, finding_index=i,
                        old_location_matches=row["location_matches"],
                        new_location_matches=current[sid]["location_matches"]))
        group_counts[gid] = counts
    selected = []
    for direction in ["gained", "lost"]:
        seen = set()
        for key in sorted(pool):
            item = pool[key]
            if item["direction"] != direction or item["sample_id"] in seen:
                continue
            selected.append(item)
            seen.add(item["sample_id"])
            if len(seen) == 4:
                break
    return dict(selection=POLICY["bench_validation"]["change_sample"],
                review_causes=POLICY["bench_validation"]["review_causes"], counts_by_group=group_counts,
                unique_changed_diagnoses=len(pool), selected=selected,
                interpretation="Location-credit changes only. Check original answer and physical reason before attributing a change to better labels; no automatic physical verdict is assigned.")


def parent_batch_receipts(root):
    """Keep the completed first batch immutable while expanding the same evaluation."""
    paths=[PARENT_BATCH+"/plan.json",PARENT_BATCH+"/scored/report.json",
           PARENT_BATCH+"/scored/summary.csv",PARENT_BATCH+"/scored/completion.json"]
    parent_plan, report, completion = read(root/paths[0]), read(root/paths[1]), read(root/paths[3])
    if parent_plan["policy"] != POLICY or completion["groups"] != 4 or completion["status"] != "completed":
        raise ValueError("Unexpected completed parent batch")
    check_hash(root/paths[0],report["plan_sha256"])
    check_hash(root/paths[1],completion["report_sha256"])
    check_hash(root/paths[2],completion["summary_sha256"])
    for path,sha in parent_plan["data_files"].items():
        check_hash(root/path,sha)
    return dict(files={p:digest(root/p) for p in paths},
        code_git_commit="50330659b12bba590478b99457cb1fb6537e0ff0",
        code_git_ref="refs/experiments/bench-rescore-20261018-recent4",
        interpretation="All39 expansion authorized after the recent4 results were seen; exploratory historical rescoring, no new generation.")


def prepare(root, plan_path, scope=RECENT_SCOPE):
    old, new, _ = reference_data(root)  # No saved experiment is scored against the new reference.
    controls = benchmark_controls(old, new)
    if not controls["all_control_checks_pass"]:
        raise ValueError("Known-case location control failed; inspect evidence, do not change labels to obtain a pass")
    ids = [r["id"] for r in old]
    frozen = [INPUTS, OLD_METRICS, BASE + "/review/blind_review.json", BASE + "/review/operator_map.json",
              RELEASE, REFERENCE, str(Path(REFERENCE).with_name("review_decisions.json")),
              BASE + "/review/assistant_case_review.json",
              "results/bench_reannotation_99_20261011/inputs.json"]
    parent = parent_batch_receipts(root) if scope == FULL_SCOPE else None
    if parent:
        frozen.extend(parent["files"])
    groups = registry(scope)
    for group in groups:
        metrics = read(root / group["metrics"])
        group["stage_failure_evidence"] = metrics.get("technical_stage_failures", [])
        group["source_contract"] = select(metrics, group["hash_keys"]) if group["hash_keys"] else None
        files, missing = group_sources(root, group)
        group["frozen_sources"] = {str(f.relative_to(root)): digest(f) for f in files}
        group["missing_at_preparation"] = missing
        group["metrics_sha256"] = digest(root / group["metrics"])
        group["normalized_predictions_sha256"] = None
        if not missing:
            predictions = load_predictions(files, group, ids)
            for prediction in predictions:
                for evidence in prediction["stage_failure_evidence"]:
                    source = evidence.get("source")
                    if source:
                        prefix="/home/visitor/huhengyuan/PhysicsVerifier/"
                        if not source["path"].startswith(prefix):
                            raise ValueError("Unknown retrieval-failure source root")
                        relative=source["path"][len(prefix):]
                        check_hash(root/relative,source["sha256"])
                        frozen.append(relative)
            group["normalized_predictions_sha256"] = object_digest(predictions)
            group["old_score_reproduction"] = validate_old_score(score(old, predictions), select(metrics, group["score_keys"]))
    code = ["scripts/rescore_frozen_bench.py", "scripts/evaluate_physics_eval_sets.py",
            "scripts/evaluate_direct_baseline_comparison.py"]
    plan = dict(schema_version=3 if parent else 2, created_utc=datetime.now(timezone.utc).isoformat(),
                status="prepared_not_rescored", policy=policy_for_scope(scope), ids=ids, groups=groups,
                benchmark_controls=controls,
                comparisons=comparisons(scope), data_files={f: digest(root/f) for f in frozen},
                code_files={f: digest(ROOT/f) for f in code},
                source_audit_scope="frozen final output files and exact old-score reproduction; not a fresh audit of every provider request")
    if parent:
        plan["parent_completed_batch"] = parent
    write_new(plan_path, plan)
    return {"groups": len(groups), "old_reproduced": sum(not g["missing_at_preparation"] for g in groups),
            "missing_groups": [g["id"] for g in groups if g["missing_at_preparation"]], "new_scores_computed": False}


def verify(root, plan):
    scope=plan["policy"]["scope"]
    if plan["policy"] != policy_for_scope(scope) or len(plan["groups"]) != len(registry(scope)) or plan["comparisons"] != comparisons(scope):
        raise ValueError("Frozen protocol changed")
    for expected, saved in zip(registry(scope), plan["groups"]):
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen experiment registry changed")
    for file, sha in plan["code_files"].items():
        check_hash(ROOT/file, sha)
    for file, sha in plan["data_files"].items():
        check_hash(root/file, sha)
    if scope == FULL_SCOPE and parent_batch_receipts(root) != plan["parent_completed_batch"]:
        raise ValueError("Completed parent batch changed")
    old, new, unresolved = reference_data(root)
    if benchmark_controls(old, new) != plan["benchmark_controls"]:
        raise ValueError("Benchmark control result changed")
    if [r["id"] for r in old] != plan["ids"]:
        raise ValueError("Frozen ID order differs")
    loaded, checks, missing = {}, {}, {}
    for group in plan["groups"]:
        check_hash(root/group["metrics"], group["metrics_sha256"])
        for file, sha in group["frozen_sources"].items():
            check_hash(root/file, sha)
        files, absent = group_sources(root, group)
        if absent:
            missing[group["id"]] = absent
            continue
        # Existing complete groups cannot silently acquire additional raw files.
        if not group["missing_at_preparation"] and set(group["frozen_sources"]) != {str(f.relative_to(root)) for f in files}:
            raise ValueError("Source inventory changed: " + group["id"])
        predictions = load_predictions(files, group, plan["ids"])
        expected = group["normalized_predictions_sha256"]
        if expected and object_digest(predictions) != expected:
            raise ValueError("Normalized saved predictions changed")
        actual = score(old, predictions)
        checks[group["id"]] = validate_old_score(actual, select(read(root/group["metrics"]), group["score_keys"]))
        loaded[group["id"]] = dict(predictions=predictions, old=actual,
                                   source_receipts={str(f.relative_to(root)): digest(f) for f in files})
    return loaded, checks, missing, new, unresolved


def bootstrap(a, b):
    import numpy as np
    if [r["id"] for r in a["details"]] != [r["id"] for r in b["details"]]:
        raise ValueError("Paired IDs differ")
    if [r["gt_error_count"] for r in a["details"]] != [r["gt_error_count"] for r in b["details"]]:
        raise ValueError("Paired reference denominators differ")
    rng = np.random.Generator(np.random.PCG64(POLICY["bootstrap"]["seed"]))
    indices = rng.integers(0, len(a["details"]), size=(POLICY["bootstrap"]["replicates"], len(a["details"])))
    estimates = []
    for group in [a, b]:
        matrix = np.array([[r["matched_error_count"], r["pred_locatable_finding_count"], r["gt_error_count"]]
                           for r in group["details"]], dtype=float)
        totals = matrix[indices].sum(axis=1)
        tp, pred, gt = totals.T
        p = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
        r = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
        f = np.divide(2*tp, pred+gt, out=np.zeros_like(tp), where=pred+gt > 0)
        estimates.append(np.stack([p, r, f], axis=1))
    differences = 100*(estimates[0]-estimates[1])
    return {name: dict(delta_pp=100*(a["summary"][name]-b["summary"][name]),
                       ci95_pp=np.quantile(differences[:, i], [.025, .975]).tolist())
            for i, name in enumerate(["precision", "recall", "f1"])}


def run(root, plan_path, out):
    plan = read(plan_path)
    loaded, checks, missing, new, unresolved = verify(root, plan)
    if missing:
        raise ValueError("Missing original output groups; do not substitute metric previews: " + ", ".join(missing))
    if out.exists():
        raise ValueError("Output already exists; inspect it instead of overwriting")
    out.mkdir(parents=True)
    # Exclusive run claim; interrupted offline runs retain this marker for review.
    write_new(out/"run_claim.json", dict(plan_sha256=digest(plan_path), status="started_offline"))
    groups = {}
    for definition in plan["groups"]:
        gid = definition["id"]
        data = loaded[gid]
        rescored = score(new, data["predictions"], unresolved)
        for field in ["published_diagnostics", "total_findings", "pred_locatable_findings", "failed_questions"]:
            if data["old"]["summary"][field] != rescored["summary"][field]:
                raise ValueError(f"Prediction-only count changed across references: {gid}/{field}")
        groups[gid] = dict(label=definition["label"], family=definition["family"], note=definition["note"],
                           old_full50=data["old"], new_full50=rescored,
                           source_receipts=data["source_receipts"])
        if "experiment_id" in definition:
            groups[gid]["experiment_id"] = definition["experiment_id"]
    parent_reproduction = {}
    if plan.get("parent_completed_batch"):
        parent_report=read(root/PARENT_BATCH/"scored/report.json")
        for gid,parent_group in parent_report["groups"].items():
            for field in ["old_full50","new_full50","source_receipts"]:
                if groups[gid][field] != parent_group[field]:
                    raise ValueError(f"Previously completed score changed: {gid}/{field}")
            parent_reproduction[gid]=True
    paired = []
    for pair in plan["comparisons"]:
        item = dict(pair)
        for version in ["old_full50", "new_full50"]:
            a, b = (groups[pair[side]][version] for side in ["treatment", "control"])
            item[version] = bootstrap(a, b)
            am = {(r["id"], m) for r in a["details"] for m in r["matched_claim_ids"]}
            bm = {(r["id"], m) for r in b["details"] for m in r["matched_claim_ids"]}
            item[version]["matched_reference_changes"] = dict(shared=len(am & bm),
                gained=sorted(am-bm), lost=sorted(bm-am))
        item["all_three_point_estimates_not_lower"] = (
            all(item["new_full50"][k]["delta_pp"] >= 0 for k in ["precision", "recall", "f1"])
            if pair["comparison_kind"] == "rule_effect" else None)
        paired.append(item)
    report = dict(status="completed_offline_location_rescore", plan_sha256=digest(plan_path),
                  policy=plan.get("policy",POLICY), old_score_reproduction=checks, groups=groups, paired=paired,
                  parent_score_reproduction=parent_reproduction,
                  benchmark_controls=plan.get("benchmark_controls", {}),
                  benchmark_validity_status="known_issue_repairs_and_operational_checks_only_not_global_accuracy",
                  match_change_audit=match_change_audit(groups, loaded, reference_data(root)[0], new),
                  limitations=["机器参考＋助手补齐，非人工金标；位置重叠不等于同一物理错误。",
                      "同一开发50题反复使用；bootstrap为开发集不确定性描述，未校正多重比较。",
                      "新参考201条含3条遗漏，位置计分不可命中；3条未决不作为正负金标。",
                      "保留原评分器局限，包括零偏移处理；不得把未匹配计数称为已确认误报。",
                      "当前评分不重新审核每个服务端原始请求，原始证据仍由各实验保存。"])
    write_new(out/"report.json", report)
    stream = io.StringIO()
    fields = ["group", "experiment_id", "family", "label", "reference", "questions", "failed", "protocol_noncompliant", "published", "locatable", "matched", "gt", "P", "R", "F1"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for gid, group in groups.items():
        for version in ["old_full50", "new_full50"]:
            s = group[version]["summary"]
            writer.writerow(dict(group=gid, experiment_id=group.get("experiment_id",""), family=group["family"], label=group["label"], reference=version, questions=s["dataset_size"],
                failed=s["failed_questions"], published=s["published_diagnostics"], locatable=s["pred_locatable_findings"],
                protocol_noncompliant=s["protocol_noncompliance_questions"],
                matched=s["matched_gt_errors"], gt=s["total_gt_errors"],
                P=f"{100*s['precision']:.2f}", R=f"{100*s['recall']:.2f}", F1=f"{100*s['f1']:.2f}"))
    (out/"summary.csv").write_text(stream.getvalue(), encoding="utf-8-sig")
    write_new(out/"completion.json", dict(status="completed", groups=len(groups), model_calls=0,
        report_sha256=digest(out/"report.json"), summary_sha256=digest(out/"summary.csv")))
    return dict(status="completed", groups=len(groups), model_calls=0, output=str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify", "run"])
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Read-only original artifacts, can be outside worktree")
    parser.add_argument("--plan", type=Path, default=ROOT/DEFAULT_PLAN)
    parser.add_argument("--scope", choices=["recent4","all39"], default="recent4", help="prepare only; verify/run read the frozen plan scope")
    parser.add_argument("--output", type=Path, help="New result directory (run), or verification JSON (verify)")
    args = parser.parse_args()
    root = args.data_root.resolve()
    try:
        if args.mode == "prepare":
            status = prepare(root, args.plan, FULL_SCOPE if args.scope == "all39" else RECENT_SCOPE)
        elif args.mode == "verify":
            _, checks, missing, _, _ = verify(root, read(args.plan))
            status = dict(status="verified" if not missing else "waiting_for_original_outputs",
                          old_reproduced=len(checks), groups=len(read(args.plan)["groups"]), old_score_reproduction=checks,
                          missing=missing, new_scores_computed=False, model_calls=0)
            if args.output:
                write_new(args.output, status)
        else:
            if args.output is None:
                raise ValueError("run requires --output in the execution workspace")
            status = run(root, args.plan, args.output)
        print(json.dumps({k:v for k,v in status.items() if k not in {"old_score_reproduction", "missing"}}, ensure_ascii=False))
        if status.get("missing"):
            print("Missing groups: " + ", ".join(status["missing"]))
            return 2
        return 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
