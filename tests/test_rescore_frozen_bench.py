import hashlib
import json

import pytest

from scripts import rescore_frozen_bench as r


def dataset():
    # Offsets deliberately not zero: this suite retains the legacy scorer.
    return [dict(id="q", question="Check the relation.", prediction="Intro. False relation. End.", physics_error_gt=[
        dict(error_id="q/E1", error_text="wrong relation", answer_quote="False relation.",
             start_char=7, end_char=22, span_valid=True, paragraph_index=-1, paragraph_valid=False),
        dict(error_id="q/E2", error_text="omitted requested part", answer_quote="",
             start_char=-1, end_char=-1, span_valid=False, paragraph_index=-1, paragraph_valid=False),
    ])]


def diagnostic(message="wrong", quote="False relation."):
    return dict(rule="rule", message=message, evidence=dict(quote=quote))


def prediction(diagnostics, failure=""):
    return dict(id="q", diagnostics=diagnostics, execution_failure=failure,
                source_status="failed" if failure else "completed")


def test_omission_in_recall_but_never_fabricated_position():
    result = r.score(dataset(), [prediction([diagnostic()])])
    s = result["summary"]
    assert (s["matched_gt_errors"], s["total_gt_errors"], s["total_gt_locatable_errors"]) == (1, 2, 1)
    assert s["recall"] == .5
    assert s["locatable_reference_sensitivity"]["recall"] == 1


def test_unlocatable_diagnosis_and_all_published_sensitivity():
    result = r.score(dataset(), [prediction([diagnostic(), diagnostic("other", "absent from answer")])])
    s = result["summary"]
    assert (s["published_diagnostics"], s["unlocatable_findings"], s["precision"]) == (2, 1, 1)
    assert s["all_published_sensitivity"]["precision"] == .5


def test_partial_failure_retains_published_diagnosis_without_becoming_success():
    result = r.score(dataset(), [prediction([diagnostic()], "partial_failure")])
    assert result["summary"]["failed_questions"] == 1
    assert result["summary"]["matched_gt_errors"] == 1
    assert result["summary"]["successful_empty_questions"] == 0


def test_failed_empty_is_distinct_from_successful_empty():
    failed = r.score(dataset(), [prediction([], "parse_failure")])["summary"]
    empty = r.score(dataset(), [prediction([])])["summary"]
    assert failed["total_gt_errors"] == empty["total_gt_errors"] == 2
    assert failed["failed_questions"] == 1 and failed["successful_empty_questions"] == 0
    assert empty["failed_questions"] == 0 and empty["successful_empty_questions"] == 1
    assert failed["precision_defined"] is False


def test_duplicate_predictions_cannot_match_same_gt_twice():
    result = r.score(dataset(), [prediction([diagnostic("a"), diagnostic("b")])])["summary"]
    assert result["matched_gt_errors"] == 1 and result["pred_locatable_findings"] == 2


def test_unknown_is_not_positive_and_has_separate_sensitivity():
    ds = dataset()
    unknown = {"q": [ds[0]["physics_error_gt"][0]]}
    ds[0]["physics_error_gt"] = ds[0]["physics_error_gt"][1:]
    result = r.score(ds, [prediction([diagnostic()])], unknown)["summary"]
    assert result["matched_gt_errors"] == 0 and result["total_gt_errors"] == 1
    assert result["unresolved_location_overlap"] == 1
    assert result["unresolved_overlap_excluded_sensitivity"]["precision_defined"] is False


def test_duplicate_or_missing_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        r.index_rows([dict(id="q"), dict(id="q")], "fixture")
    with pytest.raises(ValueError, match="Missing"):
        r.index_rows([{}], "fixture")


def test_hash_contract_rejects_changed_output(tmp_path):
    file = tmp_path/"results.json"
    file.write_text("[]")
    group = dict(paths=["results.json"], source_contract=r.digest(file))
    assert r.group_sources(tmp_path, group)[1] == []
    file.write_text("[{}]")
    with pytest.raises(ValueError, match="changed"):
        r.group_sources(tmp_path, group)


def test_missing_output_never_replaced_by_metrics(tmp_path):
    group = dict(paths=["result.json"], source_contract="a" * 64)
    files, missing = r.group_sources(tmp_path, group)
    assert files == [] and missing == [dict(path="result.json", sha256="a" * 64)]


def test_recovery_uses_first_success_even_if_later_success_differs(tmp_path):
    files = []
    for i, (status, message) in enumerate([("failed", "x"), ("completed", "first"), ("completed", "later")]):
        file = tmp_path/f"{i}.json"
        file.write_text(json.dumps(dict(id="q", status=status, trace=dict(id="q", diagnostics=[diagnostic(message)]))))
        files.append(file)
    group = dict(id="arm", kind="first_completed")
    result = r.load_predictions(files, group, ["q"])
    assert result[0]["diagnostics"][0]["message"] == "first"


def test_jev_stage_failure_remains_visible(tmp_path):
    path = tmp_path/"results.json"
    path.write_text(json.dumps({"q_F": dict(id="q", form="F", status="completed", trace=dict(id="q", diagnostics=[]))}))
    group = dict(id="jev", kind="jev", variant="F", stage_failure_evidence=[dict(
        form="F", sample_key=hashlib.sha256(b"q").hexdigest()[:16], stage="location")])
    row = r.load_predictions([path], group, ["q"])[0]
    assert row["source_status"] == "completed"
    assert row["execution_failure"] == "recorded_stage_failure"


def test_forced_protocol_violation_is_distinct_from_technical_failure(tmp_path):
    file = tmp_path/"results.json"
    file.write_text(json.dumps([dict(id="q", status="protocol_noncompliance", diagnostics=[diagnostic()])]))
    group = dict(id="forced", kind="array")
    rows = r.load_predictions([file], group, ["q"])
    scored = r.score(dataset(), rows)["summary"]
    assert scored["protocol_noncompliance_questions"] == 1
    assert scored["failed_questions"] == 0
    assert scored["matched_gt_errors"] == 1


def test_old_score_reproduction_detects_matched_claim_change():
    actual = r.score(dataset(), [prediction([diagnostic()])])
    published = json.loads(json.dumps(actual))
    published["details"][0]["location_matches"][0]["gt_error_id"] = "wrong_id"
    with pytest.raises(ValueError, match="matched claims differ"):
        r.validate_old_score(actual, published)


def test_historical_available_subset_is_not_new_full50_policy():
    actual = r.score(dataset(), [prediction([diagnostic()])])
    failed = dict(actual["details"][0], id="failed_q", execution_failure="failure", matched_error_count=0,
                  matched_pred_locatable=0, pred_finding_count=0, pred_locatable_finding_count=0,
                  published_diagnostic_count=0, location_matches=[], matched_claim_ids=[])
    combined = dict(details=actual["details"] + [failed])
    combined["summary"] = r.summarize(combined["details"])
    published = dict(summary=actual["summary"], details=actual["details"] + [dict(id="failed_q", scored=False)])
    verified = r.validate_old_score(combined, published)
    assert verified["historical_scored_questions"] == 1
    assert verified["historical_gt"] == 2 and verified["full50_gt"] == 4


def test_registry_and_pairing_do_not_mix_continuous_controls():
    groups = r.registry()
    assert len(groups) == len({g["id"] for g in groups}) == 4
    assert {g["family"] for g in groups} == {"continuous_v2", "continuous_v3"}
    assert len(r.comparisons()) == 3
    assert sum(p["comparison_kind"] == "rule_effect" for p in r.comparisons()) == 2
    pairs = {p["treatment"]: p["control"] for p in r.comparisons()}
    assert pairs["continuous_v2_C3"] == "continuous_v2_C0p"
    assert pairs["continuous_v3_C4"] == "continuous_v3_C0p"


def test_same_group_bootstrap_is_exactly_zero():
    result = r.score(dataset(), [prediction([diagnostic()])])
    output = r.bootstrap(result, result)
    for metric in output.values():
        assert metric == dict(delta_pp=0.0, ci95_pp=[0.0, 0.0])


def test_outputs_cannot_be_overwritten(tmp_path):
    path = tmp_path/"result.json"
    r.write_new(path, {"original": True})
    with pytest.raises(ValueError, match="overwrite"):
        r.write_new(path, {"original": False})
    assert r.read(path) == {"original": True}


def test_run_writes_complete_offline_artifacts_from_synthetic_data(tmp_path, monkeypatch):
    import socket
    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline scoring must not access network")
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    predictions = [prediction([diagnostic()])]
    scored = r.score(dataset(), predictions)
    loaded = {"synthetic": dict(predictions=predictions, old=scored, source_receipts={})}
    monkeypatch.setattr(r, "verify", lambda *_: (loaded, {}, {}, dataset(), {"q": []}))
    monkeypatch.setattr(r, "reference_data", lambda *_: (dataset(), dataset(), {"q": []}))
    plan = tmp_path/"plan.json"
    r.write_new(plan, dict(groups=[dict(id="synthetic", label="合成测试", family="test", note="")], comparisons=[]))
    out = tmp_path/"output"
    result = r.run(tmp_path, plan, out)
    assert result["model_calls"] == 0
    report = r.read(out/"report.json")
    assert report["groups"]["synthetic"]["new_full50"]["summary"]["matched_gt_errors"] == 1
    completion = r.read(out/"completion.json")
    assert completion["report_sha256"] == r.digest(out/"report.json")
    assert (out/"summary.csv").read_text(encoding="utf-8-sig").startswith("group,experiment_id,family,label,")


def test_run_blocks_missing_originals_before_writing_results(tmp_path, monkeypatch):
    plan = tmp_path/"plan.json"
    r.write_new(plan, {})
    monkeypatch.setattr(r, "verify", lambda *_: ({}, {}, {"missing": [{}]}, [], {}))
    out = tmp_path/"output"
    with pytest.raises(ValueError, match="Missing original output"):
        r.run(tmp_path, plan, out)
    assert not out.exists()


def test_known_case_controls_separate_reference_repair_from_semantic_scoring():
    density_id, time_id = "cl_174_96977", "cl_132_54047"
    density = dict(dataset()[0], id=density_id)
    density["physics_error_gt"] = [dict(dataset()[0]["physics_error_gt"][0], error_id=density_id+"/E1")]
    time = dict(dataset()[0], id=time_id)
    time["physics_error_gt"] = [dict(dataset()[0]["physics_error_gt"][0], error_id="claim_393743567a5af01f",
                                   start_char=-1, end_char=-1, span_valid=False)]
    old = [dict(density, physics_error_gt=[]), time]
    new = [density, dict(time, physics_error_gt=[])]
    result = r.benchmark_controls(old, new)
    assert result["all_control_checks_pass"] is True
    assert result["checks"][1]["observed"] == dict(old_hits=0, new_hits=0)
    assert result["checks"][1]["old_claim_locatable"] is False
    assert result["wrong_reason_control"]["semantically_valid"] is False
    assert result["wrong_reason_control"]["exposes_location_only_limitation"] is True


def test_match_change_audit_is_deterministic_and_keeps_unadjudicated_status():
    ds = dataset()
    predictions = [prediction([diagnostic()])]
    old_ds = [dict(ds[0], physics_error_gt=[])]
    groups = {"test": dict(old_full50=r.score(old_ds, predictions), new_full50=r.score(ds, predictions))}
    loaded = {"test": dict(predictions=predictions)}
    a = r.match_change_audit(groups, loaded, old_ds, ds)
    b = r.match_change_audit(groups, loaded, old_ds, ds)
    assert a == b
    assert a["counts_by_group"]["test"] == dict(gained=1, lost=0)
    assert a["selected"][0]["physics_review"] == "not_adjudicated"
    assert a["selected"][0]["sample_id"] == "q"


def test_full_registry_matches_e01_to_e39_and_reuses_recent_sources():
    groups = r.registry(r.FULL_SCOPE)
    assert len(groups) == len({g["id"] for g in groups}) == 39
    assert [g["experiment_id"] for g in groups] == [f"E{i:02d}" for i in range(1,40)]
    for previous,expanded in zip(r.registry(),groups[-4:]):
        assert {k:v for k,v in expanded.items() if k!="experiment_id"} == previous
    assert groups[3]["kind"] == "first_completed"
    assert groups[21]["variant"] == "F"
    assert all("budget_supplement" in g["paths"][0] for g in groups[32:35])


def test_full_pairs_keep_family_controls_and_do_not_cross_continuous_rounds():
    pairs = r.comparisons(r.FULL_SCOPE)
    ids = {g["id"] for g in r.registry(r.FULL_SCOPE)}
    assert len(pairs) == 28
    assert all(p["treatment"] in ids and p["control"] in ids for p in pairs)
    assert pairs[-3:] == r.comparisons()
    controls = {p["treatment"]:p["control"] for p in pairs}
    assert controls["joint_O4"] == controls["joint_O5"] == "joint_pure"
    assert controls["jev_F"] == controls["jev_P"] == "retrieval_3"
    assert controls["terra_candidates_literal_evidence"] == "terra_candidates_original"
    assert controls["rule_Kplus"] == "rule_Kminus"


def test_scope_expansion_preserves_every_scoring_policy():
    recent,full=r.policy_for_scope(r.RECENT_SCOPE),r.policy_for_scope(r.FULL_SCOPE)
    assert {k:v for k,v in full.items() if k!="scope"} == {k:v for k,v in recent.items() if k!="scope"}
    with pytest.raises(ValueError,match="scope"):
        r.registry("unapproved_subset")


def test_terminal_retrieval_failure_keeps_envelope_identity_and_failure(tmp_path):
    file=tmp_path/"result.json"
    row=dict(id="q",status="failed",failure_stage="retrieval",checker_executed=False,
        retrieval_failure_source=dict(path="retrieval/result.json",sha256="a"*64,status="failed"),
        trace=dict(diagnostics=[],checker_status="not_run_terminal_retrieval_failure"))
    file.write_text(json.dumps(row))
    group=dict(id="joint",kind="per_sample")
    normalized=r.load_predictions([file],group,["q"])
    summary=r.score(dataset(),normalized)["summary"]
    assert summary["failed_questions"]==1 and summary["successful_empty_questions"]==0
    assert normalized[0]["stage_failure_evidence"][0]["identity_source"]=="result_envelope"
    assert r.read(file)==row


@pytest.mark.parametrize("trace", [dict(diagnostics=[]),dict(id="different",diagnostics=[]),dict(id="q")])
def test_missing_or_conflicting_completed_trace_identity_is_still_rejected(tmp_path, trace):
    file=tmp_path/"result.json"
    file.write_text(json.dumps(dict(id="q",status="completed",trace=trace)))
    with pytest.raises(ValueError,match="Missing final diagnostics/identity"):
        r.load_predictions([file],dict(id="test",kind="per_sample"),["q"])
