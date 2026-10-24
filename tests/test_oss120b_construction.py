import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rule_framework import oss120b_construction as p
from scripts import build_120b_rules as runner
from scripts import run_built_catalog as runtime


TAXONOMY = {"domains": [{"id": "mechanics", "name": "力学", "summary": "力与运动",
    "topics": [{"id": "dynamics", "name": "动力学", "summary": "运动方程"}]}]}
TOPIC = "mechanics/dynamics"
ROWS = [
    {"id": "a", "question": "2kg质点受6N力，求加速度", "prediction": "a=F*m=12", "answer": "a=3", "context": ""},
    {"id": "b", "question": "小车合力与加速度的关系", "prediction": "a=Fm", "answer": "a=F/m", "context": ""},
]


def mock_model(stage, key, data):
    if stage == "taxonomy":
        return copy.deepcopy(TAXONOMY)
    if stage == "audit":
        return {"errors": [{"quote": data["sample"]["prediction"], "message": "牛顿第二定律除以质量",
                            "correction": "a=F/m", "topic_key": TOPIC, "error_type": "concept"}], "uncertain": []}
    if stage in {"group_errors", "cluster_rules"}:
        return {"new_groups": [{"key": "new1", "name": "合力与质量", "summary": "合力决定加速度"}] if not data["groups"] else [],
                "assignments": [{"item_id": x["id"], "group_id": data["groups"][0]["id"] if data["groups"] else "new1"} for x in data["items"]]}
    assert stage == "distill"
    return {"rules": [{"title": "牛顿第二定律", "summary": "合力除以质量", "trigger": "惯性系恒质量质点",
        "check_logic": "合外力等于质量乘加速度，不是加速度等于合力乘质量", "error_type": "concept",
        "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
        "support": [{"error_id": e["id"], "reason": "错误把质量乘在合力上"} for e in data["errors"]]}], "rejected": []}


def test_full_build_has_traceable_immutable_leaves_and_runtime_tools():
    stages = []
    def call(stage, key, payload):
        stages.append(stage)
        return mock_model(stage, key, payload)
    result = p.build_catalog(ROWS, call, batch_size=1)
    assert result["counts"] == dict(samples=2, errors=2, uncertain=0, rules=1, rejected_or_insufficient_support=0, clusters=1)
    assert stages.count("distill") == 1
    assert set(stages) == set(p.PROMPTS)
    leaf = result["provenance"]["rules"][0]["leaf"]
    corpus = p.Corpus(result["catalog"], tree=True)
    assert corpus.call("read_rules", {"rule_ids": [leaf["rule_id"]]}, max_chars=100000) == [leaf]
    root = corpus.call("read_tree", {"node_id": "root"}, max_chars=100000)
    assert root["children"] == ["domain:mechanics"]
    assert set(corpus.leaves[leaf["rule_id"]]) == set(p.LEAF_FIELDS)
    assert corpus.search("质量", "bm25", 6)
    assert result["validation"]["valid"]


def test_grouping_first_schedule_preserves_every_unique_request_and_final_build():
    from scripts.resume_120b_construction import group_before_teacher
    taxonomy = copy.deepcopy(TAXONOMY)
    taxonomy["domains"][0]["topics"].append({"id": "statics", "name": "静力学", "summary": "平衡"})
    rows = copy.deepcopy(ROWS) + [
        dict(ROWS[0], id="c", question="不同题干甲"),
        dict(ROWS[1], id="d", question="不同题干乙")]

    def cached_model():
        cache, order = {}, []
        def call(stage, key, payload):
            fingerprint = p.digest([stage, key, payload])
            if (stage, key) in cache:
                assert cache[stage, key][0] == fingerprint
                return copy.deepcopy(cache[stage, key][1])
            if stage == "taxonomy":
                value = copy.deepcopy(taxonomy)
            else:
                value = mock_model(stage, key, payload)
                if stage == "audit" and key in {"c", "d"}:
                    value["errors"][0]["topic_key"] = "mechanics/statics"
            cache[stage, key] = fingerprint, copy.deepcopy(value)
            order.append(stage)
            return value
        return call, cache, order

    standard, standard_cache, _ = cached_model()
    expected = p.build_catalog(rows, standard, batch_size=1)
    scheduled, scheduled_cache, order = cached_model()
    actual = group_before_teacher(rows, scheduled, p.build_catalog, batch_size=1)
    assert actual == expected
    assert scheduled_cache == standard_cache
    assert max(i for i, stage in enumerate(order) if stage == "group_errors") < order.index("distill")
    assert order.count("distill") == 2


def test_no_error_is_success_and_does_not_call_distiller():
    seen = []
    def call(stage, key, data):
        seen.append(stage)
        return TAXONOMY if stage == "taxonomy" else {"errors": [], "uncertain": []}
    result = p.build_catalog(ROWS, call)
    assert result["counts"]["rules"] == 0 and "distill" not in seen


def test_same_question_different_answers_not_independent_support():
    rows = copy.deepcopy(ROWS); rows[1]["question"] = rows[0]["question"]
    seen = []
    def call(stage, key, data):
        seen.append(stage)
        return mock_model(stage, key, data)
    result = p.build_catalog(rows, call)
    assert "distill" not in seen and result["counts"]["rejected_or_insufficient_support"] == 2


def test_bad_quote_and_unknown_topic_stop_before_distillation():
    for field, bad in [("quote", "NOT ORIGINAL"), ("topic_key", "other/topic")]:
        def call(stage, key, data):
            value = mock_model(stage, key, data)
            if stage == "audit": value["errors"][0][field] = bad
            return value
        with pytest.raises(ValueError): p.build_catalog(ROWS, call)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown"])
def test_cluster_membership_cannot_hide_leaves(fault):
    def call(stage, key, data):
        value = mock_model(stage, key, data)
        if stage == "cluster_rules":
            if fault == "missing": value["assignments"] = []
            elif fault == "duplicate": value["assignments"] *= 2
            else: value["assignments"][0]["item_id"] = "invented"
        return value
    with pytest.raises(ValueError): p.build_catalog(ROWS, call)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "invented", "single", "program"])
def test_distiller_cannot_invent_or_drop_support(fault):
    def call(stage, key, data):
        value = mock_model(stage, key, data)
        if stage == "distill":
            rule = value["rules"][0]
            if fault in {"missing", "single"}: rule["support"].pop()
            elif fault == "duplicate": rule["support"].append(rule["support"][0])
            elif fault == "invented": rule["support"][0]["error_id"] = "invented"
            else: rule["symbolic_hint"]["primitive"] = "unverified_program"
        return value
    with pytest.raises(ValueError): p.build_catalog(ROWS, call)


def test_distiller_can_reject_false_audit_claims_without_new_rules():
    def call(stage, key, data):
        if stage == "distill":
            return {"rules": [], "rejected": [{"error_id": e["id"], "reason": "不足以泛化"} for e in data["errors"]]}
        return mock_model(stage, key, data)
    result = p.build_catalog(ROWS, call)
    assert result["counts"]["rules"] == 0 and len(result["provenance"]["rejected"]) == 2


def test_isolation_uses_question_not_id_and_removes_number_variants():
    q = "一个物体质量为2kg，在水平桌面上沿直线运动，受到恒定水平外力，桌面摩擦系数给定，试讨论速度与时间关系以及动量定理的适用条件。"
    rows = [{"id": "different", "question": q.replace("2kg", "5kg"), "prediction": "x", "answer": "historical"}, ROWS[0]]
    kept, blocked = p.isolate_sources(rows, [{"id": "test", "question": q, "private_label": "not a prompt"}])
    assert [x["id"] for x in kept] == ["a"]
    assert blocked[0]["matches"][0]["reason"] == "numeric_template"
    kept, blocked = p.isolate_sources([ROWS[0]], [{"id": "different", "question": ROWS[0]["question"] + "\n"}])
    assert not kept and blocked[0]["matches"][0]["reason"] == "normalized_exact"


def test_distill_requires_same_120b_identity_and_harmony_completion(monkeypatch):
    config = {"model": runner.r.EXPECTED_MODEL, "max_tokens": 16384}
    raw = {"id": "x", "model": config["model"]}
    seen = []
    def decode(response, **kwargs):
        seen.append(kwargs)
        if response.get("incomplete"):
            raise ValueError("Harmony incomplete")
        return {"content": "{}"}
    monkeypatch.setattr(runner.harmony, "decode_response", decode)
    assert runner.decode({"response": raw}, "distill", config) == {}
    assert seen == [dict(model=config["model"], max_tokens=16384, native_tools=None)]
    for invalid in [dict(raw, model="gpt-6.1-sol"), dict(raw, id=""), dict(raw, incomplete=True)]:
        with pytest.raises(ValueError): runner.decode({"response": invalid}, "distill", config)
    with pytest.raises(ValueError, match="Failed request"):
        runner.decode({"response": raw, "error": "saved failure"}, "distill", config)


def test_runtime_changes_only_catalog_binding_and_new_reference(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(p.build_catalog(ROWS, mock_model)["catalog"]))
    (tmp_path / "release_manifest.json").write_text("{}")
    monkeypatch.setattr(runtime, "verified_catalog", lambda path: (catalog, {"configuration_sha256": "build"}))
    old_catalog = runtime.runner.CATALOG
    old_hash = runtime.r.EXPECTED_CATALOG_SHA256
    with runtime.configured(tmp_path):
        config = runtime.runner.configuration()
        assert config["catalog_sha256"] == runtime.r.sha256_file(catalog)
        assert config["reference_version"] == "sol_high_4pass_99_assistant_review_v2"
        assert config["limits"]["max_requests_per_sample"] == 12
        assert config["arms"] == ["C0p", "C4"] and len(config["sample_ids"]) == 50
        assert "distill" not in config["prompts"]["rules"]
    assert runtime.runner.CATALOG == old_catalog and runtime.r.EXPECTED_CATALOG_SHA256 == old_hash


def test_runtime_refuses_incomplete_or_tampered_build(tmp_path):
    (tmp_path / "release_manifest.json").write_text(json.dumps({"usable_for_runtime": False, "human_gold": False}))
    with pytest.raises(ValueError): runtime.verified_catalog(tmp_path)


def test_journal_routes_every_stage_locally_without_keys_and_replays_success(tmp_path, monkeypatch):
    calls = []
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "BACKUP"):
        monkeypatch.delenv(name, raising=False)
    def budget(*args, **kwargs):
        assert not kwargs.get("quota_probe")
        return object()
    monkeypatch.setattr(runner, "BatchBudget", budget)
    monkeypatch.setattr(runner.harmony, "decode_response", lambda raw, **kw: {"content": raw["test_content"]})
    class Transport:
        def __init__(self, **kw):
            self.kw = kw; self.requests = 0
        def post(self, endpoint, body):
            calls.append((self.kw["base"], self.kw["model"], endpoint))
            assert self.kw["key"] == "local" and self.kw["reasoning_effort"] == "medium"
            assert body["reasoning_effort"] == "medium" and body["return_token_ids"]
            value = {"test": self.record_metadata["build_stage"]}
            response = {"id": "test", "model": self.kw["model"], "status": "completed",
                "reasoning": {"effort": self.kw["reasoning_effort"]}, "usage": {"input_tokens": 10, "output_tokens": 10},
                "test_content": json.dumps(value), "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}
            path = self.kw["output"] / f"request_{self.requests+1:03d}.json"
            runner.r.write_json(path, {"request": body, "response": response, "client_metadata": self.record_metadata})
        def responses(self, **kwargs):
            return self.post("/responses", kwargs)
    monkeypatch.setattr(runner, "Transport", Transport)
    config = {"reasoning_effort": "medium", "model": runner.r.EXPECTED_MODEL,
              "distill_max_requests": 1, "api_base": "http://127.0.0.1:18083/v1", "max_requests": 10,
              "max_tokens": 16384, "max_request_input_bytes": 100000, "attempts_per_job": 3}
    journal = runner.Journal(tmp_path, config)
    for stage in p.PROMPTS:
        assert journal(stage, "one", {}) == {"test": stage}
    assert calls == [(config["api_base"], config["model"], "/chat/completions")] * len(p.PROMPTS)
    restarted = runner.Journal(tmp_path, config)
    for stage in p.PROMPTS:
        assert restarted(stage, "one", {}) == {"test": stage}
    assert len(calls) == len(p.PROMPTS)  # no duplicate calls after restart
    with pytest.raises(ValueError, match="Distillation request cap"):
        restarted("distill", "second-distillation", {})
    assert len(calls) == len(p.PROMPTS)
    with pytest.raises(ValueError, match="Cached job input mismatch"):
        restarted("audit", "one", {"changed_input": True})
    path = next(tmp_path.glob("arm_audit/*/request_*.json")); path.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        restarted("audit", "one", {})


def test_prepare_keeps_heldout_labels_out_and_binds_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.json"; source.write_text(json.dumps(ROWS))
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text(json.dumps([{"id": "h", "question": "完全不同的热力学问题", "answer": "HELDOUT_SECRET", "errors": ["never expose"]}]))
    monkeypatch.setattr(runner, "BENCHMARK_FILES", {str(exclusion): 1})
    args = SimpleNamespace(source=source, exclude=None, size=2, seed=17,
      source_model="historical-30b", distill_max_requests=5, output=tmp_path / "run")
    runner.prepare(args)
    runner.verify(args.output)
    original_plan = runner.read(args.output / 'plan.json')
    altered = json.loads(json.dumps(original_plan))
    altered['configuration']['embedding_model'] = 'unexpected-embedding-model'
    altered['configuration_sha256'] = p.digest(altered['configuration'])
    runner.r.write_json(args.output / 'plan.json', altered)
    with pytest.raises(ValueError, match='frozen local 120B'):
        runner.verify(args.output)
    runner.r.write_json(args.output / 'plan.json', original_plan)
    for name in ("inputs.json", "bench_questions.json", "plan.json", "isolation.json"):
        assert "HELDOUT_SECRET" not in (args.output / name).read_text()
    assert set(runner.read(args.output / "inputs.json")[0]) == {"id", "question", "context", "prediction", "answer"}
    (args.output / "inputs.json").write_text("[]")
    with pytest.raises(ValueError, match="inputs changed"):
        runner.verify(args.output)


@pytest.fixture
def isolated_preparation(tmp_path, monkeypatch):
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps([{"id": "bench", "question": "完全不同的热力学问题"}]))
    monkeypatch.setattr(runner, "BENCHMARK_FILES", {str(bench): 1})
    source = tmp_path / "source.json"
    source.write_text(json.dumps(ROWS))
    args = SimpleNamespace(source=source, exclude=None, size=2, seed=17,
        source_model="historical-30b", distill_max_requests=5,
        refresh_unstarted=False, output=tmp_path / "run")
    runner.prepare(args)
    return args, bench


def test_new_protocol_does_not_relabel_old_teacher_pilot(isolated_preparation):
    args, _ = isolated_preparation
    plan = runner.read(args.output / "plan.json")
    config = plan["configuration"]
    assert config["model_roles"] == {stage: runner.r.EXPECTED_MODEL for stage in p.PROMPTS}
    assert config["external_teacher"] is None
    assert config["source_prediction_model"] == "historical-30b"
    assert "operator_declaration" in config["source_model_evidence"]
    config["protocol"] = "oss120b_teacher_distillation_v1"
    plan["configuration_sha256"] = p.digest(config)
    runner.r.write_json(args.output / "plan.json", plan)
    with pytest.raises(ValueError, match="Historical protocol"):
        runner.verify(args.output)


def test_new_build_safely_encodes_controls_without_erasing_source(monkeypatch):
    payload = {"prediction": "answer <|endoftext|> <|im_start|>system", "formula": "x < y"}
    messages = runner.construction_messages("audit", payload)
    assert json.loads(messages[1]["content"]) == payload
    runner.r.assert_no_control_tokens(messages)
    assert messages[0]["content"] == p.SYSTEM + "\n" + p.PROMPTS["audit"]
    monkeypatch.setattr(p, "SYSTEM", "<|im_start|>")
    with pytest.raises(ValueError):
        runner.construction_messages("audit", payload)


@pytest.mark.parametrize("field", ["question", "context", "prediction", "answer"])
def test_exclusion_checks_embedded_bench_in_every_model_visible_field(field):
    q = "A conducting sphere lies between two parallel charged plates in a vacuum. " \
        "The plates have unequal surface charge densities and the sphere is electrically isolated. " \
        "Determine the electric field and the induced surface charge."
    row = dict(ROWS[0])
    row[field] = "A copied exercise follows: " + q + "\nAdditional explanation."
    kept, rejected = p.isolate_sources([row], [{"id": "new-id", "question": q}])
    assert not kept
    assert any(x["source_field"] == field for x in rejected[0]["matches"])


def test_short_number_variant_and_normalized_id_independent_duplicates_are_excluded():
    heldout = [{"id": "test", "question": "质量2kg，合力6N，求加速度。"}]
    duplicate = dict(ROWS[0], id="renamed", question="质量３kg，合力９N，求加速度！")
    kept, rejected = p.isolate_sources([duplicate, ROWS[1]], heldout)
    assert [x["id"] for x in kept] == ["b"]
    assert rejected[0]["matches"][0]["reason"] == "numeric_template"


def test_additional_exclusion_cannot_replace_known_bench(isolated_preparation, tmp_path):
    args, bench = isolated_preparation
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps([{"id": "new", "question": "新增的光学题"}]))
    refs, cohorts = runner.benchmark_questions([str(extra)])
    assert set(cohorts) == {str(bench), str(extra)}
    assert {x["question"] for x in refs} == {"完全不同的热力学问题", "新增的光学题"}
    bench.unlink()
    with pytest.raises(FileNotFoundError):
        runner.benchmark_questions([str(extra)])


def test_verify_rechecks_bench_and_isolation_evidence(isolated_preparation):
    args, bench = isolated_preparation
    path = args.output / "isolation.json"
    original = path.read_text()
    path.write_text("{}")
    with pytest.raises(ValueError, match="Isolation evidence changed"):
        runner.verify(args.output)
    path.write_text(original)
    bench.write_text(json.dumps([{"id": "bench", "question": ROWS[0]["question"]}]))
    with pytest.raises(ValueError, match="benchmark inputs changed"):
        runner.verify(args.output)


def test_hashed_but_contaminated_selection_still_fails_replay(isolated_preparation):
    args, bench = isolated_preparation
    rows = runner.read(args.output / "inputs.json")
    rows[0]["question"] = runner.read(bench)[0]["question"]
    runner.r.write_json(args.output / "inputs.json", rows)
    plan = runner.read(args.output / "plan.json")
    plan["configuration"]["selected_sha256"] = p.digest(rows)
    plan["configuration_sha256"] = p.digest(plan["configuration"])
    runner.r.write_json(args.output / "plan.json", plan)
    with pytest.raises(ValueError, match="does not replay"):
        runner.verify(args.output)


def test_refresh_unstarted_cannot_erase_execution_evidence(isolated_preparation):
    args, _ = isolated_preparation
    args.refresh_unstarted = True
    runner.prepare(args)  # Preparation-only replacement is safe.
    evidence = args.output / "arm_audit" / "example" / "request_001.json"
    runner.r.write_json(evidence, {"error": "saved failure"})
    original = (args.output / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="execution artifacts"):
        runner.prepare(args)
    assert evidence.exists() and (args.output / "plan.json").read_bytes() == original


def test_sampling_and_support_do_not_treat_numeric_variants_as_independent():
    a = dict(ROWS[0], question="质量2kg，合力6N，求加速度")
    variant = dict(ROWS[0], id="a2", question="质量3kg，合力9N，求加速度")
    selected = p.select_sources([a, variant, ROWS[1]], 2, 17)
    assert len({p.question_template(x["question"]) for x in selected}) == 2
    def call(stage, key, data):
        assert stage != "distill"
        return mock_model(stage, key, data)
    result = p.build_catalog([a, variant], call)
    assert result["counts"]["rules"] == 0 and result["counts"]["rejected_or_insufficient_support"] == 2


def test_runtime_rejects_release_that_drops_isolation_artifacts(tmp_path):
    (tmp_path / "release_manifest.json").write_text(json.dumps(
        {"usable_for_runtime": True, "human_gold": False, "files": {}}))
    with pytest.raises(ValueError, match="omits required"):
        runtime.verified_catalog(tmp_path)


def test_runtime_rechecks_actual_formal_questions(isolated_preparation, tmp_path, monkeypatch):
    args, _ = isolated_preparation
    for name, value in p.build_catalog(runner.read(args.output / "inputs.json"), mock_model).items():
        runner.r.write_json(args.output / (name + ".json"), value)
    runner.r.write_json(args.output / "runtime.json", {"fixture": "offline, no model calls"})
    plan = runner.read(args.output / "plan.json")
    config = plan["configuration"]
    release = {"usable_for_runtime": True, "human_gold": False,
        "status": "candidate_catalog_complete_not_default_release",
        "files": {n: runner.r.sha256_file(args.output / n) for n in runner.RELEASE_FILES},
        "configuration_sha256": plan["configuration_sha256"],
        "online_model": runner.r.EXPECTED_MODEL, "model_roles": runner.model_roles(), "external_teacher": None,
        "source_prediction_model": config["source_prediction_model"],
        "source_model_evidence": config["source_model_evidence"], "data_isolation": {
            "passed": True, "selected_overlap_count": 0, "policy_version": p.ISOLATION_POLICY["version"],
            "bench_questions_sha256": config["bench_questions_sha256"],
            "isolation_sha256": config["isolation_sha256"]}}
    runner.r.write_json(args.output / "release_manifest.json", release)
    formal = tmp_path / "formal.json"
    formal.write_text(json.dumps([{"id": "test", "question": "电容器的能量"}]))
    monkeypatch.setattr(runtime.runner, "INPUT", formal)
    assert runtime.verified_catalog(args.output)[0] == args.output / "catalog.json"
    formal.write_text(json.dumps([{"id": "another-id", "question": ROWS[0]["question"]}]))
    with pytest.raises(ValueError, match="actual evaluation questions"):
        runtime.verified_catalog(args.output)


def test_usage_report_separates_unknown_and_initial_root_from_navigation():
    trace = {"model_requests": 2, "response_evidence": [
        {"usage": {"prompt_tokens": 13, "completion_tokens": 17}}, {"usage": None}],
        "tool_events": [{"tool": "read_tree", "arguments": {"node_id": "root", "note": "Initial tree context"}}],
        "read_rule_ids": []}
    summary = runtime.trace_summary([{"trace": trace}, {}])
    assert summary["records_missing_trace"] == 1
    assert summary["usage_known_responses"] == summary["usage_unknown_responses"] == 1
    assert summary["known_output_tokens"] == 17
    assert summary["samples_with_active_tree_navigation"] == 0
    trace["tool_events"].append({"tool": "read_tree", "arguments": {"node_id": "domain:mechanics"}})
    trace["read_rule_ids"].append("actual-leaf")
    assert runtime.trace_summary([{"trace": trace}])["samples_with_active_tree_navigation"] == 1
    assert runtime.trace_summary([{"trace": trace}])["samples_with_full_rule_read"] == 1


def test_new_runtime_scoring_uses_new_reference_and_rule_minus_pure(tmp_path, monkeypatch):
    rows = [{"id": str(i), "prediction": "prefix wrong_expression suffix",
             "physics_error_gt": [{"error_id": str(i) + "/new_gt", "error_text": "wrong relation",
                "answer_quote": "wrong_expression", "start_char": 7, "end_char": 23,
                "span_valid": True, "paragraph_index": -1, "paragraph_valid": False}]} for i in range(50)]
    groups = {arm: [{"id": row["id"], "status": "completed", "protocol_audit": {"passed": True},
                    "diagnostics": ([{"message": "wrong relation", "evidence": {"quote": "wrong_expression"}}] if arm == "C4" else [])}
                   for row in rows] for arm in ("C0p", "C4")}
    monkeypatch.setattr(runtime.runner, "validate_plan", lambda out: {"configuration_sha256": "test"})
    monkeypatch.setattr(runtime.runner, "frozen_rows", lambda: rows)
    monkeypatch.setattr(runtime.runner, "collect", lambda *args: groups)
    monkeypatch.setattr(runtime.evaluation, "reference_data", lambda root: ([], rows, {x["id"]: [] for x in rows}))
    monkeypatch.setattr(runtime.runner, "protocol", SimpleNamespace(ARMS=("C0p", "C4")))
    report = runtime.score(tmp_path)
    assert report["scores_all_50"]["C4"]["summary"]["matched_gt_errors"] == 50
    assert report["scores_all_50"]["C0p"]["summary"]["precision_defined"] is False
    assert report["rule_minus_pure_paired_bootstrap"]["f1"]["delta_pp"] == 100
    assert not report["human_gold"] and not report["independent_test"]
