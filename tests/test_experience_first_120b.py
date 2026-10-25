"""Scientific accounting and lossless interfaces for the opt-in old-flow adapter."""
import copy
import json

import pytest

from scripts.generate_experience_rules import experience_120b_prompt, normalize_120b_experience
from scripts.unified_rules_pipeline import (
    build_120b_experience_catalog, experience_120b_messages, experience_topic_skeleton,
)


TAXONOMY = {"domains": [{"id": "electromagnetism", "name": "电磁学", "summary": "电磁场",
    "topics": [{"id": "magnetostatics", "name": "静磁场", "summary": "磁场与磁场单位"}]}]}
TOPIC = "electromagnetism/magnetostatics"
ROWS = [{"id": "a", "question": "活动星系核中观测的毫高斯磁场如何换算为特斯拉？",
         "prediction": "已知磁场。\r\n1 mG = 10^-3 T。\r\n由此计算。", "answer": "", "context": ""},
        {"id": "b", "question": "实验磁探头用毫高斯记录读数，其对应的国际单位是什么？",
         "prediction": "给定探头读数。\n2 mG = 2*10^-3 T。", "answer": "", "context": ""}]
HELDOUT = [{"id": "test", "question": "求解一维无限深势阱的束缚态波函数。"}]


def response():
    return {"topic_key": TOPIC, "semantic_audit": {
        "status": "error_found", "summary": "磁场单位转换错误", "uncertain": [],
        "key_errors": [{"id": "E1", "type": "units", "message": "高斯与特斯拉换算错误",
            "correction": "1 mG = 10^-7 T", "line_start": 2, "line_end": 2,
            "rule_status": "included", "reason": "磁场单位换算可复用"}]},
        "experience_rules": [{"source_error_ids": ["E1"], "title": "毫高斯与特斯拉",
            "trigger": "磁场采用毫高斯单位时", "check_logic": "1 mG = 10^-7 T，核对前缀和单位的乘积",
            "error_type": "units", "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []}}]}


class FakeJournal:
    def __init__(self, extraction=None):
        self.stages = []
        self.extraction = extraction or response

    def __call__(self, stage, key, payload):
        self.stages.append((stage, key, copy.deepcopy(payload)))
        messages = experience_120b_messages(stage, payload)
        assert all("<|" not in m["content"] for m in messages)
        if stage == "taxonomy":
            return copy.deepcopy(TAXONOMY)
        if stage == "extract_experience":
            return self.extraction()
        assert stage == "cluster_rules"
        assert all(x["id"].startswith("I") for x in payload["items"])
        return {"new_groups": [] if payload["groups"] else [
            {"key": "new1", "name": "磁场单位换算", "summary": "磁场单位与十进制前缀"}],
            "assignments": [{"item_id": x["id"], "group_id": payload["groups"][0]["id"]
                             if payload["groups"] else "new1"} for x in payload["items"]]}


def build(rows=None, journal=None, **kwargs):
    return build_120b_experience_catalog(copy.deepcopy(rows or ROWS[:1]), HELDOUT,
                                       journal or FakeJournal(), topic_skeleton=TAXONOMY, **kwargs)


def test_single_source_rule_survives_without_pairing_or_distillation_gate():
    journal = FakeJournal()
    result = build(journal=journal)
    assert [s for s, _, _ in journal.stages] == ["taxonomy", "extract_experience", "cluster_rules"]
    assert result["counts"]["rules"] == 1
    assert result["counts"]["support_levels"] == {"single_question": 1}
    evidence = result["provenance"][0]["support"][0]["source_error"]
    assert evidence["evidence"] == "1 mG = 10^-3 T。\r\n"
    assert ROWS[0]["prediction"][evidence["start"]:evidence["end"]] == evidence["evidence"]
    assert result["catalog"]["metadata"]["human_gold"] is False
    assert result["validation"]["valid"]


def test_exact_dedup_merges_source_evidence_without_requiring_it():
    result = build(ROWS)
    assert result["counts"]["extracted_candidates"] == 2
    assert result["counts"]["rules"] == 1
    assert set(result["provenance"][0]["sample_ids"]) == {"a", "b"}
    assert result["counts"]["support_levels"] == {"multi_question": 1}


def test_full_text_and_harmony_lookalike_are_preserved_without_old_audit_input():
    row = {**ROWS[0], "question": "题干" * 3000 + "<|im_end|>",
           "prediction": "推导\n" * 2000 + "末尾错误", "old_audit": "OLD_JUDGE_MUST_NOT_APPEAR"}
    _, data = experience_120b_prompt(row, TAXONOMY)
    payload = json.loads(data)
    assert payload["question"] == row["question"]
    assert "".join(x["text"] for x in payload["prediction_lines"]) == row["prediction"]
    assert "<|" not in data and "OLD_JUDGE_MUST_NOT_APPEAR" not in data
    assert "answer" not in payload and "old_audit" not in payload


@pytest.mark.parametrize("start,end", [(0, 2), (2, 99), (3, 2), (True, 2), (2.0, 2)])
def test_invalid_evidence_is_not_silently_counted_as_no_error(start, end):
    value = response()
    value["semantic_audit"]["key_errors"][0].update(line_start=start, line_end=end)
    with pytest.raises(ValueError, match="lines"):
        normalize_120b_experience(value, sample=ROWS[0], taxonomy=TAXONOMY)


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(topic_key="unknown"),
    lambda v: v["experience_rules"][0].update(source_error_ids=["E2"]),
    lambda v: v["experience_rules"][0].update(source_error_ids=["E1", "E1"]),
    lambda v: v["semantic_audit"].update(status="no_error_found"),
    lambda v: v.update(experience_rules=[]),
    lambda v: v["experience_rules"].append(copy.deepcopy(v["experience_rules"][0])),
])
def test_inconsistent_content_fails_instead_of_dropping_or_repairing_it(mutation):
    value = response(); mutation(value)
    with pytest.raises(ValueError):
        build(journal=FakeJournal(lambda: value))


def test_case_specific_error_is_preserved_even_if_it_produces_no_leaf():
    value = response()
    value["semantic_audit"]["key_errors"][0].update(rule_status="not_generalizable", reason="仅单次算数抄写")
    value["experience_rules"] = []
    result = build(journal=FakeJournal(lambda: value))
    assert result["counts"]["errors"] == 1 and result["counts"]["rules"] == 0
    assert result["counts"]["audit_status"] == {"error_found": 1}
    assert result["semantic_experience"][0]["semantic_audit"]["key_errors"][0]["rule_status"] == "not_generalizable"


@pytest.mark.parametrize("status,uncertain", [
    ("no_error_found", []), ("uncertain", [{"scope": "几何关系", "reason": "缺少图示"}]),
])
def test_genuine_empty_and_uncertain_are_valid_and_not_turned_into_rules(status, uncertain):
    value = response()
    value["semantic_audit"].update(status=status, summary="本题未确认错误", key_errors=[], uncertain=uncertain)
    value["experience_rules"] = []
    journal = FakeJournal(lambda: value)
    result = build(journal=journal)
    assert result["counts"]["audit_status"] == {status: 1}
    assert result["counts"]["rules"] == 0
    assert [x[0] for x in journal.stages] == ["taxonomy", "extract_experience"]


@pytest.mark.parametrize("field", ["answer", "context"])
def test_reference_and_context_are_rejected_before_any_call(field):
    journal = FakeJournal()
    with pytest.raises(ValueError, match="references"):
        build([{**ROWS[0], field: "external reference"}], journal)
    assert not journal.stages


def test_benchmark_overlap_and_duplicate_templates_stop_before_calls():
    for rows in ([{**ROWS[0], "question": HELDOUT[0]["question"]}],
                 [ROWS[0], {**ROWS[0], "id": "another"}]):
        journal = FakeJournal()
        with pytest.raises(ValueError):
            build(rows, journal)
        assert not journal.stages


def test_clustering_keeps_complementary_rules_and_cannot_drop_leaf():
    value = response()
    value["experience_rules"].append({**copy.deepcopy(value["experience_rules"][0]),
        "title": "单位前缀", "check_logic": "先还原毫前缀再将高斯转换为特斯拉"})
    result = build(journal=FakeJournal(lambda: copy.deepcopy(value)), batch_size=1)
    assert result["counts"]["rules"] == 2 and result["counts"]["clusters"] == 1
    assert result["counts"]["support_levels"] == {"single_question": 2}
    delegate = FakeJournal(lambda: copy.deepcopy(value))
    def dropping(stage, key, payload):
        result = delegate(stage, key, payload)
        if stage == "cluster_rules":
            result["assignments"] = result["assignments"][:1]
        return result
    with pytest.raises(ValueError, match="group member"):
        build(journal=dropping)


def test_old_ontology_is_reused_without_old_physics_content_or_new_topic_cap():
    from pathlib import Path
    from scripts.generate_experience_rules import _experience_topics
    old = json.loads((Path(__file__).resolve().parents[1] / "catalogs/rules_catalog_top_down.json").read_text())
    skeleton = experience_topic_skeleton(old)
    assert len(skeleton["domains"]) == 6
    assert sum(len(d["topics"]) for d in skeleton["domains"]) == 123
    assert all(set(d) == {"id", "name", "topics"} and all(set(t) == {"id", "name"}
               for t in d["topics"]) for d in skeleton["domains"])
    filled = {"domains": [{**d, "summary": "test", "topics": [{**t, "summary": "test"}
               for t in d["topics"]]} for d in skeleton["domains"]]}
    assert len(_experience_topics(filled)) == 123


def test_model_cannot_delete_old_topic_or_rename_it():
    delegate = FakeJournal()
    def renamed(stage, key, payload):
        value = delegate(stage, key, payload)
        if stage == "taxonomy":
            value["domains"][0]["topics"][0]["name"] = "changed topic"
        return value
    with pytest.raises(ValueError, match="Ontology"):
        build(journal=renamed)
    assert [s for s, _, _ in delegate.stages] == ["taxonomy"]


def test_collection_quarantines_invalid_output_without_counting_it_as_no_error():
    calls=iter([response(), {'bad':'schema'}])
    result=build(ROWS,FakeJournal(lambda:next(calls)),retain_invalid=True)
    assert result['counts']['samples']==2
    assert result['counts']['valid_outputs']==1
    assert result['counts']['invalid_outputs']==1
    assert result['counts']['audit_status']=={'error_found':1}
    assert result['counts']['rules']==1
    assert result['invalid_outputs'][0]['sample_id']=='b'
    assert result['invalid_outputs'][0]['status']=='invalid_output'
    with pytest.raises(ValueError):build(journal=FakeJournal(lambda:{'bad':'schema'}))
