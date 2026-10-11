import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import run_gpt_oss_selfcheck as r
from scripts import continuous_context_v2 as p
from scripts import gpt_oss_harmony_transport as transport
from scripts import run_continuous_context_v2 as runner

CATALOG = json.loads(runner.CATALOG.read_text())
ROW = {"id": "external-unit-test", "question": "检查以下表达。",
       "prediction": r"原句：x = \frac{a}{b}。另一个独立句子。"}
BAD = {"diagnostics": [{"message": "A fixed diagnostic reason.",
                       "evidence": {"quote": "x=a/b"}, "rule_refs": []}]}
GOOD_QUOTE = r"x = \frac{a}{b}"
PATCH = {"quote_patches": [{"index": 0, "quote": GOOD_QUOTE}]}


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [0] * sum(max(1, len(x.get("content", "")) // 5) for x in messages)

    def encode(self, text, **kwargs):
        return list(text)


@pytest.fixture
def isolated_reference(tmp_path, monkeypatch):
    """Plan/scorer tests do not depend on ignored private benchmark files."""
    root = tmp_path / "synthetic_reference"
    inputs = root / "frozen50/inputs.json"
    rows = [{"id": f"test_{i}", "question": "Synthetic fixture", "prediction": "Fixture: x = 2"} for i in range(50)]
    # The frozen legacy scorer loses offset zero via `or -1`; this test checks
    # integration without changing that scoring behavior for the experiment.
    claims = [{"claim_id": f"claim_{i}", "reported_quote": "x = 2", "reported_offset_matches_original": True,
               "reported_start_char": 9, "reported_end_char": 14, "unreviewed_error_claim": "Fixture only"}
              for i in range(187)]
    mapping = {c["claim_id"]: {"sample_id": rows[i % 50]["id"]} for i, c in enumerate(claims)}
    r.write_json(inputs, rows)
    r.write_json(root / "review/blind_review.json", {"reference_claims": claims})
    r.write_json(root / "review/operator_map.json", {"reference_claims": mapping})
    r.write_json(root / runner.SCORING_FILES[0], {"sample": {
        "blind_review_sha256": r.sha256_file(root / "review/blind_review.json"),
        "operator_mapping_sha256": r.sha256_file(root / "review/operator_map.json"),
        "frozen_inputs_sha256": r.sha256_file(inputs)}})
    monkeypatch.setattr(runner, "REFERENCE", root)
    monkeypatch.setattr(runner, "INPUT", inputs)
    monkeypatch.setattr(r, "EXPECTED_INPUT_SHA256", r.sha256_file(inputs))


def run(tmp_path, arm, sequence, *, row=None, signature="test", **limits):
    calls = []

    def respond(client, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        value = sequence[len(calls) - 1]
        r.write_json(kwargs["request_path"], {"request": r.request_body(
            kwargs["model"], kwargs["messages"], kwargs["max_tokens"], "vllm", kwargs["native_tools"])})
        kwargs["raw_path"].write_text("mock raw response\n")
        if isinstance(value, Exception):
            r.write_json(kwargs["response_path"], {"error": str(value)})
            raise value
        record = {"content": json.dumps(value), "model_ids": [r.EXPECTED_MODEL],
                  "usage": {"completion_tokens": 50, "prompt_tokens": 1},
                  "finish_reasons": ["stop"], "reasoning_content": "Visible tool analysis."}
        if "action" in value:
            record["tool_calls"] = [{"id": "call_" + str(len(calls)), "type": "function",
                "function": {"name": value["action"], "arguments": json.dumps(value["arguments"])}}]
        r.write_json(kwargs["response_path"], record)
        return record

    with patch.object(transport, "chat_response", respond):
        result = r.run_sample(ROW if row is None else row, arm=arm, catalog=CATALOG, tokenizer=Tokenizer(), client=None,
            endpoint="http://offline.invalid", model=r.EXPECTED_MODEL, root=tmp_path,
            signature=signature, limits=dict(runner.LIMITS, **limits))
    return result, calls


def test_same_base_no_reference_leak():
    row = dict(ROW, answer="SECRET ANSWER", physics_error_gt=["SECRET GOLD"])
    pure = p.messages(row, rules=False, tree_root=None)
    assisted = p.messages(row, rules=True, tree_root={"id": "root"})
    assert len(pure) == len(assisted) == 2
    assert assisted[0]["content"] == pure[0]["content"] + "\n\n" + p.RULES
    assert "SECRET" not in json.dumps(assisted)
    assert ROW["prediction"] in pure[1]["content"]


def test_pure_repair_is_quote_only_and_same_context(tmp_path):
    result, calls = run(tmp_path, "C0p", [BAD, PATCH])
    assert result["status"] == "completed"
    assert result["diagnostics"] == [dict(BAD["diagnostics"][0], evidence={"quote": GOOD_QUOTE})]
    assert result["trace"]["output_token_limit_reserved"] == 100
    assert all(x["native_tools"] is None for x in calls)
    assert calls[1]["messages"][:2] == calls[0]["messages"]
    assert result["quote_repair"]["attempts"] == 1
    assert json.loads(next(tmp_path.glob("samples/*/*/first_diagnosis.json")).read_text()) == BAD
    again, calls = run(tmp_path, "C0p", [])
    assert again == result and not calls


def test_rule_history_and_repair_tools_disabled(tmp_path):
    rid = r.Corpus(CATALOG, tree=True).ids[0]
    draft = copy.deepcopy(BAD)
    draft["diagnostics"][0]["rule_refs"] = [rid]
    result, calls = run(tmp_path, "C3", [
        {"action": "search_rules", "arguments": {"query": "牛顿", "mode": "bm25", "limit": 3}},
        {"action": "read_rules", "arguments": {"rule_ids": [rid]}},
        draft, PATCH])
    assert result["status"] == "completed" and result["trace"]["read_rule_ids"] == [rid]
    assert calls[0]["native_tools"] and calls[-1]["native_tools"] is None
    assert sum(x["role"] == "tool" for x in calls[-1]["messages"]) == 2
    assert any(x.get("content") == "Visible tool analysis." for x in calls[-1]["messages"])
    assert result["diagnostics"][0]["rule_refs"] == [rid]
    assert result["trace"]["output_token_limit_reserved"] == 200


@pytest.mark.parametrize("bad_patch", [
    {"diagnostics": []}, {"quote_patches": []},
    {"quote_patches": [{"index": 0, "quote": GOOD_QUOTE}, {"index": 0, "quote": GOOD_QUOTE}]},
    {"quote_patches": [{"index": 0, "quote": GOOD_QUOTE, "message": "Changed physics"}]},
    {"quote_patches": [{"index": False, "quote": GOOD_QUOTE}]},
    {"quote_patches": [{"index": 1, "quote": GOOD_QUOTE}]},
])
def test_bad_patch_cannot_edit_or_erase_diagnosis(tmp_path, bad_patch):
    result, calls = run(tmp_path, "C3", [BAD, bad_patch])
    assert result["diagnostics"] == BAD["diagnostics"]
    assert result["quote_repair"]["status"] == "rejected_patch"
    assert len(calls) == 2


@pytest.mark.parametrize("quote", [None, "", "still not in the original"])
def test_unresolved_is_not_filtered(tmp_path, quote):
    result, calls = run(tmp_path, "C0p", [BAD, {"quote_patches": [{"index": 0, "quote": quote}]}])
    assert result["status"] == "completed" and result["diagnostics"] == BAD["diagnostics"]
    assert result["quote_repair"]["unresolved_indices"] == [0] and len(calls) == 2


@pytest.mark.parametrize("limits", [{"output_tokens_per_sample": 50}, {"max_requests_per_sample": 1}])
def test_no_extra_budget_for_repair(tmp_path, limits):
    result, calls = run(tmp_path, "C0p", [BAD], **limits)
    assert result["diagnostics"] == BAD["diagnostics"]
    assert result["quote_repair"]["status"] == "not_attempted_budget"
    assert len(calls) == 1 and result["trace"]["output_token_limit_reserved"] == 50


def test_network_failure_retains_first_diagnosis_without_retry(tmp_path):
    result, calls = run(tmp_path, "C3", [BAD, TimeoutError("network timeout")])
    assert result["diagnostics"] == BAD["diagnostics"]
    assert result["quote_repair"]["status"] == "technical_failure"
    assert not runner.healthy_result(result)
    assert result["trace"]["pending_request"]
    again, calls = run(tmp_path, "C3", [])
    assert again == result and not calls


@pytest.mark.parametrize("arm", ["C0p", "C3"])
def test_empty_diagnosis_and_no_lookup_are_valid(tmp_path, arm):
    result, calls = run(tmp_path, arm, [{"diagnostics": []}])
    assert result["status"] == "completed" and result["protocol_audit"]["passed"]
    assert len(calls) == 1


def test_seventh_leaf_not_returned(tmp_path):
    ids = r.Corpus(CATALOG, tree=True).ids[:7]
    result, calls = run(tmp_path, "C3", [
        {"action": "read_rules", "arguments": {"rule_ids": ids}}, {"diagnostics": []}])
    assert not result["trace"]["read_rule_ids"]
    assert any("Distinct full-leaf budget" in x.get("content", "") for x in calls[-1]["messages"])


def test_failed_physics_generation_is_not_a_true_empty_result(tmp_path):
    result, calls = run(tmp_path, "C0p", [TimeoutError("upstream failed")])
    assert result["status"] == "failed" and not runner.healthy_result(result)
    again, calls = run(tmp_path, "C0p", [])
    assert again == result and not calls


def test_raw_harmony_does_not_infer_tools_and_accepts_constrained_final(monkeypatch):
    import sys
    import types
    module = types.ModuleType("vllm.entrypoints.harmony_utils")
    def message(channel, recipient, text):
        return SimpleNamespace(channel=channel, recipient=recipient,
                               content=[SimpleNamespace(text=text)])
    messages = [message("final", "<|constrain|>json", '{"diagnostics":[]}')]
    module.parse_output_into_messages = lambda tokens: SimpleNamespace(messages=messages, current_content="")
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.harmony_utils", module)
    raw = {"model": r.EXPECTED_MODEL, "usage": {"prompt_tokens": 10, "completion_tokens": 2},
           "choices": [{"token_ids": [5, 200002], "finish_reason": "stop"}]}
    args = dict(model=r.EXPECTED_MODEL, max_tokens=100, native_tools=runner.native_tools(True))
    assert transport.decode_response(raw, **args)["content"] == '{"diagnostics":[]}'
    messages[:] = [message("analysis", None, 'Call search_rules with {"query":"x"}')]
    raw["choices"][0]["token_ids"][-1] = 200012
    with pytest.raises(ValueError):
        transport.decode_response(raw, **args)
    messages[:] = [message("analysis", "functions.search_rules", '{"query":"x","mode":"bm25","limit":1}')]
    assert transport.decode_response(raw, **args)["tool_calls"][0]["function"]["name"] == "search_rules"
    raw["model"] = "another-model"
    with pytest.raises(ValueError):
        transport.decode_response(raw, **args)


def test_plan_freezes_prompts_and_full_50_without_calls(tmp_path, monkeypatch, isolated_reference):
    runner.prepare(tmp_path)
    plan = runner.validate_plan(tmp_path)
    assert len(plan["configuration"]["sample_ids"]) == 50
    assert plan["configuration"]["arms"] == ["C0p", "C3"]
    monkeypatch.setattr(p, "BASE", p.BASE + " changed")
    with pytest.raises(ValueError):
        runner.validate_plan(tmp_path)


def test_first_diagnosis_and_transport_evidence_are_checked(tmp_path):
    result, _ = run(tmp_path, "C0p", [BAD, PATCH])
    runner.audit_result(result)
    Path(result["first_diagnosis_path"]).write_text('{"diagnostics":[]}')
    with pytest.raises(ValueError, match="First diagnosis changed"):
        runner.audit_result(result)


def test_copy_repair_preserves_message_whitespace_and_unread_refs(tmp_path):
    draft = copy.deepcopy(BAD)
    draft["diagnostics"][0]["message"] = "  Original reason.  "
    draft["diagnostics"][0]["rule_refs"] = ["unread-reference"]
    result, _ = run(tmp_path, "C3", [draft, PATCH])
    assert result["diagnostics"][0]["message"] == "  Original reason.  "
    assert result["diagnostics"][0]["rule_refs"] == ["unread-reference"]
    assert result["protocol_audit"]["violations"] == ["unread_rule_reference"]


def test_partial_batch_does_not_publish_full_50_metrics(tmp_path, isolated_reference):
    runner.prepare(tmp_path)
    report = runner.score(tmp_path)
    assert report["scores_all_50"] == report["scores_before_quote_repair"] == {}
    assert report["not_scored_arms"] == ["C0p", "C3"]
    assert "all_three_point_estimates_not_lower" not in report


def test_complete_mock_batch_scores_first_and_repaired_diagnoses_separately(tmp_path, isolated_reference):
    runner.prepare(tmp_path)
    signature = runner.validate_plan(tmp_path)["configuration_sha256"]
    final = {"diagnostics": [{"message": "Synthetic check", "evidence": {"quote": "x = 2"}, "rule_refs": []}]}
    for row in runner.frozen_rows():
        run(tmp_path / "formal", "C0p", [final], row=row, signature=signature)
        run(tmp_path / "formal", "C3", [BAD, {"quote_patches": [{"index": 0, "quote": "x = 2"}]}],
            row=row, signature=signature)
    report = runner.score(tmp_path)
    assert report["not_scored_arms"] == []
    assert report["counts"] == {"C0p": {"completed": 50}, "C3": {"completed": 50}}
    assert report["scores_before_quote_repair"]["C3"]["summary"]["recall"] == 0
    assert report["scores_all_50"]["C3"]["summary"]["recall"] > 0
    assert report["all_three_point_estimates_not_lower"] is True
    assert report["usage"]["C0p"]["completed_responses"] == 50
    assert report["usage"]["C3"]["completed_responses"] == 100


def test_failure_during_second_turn_preserves_successful_prefix(tmp_path):
    result, calls = run(tmp_path, "C3", [
        {"action": "search_rules", "arguments": {"query": "牛顿", "mode": "bm25", "limit": 3}},
        TimeoutError("network timeout")])
    assert result["status"] == "failed" and len(calls) == 2
    assert len(result["trace"]["response_evidence"]) == 1
    assert len(result["trace"]["tool_events"]) >= 2
    assert result["trace"]["output_token_limit_reserved"] == runner.LIMITS["output_tokens_per_sample"]
    assert len(list(tmp_path.glob("samples/*/*/turn_*/request.json"))) == 2
    again, calls = run(tmp_path, "C3", [])
    assert again == result and not calls


@pytest.mark.parametrize("change", [
    {"usage": {"completion_tokens": 3, "prompt_tokens": 10}},
    {"choices": [{"token_ids": [5, 200002], "finish_reason": "length"}]},
    {"choices": [{"token_ids": [5, 200002], "finish_reason": "stop", "stop_reason": 200012}]},
])
def test_incomplete_or_unmetered_harmony_is_not_scored(change):
    raw = {"model": r.EXPECTED_MODEL, "usage": {"prompt_tokens": 10, "completion_tokens": 2},
           "choices": [{"token_ids": [5, 200002], "finish_reason": "stop"}]}
    raw.update(change)
    with pytest.raises(ValueError):
        transport.decode_response(raw, model=r.EXPECTED_MODEL, max_tokens=100,
                                  native_tools=runner.native_tools(True))
