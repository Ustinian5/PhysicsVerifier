from scripts import continuous_context_v2 as old
from scripts import continuous_context_v3 as protocol
from scripts import run_continuous_context_v2 as runner
from scripts.run_continuous_context_v3 import configured
from tests.test_continuous_context_v2 import run, ROW, BAD, PATCH


def test_pure_prompt_is_exact_previous_comparator():
    assert protocol.messages(ROW, rules=False, tree_root=None) == old.messages(ROW, rules=False, tree_root=None)


def test_rule_cycle_keeps_context_and_common_quote_repair(tmp_path):
    result, calls = run(tmp_path, "C4", [
        {"action": "search_rules", "arguments": {"query": "牛顿 合力", "mode": "bm25", "limit": 6}},
        BAD, PATCH], review_protocol=protocol.PROTOCOL)
    assert result["status"] == "completed"
    assert result["trace"]["tree_access"]
    assert result["protocol_audit"]["search_instruction_followed"]
    assert calls[0]["native_tools"] and calls[-1]["native_tools"] is None
    assert protocol.RULES in calls[0]["messages"][0]["content"]
    assert any(m["role"] == "tool" for m in calls[-1]["messages"])
    assert result["diagnostics"][0]["message"] == BAD["diagnostics"][0]["message"]
    assert result["trace"]["output_token_limit_reserved"] == 150


def test_nonuse_is_recorded_without_reroll(tmp_path):
    result, calls = run(tmp_path, "C4", [{"diagnostics": []}], review_protocol=protocol.PROTOCOL)
    assert result["status"] == "completed" and len(calls) == 1
    assert result["protocol_audit"]["search_instruction_followed"] is False
    same, calls = run(tmp_path, "C4", [], review_protocol=protocol.PROTOCOL)
    assert same == result and not calls


def test_experiment_adapter_restores_previous_defaults():
    limits = runner.LIMITS
    original = runner.protocol
    with configured():
        config = runner.configuration()
        assert config["arms"] == ["C0p", "C4"]
        assert config["limits"]["output_tokens_per_sample"] == 16384
        assert len(config["sample_ids"]) == 50
        assert config["prompts"]["base"] == old.BASE
    assert runner.LIMITS is limits and runner.protocol is original
