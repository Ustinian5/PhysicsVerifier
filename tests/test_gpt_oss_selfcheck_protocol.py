import unittest
from scripts.run_gpt_oss_selfcheck import BASE_SYSTEM, RULE_SYSTEM, parse_json, prompt_messages


class SelfCheckProtocolTest(unittest.TestCase):
    def test_prompts_request_unprefixed_json(self):
        assert "raw JSON object" in BASE_SYSTEM
        assert "RESULT " not in BASE_SYSTEM
        assert "raw JSON object" in RULE_SYSTEM
        assert "RESULT " not in RULE_SYSTEM


    def test_parser_accepts_raw_diagnostics_and_tool_action_json(self):
        assert parse_json('{"diagnostics":[]}') == {"diagnostics": []}
        assert parse_json('{"action":"read_tree","arguments":{"node_id":"root"}}') == {
            "action": "read_tree", "arguments": {"node_id": "root"}
        }


    def test_parser_repairs_escaped_diagnostic_object_boundaries_without_editing_text(self):
        from scripts import run_gpt_oss_selfcheck as runner

        raw = r'{"diagnostics":[{"message":"first","evidence":{"quote":"x"}}, "{ \"message\": \"Second \\omega\",\"evidence\":{\"quote\":\"\\\\omega\"}},{\"message\":\"Third\",\"evidence\":{\"quote\":\"q\"}}]}'
        value = runner.parse_json(raw)
        diagnostics = runner.validate_diagnostics(value)
        assert diagnostics == [
            {"message": "first", "evidence": {"quote": "x"}},
            {"message": r"Second \omega", "evidence": {"quote": r"\\omega"}},
            {"message": "Third", "evidence": {"quote": "q"}},
        ]
        assert runner.LAST_JSON_REPAIRS == [{
            "type": "normalize_escaped_diagnostic_object_boundaries",
            "objects": 2,
            "field_text_preserved": True,
        }]


    def test_parser_rejects_preamble_and_markdown_wrappers(self):
        with self.assertRaises(ValueError):
            parse_json('analysis {"diagnostics":[]}')
        with self.assertRaises(ValueError):
            parse_json('```json\n{"diagnostics":[]}\n```')


    def test_prompts_keep_same_final_diagnostics_schema(self):
        row = {"id": "synthetic", "question": "q", "prediction": "a"}
        model_only = prompt_messages(row, rules=False)
        self_check = prompt_messages(row, rules=True)
        assert "diagnostics" in model_only[0]["content"]
        assert "diagnostics" in self_check[0]["content"]
        assert "action" in self_check[0]["content"]


if __name__ == "__main__":
    unittest.main()
