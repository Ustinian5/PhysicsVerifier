from __future__ import annotations

import unittest

from core.semantic_rule_checker import SemanticRuleChecker


class SemanticRuleCheckerPromptTest(unittest.TestCase):
    def test_rule_check_prompt_includes_problem_and_rejects_method_omission_flags(self) -> None:
        checker = SemanticRuleChecker(llm_model=None, enable_cache=False)

        _, prompt = checker._get_check_prompt(
            srd="Use conservation of angular momentum when applicable.",
            raw_answer="The student solution.",
            problem_text="Find the oscillation period.",
            context_summary="{}",
            rule_id="rule_1",
        )

        self.assertIn("Find the oscillation period.", prompt)
        self.assertIn("conditional diagnostic aid", prompt)
        self.assertIn("Do NOT penalize", prompt)
        self.assertIn("alternative derivation", prompt)


if __name__ == "__main__":
    unittest.main()
