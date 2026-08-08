from __future__ import annotations

import unittest

from core.physics_rule_verifier import PhysicsRuleVerifier
from scripts.evaluate_physics_eval_sets import (
    _collect_pred_findings as collect_error_findings,
    _execution_failure_reason as error_failure_reason,
)
from scripts.evaluate_question_level_sets import (
    _collect_pred_findings as collect_question_findings,
    _execution_failure_reason as question_failure_reason,
)


class _NonNegativeChecker:
    @staticmethod
    def _is_negative_or_uncertain_diagnostic(_diagnostic):
        return False


def _release_verifier(mode: str) -> PhysicsRuleVerifier:
    verifier = object.__new__(PhysicsRuleVerifier)
    verifier.checker_gate_mode = mode
    verifier.precision_mode = "strict"
    verifier.min_diagnostic_rule_score = 0.0
    verifier.quote_required_symbol_ratio = 0.0
    verifier.semantic_checker = _NonNegativeChecker()
    return verifier


def _diagnostic(passed=True):
    return {
        "severity": "error",
        "rule": "synthetic_rule",
        "symbol": "v",
        "message": "The stated relation contradicts the selected rule.",
        "evidence": {
            "quote": "v = 0",
            "location": {
                "start_char": 0,
                "end_char": 5,
                "span_valid": True,
                "paragraph_index": 1,
                "paragraph_valid": True,
                "locatable_valid": True,
            },
        },
        "checker_gate_mode": "dual_evidence",
        "checker_evidence_gate": {"passed": passed, "reasons": []},
    }


def _rule_record():
    return {
        "rule": {"id": "synthetic_rule"},
        "score": 1.0,
        "topic_rank": 0,
        "publish_gate": {
            "publishable": True,
            "reasons": [],
            "score_kind": "semantic_0_1",
            "min_publish_score": 0.0,
        },
    }


class CheckerReleaseIntegrationTests(unittest.TestCase):
    def test_dual_release_gate_requires_exact_true_and_matching_mode(self) -> None:
        verifier = _release_verifier("dual_evidence")

        self.assertTrue(verifier._diagnostic_release_gate(_diagnostic(True), _rule_record())["publishable"])

        string_true = verifier._diagnostic_release_gate(_diagnostic("true"), _rule_record())
        self.assertFalse(string_true["publishable"])
        self.assertIn("checker_evidence_gate_failed", string_true["reasons"])

        wrong_mode = _diagnostic(True)
        wrong_mode["checker_gate_mode"] = "dual_evidence_consistency"
        rejected = verifier._diagnostic_release_gate(wrong_mode, _rule_record())
        self.assertFalse(rejected["publishable"])
        self.assertIn("checker_gate_mode_mismatch", rejected["reasons"])

    def test_legacy_release_gate_does_not_require_dual_fields(self) -> None:
        verifier = _release_verifier("legacy")
        diagnostic = _diagnostic(True)
        diagnostic.pop("checker_gate_mode")
        diagnostic.pop("checker_evidence_gate")

        self.assertTrue(verifier._diagnostic_release_gate(diagnostic, _rule_record())["publishable"])

    def test_dual_gate_does_not_turn_runtime_metadata_into_a_hard_reject(self) -> None:
        verifier = _release_verifier("dual_evidence")
        record = _rule_record()
        record["rule"]["evidence_requirements"] = ["unrelated metadata token"]
        record["rule"]["negative_conditions"] = ["v = 0"]

        gate = verifier._diagnostic_release_gate(_diagnostic(True), record)

        self.assertTrue(gate["publishable"])
        self.assertFalse(gate["metadata_hard_gate_enabled"])

    def test_target_binding_is_preselected_without_becoming_semantic(self) -> None:
        verifier = _release_verifier("dual_evidence")
        verifier.min_diagnostic_rule_score = 0.99
        record = _rule_record()
        record["score"] = 0.0
        record["retrieval_strategy"] = "target_rule_binding"
        record["publish_gate"]["score_kind"] = "fixed_control_0_1"

        kept, suppressed = verifier._filter_low_confidence_unified_diagnostics(
            [_diagnostic(True)], [record]
        )
        gate = verifier._diagnostic_release_gate(_diagnostic(True), record)

        self.assertEqual(len(kept), 1)
        self.assertEqual(suppressed, [])
        self.assertEqual(gate["score_kind"], "fixed_control_0_1")

    def test_dual_metrics_do_not_promote_symbolic_audit_findings(self) -> None:
        pred = {
            "id": "synthetic",
            "checker_gate_mode": "dual_evidence",
            "diagnostics": [],
        }
        audit = {
            "experience_code_checks": [
                {
                    "result": "fail",
                    "rule": "experience_code::synthetic_rule",
                    "message": "synthetic failure",
                    "evidence": "v = 0",
                }
            ]
        }

        self.assertEqual(collect_question_findings(pred, audit), [])
        self.assertEqual(collect_error_findings(pred, audit), [])

    def test_checker_failures_are_not_scored_as_empty_predictions(self) -> None:
        failed = {"checker_status": "partial_failure", "checker_failure_count": 1}
        self.assertEqual(question_failure_reason(failed), "checker_failure")
        self.assertEqual(error_failure_reason(failed), "checker_failure")
        self.assertEqual(question_failure_reason(None), "missing_result")
        self.assertEqual(error_failure_reason(None), "missing_result")


if __name__ == "__main__":
    unittest.main()
