from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from core.semantic_rule_checker import SemanticRuleChecker


RULE_ID = "generic_rule"
QUESTION = "  The task states condition alpha is active."
CONTEXT = "Background information keeps condition beta fixed."
PREDICTION = "The submission asserts outcome gamma as its conclusion."
APP_QUOTE = "condition alpha is active"
VIOLATION_QUOTE = "asserts outcome gamma"


def _evidence(source: str, quote: str, start: int = -1, end: int = -1) -> Dict[str, Any]:
    return {
        "source": source,
        "quote": quote,
        "location": {"start_char": start, "end_char": end},
    }


def _diagnostic(
    *,
    message: str = "The current conclusion contradicts the applicable condition.",
    violation_source: str = "prediction",
    violation_quote: str = VIOLATION_QUOTE,
    violation_present: bool = True,
    violation_confidence: float = 0.95,
    consistency_status: str | None = None,
    consistency_reason: str = "The quoted claim remains asserted.",
) -> Dict[str, Any]:
    diagnostic: Dict[str, Any] = {
        "severity": "error",
        "symbol": None,
        "message": message,
        "violation": {
            "present": violation_present,
            "confidence": violation_confidence,
            "evidence": [_evidence(violation_source, violation_quote)],
        },
    }
    if consistency_status is not None:
        diagnostic["consistency"] = {
            "status": consistency_status,
            "confidence": 0.95,
            "reason": consistency_reason,
        }
    return diagnostic


def _payload(
    *,
    status: str = "violation",
    rule_id: str = RULE_ID,
    applies: bool = True,
    applicability_confidence: float = 0.95,
    applicability_source: str = "question",
    applicability_quote: str = APP_QUOTE,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if diagnostics is None:
        diagnostics = [_diagnostic()] if status == "violation" else []
    return {
        "schema_version": SemanticRuleChecker.DUAL_EVIDENCE_SCHEMA_VERSION,
        "rule_id": rule_id,
        "status": status,
        "applicability": {
            "applies": applies,
            "confidence": applicability_confidence,
            "evidence": (
                [_evidence(applicability_source, applicability_quote)]
                if applicability_quote
                else []
            ),
        },
        "diagnostics": diagnostics,
    }


def _sample(**overrides: Any) -> Dict[str, Any]:
    sample: Dict[str, Any] = {
        "id": "synthetic",
        "question": QUESTION,
        "context": CONTEXT,
        "prediction": PREDICTION,
        # This would make the legacy shortcut pass. Strict modes must ignore it.
        "answer": PREDICTION,
    }
    sample.update(overrides)
    return sample


def _strict_checker(
    responses: List[Any],
    *,
    mode: str = SemanticRuleChecker.CHECKER_MODE_DUAL_EVIDENCE,
    attempts: int = 1,
) -> SemanticRuleChecker:
    checker = SemanticRuleChecker(
        llm_model=None,
        rules=[RULE_ID],
        enable_cache=False,
        use_symbol_graph=False,
        checker_mode=mode,
        checker_json_attempts=attempts,
    )
    checker.rule_translations = {RULE_ID: {"srd": "Apply only when the stated condition holds."}}
    queue = list(responses)
    prompts: List[str] = []

    def _request(_system_prompt: str, user_prompt: str) -> str:
        prompts.append(user_prompt)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return item
        return json.dumps(item, ensure_ascii=False)

    checker._request_json_object_text = _request  # type: ignore[method-assign]
    checker._test_prompts = prompts  # type: ignore[attr-defined]
    return checker


class SemanticRuleCheckerDualEvidenceTest(unittest.TestCase):
    def test_mode_contract_and_constructor_validation(self) -> None:
        self.assertEqual(
            SemanticRuleChecker.CHECKER_MODES,
            frozenset({"legacy", "dual_evidence", "dual_evidence_consistency"}),
        )
        with self.assertRaises(ValueError):
            SemanticRuleChecker(checker_mode="unknown")
        with self.assertRaises(ValueError):
            SemanticRuleChecker(checker_json_attempts=0)
        with self.assertRaises(ValueError):
            SemanticRuleChecker(checker_json_attempts=6)
        with self.assertRaises(ValueError):
            SemanticRuleChecker(checker_min_confidence=1.1)

    def test_prompt_preserves_source_coordinates_and_omits_mixed_summary(self) -> None:
        checker = _strict_checker([])
        system_prompt, user_prompt = checker._get_dual_evidence_prompt(
            srd="A generic conditional rule.",
            raw_answer="  prediction text",
            question_text="  question text",
            context_text="  context text",
            rule_id=RULE_ID,
        )
        self.assertIn("exactly one JSON object", system_prompt)
        self.assertIn("\n  question text\n", user_prompt)
        self.assertIn("\n  context text\n", user_prompt)
        self.assertIn("\n  prediction text\n", user_prompt)
        self.assertNotIn("Structured extraction summary", user_prompt)
        self.assertIn('"schema_version": "semantic-rule-checker-dual-evidence-v1"', user_prompt)

    def test_dual_evidence_repairs_unique_spans_and_ignores_reference_answer(self) -> None:
        checker = _strict_checker([_payload()])
        result = checker.analyze(_sample())

        self.assertFalse(result["answer_correct"])
        self.assertEqual(result["checker_status"], "valid_with_diagnostics")
        self.assertEqual(result["checker_failures"], [])
        self.assertEqual(len(result["diagnostics"]), 1)
        diagnostic = result["diagnostics"][0]
        self.assertEqual(diagnostic["rule"], RULE_ID)
        self.assertEqual(diagnostic["checker_gate_mode"], "dual_evidence")
        self.assertIs(diagnostic["checker_evidence_gate"]["passed"], True)
        self.assertEqual(diagnostic["applicability"]["evidence"][0]["source"], "question")
        self.assertEqual(diagnostic["evidence"]["source"], "prediction")
        self.assertTrue(diagnostic["applicability"]["evidence"][0]["location"]["span_repaired"])
        self.assertTrue(diagnostic["evidence"]["location"]["span_repaired"])
        self.assertEqual(
            diagnostic["evidence"]["location"]["start_char"],
            PREDICTION.index(VIOLATION_QUOTE),
        )

    def test_wrong_role_sources_are_schema_failures_even_for_valid_empty(self) -> None:
        wrong_app_source = _payload(
            status="no_violation",
            applicability_source="prediction",
            applicability_quote=VIOLATION_QUOTE,
            diagnostics=[],
        )
        result = _strict_checker([wrong_app_source]).analyze(_sample())
        self.assertEqual(result["checker_status"], "failed")
        self.assertEqual(result["checker_failures"][0]["status"], "schema_failure")
        self.assertIn(
            "applicability.evidence[0]:wrong_source_for_role",
            result["checker_failures"][0]["errors"],
        )

        wrong_violation_source = _payload(
            diagnostics=[
                _diagnostic(
                    violation_source="question",
                    violation_quote=APP_QUOTE,
                )
            ]
        )
        result = _strict_checker([wrong_violation_source]).analyze(_sample())
        self.assertEqual(result["checker_failures"][0]["status"], "schema_failure")
        self.assertIn(
            "diagnostics[0].violation.evidence[0]:wrong_source_for_role",
            result["checker_failures"][0]["errors"],
        )

    def test_hallucinated_and_ambiguous_quotes_fail_closed(self) -> None:
        cases = [
            (
                _payload(
                    diagnostics=[_diagnostic(violation_quote="quote absent from submission")]
                ),
                _sample(),
                "violation_evidence_quote_not_found",
            ),
            (
                _payload(
                    applicability_quote="repeated anchor",
                    diagnostics=[_diagnostic()],
                ),
                _sample(question="repeated anchor and repeated anchor"),
                "applicability_evidence_quote_ambiguous",
            ),
        ]
        for payload, sample, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                result = _strict_checker([payload]).analyze(sample)
                self.assertEqual(result["checker_status"], "valid_empty")
                self.assertEqual(result["diagnostics"], [])
                self.assertEqual(len(result["checker_suppressed"]), 1)
                gate = result["checker_suppressed"][0]["checker_evidence_gate"]
                self.assertIs(gate["passed"], False)
                self.assertIn(expected_reason, gate["reasons"])

    def test_consistency_mode_requires_current_confirmed_violation(self) -> None:
        diagnostics = [
            _diagnostic(
                message="Candidate one.",
                consistency_status="confirmed_violation",
                consistency_reason="The derivation is correct and remains a valid alternative.",
            ),
            _diagnostic(
                message="Candidate two.",
                consistency_status="self_corrected",
                consistency_reason="The quoted claim was subsequently corrected.",
            ),
            _diagnostic(
                message="Candidate three.",
                consistency_status="confirmed_violation",
                consistency_reason="The quoted claim remains asserted as the conclusion.",
            ),
            _diagnostic(
                message="Candidate four.",
                consistency_status="confirmed_violation",
                consistency_reason="The claim was not subsequently corrected and is not equivalent.",
            ),
            _diagnostic(
                message="Candidate five.",
                consistency_status="confirmed_violation",
                consistency_reason="The assertion that the derivation is correct is false and remains asserted.",
            ),
        ]
        checker = _strict_checker(
            [_payload(diagnostics=diagnostics)],
            mode=SemanticRuleChecker.CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY,
        )
        result = checker.analyze(_sample())

        self.assertEqual(
            [d["message"] for d in result["diagnostics"]],
            ["Candidate three.", "Candidate four.", "Candidate five."],
        )
        self.assertEqual(len(result["checker_suppressed"]), 2)
        first_reasons = result["checker_suppressed"][0]["checker_evidence_gate"]["reasons"]
        second_reasons = result["checker_suppressed"][1]["checker_evidence_gate"]["reasons"]
        self.assertIn("consistency_reason_contradicts_status", first_reasons)
        self.assertIn("consistency_not_confirmed", second_reasons)

    def test_student_negative_words_do_not_change_non_consistency_arms(self) -> None:
        cases = [
            (
                "The derivation is correct even though its stated premise is absent.",
                "The derivation is correct",
            ),
            (
                "Conservation does not apply, so the isolated system gains momentum.",
                "Conservation does not apply",
            ),
        ]
        for prediction, quote in cases:
            with self.subTest(quote=quote):
                payload = _payload(
                    diagnostics=[
                        _diagnostic(
                            message="The quoted claim contradicts the applicable condition.",
                            violation_quote=quote,
                        )
                    ]
                )
                result = _strict_checker([payload]).analyze(
                    _sample(prediction=prediction)
                )
                self.assertEqual(result["checker_status"], "valid_with_diagnostics")
                self.assertEqual(len(result["diagnostics"]), 1)

    def test_retry_uses_only_generic_error_category_and_records_attempts(self) -> None:
        for first_response, expected_category in [
            ("not-json", "parse"),
            (_payload(rule_id="other_rule"), "schema"),
        ]:
            with self.subTest(expected_category=expected_category):
                checker = _strict_checker([first_response, _payload()], attempts=2)
                result = checker.analyze(_sample())
                decision = result["checker_decisions"][0]
                self.assertEqual(decision["attempt_count"], 2)
                self.assertEqual(decision["attempts"][1]["retry_category"], expected_category)
                retry_prompt = checker._test_prompts[1]  # type: ignore[attr-defined]
                self.assertIn(f"generic {expected_category} error", retry_prompt)
                self.assertNotIn("not-json", retry_prompt)
                self.assertEqual(result["checker_status"], "valid_with_diagnostics")

    def test_valid_empty_is_distinct_from_parse_and_transport_failure(self) -> None:
        valid_empty = _strict_checker(
            [_payload(status="no_violation", diagnostics=[])]
        ).analyze(_sample())
        self.assertEqual(valid_empty["checker_status"], "valid_empty")
        self.assertEqual(valid_empty["checker_failures"], [])

        parse_failure = _strict_checker(["not-json"]).analyze(_sample())
        self.assertEqual(parse_failure["checker_status"], "failed")
        self.assertEqual(parse_failure["checker_failures"][0]["status"], "parse_failure")

        transport_failure = _strict_checker([RuntimeError("endpoint unavailable")]).analyze(
            _sample()
        )
        self.assertEqual(transport_failure["checker_status"], "failed")
        self.assertEqual(
            transport_failure["checker_failures"][0]["status"],
            "transport_failure",
        )

    def test_every_requested_rule_gets_a_decision_and_failed_is_not_a_sentinel(self) -> None:
        checker = _strict_checker([_payload()])
        checker.rule_translations[RULE_ID]["srd"] = (
            "A previous approach failed, so check the current physical claim directly."
        )
        result = checker.analyze(_sample())
        self.assertEqual(result["checker_status"], "valid_with_diagnostics")
        self.assertEqual(len(result["checker_decisions"]), 1)

        missing = _strict_checker([])
        missing.rule_translations = {}
        result = missing.analyze(_sample())
        self.assertEqual(result["checker_status"], "failed")
        self.assertEqual(len(result["checker_decisions"]), 1)
        self.assertEqual(
            result["checker_failures"][0]["status"],
            "configuration_failure",
        )

    def test_strict_schema_rejects_wrong_types_unknown_keys_and_arrays(self) -> None:
        wrong_type = _payload()
        wrong_type["applicability"]["applies"] = "true"
        unknown_key = _payload()
        unknown_key["extra"] = "not allowed"
        cases: List[Any] = [wrong_type, unknown_key, [_payload()]]
        for response in cases:
            with self.subTest(response_type=type(response).__name__):
                result = _strict_checker([response]).analyze(_sample())
                self.assertEqual(result["checker_status"], "failed")
                self.assertEqual(result["checker_failures"][0]["status"], "schema_failure")

        huge_integer = _payload()
        huge_integer["applicability"]["confidence"] = 10**400
        result = _strict_checker([huge_integer]).analyze(_sample())
        self.assertEqual(result["checker_status"], "failed")
        self.assertEqual(result["checker_failures"][0]["status"], "schema_failure")
        self.assertIn(
            "applicability:invalid_confidence",
            result["checker_failures"][0]["errors"],
        )

        empty_reason = _payload(
            diagnostics=[
                _diagnostic(
                    consistency_status="confirmed_violation",
                    consistency_reason="",
                )
            ]
        )
        result = _strict_checker(
            [empty_reason],
            mode=SemanticRuleChecker.CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY,
        ).analyze(_sample())
        self.assertEqual(result["checker_status"], "failed")
        self.assertIn(
            "diagnostics[0].consistency:reason_must_be_nonempty_string",
            result["checker_failures"][0]["errors"],
        )

    def test_legacy_mode_preserves_output_but_audits_schema_and_failures(self) -> None:
        checker = SemanticRuleChecker(
            llm_model=None,
            rules=[RULE_ID],
            enable_cache=False,
            use_symbol_graph=False,
            checker_mode="legacy",
        )
        checker.rule_translations = {RULE_ID: {"srd": "A generic legacy rule."}}

        def _valid_call(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": True,
                "status": "valid_json",
                "data": {
                    "severity": "error",
                    "rule": RULE_ID,
                    "symbol": None,
                    "message": "A concrete legacy finding.",
                    "evidence": {"quote": VIOLATION_QUOTE},
                },
                "errors": [],
                "cache_hit": False,
                "attempt_count": 1,
            }

        checker._llm_json = _valid_call  # type: ignore[method-assign]
        result = checker.analyze(_sample(answer=""))
        self.assertEqual(result["checker_status"], "valid_with_diagnostics")
        self.assertEqual(result["diagnostics"][0]["rule"], RULE_ID)
        self.assertNotIn("checker_evidence_gate", result["diagnostics"][0])

        def _wrong_rule_call(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            result = _valid_call()
            result["data"] = {**result["data"], "rule": "other_rule"}
            return result

        checker._llm_json = _wrong_rule_call  # type: ignore[method-assign]
        result = checker.analyze(_sample(answer=""))
        self.assertEqual(result["checker_status"], "failed")
        self.assertEqual(result["checker_failures"][0]["status"], "schema_failure")

        def _parse_failure_call(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": False,
                "status": "parse_failure",
                "data": [],
                "errors": ["response_is_not_valid_json"],
                "cache_hit": False,
                "attempt_count": 1,
            }

        checker._llm_json = _parse_failure_call  # type: ignore[method-assign]
        result = checker.analyze(_sample(answer=""))
        self.assertEqual(result["checker_status"], "failed")
        self.assertEqual(result["checker_failures"][0]["status"], "parse_failure")

    def test_legacy_location_requires_real_unambiguous_span(self) -> None:
        checker = SemanticRuleChecker(llm_model=None, enable_cache=False)

        repaired = checker._normalize_diagnostic_location(
            {
                "evidence": {
                    "quote": "unique quote",
                    "location": {"start_char": 0, "end_char": 4, "paragraph_index": 1},
                }
            },
            "prefix unique quote suffix",
        )
        self.assertTrue(repaired["evidence"]["location"]["span_repaired"])
        self.assertTrue(repaired["evidence"]["location"]["locatable_valid"])

        hallucinated = checker._normalize_diagnostic_location(
            {
                "evidence": {
                    "quote": "absent quote",
                    "location": {"start_char": 0, "end_char": 6, "paragraph_index": 1},
                }
            },
            "source text",
        )
        self.assertFalse(hallucinated["evidence"]["location"]["span_valid"])
        self.assertFalse(hallucinated["evidence"]["location"]["locatable_valid"])

        ambiguous = checker._normalize_diagnostic_location(
            {"evidence": {"quote": "repeat", "location": {"start_char": -1, "end_char": -1}}},
            "repeat then repeat",
        )
        self.assertTrue(ambiguous["evidence"]["location"]["span_ambiguous"])
        self.assertFalse(ambiguous["evidence"]["location"]["locatable_valid"])

        at_zero = checker._normalize_diagnostic_location(
            {"evidence": {"quote": "start", "location": {"start_char": 0, "end_char": 5}}},
            "start here",
        )
        self.assertTrue(at_zero["evidence"]["location"]["span_valid"])
        self.assertTrue(at_zero["evidence"]["location"]["locatable_valid"])

    def test_legacy_schema_failure_is_not_cached_and_config_changes_cache_key(self) -> None:
        class _FakeCompletions:
            def __init__(self) -> None:
                self.contents: List[str] = []
                self.calls = 0

            def create(self, **_kwargs: Any) -> Any:
                self.calls += 1
                content = self.contents.pop(0)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        completions = _FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = SemanticRuleChecker(
                llm_model=None,
                rules=[RULE_ID],
                enable_cache=True,
                use_symbol_graph=False,
                checker_mode="legacy",
            )
            checker.llm_model = "fake-model"
            checker._llm = fake_client
            checker._cache = {}
            checker._cache_path = Path(tmpdir) / "cache.json"
            validator = lambda payload: checker._validate_legacy_diagnostics_payload(
                payload,
                expected_rule_id=RULE_ID,
            )[1]

            completions.contents.append(json.dumps({"rule": "other_rule"}))
            failed = checker._llm_json(
                "system",
                "user",
                return_meta=True,
                json_validator=validator,
            )
            self.assertEqual(failed["status"], "schema_failure")
            self.assertEqual(checker._cache, {})

            valid_payload = {
                "severity": "error",
                "rule": RULE_ID,
                "symbol": None,
                "message": "Finding.",
            }
            completions.contents.append(json.dumps(valid_payload))
            succeeded = checker._llm_json(
                "system",
                "user",
                return_meta=True,
                json_validator=validator,
            )
            self.assertTrue(succeeded["ok"])
            self.assertEqual(len(checker._cache), 1)

            checker.llm_temperature = 0.2
            completions.contents.append(json.dumps(valid_payload))
            checker._llm_json(
                "system",
                "user",
                return_meta=True,
                json_validator=validator,
            )
            self.assertEqual(completions.calls, 3)
            self.assertEqual(len(checker._cache), 2)

    def test_legacy_trace_records_checker_mode_for_success_and_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = SemanticRuleChecker(
                llm_model=None,
                rules=[RULE_ID],
                enable_cache=False,
                use_symbol_graph=False,
                checker_mode="legacy",
            )
            checker.llm_model = "fake-model"
            trace_path = Path(tmpdir) / "legacy.trace.jsonl"
            checker.llm_trace_path = str(trace_path)
            checker._llm = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kwargs: SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
                        )
                    )
                )
            )

            succeeded = checker._llm_json(
                "system",
                "user",
                trace_meta={"sample_id": "sample-1", "rule_id": RULE_ID},
                return_meta=True,
            )
            self.assertTrue(succeeded["ok"])

            def _raise_transport(**_kwargs: Any) -> Any:
                raise RuntimeError("synthetic transport failure")

            checker._llm.chat.completions.create = _raise_transport
            failed = checker._llm_json(
                "system",
                "user",
                trace_meta={"sample_id": "sample-2", "rule_id": RULE_ID},
                return_meta=True,
            )
            self.assertEqual(failed["status"], "transport_failure")

            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["checker_mode"] for record in records], ["legacy", "legacy"])
            self.assertEqual(records[0]["parse_status"], "json.loads_ok")
            self.assertEqual(records[1]["parse_status"], "exception")


if __name__ == "__main__":
    unittest.main()
