from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException

from training.reward_server import physics_reward_server as server


class _FakeVerifier:
    def __init__(self, result=None) -> None:
        self.calls = 0
        self.result = result

    def verify(self, sample):
        self.calls += 1
        if self.result is not None:
            return self.result
        pred = str(sample.get("prediction") or "")
        loc = 0 if not pred else min(12, max(0, len(pred) // 3))
        return {
            "checker_status": "valid_with_diagnostics",
            "checker_failures": [],
            "diagnostics": [
                {
                    "severity": "error",
                    "rule": "test_rule",
                    "message": "test error",
                    "start_char": loc,
                    "end_char": loc + 4,
                    "location": {"start_char": loc, "end_char": loc + 4, "paragraph_index": 1},
                }
            ]
        }


class RewardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_on_wrong = server.VERIFIER_ON_WRONG
        self.original_get_verifier = server._get_verifier
        self.original_append_metrics = server._append_metrics
        self.original_failure_policy = server.VERIFIER_FAILURE_POLICY
        self.original_sample_rate = server.VERIFIER_SAMPLE_RATE
        self.original_semaphore = server._semaphore
        server._append_metrics = lambda record: None
        server._semaphore = None
        server.reset_reward_cache(maxsize=64)
        self.original_llm_judge = server._get_llm_step_judge

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server.VERIFIER_ON_WRONG = self.original_on_wrong
        server._get_verifier = self.original_get_verifier
        server._append_metrics = self.original_append_metrics
        server.VERIFIER_FAILURE_POLICY = self.original_failure_policy
        server.VERIFIER_SAMPLE_RATE = self.original_sample_rate
        server._semaphore = self.original_semaphore
        server._get_llm_step_judge = self.original_llm_judge

    def test_wrong_answer_skips_verifier(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_ON_WRONG = False
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{2}", label="1")
            )
        )
        self.assertFalse(result["acc"])
        self.assertEqual(result["verifier_mode"], "skipped")
        self.assertEqual(fake.calls, 0)

    def test_process_paragraph_runs_verifier_on_wrong_answer(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "process_paragraph"
        server.VERIFIER_ON_WRONG = True
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(
                    prompt="question",
                    response="A derivation that is locally wrong but has no boxed match.",
                    label="1",
                )
            )
        )
        self.assertFalse(result["acc"])
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(fake.calls, 1)
        self.assertGreater(float(result["score"]), 0.0)

    def test_process_paragraph_ignores_final_answer(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "process_paragraph"
        server.VERIFIER_ON_WRONG = True
        server._get_verifier = lambda: fake
        text = (
            "A derivation that uses F = ma and conservation of energy. " * 4
            + r" \boxed{42}"
        )
        wrong = asyncio.run(
            server.score_one(server.ScoreRequest(prompt="question", response=text, label="1"))
        )
        right = asyncio.run(
            server.score_one(server.ScoreRequest(prompt="question", response=text, label="42"))
        )
        self.assertFalse(wrong["acc"])
        self.assertTrue(right["acc"])
        self.assertEqual(wrong["score"], right["score"])
        self.assertEqual(wrong["reward_components"]["weights"]["answer"], 0.0)
        self.assertEqual(wrong["reward_components"]["weights"]["format"], 0.0)
        self.assertEqual(fake.calls, 2)

    def test_reward_cache_dedupes_identical_completions(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "process_paragraph"
        server.VERIFIER_ON_WRONG = True
        server._get_verifier = lambda: fake
        text = "A derivation that uses F = ma and conservation of energy. " * 4
        req = server.OpenRLHFRewardRequest(
            query=["q" + text, "q" + text, "q" + text + " extra"],
            prompts=["q", "q", "q"],
            labels=["1", "1", "1"],
        )
        payload = asyncio.run(server.openrlhf_get_reward(req))
        self.assertEqual(len(payload["rewards"]), 3)
        self.assertEqual(payload["rewards"][0], payload["rewards"][1])
        self.assertEqual(fake.calls, 2)
        self.assertEqual(payload["extra_logs"]["physics_reward_batch_unique_scored"], 2.0)

    def test_group_indices_by_key_preserves_order(self) -> None:
        groups = server.group_indices_by_key(["a", "b", "a", "c", "b"])
        self.assertEqual(groups, [[0, 2], [1, 4], [3]])

    def test_correct_answer_calls_verifier(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "answer_low_verifier"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertTrue(result["acc"])
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(result["n_errors"], 1)
        self.assertEqual(fake.calls, 1)

    def test_semantic_retrieval_failure_is_not_scored_as_clean(self) -> None:
        fake = _FakeVerifier(
            {
                "selection_strategy": "semantic_error",
                "semantic_selection_error": "judge timeout",
                "checker_status": "complete_no_rules",
                "checker_failures": [],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_FAILURE_POLICY = "raise"
        server._get_verifier = lambda: fake
        with self.assertRaises(server.VerifierExecutionError):
            asyncio.run(
                server.score_one(
                    server.ScoreRequest(
                        prompt="question", response=r"\boxed{1}", label="1"
                    )
                )
            )

    def test_checker_partial_failure_is_not_scored_as_clean(self) -> None:
        fake = _FakeVerifier(
            {
                "semantic_selection_error": "",
                "checker_status": "partial_failure",
                "checker_failures": [{"rule_id": "r1", "status": "transport_failure"}],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_FAILURE_POLICY = "raise"
        server._get_verifier = lambda: fake
        with self.assertRaises(server.VerifierExecutionError):
            asyncio.run(
                server.score_one(
                    server.ScoreRequest(
                        prompt="question", response=r"\boxed{1}", label="1"
                    )
                )
            )

    def test_explicit_zero_reward_policy_neutralizes_failed_check(self) -> None:
        fake = _FakeVerifier(
            {
                "semantic_selection_error": "",
                "checker_status": "failed",
                "checker_failures": [{"rule_id": "r1"}],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_FAILURE_POLICY = "zero_reward"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["score_noxverify"], 0.0)
        self.assertEqual(result["verifier_mode"], "failed")
        self.assertTrue(result["reward_components"]["verifier_failed"])

    def test_valid_empty_check_remains_successful(self) -> None:
        fake = _FakeVerifier(
            {
                "semantic_selection_error": "",
                "selection_strategy": "semantic_tree_empty",
                "checker_status": "complete_no_rules",
                "checker_failures": [],
                "diagnostics": [],
            }
        )
        server.REWARD_MODE = "answer_low_verifier"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(result["n_errors"], 0)

    def test_query_response_is_removed_only_when_prompt_is_prefix(self) -> None:
        self.assertEqual(server._response_from_query("promptanswer", "prompt"), "answer")
        with self.assertRaises(ValueError):
            server._response_from_query("answer mentions prompt", "prompt")

    def test_openrlhf_rejects_misaligned_payload(self) -> None:
        request = server.OpenRLHFRewardRequest(
            query=["q1", "q2"], prompts=["p1"], labels=["1", "2"]
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server.openrlhf_get_reward(request))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_verifier_sampling_is_deterministic_and_not_position_quantized(self) -> None:
        server.VERIFIER_SAMPLE_RATE = 0.37
        first = [
            server._should_run_verifier(True, index % 2, f"generic-sample-{index}")
            for index in range(2000)
        ]
        second = [
            server._should_run_verifier(True, index % 2, f"generic-sample-{index}")
            for index in range(2000)
        ]
        self.assertEqual(first, second)
        observed_rate = sum(first) / len(first)
        self.assertGreater(observed_rate, 0.33)
        self.assertLess(observed_rate, 0.41)


class _FakeLLMJudge:
    prompt_version = "llm_step_v1"

    def __init__(self) -> None:
        self.calls: list = []

    def score_group(self, question, solutions):
        self.calls.append((question, tuple(solutions)))
        out = []
        for i, _sol in enumerate(solutions):
            out.append(
                {
                    "id": f"c{i}",
                    "raw_score": 5.0 + i,
                    "score": (5.0 + i) / 10.0,
                    "fatal_error": False,
                    "answer_only": False,
                    "step_assessments": [],
                    "brief_reason": "ok",
                }
            )
        return out

    def metrics_snapshot(self):
        return {"llm_step_api_calls": float(len(self.calls))}


class LLMStepRewardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_get_verifier = server._get_verifier
        self.original_judge = server._get_llm_step_judge
        server._append_metrics = lambda record: None
        server.reset_reward_cache(maxsize=64)

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server._get_verifier = self.original_get_verifier
        server._get_llm_step_judge = self.original_judge

    def test_labels_do_not_change_cache_or_reward(self) -> None:
        fake = _FakeLLMJudge()
        server.REWARD_MODE = "llm_step_score"
        server._get_llm_step_judge = lambda: fake
        a = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(query=["qsol"], prompts=["q"], labels=["GOLD1"])
            )
        )
        b = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(query=["qsol"], prompts=["q"], labels=["GOLD2"])
            )
        )
        self.assertEqual(a["rewards"], b["rewards"])
        self.assertEqual(len(fake.calls), 1)

    def test_same_question_group_is_one_call_and_order_restored(self) -> None:
        fake = _FakeLLMJudge()
        server.REWARD_MODE = "llm_step_score"
        server._get_llm_step_judge = lambda: fake
        payload = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(
                    query=["qas0", "qbt0", "qas1"],
                    prompts=["qa", "qb", "qa"],
                    labels=["1", "2", "3"],
                )
            )
        )
        self.assertEqual(len(fake.calls), 2)
        self.assertAlmostEqual(payload["rewards"][0], 0.5)
        self.assertAlmostEqual(payload["rewards"][2], 0.6)
        self.assertEqual(len(payload["rewards"]), 3)

    def test_llm_mode_never_instantiates_rule_verifier(self) -> None:
        fake = _FakeLLMJudge()
        server.REWARD_MODE = "llm_step_score"
        server._get_llm_step_judge = lambda: fake

        def boom():
            raise AssertionError("rule verifier should not be created")

        server._get_verifier = boom
        payload = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(query=["qs0", "qs1"], prompts=["q", "q"], labels=["x", "y"])
            )
        )
        self.assertEqual(payload["extra_logs"]["physics_llm_step_mode"], 1.0)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
