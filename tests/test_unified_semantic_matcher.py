from __future__ import annotations

import copy
import hashlib
import json
import unittest

from core.unified_semantic_matcher import SemanticSelectionError, UnifiedSemanticMatcher


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_FakeChoice(content, finish_reason)]
        self.model = "fake-model"
        self.id = "fake-response-id"


_FakeReply = str | tuple[str, str] | Exception


class _FakeAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, body: object | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class _FakeCompletions:
    def __init__(self, responses: list[_FakeReply]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> _FakeResponse:
        self.requests.append(request)
        confirmation = self._confirmation_response(request)
        if confirmation is not None:
            return _FakeResponse(confirmation)
        if not self._responses:
            raise AssertionError("No fake responses left.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            content, finish_reason = response
            return _FakeResponse(self._adapt_rule_contract(content, request), finish_reason)
        return _FakeResponse(self._adapt_rule_contract(response, request))

    @staticmethod
    def _confirmation_response(request: dict[str, object]) -> str | None:
        messages = request.get("messages")
        if not isinstance(messages, list):
            return None
        user_content = next(
            (
                str(message.get("content") or "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            "",
        )
        try:
            payload = json.loads(user_content)
        except (TypeError, ValueError):
            return None
        if payload.get("selection_phase") != "background_only_rule_precision_confirmation":
            return None
        candidates = payload.get("preliminary_rules")
        candidates = candidates if isinstance(candidates, list) else []
        decisions = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("rule_id"):
                continue
            decisions.append(
                {
                    "rule_id": candidate["rule_id"],
                    "decision": "confirm",
                    "background_anchor_index": 0,
                }
            )
        return json.dumps({"decisions": decisions})

    @staticmethod
    def _adapt_rule_contract(content: str, request: dict[str, object]) -> str:
        """Keep navigation fixtures focused while production Rule uses source anchors."""
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            return content
        rules = parsed.get("rules") if isinstance(parsed, dict) else None
        if not isinstance(rules, list) or not any(
            isinstance(item, dict) and "applicable" in item for item in rules
        ):
            return content
        response_format = request.get("response_format")
        schema = (
            response_format.get("json_schema", {}).get("schema", {})
            if isinstance(response_format, dict)
            else {}
        )
        item_properties = (
            schema.get("properties", {})
            .get("rules", {})
            .get("items", {})
            .get("properties", {})
            if isinstance(schema, dict)
            else {}
        )
        if "background_anchor_index" not in item_properties:
            return content
        background_values = item_properties.get("background_anchor_index", {}).get("enum", [])
        claim_values = item_properties.get("claim_anchor_index", {}).get("enum", [])
        if not background_values or not claim_values:
            return content
        compact_rules = []
        for item in rules:
            if not isinstance(item, dict) or item.get("applicable") is not True:
                continue
            compact_rules.append(
                {
                    "rule_id": item.get("rule_id"),
                    "score": max(0.8, float(item.get("score") or 0.8)),
                    "background_anchor_index": background_values[0],
                    "claim_anchor_index": claim_values[0],
                }
            )
        return json.dumps({"rules": compact_rules})


class _FakeChat:
    def __init__(self, responses: list[_FakeReply]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[_FakeReply]) -> None:
        self.chat = _FakeChat(responses)

    @property
    def requests(self) -> list[dict[str, object]]:
        return self.chat.completions.requests


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.function = _FakeToolFunction(name, arguments)


class _FakeToolMessage:
    def __init__(self, name: str, arguments: str) -> None:
        self.content = None
        self.tool_calls = [_FakeToolCall(name, arguments)]


class _FakeToolChoice:
    def __init__(self, name: str, arguments: str, finish_reason: str = "stop") -> None:
        self.message = _FakeToolMessage(name, arguments)
        self.finish_reason = finish_reason


class _FakeToolResponse:
    def __init__(self, name: str, arguments: str, finish_reason: str = "stop") -> None:
        self.choices = [_FakeToolChoice(name, arguments, finish_reason)]


class _FakeToolCompletions:
    def __init__(self, arguments: str, finish_reason: str = "stop") -> None:
        self.arguments = arguments
        self.finish_reason = finish_reason
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> _FakeToolResponse:
        self.requests.append(request)
        tool_choice = request["tool_choice"]
        assert isinstance(tool_choice, dict)
        function = tool_choice["function"]
        assert isinstance(function, dict)
        return _FakeToolResponse(
            str(function["name"]),
            self.arguments,
            self.finish_reason,
        )


class _FakeToolClient:
    def __init__(self, arguments: str, finish_reason: str = "stop") -> None:
        self.chat = type("FakeToolChat", (), {})()
        self.chat.completions = _FakeToolCompletions(arguments, finish_reason)

    @property
    def requests(self) -> list[dict[str, object]]:
        return self.chat.completions.requests


def _catalog() -> dict:
    return {
        "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2", "schema_profile": "semantic_navigation_tree_minimal"},
        "domains": [
            {
                "id": "mechanics",
                "name": "Mechanics",
                "summary": "Classical mechanics topics.",
                "topics": [
                    {
                        "id": "mechanics.gravitation_and_keplers_laws",
                        "name": "Gravitation and Kepler's Laws",
                        "summary": "Orbital motion and gravity-governed trajectories.",
                        "rules": [
                            {
                                "rule_id": "exp_orbit_decay",
                                "title": "Orbital decay energy accounting",
                                "summary": "Check satellite orbital decay energy accounting.",
                                "trigger": "satellite orbital decay with drag",
                                "check_logic": "track orbital energy loss consistently",
                                "error_type": "logic",
                                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["G", "M", "r"]},
                            }
                        ],
                        "scenario_clusters": [
                            {
                                "id": "orbital_decay_and_orbit_accounting",
                                "name": "Orbital Decay and Orbit Accounting",
                                "summary": "Satellite orbit change and orbital-energy bookkeeping.",
                                "rule_groups": [
                                    {
                                        "id": "orbit_decay_core_checks",
                                        "name": "Orbit Decay Core Checks",
                                        "summary": "Checks orbital decay accounting.",
                                        "activation_condition": "Use for orbital decay scenarios.",
                                        "rule_ids": ["exp_orbit_decay"],
                                    }
                                ],
                                "rule_ids": ["exp_orbit_decay"],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "modern_physics",
                "name": "Modern Physics",
                "summary": "Relativity and quantum topics.",
                "topics": [
                    {
                        "id": "modern_physics.special_relativity_time_dilation_length_contraction",
                        "name": "Special Relativity (Time Dilation, Length Contraction)",
                        "summary": "Relativistic observation and frame effects.",
                        "rules": [
                            {
                                "rule_id": "exp_pinhole",
                                "title": "Pinhole simultaneity",
                                "summary": "Check simultaneity assumptions in pinhole observations.",
                                "trigger": "pinhole camera sees moving rod",
                                "check_logic": "treat exposure as simultaneous in observer frame",
                                "error_type": "logic",
                                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["L", "v"]},
                            }
                        ],
                        "scenario_clusters": [
                            {
                                "id": "observation_and_projection",
                                "name": "Observation and Projection Geometry",
                                "summary": "Observation tasks where camera timing and projection are central.",
                                "rule_groups": [
                                    {
                                        "id": "projection_checks",
                                        "name": "Projection Checks",
                                        "summary": "Checks relativistic observation geometry.",
                                        "activation_condition": "Use for observed length problems.",
                                        "rule_ids": ["exp_pinhole"],
                                    }
                                ],
                                "rule_ids": ["exp_pinhole"],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _background_analysis(task_focus: str = "Classify the stated physics task.") -> dict:
    return {
        "task_focus": task_focus,
        "objects": [],
        "processes": [],
        "conditions": [],
        "target_quantity": "",
        "symbols_and_units": [],
        "missing_information": [],
        "inactive_context": [],
    }


def _domain_json(domains: list[dict], task_focus: str = "Classify the stated physics task.") -> str:
    return json.dumps(
        {
            "background_analysis": _background_analysis(task_focus),
            "domains": domains,
        }
    )


def _navigation_role_catalog() -> dict:
    def rule(rule_id: str, title: str) -> dict:
        return {
            "rule_id": rule_id,
            "title": title,
            "summary": f"Audit {title.lower()}.",
            "trigger": f"Trigger for {title.lower()}.",
            "check_logic": f"Check {title.lower()} consistently.",
            "error_type": "logic",
            "symbolic_hint": {
                "primitive": "none",
                "canonical": f"canonical {rule_id}",
                "required_symbols": [],
            },
        }

    def cluster(cluster_id: str, name: str, rule_id: str) -> dict:
        return {
            "id": cluster_id,
            "name": name,
            "summary": f"Scenario for {name.lower()}.",
            "rule_ids": [rule_id],
            "rule_groups": [
                {
                    "id": f"{cluster_id}_checks",
                    "name": f"{name} checks",
                    "summary": f"Checks for {name.lower()}.",
                    "activation_condition": f"Use for {name.lower()}.",
                    "rule_ids": [rule_id],
                }
            ],
        }

    rules = [
        rule("r_initial", "Initial scenario rule"),
        rule("r_alt_one", "First alternative rule"),
        rule("r_alt_two", "Second alternative rule"),
        rule("r_embedding", "Embedding fallback rule"),
        rule("r_residual", "Residual fallback rule"),
        rule("r_general", "General reasoning rule"),
    ]
    clusters = [
        cluster("initial_primary", "Initial Primary", "r_initial"),
        cluster("alternative_primary_one", "Alternative Primary One", "r_alt_one"),
        cluster("alternative_primary_two", "Alternative Primary Two", "r_alt_two"),
        cluster("embedding_cluster_01", "Embedding Cluster 01", "r_embedding"),
        cluster("residual_rules_01", "Residual Rules 01", "r_residual"),
        cluster("general_reasoning", "General Reasoning", "r_general"),
    ]
    return {
        "metadata": {
            "version": "2.0",
            "catalog_type": "unified_rules_v2",
            "schema_profile": "semantic_navigation_tree_minimal",
        },
        "domains": [
            {
                "id": "mechanics",
                "name": "Mechanics",
                "summary": "Classical mechanics topics.",
                "topics": [
                    {
                        "id": "mechanics.bounded_navigation",
                        "name": "Bounded Navigation",
                        "summary": "A topic used to test bounded semantic navigation.",
                        "rules": rules,
                        "scenario_clusters": clusters,
                    }
                ],
            }
        ],
    }


class UnifiedSemanticMatcherTests(unittest.TestCase):
    def test_required_provider_identity_rejects_wrong_model(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="expected-model",
            client=_FakeClient(['{"items":[]}']),
            json_retries=0,
            require_provider_identity=True,
        )

        with self.assertRaisesRegex(RuntimeError, "provider model mismatch"):
            matcher._chat_json(
                system_prompt="system",
                user_prompt="user",
                list_key="items",
            )

    def test_request_timeout_is_bounded_and_visible_in_trace(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient([]),
            request_timeout=37,
        )

        self.assertEqual(matcher.request_timeout, 37.0)
        self.assertEqual(
            matcher.last_trace["model_config"]["request_timeout_seconds"],
            37.0,
        )
        self.assertEqual(matcher.last_trace["model_config"]["sdk_max_retries"], 0)

    @staticmethod
    def _selection_schema(list_key: str) -> dict:
        return {
            "type": "object",
            "properties": {
                list_key: {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
            "required": [list_key],
            "additionalProperties": False,
        }

    def test_chat_json_accepts_fenced_json_object(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(['```json\n{"domains":[{"domain":"Mechanics","relevant":true,"score":0.9,"reason":"motion"}]}\n```']),
        )

        result = matcher._chat_json(system_prompt="system", user_prompt="user")

        self.assertEqual(result["domains"][0]["domain"], "Mechanics")

    def test_chat_json_wraps_array_response_for_expected_selection_key(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(['[{"domain":"Mechanics","topic":"Gravitation and Kepler\'s Laws","relevant":true,"score":0.95,"reason":"orbit"}]']),
        )

        result = matcher._chat_json(system_prompt="system", user_prompt="user", list_key="topics")

        self.assertEqual(result["topics"][0]["topic"], "Gravitation and Kepler's Laws")

    def test_chat_json_retries_invalid_content_and_records_bounded_attempts(self) -> None:
        client = _FakeClient(
            [
                "0.00000000000021",
                "not json",
                '{"domains":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            api_key="must-not-appear-in-trace",
            json_retries=3,
            max_response_tokens=999_999,
        )

        result = matcher._chat_json(system_prompt="system", user_prompt="user", list_key="domains")

        self.assertEqual(result, {"domains": []})
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(client.requests[0]["max_tokens"], matcher.HARD_MAX_RESPONSE_TOKENS)
        self.assertTrue(all(request["max_tokens"] <= 512 for request in client.requests[1:]))
        attempts = matcher.last_trace["stages"]["chat_json"]["api_attempts"]
        expected_raw = ["0.00000000000021", "not json", '{"domains":[]}']
        self.assertEqual([item["raw_preview"] for item in attempts], expected_raw)
        self.assertEqual([item["raw_length"] for item in attempts], [len(item) for item in expected_raw])
        self.assertEqual(
            [item["raw_sha256"] for item in attempts],
            [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in expected_raw],
        )
        self.assertTrue(all("raw" not in item for item in attempts))
        self.assertEqual([item["finish_reason"] for item in attempts], ["stop", "stop", "stop"])
        self.assertNotIn("must-not-appear-in-trace", json.dumps(matcher.last_trace, ensure_ascii=False))
        retry_messages = client.requests[1]["messages"]
        self.assertIn("top-level JSON object", retry_messages[-1]["content"])
        self.assertIn("'domains' field is an array", retry_messages[-1]["content"])

    def test_chat_json_requests_strict_json_schema(self) -> None:
        schema = self._selection_schema("domains")
        client = _FakeClient(['{"domains":[]}'])
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher._chat_json(
            system_prompt="system",
            user_prompt="user",
            list_key="domains",
            response_schema=schema,
        )

        self.assertEqual(result, {"domains": []})
        response_format = client.requests[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], schema)
        self.assertTrue(matcher._json_schema_supported)

    def test_rule_anchor_indices_must_be_bounded_integers(self) -> None:
        valid = {
            "rule_id": "r_exact",
            "score": 0.95,
            "background_anchor_index": 0,
            "claim_anchor_index": 0,
        }
        cases = [
            ("background_anchor_index", -1),
            ("background_anchor_index", 1),
            ("background_anchor_index", True),
            ("claim_anchor_index", -1),
            ("claim_anchor_index", 1),
            ("claim_anchor_index", True),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                item = dict(valid)
                item[field] = value
                with self.assertRaises(RuntimeError) as raised:
                    UnifiedSemanticMatcher._validate_rule_selection_contract(
                        {"rules": [item]},
                        resolve_id=lambda candidate: (
                            "r_exact" if candidate.get("rule_id") == "r_exact" else ""
                        ),
                        background_source="A charged conductor is split into two halves.",
                        claim_source="Use electrostatic pressure on the projected area.",
                        allowed_background_anchors=[
                            "A charged conductor is split into two halves."
                        ],
                        allowed_claim_anchors=[
                            "Use electrostatic pressure on the projected area."
                        ],
                        max_items=1,
                    )
                self.assertIn(field, str(raised.exception))

    def test_rule_selection_deduplicates_known_rule_id_after_full_validation(self) -> None:
        response = {
            "rules": [
                {
                    "rule_id": "r_exact",
                    "score": 0.9,
                    "background_anchor_index": 0,
                    "claim_anchor_index": 0,
                },
                {
                    "rule_id": "r_exact",
                    "score": 0.96,
                    "background_anchor_index": 0,
                    "claim_anchor_index": 0,
                },
                {
                    "rule_id": "r_exact",
                    "score": 0.96,
                    "background_anchor_index": 1,
                    "claim_anchor_index": 1,
                },
            ]
        }

        UnifiedSemanticMatcher._validate_rule_selection_contract(
            response,
            resolve_id=lambda item: (
                "r_exact" if item.get("rule_id") == "r_exact" else ""
            ),
            background_source="First background fact. Second background fact.",
            claim_source="First claim. Second claim.",
            allowed_background_anchors=[
                "First background fact.",
                "Second background fact.",
            ],
            allowed_claim_anchors=["First claim.", "Second claim."],
            max_items=1,
        )

        self.assertEqual(len(response["rules"]), 1)
        self.assertEqual(response["rules"][0]["rule_id"], "r_exact")
        self.assertEqual(response["rules"][0]["score"], 0.96)
        self.assertEqual(response["rules"][0]["background_anchor_index"], 0)

    def test_rule_selection_validates_duplicate_items_before_deduplication(self) -> None:
        valid = {
            "rule_id": "r_exact",
            "score": 0.95,
            "background_anchor_index": 0,
            "claim_anchor_index": 0,
        }
        invalid_duplicate = dict(valid)
        invalid_duplicate["claim_anchor_index"] = 1

        with self.assertRaisesRegex(RuntimeError, "claim_anchor_index"):
            UnifiedSemanticMatcher._validate_rule_selection_contract(
                {"rules": [valid, invalid_duplicate]},
                resolve_id=lambda item: (
                    "r_exact" if item.get("rule_id") == "r_exact" else ""
                ),
                background_source="A valid background fact.",
                claim_source="A valid claim.",
                allowed_background_anchors=["A valid background fact."],
                allowed_claim_anchors=["A valid claim."],
                max_items=1,
            )

    def test_rule_selection_still_rejects_unknown_ids_and_unique_overflow(self) -> None:
        def rule(rule_id: str) -> dict:
            return {
                "rule_id": rule_id,
                "score": 0.95,
                "background_anchor_index": 0,
                "claim_anchor_index": 0,
            }

        kwargs = {
            "resolve_id": lambda item: (
                item.get("rule_id") if item.get("rule_id") in {"r_one", "r_two"} else ""
            ),
            "background_source": "A valid background fact.",
            "claim_source": "A valid claim.",
            "allowed_background_anchors": ["A valid background fact."],
            "allowed_claim_anchors": ["A valid claim."],
            "max_items": 1,
        }
        with self.assertRaisesRegex(RuntimeError, "unknown candidate"):
            UnifiedSemanticMatcher._validate_rule_selection_contract(
                {"rules": [rule("r_unknown")]},
                **kwargs,
            )
        with self.assertRaisesRegex(RuntimeError, "unique selected items"):
            UnifiedSemanticMatcher._validate_rule_selection_contract(
                {"rules": [rule("r_one"), rule("r_two")]},
                **kwargs,
            )

    def test_global_rule_confirmation_prunes_cross_topic_false_positive(self) -> None:
        class ConfirmationCompletions:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def create(self, **request: object) -> _FakeResponse:
                self.requests.append(request)
                return _FakeResponse(
                    json.dumps(
                        {
                            "decisions": [
                                {
                                    "rule_id": "r_exact",
                                    "decision": "confirm",
                                    "background_anchor_index": 0,
                                },
                                {
                                    "rule_id": "r_torque_fp",
                                    "decision": "reject_different_configuration",
                                    "background_anchor_index": -1,
                                },
                            ]
                        }
                    )
                )

        completions = ConfirmationCompletions()
        client = type(
            "ConfirmationClient",
            (),
            {"chat": type("ConfirmationChat", (), {"completions": completions})()},
        )()
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            json_retries=0,
            max_selected_rules=6,
        )
        sample = {
            "question": "A charged conductor is split into two halves.",
            "context": "",
            "prediction": "Use electrostatic pressure on the projected area.",
        }

        def judgment(rule_id: str, domain: str, topic: str, trigger: str) -> dict:
            return {
                "rule_id": rule_id,
                "title": rule_id,
                "domain": domain,
                "topic": topic,
                "topic_id": topic,
                "cluster_id": "cluster",
                "cluster": "cluster",
                "score": 0.95,
                "background_anchor": sample["question"],
                "claim_anchor": sample["prediction"],
                "rule_obj": {
                    "rule_id": rule_id,
                    "title": rule_id,
                    "summary": trigger,
                    "trigger": trigger,
                    "check_logic": "Audit the stated step.",
                    "preconditions": [],
                    "negative_conditions": [],
                    "evidence_requirements": [],
                },
            }

        confirmed = matcher._confirm_rule_judgments(
            sample=sample,
            background_analysis=_background_analysis("Compute conductor force."),
            judgments=[
                judgment(
                    "r_exact",
                    "Electromagnetism",
                    "Conductors",
                    "charged conductor halves and electrostatic pressure",
                ),
                judgment(
                    "r_torque_fp",
                    "Mechanics",
                    "Torque",
                    "rigid-body pivot and moment-arm balance",
                ),
            ],
        )

        self.assertEqual([item["rule_id"] for item in confirmed], ["r_exact"])
        payload = json.loads(completions.requests[0]["messages"][-1]["content"])
        self.assertEqual(payload["selection_phase"], "background_only_rule_precision_confirmation")
        self.assertNotIn("student_solution", payload)
        self.assertNotIn("max_confirmed_rules", payload)
        self.assertNotIn(sample["prediction"], json.dumps(payload))
        confirmation_system_prompt = completions.requests[0]["messages"][0]["content"]
        self.assertNotIn("claim option", confirmation_system_prompt)
        self.assertNotIn("reject_redundant_or_weaker", confirmation_system_prompt)
        self.assertEqual(
            {item["rule_id"] for item in payload["preliminary_rules"]},
            {"r_exact", "r_torque_fp"},
        )
        rejected = matcher.last_trace["stages"]["rule_confirmation"]["rejected"]
        self.assertTrue(
            any(
                record["reason"] == "reject_different_configuration"
                and record["item"]["rule_id"] == "r_torque_fp"
                for record in rejected
            )
        )
        self.assertEqual(matcher._active_stage, "rule")

    def test_forced_tool_call_extracts_schema_valid_arguments(self) -> None:
        schema = self._selection_schema("rules")
        for finish_reason in ("stop", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                client = _FakeToolClient('{"rules":[]}', finish_reason=finish_reason)
                matcher = UnifiedSemanticMatcher(
                    model="fake-model",
                    client=client,
                    structured_output_adapter="forced_tool_call",
                )

                result = matcher._chat_json(
                    system_prompt="system",
                    user_prompt="user",
                    list_key="rules",
                    response_schema=schema,
                    schema_name="rule_selection",
                )

                self.assertEqual(result, {"rules": []})
                request = client.requests[0]
                self.assertNotIn("response_format", request)
                self.assertEqual(request["tools"][0]["function"]["parameters"], schema)
                self.assertTrue(request["tools"][0]["function"]["strict"])
                self.assertFalse(request["parallel_tool_calls"])
                self.assertEqual(
                    request["tool_choice"]["function"]["name"],
                    request["tools"][0]["function"]["name"],
                )
                attempt = matcher.last_trace["stages"]["chat_json"]["api_attempts"][0]
                self.assertEqual(attempt["response_format"], "forced_tool_call")
                self.assertEqual(attempt["finish_reason"], finish_reason)
                self.assertEqual(attempt["raw_preview"], '{"rules":[]}')
                self.assertEqual(attempt["tool_call_count"], 1)
                self.assertEqual(
                    attempt["tool_call_names"],
                    [request["tool_choice"]["function"]["name"]],
                )

    def test_forced_tool_call_repairs_only_surplus_closing_delimiters(self) -> None:
        schema = self._selection_schema("rules")
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeToolClient('{"rules":[]} ]'),
            structured_output_adapter="forced_tool_call",
            json_retries=0,
        )

        result = matcher._chat_json(
            system_prompt="system",
            user_prompt="user",
            list_key="rules",
            response_schema=schema,
            schema_name="rule_selection",
        )

        self.assertEqual(result, {"rules": []})
        attempt = matcher.last_trace["stages"]["chat_json"]["api_attempts"][0]
        self.assertEqual(
            attempt["structured_repair"],
            "ignored_surplus_closing_delimiters",
        )

        invalid = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeToolClient('{"rules":[]} trailing prose'),
            structured_output_adapter="forced_tool_call",
            json_retries=0,
        )
        with self.assertRaisesRegex(RuntimeError, "exact JSON object"):
            invalid._chat_json(
                system_prompt="system",
                user_prompt="user",
                list_key="rules",
                response_schema=schema,
                schema_name="rule_selection",
            )

    def test_selection_schema_rejects_empty_candidate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "without candidate IDs"):
            UnifiedSemanticMatcher._selection_response_schema(
                list_key="rules",
                id_key="rule_id",
                bool_key="applicable",
                allowed_ids=["", "  "],
                max_items=2,
            )

    def test_forced_tool_call_fails_closed_when_endpoint_returns_content(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient([('{"rules":[]}', "tool_calls")]),
            structured_output_adapter="forced_tool_call",
            json_retries=0,
        )

        with self.assertRaisesRegex(RuntimeError, "Forced semantic tool call"):
            matcher._chat_json(
                system_prompt="system",
                user_prompt="user",
                list_key="rules",
                response_schema=self._selection_schema("rules"),
            )

        attempt = matcher.last_trace["stages"]["chat_json"]["api_attempts"][0]
        self.assertEqual(attempt["finish_reason"], "tool_calls")
        self.assertEqual(attempt["raw_preview"], '{"rules":[]}')
        self.assertIn("tool call was not returned", attempt["error"])

    def test_invalid_json_schema_response_is_not_marked_supported(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(["0.00000000000021"]),
            json_retries=0,
        )

        with self.assertRaisesRegex(RuntimeError, "not enforced"):
            matcher._chat_json(
                system_prompt="system",
                user_prompt="user",
                list_key="rules",
                response_schema=self._selection_schema("rules"),
            )

        self.assertFalse(matcher._json_schema_supported)

    def test_unsupported_json_schema_falls_back_for_current_request_and_is_cached(self) -> None:
        client = _FakeClient(
            [
                _FakeAPIError(
                    "response_format type json_schema is unsupported",
                    status_code=400,
                    body={"error": "structured output is not supported"},
                ),
                '{"domains":[]}',
                '{"domains":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            json_retries=0,
            allow_json_object_fallback=True,
        )
        schema = self._selection_schema("domains")

        first = matcher._chat_json(
            system_prompt="system",
            user_prompt="first request",
            list_key="domains",
            response_schema=schema,
        )
        second = matcher._chat_json(
            system_prompt="system",
            user_prompt="second request",
            list_key="domains",
            response_schema=schema,
        )

        self.assertEqual(first, {"domains": []})
        self.assertEqual(second, {"domains": []})
        self.assertEqual(
            [request["response_format"]["type"] for request in client.requests],
            ["json_schema", "json_object", "json_object"],
        )
        self.assertFalse(matcher._json_schema_supported)
        attempts = matcher.last_trace["stages"]["chat_json"]["api_attempts"]
        self.assertEqual(attempts[0]["error"], "json_schema_unsupported_fallback")
        self.assertEqual(attempts[0]["response_format"], "json_schema")

    def test_unsupported_json_schema_fails_closed_by_default(self) -> None:
        client = _FakeClient(
            [
                _FakeAPIError(
                    "response_format type json_schema is unsupported",
                    status_code=400,
                )
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=3)

        with self.assertRaisesRegex(RuntimeError, "strict json_schema"):
            matcher._chat_json(
                system_prompt="system",
                user_prompt="request",
                list_key="rules",
                response_schema=self._selection_schema("rules"),
            )

        self.assertEqual(len(client.requests), 1)
        self.assertIsNone(matcher._json_schema_supported)
        attempt = matcher.last_trace["stages"]["chat_json"]["api_attempts"][0]
        self.assertEqual(attempt["error"], "json_schema_required_but_unsupported")

    def test_non_compatibility_api_errors_do_not_fall_back_from_json_schema(self) -> None:
        cases = [
            (500, "response_format json_schema is unsupported due to an internal failure"),
            (401, "authentication failed for the supplied API key"),
        ]
        for status_code, message in cases:
            with self.subTest(status_code=status_code):
                client = _FakeClient(
                    [_FakeAPIError(message, status_code=status_code, body={"error": message})]
                )
                matcher = UnifiedSemanticMatcher(
                    model="fake-model",
                    client=client,
                    json_retries=3,
                )

                with self.assertRaises(_FakeAPIError):
                    matcher._chat_json(
                        system_prompt="system",
                        user_prompt="request",
                        list_key="rules",
                        response_schema=self._selection_schema("rules"),
                    )

                self.assertEqual(len(client.requests), 1)
                self.assertEqual(client.requests[0]["response_format"]["type"], "json_schema")
                self.assertIsNone(matcher._json_schema_supported)
                attempt = matcher.last_trace["stages"]["chat_json"]["api_attempts"][0]
                self.assertEqual(attempt["response_format"], "json_schema")
                self.assertIn(message, attempt["error"])

    def test_length_numeric_stream_fails_fast_without_retry(self) -> None:
        numeric_stream = "1." + ("0" * 5_000)
        client = _FakeClient(
            [
                (numeric_stream, "length"),
                ('{"rules":[]}', "stop"),
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            json_retries=1,
            max_response_tokens=2_048,
        )

        with self.assertRaisesRegex(RuntimeError, "numeric_stream_degeneration"):
            matcher._chat_json(
                system_prompt="system",
                user_prompt="select applicable rules",
                list_key="rules",
                response_schema=self._selection_schema("rules"),
            )

        self.assertEqual(len(client.requests), 1)
        attempts = matcher.last_trace["stages"]["chat_json"]["api_attempts"]
        self.assertEqual([item["finish_reason"] for item in attempts], ["length"])
        self.assertEqual(
            attempts[0]["format_violation_kind"],
            "numeric_stream_degeneration",
        )

    def test_long_invalid_response_trace_is_bounded_and_hashed(self) -> None:
        numeric_stream = "0." + ("1234567890" * 1_000)
        client = _FakeClient([(numeric_stream, "length")])
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            json_retries=0,
        )

        with self.assertRaisesRegex(RuntimeError, "numeric_stream_degeneration"):
            matcher._chat_json(
                system_prompt="system",
                user_prompt="select relevant clusters",
                list_key="clusters",
                response_schema=self._selection_schema("clusters"),
            )

        attempt = matcher.last_trace["stages"]["chat_json"]["api_attempts"][0]
        self.assertNotIn("raw", attempt)
        self.assertEqual(attempt["raw_length"], len(numeric_stream))
        self.assertLessEqual(len(attempt["raw_preview"]), 300)
        self.assertNotEqual(attempt["raw_preview"], numeric_stream)
        self.assertEqual(
            attempt["raw_sha256"],
            hashlib.sha256(numeric_stream.encode("utf-8")).hexdigest(),
        )

    def test_chat_json_stops_after_initial_request_plus_three_retries(self) -> None:
        client = _FakeClient(["bad-1", "bad-2", "bad-3", "bad-4", '{"rules":[]}'])
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=99)

        with self.assertRaisesRegex(RuntimeError, "after 4 attempts"):
            matcher._chat_json(system_prompt="system", user_prompt="user", list_key="rules")

        self.assertEqual(len(client.requests), 4)
        self.assertEqual(len(matcher.last_trace["stages"]["chat_json"]["api_attempts"]), 4)

    def test_chat_json_retries_missing_or_non_array_selection_key(self) -> None:
        client = _FakeClient(
            [
                '{"wrong":[]}',
                '{"domains":{}}',
                '{"domains":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=2)

        result = matcher._chat_json(system_prompt="system", user_prompt="user", list_key="domains")

        self.assertEqual(result, {"domains": []})
        self.assertEqual(len(client.requests), 3)
        errors = [
            item.get("error", "")
            for item in matcher.last_trace["stages"]["chat_json"]["api_attempts"]
        ]
        self.assertIn("'domains' array", errors[0])
        self.assertIn("'domains' array", errors[1])

    def test_domain_retries_missing_or_wrong_background_analysis(self) -> None:
        domain_item = {
            "domain": "Mechanics",
            "relevant": True,
            "score": 0.9,
            "reason": "orbital motion",
        }
        valid_analysis = _background_analysis("Classify a satellite-orbit problem.")
        client = _FakeClient(
            [
                json.dumps({"domains": [domain_item]}),
                json.dumps({"background_analysis": [], "domains": [domain_item]}),
                json.dumps({"background_analysis": valid_analysis, "domains": [domain_item]}),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=3)

        result = matcher.select_tree_semantically(
            {"question": "A satellite moves in orbit.", "context": "", "prediction": "claim"},
            _catalog(),
        )

        self.assertEqual(result["background_analysis"], valid_analysis)
        attempts = result["navigation_trace"]["stages"]["domain"]["api_attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertIn("background_analysis", attempts[0]["error"])
        self.assertIn("background_analysis", attempts[1]["error"])
        topic_payload = json.loads(client.requests[-2]["messages"][-1]["content"])
        self.assertEqual(topic_payload["background_analysis"], valid_analysis)

    def test_background_analysis_and_topic_id_flow_through_navigation(self) -> None:
        catalog = copy.deepcopy(_catalog())
        topic = catalog["domains"][1]["topics"][0]
        topic["retrieval_hints"] = {
            "scene_keywords": ["pinhole observation", "moving rod"],
            "llm_discriminative_terms": ["observer-frame exposure"],
        }
        analysis = {
            "task_focus": "audit the observed rod length",
            "objects": ["rod", "pinhole camera"],
            "processes": ["relativistic observation"],
            "conditions": ["camera frame is specified"],
            "target_quantity": "observed length",
            "symbols_and_units": ["v: speed"],
            "missing_information": [],
            "inactive_context": ["unrelated apparatus history"],
        }
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "background_analysis": analysis,
                        "domains": [
                            {
                                "domain_id": "modern_physics",
                                "relevant": True,
                                "score": 0.95,
                                "reason": "relativistic observation",
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": "modern_physics.special_relativity_time_dilation_length_contraction",
                                "relevant": True,
                                "score": 0.94,
                                "reason": "moving rod observation",
                            }
                        ]
                    }
                ),
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.9,"reason":"pinhole"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"auditable"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        sample = {
            "question": "RAW_QUESTION: A pinhole camera observes a moving rod.",
            "context": "RAW_CONTEXT: Work in the camera frame.",
            "prediction": "UNTRUSTED_PREDICTION: exposure is simultaneous.",
        }

        result = matcher.select_tree_semantically(sample, catalog)

        self.assertEqual(result["background_analysis"], analysis)
        self.assertEqual(result["domain_judgments"][0]["domain_id"], "modern_physics")
        self.assertEqual(
            result["selected_topics"][0]["topic_id"],
            "modern_physics.special_relativity_time_dilation_length_contraction",
        )
        payloads = [
            json.loads(next(message for message in request["messages"] if message["role"] == "user")["content"])
            for request in client.requests
        ]
        expected_list_keys = ["domains", "topics", "clusters", "rules"]
        for request, list_key in zip(client.requests, expected_list_keys):
            response_format = request["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            response_schema = response_format["json_schema"]["schema"]
            self.assertIn(list_key, response_schema["required"])
            self.assertFalse(response_schema["additionalProperties"])
            self.assertGreater(response_schema["properties"][list_key]["maxItems"], 0)
        topic_candidate = payloads[1]["candidate_topics"][0]
        self.assertEqual(topic_candidate["topic_id"], result["selected_topics"][0]["topic_id"])
        self.assertIn("retrieval_hints", topic_candidate)
        self.assertIn("cluster_previews", topic_candidate)
        for payload in payloads[1:]:
            self.assertEqual(payload["background_analysis"], analysis)
            self.assertIn(sample["question"], payload["problem_background"])
            self.assertIn(sample["context"], payload["problem_background"])
        self.assertNotIn(sample["prediction"], json.dumps(payloads[:3], ensure_ascii=False))
        self.assertEqual(payloads[3]["student_solution"], sample["prediction"])
        stages = result["navigation_trace"]["stages"]
        self.assertEqual(
            {item["domain_id"] for item in stages["domain"]["candidates"]},
            {"mechanics", "modern_physics"},
        )
        self.assertIn(
            "mechanics",
            {item["domain_id"] for item in stages["domain"]["not_selected"]},
        )
        self.assertEqual(stages["domain"]["not_selected"][0]["reason"], "not_returned_by_model")
        self.assertIn("exp_pinhole", {item["rule_id"] for item in stages["rule"]["candidates"]})
        self.assertEqual(stages["rule"]["candidates"][0]["batch_index"], 1)

    def test_invalid_domain_items_trigger_contract_retry(self) -> None:
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "background_analysis": _background_analysis(),
                        "domains": [
                            {"domain": "Mechanics", "relevant": "false", "score": 1.0, "reason": "bad bool"},
                            {"domain": "Unknown Domain", "relevant": True, "score": 1.0, "reason": "unknown"},
                        ]
                    }
                ),
                _domain_json(
                    [{"domain": "Mechanics", "relevant": True, "score": 0.9, "reason": "valid"}]
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A satellite moves in orbit.", "context": "", "prediction": ""},
            _catalog(),
        )

        self.assertEqual(result["selected_domains"], ["Mechanics"])
        attempts = result["navigation_trace"]["stages"]["domain"]["api_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIn("JSON boolean", attempts[0]["error"])

    def test_out_of_range_selection_score_triggers_retry(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Mechanics", "relevant": True, "score": 95, "reason": "invalid scale"}]
                ),
                _domain_json(
                    [{"domain": "Mechanics", "relevant": True, "score": 0.95, "reason": "valid scale"}]
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A mechanics problem.", "context": "", "prediction": "claim"},
            _catalog(),
        )

        attempts = result["navigation_trace"]["stages"]["domain"]["api_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIn("0 to 1", attempts[0]["error"])

    def test_stage_failure_exposes_partial_result_and_raw_attempts(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 0.9, "reason": "modern"}]
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":0.9,"reason":"relativity"}]}',
                "0.1",
                "0.2",
                "0.3",
                "0.4",
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        with self.assertRaises(SemanticSelectionError) as caught:
            matcher.select_tree_semantically(
                {"question": "A pinhole camera observes a moving rod.", "context": "", "prediction": "claim"},
                _catalog(),
            )

        error = caught.exception
        self.assertEqual(error.stage, "cluster")
        self.assertEqual(error.partial_result["selected_domains"], ["Modern Physics"])
        self.assertEqual(len(error.partial_result["selected_topics"]), 1)
        self.assertEqual(error.trace["status"], "failed")
        self.assertEqual(error.trace["terminal_stage"], "cluster")
        attempts = error.trace["stages"]["cluster"]["api_attempts"]
        self.assertEqual([item["raw_preview"] for item in attempts], ["0.1"])
        self.assertEqual([item["raw_length"] for item in attempts], [3])
        self.assertEqual(attempts[0]["format_violation_kind"], "non_object_json_root")
        self.assertTrue(all("raw" not in item for item in attempts))

    def test_second_topic_cluster_failure_keeps_first_topic_cluster(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [
                        {"domain": "Mechanics", "relevant": True, "score": 1.0, "reason": "orbit"},
                        {"domain": "Modern Physics", "relevant": True, "score": 0.9, "reason": "observation"},
                    ]
                ),
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": "mechanics.gravitation_and_keplers_laws",
                                "relevant": True,
                                "score": 1.0,
                                "reason": "orbit",
                            },
                            {
                                "topic_id": "modern_physics.special_relativity_time_dilation_length_contraction",
                                "relevant": True,
                                "score": 0.9,
                                "reason": "observation",
                            },
                        ]
                    }
                ),
                '{"clusters":[{"cluster_id":"orbital_decay_and_orbit_accounting","relevant":true,"score":0.95,"reason":"orbit decay"}]}',
                "bad-1",
                "bad-2",
                "bad-3",
                "bad-4",
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        with self.assertRaises(SemanticSelectionError) as caught:
            matcher.select_tree_semantically(
                {"question": "Compare an orbit and an observation setup.", "context": "", "prediction": "claim"},
                _catalog(),
            )

        self.assertEqual(caught.exception.stage, "cluster")
        self.assertEqual(
            [item["cluster_id"] for item in caught.exception.partial_result["selected_clusters"]],
            ["orbital_decay_and_orbit_accounting"],
        )

    def test_later_rule_batch_failure_keeps_earlier_rule_hit(self) -> None:
        rules = [
            {
                "rule_id": f"r{index}",
                "title": f"Rule {index}",
                "summary": "A candidate check.",
                "trigger": "stated setup",
                "check_logic": "audit the claim",
                "error_type": "logic",
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
            }
            for index in range(2)
        ]
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": "test_domain",
                    "name": "Test Domain",
                    "summary": "A test domain.",
                    "topics": [
                        {
                            "id": "test_domain.test_topic",
                            "name": "Test Topic",
                            "summary": "A clusterless test topic.",
                            "rules": rules,
                            "scenario_clusters": [],
                        }
                    ],
                }
            ],
        }
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain_id": "test_domain", "relevant": True, "score": 1.0, "reason": "test"}]
                ),
                '{"topics":[{"topic_id":"test_domain.test_topic","relevant":true,"score":1.0,"reason":"test"}]}',
                '{"rules":[{"rule_id":"r0","applicable":true,"score":0.9,"reason":"first hit"}]}',
                "bad-1",
                "bad-2",
                "bad-3",
                "bad-4",
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            rule_candidate_batch_size=1,
        )

        with self.assertRaises(SemanticSelectionError) as caught:
            matcher.select_tree_semantically(
                {"question": "A test setup.", "context": "", "prediction": "A claim."},
                catalog,
            )

        self.assertEqual(caught.exception.stage, "rule")
        self.assertEqual(
            [item["rule_id"] for item in caught.exception.partial_result["selected_rules"]],
            ["r0"],
        )
        rule_trace = caught.exception.trace["stages"]["rule"]
        failed_request_index = max(
            item["request_index"] for item in rule_trace["api_attempts"] if item.get("error")
        )
        candidate_keys = {
            (item["request_index"], item["context_id"], item["rule_id"])
            for item in rule_trace["candidates"]
            if item["request_index"] == failed_request_index
        }
        failed_items = [
            item
            for item in rule_trace["not_selected"]
            if item.get("reason") == "stage_failed_before_classification"
        ]
        failed_keys = {
            (item["request_index"], item["context_id"], item["rule_id"])
            for item in failed_items
        }
        self.assertEqual(failed_keys, candidate_keys)
        self.assertEqual({item["rule_id"] for item in failed_items}, {"r1"})
        self.assertEqual({item["request_index"] for item in failed_items}, {2})
        self.assertTrue(all(item["context_id"] for item in failed_items))
        self.assertTrue(all(item["failed_stage"] == "rule" for item in failed_items))

    def test_large_topic_pool_uses_shortlist_before_detailed_selection(self) -> None:
        topics = []
        for index in range(9):
            rule_id = f"rule_{index}"
            topics.append(
                {
                    "id": f"mechanics.topic_{index}",
                    "name": f"Mechanics Topic {index}",
                    "summary": f"Physical mechanism {index}.",
                    "retrieval_hints": {
                        "scene_keywords": [f"scene {index}"],
                        "llm_discriminative_terms": [f"mechanism {index}"],
                    },
                    "rules": [{"rule_id": rule_id}],
                    "scenario_clusters": [
                        {
                            "id": f"scenario_{index}",
                            "name": f"Scenario {index}",
                            "summary": f"Scenario summary {index}.",
                            "rule_ids": [rule_id],
                            "rule_groups": [],
                        }
                    ],
                }
            )
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": "mechanics",
                    "name": "Mechanics",
                    "summary": "Mechanics topics.",
                    "topics": topics,
                }
            ],
        }
        shortlist_ids = [f"mechanics.topic_{index}" for index in range(6)]
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": topic_id,
                                "relevant": True,
                                "score": 1.0 - index * 0.05,
                                "reason": "keep for detailed confirmation",
                            }
                            for index, topic_id in enumerate(shortlist_ids)
                        ]
                    }
                ),
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": "mechanics.topic_5",
                                "relevant": True,
                                "score": 0.95,
                                "reason": "best detailed match",
                            }
                        ]
                    }
                ),
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_topics_semantically(
            {"question": "A mechanics mechanism must be classified.", "context": ""},
            catalog,
            ["Mechanics"],
            _background_analysis("Classify the active mechanics mechanism."),
        )

        self.assertEqual(len(client.requests), 2)
        payloads = [
            json.loads(request["messages"][-1]["content"])
            for request in client.requests
        ]
        shortlist_payload, detailed_payload = payloads
        self.assertEqual(shortlist_payload["selection_phase"], "recall_first_topic_shortlist")
        self.assertEqual(len(shortlist_payload["candidate_topics"]), 9)
        self.assertTrue(
            all(
                "retrieval_hints" not in candidate and "cluster_previews" not in candidate
                for candidate in shortlist_payload["candidate_topics"]
            )
        )
        self.assertTrue(
            all("coarse_anchors" in candidate for candidate in shortlist_payload["candidate_topics"])
        )
        self.assertLessEqual(
            len(detailed_payload["candidate_topics"]),
            matcher.MAX_TOPIC_SHORTLIST,
        )
        self.assertEqual(
            {candidate["topic_id"] for candidate in detailed_payload["candidate_topics"]},
            set(shortlist_ids),
        )
        self.assertTrue(
            all(
                "retrieval_hints" in candidate and "cluster_previews" in candidate
                for candidate in detailed_payload["candidate_topics"]
            )
        )
        self.assertEqual(
            [item["topic_id"] for item in result["selected_topics"]],
            ["mechanics.topic_5"],
        )
        self.assertEqual(
            matcher.last_trace["stages"]["topic_shortlist"]["candidate_count"],
            9,
        )
        self.assertEqual(
            matcher.last_trace["stages"]["topic"]["candidate_count"],
            matcher.MAX_TOPIC_SHORTLIST,
        )

    def test_initial_cluster_prompt_excludes_fallback_roles_and_has_representatives(self) -> None:
        catalog = _navigation_role_catalog()
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "clusters": [
                            {
                                "cluster_id": "initial_primary",
                                "relevant": True,
                                "score": 0.95,
                                "reason": "initial concrete scenario",
                            }
                        ]
                    }
                )
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        topic_match = matcher._build_topic_candidates(catalog, ["Mechanics"])[0]

        result = matcher.select_clusters_semantically(
            {"question": "Use the initial concrete mechanics scenario.", "context": ""},
            [topic_match],
            _background_analysis("Identify the concrete mechanics scenario."),
        )

        self.assertEqual(len(client.requests), 1)
        payload = json.loads(client.requests[0]["messages"][-1]["content"])
        prompt_candidates = payload["candidate_clusters"]
        self.assertEqual(
            {item["cluster_id"] for item in prompt_candidates},
            {
                "initial_primary",
                "alternative_primary_one",
                "alternative_primary_two",
            },
        )
        self.assertTrue(
            all(item["navigation_role"] == "primary" for item in prompt_candidates)
        )
        self.assertTrue(all(item["representative_rules"] for item in prompt_candidates))
        self.assertEqual(
            prompt_candidates[0]["representative_rules"][0]["rule_id"],
            "r_initial",
        )
        deferred = {
            item["cluster_id"]
            for item in matcher.last_trace["stages"]["cluster"]["not_selected"]
            if item.get("reason") == "deferred_by_navigation_role"
        }
        self.assertEqual(
            deferred,
            {"embedding_cluster_01", "residual_rules_01", "general_reasoning"},
        )
        self.assertEqual(
            [item["cluster_id"] for item in result["selected_clusters"]],
            ["initial_primary"],
        )

    def test_deferred_bucket_exposes_more_representative_rules_than_primary(self) -> None:
        catalog = _navigation_role_catalog()
        topic = catalog["domains"][0]["topics"][0]
        residual = next(
            cluster
            for cluster in topic["scenario_clusters"]
            if cluster["id"] == "residual_rules_01"
        )
        for index in range(2, 9):
            rule_id = f"r_residual_{index}"
            topic["rules"].append(
                {
                    "rule_id": rule_id,
                    "title": f"Residual rule {index}",
                    "summary": f"Audit residual mechanism {index}.",
                    "trigger": f"Residual mechanism {index}.",
                    "check_logic": f"Check residual mechanism {index}.",
                    "error_type": "logic",
                    "symbolic_hint": {
                        "primitive": "none",
                        "canonical": "",
                        "required_symbols": [],
                    },
                }
            )
            residual["rule_ids"].append(rule_id)

        matcher = UnifiedSemanticMatcher(model="fake-model", client=_FakeClient([]))
        topic_match = matcher._build_topic_candidates(catalog, ["Mechanics"])[0]
        candidates = {
            item["cluster_id"]: item
            for item in matcher._build_cluster_candidates(topic_match)
        }

        self.assertEqual(
            len(candidates["initial_primary"]["representative_rules"]),
            1,
        )
        self.assertEqual(
            len(candidates["residual_rules_01"]["representative_rules"]),
            8,
        )
        self.assertEqual(
            candidates["residual_rules_01"]["representative_rules"][-1]["rule_id"],
            "r_residual_8",
        )

    def test_topic_shortlist_keeps_one_candidate_per_selected_domain(self) -> None:
        domains = []
        returned_topics = []
        for domain_index, domain_name in enumerate(("Domain A", "Domain B")):
            topics = []
            for topic_index in range(6):
                topic_id = f"domain_{domain_index}.topic_{topic_index}"
                rule_id = f"rule_{domain_index}_{topic_index}"
                topics.append(
                    {
                        "id": topic_id,
                        "name": f"Topic {domain_index}-{topic_index}",
                        "summary": "A candidate mechanism.",
                        "rules": [{"rule_id": rule_id}],
                        "scenario_clusters": [],
                    }
                )
                returned_topics.append(
                    {
                        "topic_id": topic_id,
                        "relevant": True,
                        "score": 0.99 - domain_index * 0.8 - topic_index * 0.01,
                        "reason": "shortlist candidate",
                    }
                )
            domains.append(
                {
                    "id": f"domain_{domain_index}",
                    "name": domain_name,
                    "summary": "A selected physics domain.",
                    "topics": topics,
                }
            )
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": domains,
        }
        client = _FakeClient(
            [
                json.dumps({"topics": returned_topics[:6]}),
                json.dumps({"topics": [returned_topics[0], returned_topics[6]]}),
                '{"topics":[{"topic_id":"domain_1.topic_0","relevant":true,"score":0.9,"reason":"domain B mechanism"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_topics_semantically(
            {"question": "A coupled two-domain problem.", "context": ""},
            catalog,
            ["Domain A", "Domain B"],
            _background_analysis("Classify both active mechanisms."),
        )

        self.assertEqual(len(client.requests), 3)
        detailed_payload = json.loads(client.requests[2]["messages"][-1]["content"])
        detailed_ids = {
            item["topic_id"] for item in detailed_payload["candidate_topics"]
        }
        self.assertEqual(len(detailed_ids), 2)
        self.assertIn("domain_0.topic_0", detailed_ids)
        self.assertIn("domain_1.topic_0", detailed_ids)
        self.assertEqual(
            [item["topic_id"] for item in result["selected_topics"]],
            ["domain_1.topic_0"],
        )

    def test_empty_rules_use_one_primary_backtrack_and_one_deferred_fallback(self) -> None:
        catalog = _navigation_role_catalog()
        client = _FakeClient(
            [
                '{"rules":[]}',
                '{"clusters":[{"cluster_id":"alternative_primary_one","relevant":true,"score":0.9,"reason":"bounded alternative"}]}',
                '{"rules":[]}',
                '{"clusters":[{"cluster_id":"embedding_cluster_01","relevant":true,"score":0.85,"reason":"bounded deferred bucket"}]}',
                '{"rules":[{"rule_id":"r_embedding","applicable":true,"score":0.92,"reason":"fallback rule applies"}]}',
                '{"rules":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        topic_match = matcher._build_topic_candidates(catalog, ["Mechanics"])[0]
        initial_cluster = next(
            item
            for item in matcher._build_cluster_candidates(topic_match)
            if item["cluster_id"] == "initial_primary"
        )

        result = matcher.select_rules_semantically(
            {
                "question": "The initial scenario does not match; inspect bounded alternatives.",
                "context": "",
                "prediction": "A proposed mechanics derivation.",
            },
            [topic_match],
            [initial_cluster],
            _background_analysis("Audit the active mechanics scenario."),
        )

        self.assertEqual(len(client.requests), 6)
        payloads = [
            json.loads(request["messages"][-1]["content"])
            for request in client.requests
        ]
        cluster_payloads = [payload for payload in payloads if "candidate_clusters" in payload]
        self.assertEqual(
            [payload["selection_phase"] for payload in cluster_payloads],
            ["primary_cluster_backtrack", "deferred_cluster_fallback"],
        )
        self.assertEqual(
            [
                matcher.last_trace["stages"][stage]["chat_call_count"]
                for stage in ("cluster_backtrack", "cluster_fallback")
            ],
            [1, 1],
        )
        rule_payloads = [payload for payload in payloads if "candidate_rules" in payload]
        self.assertEqual(
            [payload["candidate_source"] for payload in rule_payloads],
            [
                "selected_cluster",
                "primary_cluster_backtrack",
                "deferred_cluster_fallback",
            ],
        )
        attempted_rule_ids = {
            candidate["rule_id"]
            for payload in rule_payloads
            for candidate in payload["candidate_rules"]
        }
        self.assertEqual(attempted_rule_ids, {"r_initial", "r_alt_one", "r_embedding"})
        self.assertNotIn("topic_fallback", {payload["candidate_source"] for payload in rule_payloads})
        self.assertEqual(
            [item["rule_id"] for item in result["selected_rules"]],
            ["r_embedding"],
        )
        self.assertEqual(
            [item["cluster_id"] for item in result["navigation_clusters"]],
            ["alternative_primary_one", "embedding_cluster_01"],
        )

    def test_confirmation_rejection_recovers_to_untried_deferred_bucket(self) -> None:
        class RecoveryCompletions:
            def __init__(self) -> None:
                self.responses = [
                    '{"rules":[{"rule_id":"r_initial","applicable":true,"score":0.95,"reason":"provisional"}]}',
                    '{"clusters":[{"cluster_id":"alternative_primary_one","relevant":true,"score":0.9,"reason":"bounded primary recovery"}]}',
                    '{"rules":[{"rule_id":"r_alt_one","applicable":true,"score":0.9,"reason":"provisional alternative"}]}',
                    '{"clusters":[{"cluster_id":"residual_rules_01","relevant":true,"score":0.92,"reason":"matching deferred rule"}]}',
                    '{"rules":[{"rule_id":"r_residual","applicable":true,"score":0.96,"reason":"matching rule"}]}',
                ]
                self.requests: list[dict[str, object]] = []

            def create(self, **request: object) -> _FakeResponse:
                self.requests.append(request)
                messages = request.get("messages")
                assert isinstance(messages, list)
                payload = json.loads(messages[-1]["content"])
                if (
                    payload.get("selection_phase")
                    == "background_only_rule_precision_confirmation"
                ):
                    decisions = []
                    for candidate in payload["preliminary_rules"]:
                        confirmed = candidate["rule_id"] == "r_residual"
                        decisions.append(
                            {
                                "rule_id": candidate["rule_id"],
                                "decision": (
                                    "confirm"
                                    if confirmed
                                    else "reject_missing_background"
                                ),
                                "background_anchor_index": 0 if confirmed else -1,
                            }
                        )
                    return _FakeResponse(json.dumps({"decisions": decisions}))
                if not self.responses:
                    raise AssertionError("No fake recovery response remains.")
                content = self.responses.pop(0)
                return _FakeResponse(
                    _FakeCompletions._adapt_rule_contract(content, request)
                )

        completions = RecoveryCompletions()
        client = type(
            "RecoveryClient",
            (),
            {
                "chat": type(
                    "RecoveryChat",
                    (),
                    {"completions": completions},
                )()
            },
        )()
        catalog = _navigation_role_catalog()
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        topic_match = matcher._build_topic_candidates(catalog, ["Mechanics"])[0]
        initial_cluster = next(
            item
            for item in matcher._build_cluster_candidates(topic_match)
            if item["cluster_id"] == "initial_primary"
        )

        result = matcher.select_rules_semantically(
            {
                "question": "Use the residual mechanics configuration.",
                "context": "",
                "prediction": "A residual mechanism is used in this derivation.",
            },
            [topic_match],
            [initial_cluster],
            _background_analysis("Audit the residual mechanics configuration."),
        )

        self.assertEqual(
            [item["rule_id"] for item in result["selected_rules"]],
            ["r_residual"],
        )
        self.assertEqual(
            [item["cluster_id"] for item in result["navigation_clusters"]],
            ["alternative_primary_one", "residual_rules_01"],
        )
        stages = matcher.last_trace["stages"]
        self.assertEqual(stages["cluster_confirmation_backtrack"]["chat_call_count"], 1)
        self.assertEqual(stages["cluster_confirmation_fallback"]["chat_call_count"], 1)
        self.assertEqual(stages["rule_confirmation"]["chat_call_count"], 3)
        self.assertEqual(stages["rule_confirmation"]["empty_reason"], "")

    @staticmethod
    def _catalog_with_general_reasoning() -> dict:
        catalog = copy.deepcopy(_catalog())
        topic = catalog["domains"][1]["topics"][0]
        topic["rules"].append(
            {
                "rule_id": "exp_general_reasoning",
                "title": "General consistency check",
                "summary": "Audit a general logical consistency condition.",
                "trigger": "a derived claim",
                "check_logic": "compare the claim with stated conditions",
                "error_type": "logic",
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
            }
        )
        topic["scenario_clusters"].append(
            {
                "id": "general_reasoning",
                "name": "General Reasoning",
                "summary": "Cross-scenario consistency checks.",
                "rule_groups": [],
                "rule_ids": ["exp_general_reasoning"],
            }
        )
        return catalog

    def test_general_reasoning_is_used_after_specific_cluster_returns_no_rule(self) -> None:
        catalog = self._catalog_with_general_reasoning()
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 1.0, "reason": "modern"}]
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":1.0,"reason":"relativity"}]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":1.0,"reason":"specific"}]}',
                '{"rules":[]}',
                '{"rules":[{"rule_id":"exp_general_reasoning","applicable":true,"score":0.9,"reason":"general audit"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A pinhole camera observes a moving rod.", "context": "", "prediction": "A claim."},
            catalog,
        )

        self.assertEqual([item["rule_id"] for item in result["selected_rules"]], ["exp_general_reasoning"])
        self.assertEqual(result["selected_rules"][0]["candidate_source"], "general_reasoning_fallback")
        self.assertEqual(len(client.requests), 6)

    def test_select_tree_semantically_returns_structured_results(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Modern Physics", "relevant": True, "score": 0.91, "reason": "relativity setup"}]
                    ),
                    '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                    '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.87,"reason":"camera observation cluster"}]}',
                    '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"matches pinhole moving rod scenario"}]}',
                ]
            ),
        )
        sample = {
            "id": "29185",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        result = matcher.select_tree_semantically(sample, _catalog())
        self.assertEqual(result["selected_domains"], ["Modern Physics"])
        self.assertEqual(len(result["selected_topics"]), 1)
        self.assertEqual(result["selected_topics"][0]["topic"], "Special Relativity (Time Dilation, Length Contraction)")
        self.assertEqual(len(result["selected_clusters"]), 1)
        self.assertEqual(result["selected_clusters"][0]["cluster_id"], "observation_and_projection")
        self.assertEqual(len(result["selected_rules"]), 1)
        self.assertEqual(result["selected_rules"][0]["rule_id"], "exp_pinhole")

    def test_navigation_uses_background_and_only_rule_selection_receives_prediction(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 0.91, "reason": "relativity setup"}]
                ),
                '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.87,"reason":"camera observation cluster"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"matches the student claim"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        sample = {
            "id": "input_role_boundary",
            "question": "QUESTION_ONLY_SENTINEL: A pinhole camera observes a moving rod.",
            "context": "CONTEXT_ONLY_SENTINEL: The camera frame is specified.",
            "prediction": "PREDICTION_ONLY_SENTINEL: Treat the exposure as simultaneous.",
            "answer": "",
        }

        result = matcher.select_tree_semantically(sample, _catalog())

        self.assertEqual(len(client.requests), 5)
        payloads = []
        for request in client.requests:
            messages = request["messages"]
            self.assertIsInstance(messages, list)
            user_message = next(message for message in messages if message["role"] == "user")
            payloads.append(json.loads(user_message["content"]))

        for payload in payloads[:3]:
            self.assertNotIn("sample", payload)
            self.assertNotIn("student_solution", payload)
            self.assertIn(sample["question"], payload["problem_background"])
            self.assertIn(sample["context"], payload["problem_background"])
            self.assertNotIn(sample["prediction"], json.dumps(payload, ensure_ascii=False))

        rule_payloads = [payload for payload in payloads if "candidate_rules" in payload]
        self.assertTrue(rule_payloads)
        for rule_payload in rule_payloads:
            self.assertNotIn("sample", rule_payload)
            self.assertIn(sample["question"], rule_payload["problem_background"])
            self.assertIn(sample["context"], rule_payload["problem_background"])
            self.assertNotIn(sample["prediction"], rule_payload["problem_background"])
            self.assertEqual(rule_payload["student_solution"], sample["prediction"])
        confirmation_payload = next(
            payload
            for payload in payloads
            if payload.get("selection_phase") == "background_only_rule_precision_confirmation"
        )
        self.assertNotIn("student_solution", confirmation_payload)
        self.assertNotIn(sample["prediction"], json.dumps(confirmation_payload, ensure_ascii=False))
        self.assertEqual(result["input_policy"], "background_navigation_prediction_rule_only")

    def test_rule_candidate_order_uses_background_but_never_filters(self) -> None:
        candidates = [
            {
                "rule_id": "r_generic",
                "title": "generic",
                "rule_obj": {
                    "rule_id": "r_generic",
                    "title": "generic",
                    "match_features": {"trigger_keywords": ["unrelated cavity"]},
                },
            },
            {
                "rule_id": "r_orbit",
                "title": "orbit",
                "rule_obj": {
                    "rule_id": "r_orbit",
                    "title": "orbit",
                    "match_features": {"trigger_keywords": ["orbital decay"]},
                    "llm_hints": {"discriminative_terms": ["atmospheric drag"]},
                },
            },
            {
                "rule_id": "r_other",
                "title": "other",
                "rule_obj": {"rule_id": "r_other", "title": "other"},
            },
        ]

        ranked = UnifiedSemanticMatcher._rank_rule_candidates_by_background(
            candidates,
            "Find orbital decay caused by atmospheric drag.",
        )

        self.assertEqual(ranked[0]["rule_id"], "r_orbit")
        self.assertEqual({item["rule_id"] for item in ranked}, {"r_generic", "r_orbit", "r_other"})
        self.assertGreater(ranked[0]["background_order_score"], ranked[-1]["background_order_score"])

    def test_long_source_anchors_keep_tail_and_focus_relevant_chunks(self) -> None:
        source = " ".join(
            [f"Introductory ISS detail {index}." for index in range(40)]
            + [
                "Part B orbital deceleration and station descent rate.",
                "Assume a constant friction force acts on the satellite.",
                "Intermediate notation is introduced here.",
                "B.5 asks for descent H per revolution and total time Tn.",
                "End of the problem statement.",
            ]
        )

        anchors = UnifiedSemanticMatcher._source_anchor_candidates(
            source,
            max_items=12,
            focus="orbital deceleration constant friction descent H Tn",
        )

        self.assertEqual(len(anchors), 12)
        joined = " ".join(anchors)
        self.assertIn("Introductory ISS detail 0", joined)
        self.assertIn("constant friction force", joined)
        self.assertIn("descent H per revolution", joined)
        self.assertIn("End of the problem statement", joined)

    def test_rule_first_pass_keeps_a_larger_provisional_recall_pool(self) -> None:
        rules = [
            {
                "rule_id": f"r_{index}",
                "title": f"Rule {index}",
                "summary": "Audit orbital decay energy accounting.",
                "trigger": "Atmospheric drag causes orbital decay.",
                "check_logic": "Check the stated orbital-energy relation.",
                "error_type": "logic",
                "preconditions": [],
                "violation_signatures": [],
                "negative_conditions": [],
                "evidence_requirements": [],
                "symbolic_hint": {},
            }
            for index in range(6)
        ]
        topic = {
            "id": "mechanics.orbit",
            "name": "Orbit",
            "summary": "Orbital mechanics.",
            "rules": rules,
        }
        candidates = UnifiedSemanticMatcher._build_rule_candidates(
            {"domain": "Mechanics", "topic": "Orbit", "topic_obj": topic}
        )
        provisional_count = 6
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "rules": [
                            {
                                "rule_id": f"r_{index}",
                                "applicable": True,
                                "score": 0.9,
                                "reason": "independently applicable",
                            }
                            for index in range(provisional_count)
                        ]
                    }
                )
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            max_selected_rules=3,
        )
        sample = {
            "question": "Atmospheric drag causes orbital decay.",
            "context": "",
            "prediction": "Use the stated orbital-energy relation.",
        }

        selected = matcher._select_rules_for_context(
            sample=sample,
            background_analysis=_background_analysis("Find orbital decay."),
            context_domain="Mechanics",
            context_topic="Orbit",
            topic_obj=topic,
            rule_candidates=candidates,
        )

        self.assertEqual(len(selected), provisional_count)
        request = client.requests[0]
        payload = json.loads(request["messages"][-1]["content"])
        self.assertEqual(payload["max_provisional_rules"], provisional_count)
        self.assertEqual(
            request["response_format"]["json_schema"]["schema"]
            ["properties"]["rules"]["maxItems"],
            provisional_count,
        )
        self.assertIn("recall-oriented provisional pass", request["messages"][0]["content"])

    def test_empty_domain_selection_short_circuits_the_tree(self) -> None:
        client = _FakeClient(
            [
                _domain_json([], "Identify an ambiguous physical task."),
                _domain_json([], "Identify an ambiguous physical task."),
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {
                "id": "no_domain",
                "question": "An intentionally ambiguous problem statement.",
                "context": "No usable physical setup is provided.",
                "prediction": "This must not be used to reopen the full catalog.",
            },
            _catalog(),
        )

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(result["selected_domains"], [])
        self.assertEqual(result["selected_topics"], [])
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual(result["selected_rules"], [])

    def test_empty_topic_selection_short_circuits_before_cluster_and_rule(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 0.9, "reason": "possible modern setup"}]
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {
                "id": "no_topic",
                "question": "A modern-physics prompt with no matching catalog topic.",
                "context": "",
                "prediction": "This must not trigger cluster or rule expansion.",
            },
            _catalog(),
        )

        self.assertEqual(len(client.requests), 3)
        self.assertEqual(result["selected_domains"], ["Modern Physics"])
        self.assertEqual(result["selected_topics"], [])
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual(result["selected_rules"], [])

    def test_empty_domain_and_topic_are_rechecked_once_before_acceptance(self) -> None:
        client = _FakeClient(
            [
                "not-json-domain-response",
                _domain_json([], "Audit a relativistic observation."),
                _domain_json(
                    [
                        {
                            "domain": "Modern Physics",
                            "relevant": True,
                            "score": 0.95,
                            "reason": "relativistic observation",
                        }
                    ],
                    "Audit a relativistic observation.",
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":false,"score":0.2,"reason":"first pass unsure"}]}',
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":0.94,"reason":"moving rod"}]}',
                '{"clusters":[]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.93,"reason":"camera"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.92,"reason":"auditable"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {
                "question": "A pinhole camera observes a moving rod.",
                "context": "",
                "prediction": "Treat the exposure as simultaneous.",
            },
            _catalog(),
        )

        self.assertEqual([item["rule_id"] for item in result["selected_rules"]], ["exp_pinhole"])
        self.assertEqual(len(result["navigation_trace"]["stages"]["domain"]["api_attempts"]), 3)
        self.assertEqual(len(result["navigation_trace"]["stages"]["topic"]["api_attempts"]), 2)
        self.assertEqual(len(result["navigation_trace"]["stages"]["cluster"]["api_attempts"]), 2)
        self.assertIn(
            "no positive selection",
            result["navigation_trace"]["stages"]["topic"]["api_attempts"][0]["error"],
        )

    def test_domain_topic_and_cluster_outputs_obey_hard_limits(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": f"d{domain_index}",
                    "name": f"Domain {domain_index}",
                    "summary": f"Domain {domain_index} summary.",
                    "topics": [
                        {
                            "id": f"d{domain_index}.t{topic_index}",
                            "name": f"Topic {domain_index}-{topic_index}",
                            "summary": f"Topic {domain_index}-{topic_index} summary.",
                            "rules": [
                                {
                                    "rule_id": f"rule_{domain_index}_{topic_index}_{cluster_index}",
                                }
                                for cluster_index in range(2)
                            ],
                            "scenario_clusters": [
                                {
                                    "id": f"cluster_{domain_index}_{topic_index}_{cluster_index}",
                                    "name": f"Cluster {domain_index}-{topic_index}-{cluster_index}",
                                    "summary": "A candidate scenario.",
                                    "rule_groups": [],
                                    "rule_ids": [
                                        f"rule_{domain_index}_{topic_index}_{cluster_index}"
                                    ],
                                }
                                for cluster_index in range(2)
                            ],
                        }
                        for topic_index in range(2)
                    ],
                }
                for domain_index in range(3)
            ],
        }
        domain_response = {
            "background_analysis": _background_analysis("Classify a broad multi-domain task."),
            "domains": [
                {
                    "domain": f"Domain {index}",
                    "relevant": True,
                    "score": 1.0 - index * 0.1,
                    "reason": "candidate",
                }
                for index in range(3)
            ]
        }
        topic_response = {
            "topics": [
                {
                    "domain": f"Domain {domain_index}",
                    "topic": f"Topic {domain_index}-{topic_index}",
                    "relevant": True,
                    "score": 1.0 - (domain_index * 2 + topic_index) * 0.05,
                    "reason": "candidate",
                }
                for domain_index in range(2)
                for topic_index in range(2)
            ]
        }
        selected_topic_keys = [(0, 0), (0, 1), (1, 0)]
        cluster_responses = [
            {
                "clusters": [
                    {
                        "cluster_id": f"cluster_{domain_index}_{topic_index}_{cluster_index}",
                        "relevant": True,
                        "score": 1.0 - cluster_index * 0.1,
                        "reason": "candidate",
                    }
                    for cluster_index in range(2)
                ]
            }
            for domain_index, topic_index in selected_topic_keys
        ]
        client = _FakeClient(
            [
                json.dumps(domain_response),
                json.dumps(topic_response),
                *[json.dumps(response) for response in cluster_responses],
                '{"rules":[{"rule_id":"rule_0_0_0","applicable":true,"score":0.95}]}',
                '{"rules":[{"rule_id":"rule_0_1_0","applicable":true,"score":0.94}]}',
                '{"rules":[{"rule_id":"rule_1_0_0","applicable":true,"score":0.93}]}',
                '{"rules":[{"rule_id":"rule_0_0_1","applicable":true,"score":0.92}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        sample = {
            "question": "A broad multi-domain setup.",
            "context": "",
            "prediction": "A proposed solution.",
        }
        result = matcher.select_tree_semantically(sample, catalog)

        self.assertEqual(len(result["selected_domains"]), matcher.MAX_SELECTED_DOMAINS)
        self.assertEqual(len(result["selected_topics"]), matcher.MAX_SELECTED_TOPICS)
        self.assertEqual(len(result["selected_clusters"]), matcher.MAX_SELECTED_CLUSTERS)
        self.assertEqual(len(result["selected_rules"]), 4)
        stages = result["navigation_trace"]["stages"]
        self.assertEqual(stages["domain"]["chat_call_count"], 1)
        self.assertEqual(stages["topic"]["chat_call_count"], 1)
        self.assertEqual(stages["cluster"]["chat_call_count"], 3)
        self.assertEqual(stages["rule"]["chat_call_count"], 4)

    def test_runtime_navigation_filters_non_executable_topics_and_clusters(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": "empty_domain",
                    "name": "Empty Domain",
                    "topics": [
                        {
                            "id": "empty_domain.empty_topic",
                            "name": "Empty Topic",
                            "rules": [],
                            "scenario_clusters": [],
                        }
                    ],
                },
                {
                    "id": "mechanics",
                    "name": "Mechanics",
                    "topics": [
                        {
                            "id": "mechanics.empty_topic",
                            "name": "Empty Topic",
                            "rules": [],
                            "scenario_clusters": [
                                {"id": "empty", "rule_ids": []}
                            ],
                        },
                        {
                            "id": "mechanics.kinematics",
                            "name": "Kinematics",
                            "rules": [{"rule_id": "r1", "title": "Check motion"}],
                            "scenario_clusters": [
                                {"id": "empty", "rule_ids": []},
                                {"id": "stale", "rule_ids": ["unknown"]},
                                {
                                    "id": "motion",
                                    "rule_ids": ["r1", "unknown"],
                                    "rule_groups": [
                                        {"id": "valid", "rule_ids": ["r1"]},
                                        {"id": "stale", "rule_ids": ["unknown"]},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }

        matcher = UnifiedSemanticMatcher(model="fake-model", client=_FakeClient([]))
        domains = matcher._build_domain_candidates(catalog)
        self.assertEqual([item["domain"] for item in domains], ["Mechanics"])
        topics = matcher._build_topic_candidates(catalog, ["Mechanics"])
        self.assertEqual([item["topic"] for item in topics], ["Kinematics"])
        self.assertEqual(
            [item["cluster_id"] for item in topics[0]["cluster_previews"]],
            ["motion"],
        )
        clusters = matcher._build_cluster_candidates(topics[0])
        self.assertEqual([item["cluster_id"] for item in clusters], ["motion"])
        self.assertEqual(clusters[0]["rule_ids"], ["r1"])
        self.assertEqual(
            [group["group_id"] for group in clusters[0]["rule_groups"]],
            ["valid"],
        )

    def test_large_rule_candidates_are_split_by_character_budget(self) -> None:
        rules = [
            {
                "rule_id": f"batch_rule_{index}",
                "title": f"Batch rule {index}",
                "summary": f"Candidate {index}: " + ("long semantic description " * 80),
                "trigger": "A repeated physical trigger.",
                "check_logic": "Check the corresponding physical claim.",
                "error_type": "logic",
                "preconditions": ["The physical setup activates this candidate."],
                "violation_signatures": ["The proposed solution violates this candidate."],
                "negative_conditions": [],
                "evidence_requirements": ["Quote the relevant solution step."],
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
            }
            for index in range(5)
        ]
        topic = {
            "id": "batch.topic",
            "name": "Batch Topic",
            "summary": "A topic with many verbose candidate rules.",
            "rules": rules,
            "scenario_clusters": [],
        }
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": "batch",
                    "name": "Batch Domain",
                    "summary": "Batching test domain.",
                    "topics": [topic],
                }
            ],
        }
        batch_chars = 2_600
        probe = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient([]),
            rule_candidate_batch_size=100,
            rule_candidate_batch_chars=batch_chars,
        )
        candidates = probe._build_rule_candidates(
            {"domain": "Batch Domain", "topic": "Batch Topic", "topic_obj": topic}
        )
        expected_batches = probe._batch_rule_candidates(candidates)
        self.assertGreater(len(expected_batches), 1)

        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Batch Domain", "relevant": True, "score": 1.0, "reason": "batch domain"}]
                ),
                '{"topics":[{"domain":"Batch Domain","topic":"Batch Topic","relevant":true,"score":1.0,"reason":"batch topic"}]}',
                *['{"rules":[]}' for _ in expected_batches],
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            rule_candidate_batch_size=100,
            rule_candidate_batch_chars=batch_chars,
        )
        sample = {
            "question": "QUESTION_BATCH_SENTINEL",
            "context": "CONTEXT_BATCH_SENTINEL",
            "prediction": "PREDICTION_BATCH_SENTINEL",
        }

        result = matcher.select_tree_semantically(sample, catalog)

        rule_requests = client.requests[2:]
        self.assertEqual(len(rule_requests), len(expected_batches))
        seen_rule_ids = []
        for request in rule_requests:
            messages = request["messages"]
            user_message = next(message for message in messages if message["role"] == "user")
            payload = json.loads(user_message["content"])
            self.assertIn(sample["question"], payload["problem_background"])
            self.assertIn(sample["context"], payload["problem_background"])
            self.assertNotIn(sample["prediction"], payload["problem_background"])
            self.assertEqual(payload["student_solution"], sample["prediction"])
            self.assertLessEqual(
                len(json.dumps(payload["candidate_rules"], ensure_ascii=False, separators=(",", ":"))),
                batch_chars,
            )
            seen_rule_ids.extend(candidate["rule_id"] for candidate in payload["candidate_rules"])
        self.assertEqual(seen_rule_ids, [rule["rule_id"] for rule in rules])
        self.assertEqual(result["selected_rules"], [])

    def test_rule_selection_does_not_broaden_past_existing_cluster_boundaries(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Modern Physics", "relevant": True, "score": 0.91, "reason": "relativity setup"}]
                    ),
                    '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                    '{"clusters":[]}',
                    '{"clusters":[]}',
                    '{"clusters":[]}',
                ]
            ),
        )
        sample = {
            "id": "29185",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        result = matcher.select_tree_semantically(sample, _catalog())
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual(result["selected_rules"], [])
        self.assertEqual(len(matcher._client.requests), 5)

    def test_clusterless_topic_rules_are_capped(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2", "schema_profile": "semantic_navigation_tree_minimal"},
            "domains": [
                {
                    "id": "electromagnetism",
                    "name": "Electromagnetism",
                    "summary": "Electric and magnetic systems.",
                    "topics": [
                        {
                            "id": "electromagnetism.dc_circuits_and_kirchhoffs_laws",
                            "name": "DC Circuits and Kirchhoff's Laws",
                            "summary": "Static circuit solving with loop and node constraints.",
                            "rules": [
                                {
                                    "rule_id": f"exp_dc_{index}",
                                    "title": f"DC rule {index}",
                                    "summary": f"Check DC loop and node consistency {index}.",
                                    "trigger": "Kirchhoff branch current",
                                    "check_logic": f"Loop and node consistency {index}",
                                    "error_type": "logic",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["I", "V"]},
                                }
                                for index in range(3)
                            ],
                            "scenario_clusters": [],
                        }
                    ],
                }
            ],
        }
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Electromagnetism", "relevant": True, "score": 0.95, "reason": "circuit problem"}]
                    ),
                    '{"topics":[{"domain":"Electromagnetism","topic":"DC Circuits and Kirchhoff\'s Laws","relevant":true,"score":0.94,"reason":"static circuit solving"}]}',
                    '{"rules":[{"rule_id":"exp_dc_0","applicable":true,"score":0.95,"reason":"strongly applicable"},{"rule_id":"exp_dc_1","applicable":true,"score":0.9,"reason":"also applicable"},{"rule_id":"exp_dc_2","applicable":true,"score":0.85,"reason":"weaker but still relevant"}]}',
                ]
            ),
        )
        sample = {
            "id": "dc_clusterless_cap",
            "question": "Use Kirchhoff laws to solve branch currents in a DC circuit.",
            "prediction": "Write loop and node equations.",
            "answer": "",
        }
        result = matcher.select_tree_semantically(sample, catalog)
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual([item["rule_id"] for item in result["selected_rules"]], ["exp_dc_0", "exp_dc_1"])

    def test_selected_rules_are_globally_capped_across_multiple_clusters(self) -> None:
        topic_names = ["Topic A", "Topic B", "Topic C"]
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2", "schema_profile": "semantic_navigation_tree_minimal"},
            "domains": [
                {
                    "id": "mechanics",
                    "name": "Mechanics",
                    "summary": "Mechanics domain.",
                    "topics": [
                        {
                            "id": f"mechanics.topic_{topic_index}",
                            "name": topic_name,
                            "summary": f"{topic_name} summary.",
                            "rules": [
                                {
                                    "rule_id": f"r{topic_index}_{rule_index}",
                                    "title": f"Rule {topic_index}-{rule_index}",
                                    "summary": f"Rule {topic_index}-{rule_index} summary.",
                                    "trigger": "shared trigger",
                                    "check_logic": "shared check",
                                    "error_type": "logic",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
                                }
                                for rule_index in range(3)
                            ],
                            "scenario_clusters": [
                                {
                                    "id": f"cluster_{topic_index}",
                                    "name": f"Cluster {topic_index}",
                                    "summary": f"Cluster {topic_index} summary.",
                                    "rule_groups": [
                                        {
                                            "id": f"group_{topic_index}",
                                            "name": f"Group {topic_index}",
                                            "summary": f"Group {topic_index} summary.",
                                            "activation_condition": "Use for shared trigger.",
                                            "rule_ids": [f"r{topic_index}_{rule_index}" for rule_index in range(3)],
                                        }
                                    ],
                                    "rule_ids": [f"r{topic_index}_{rule_index}" for rule_index in range(3)],
                                }
                            ],
                        }
                        for topic_index, topic_name in enumerate(topic_names)
                    ],
                }
            ],
        }
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Mechanics", "relevant": True, "score": 1.0, "reason": "mechanics"}]
                    ),
                    '{"topics":[{"domain":"Mechanics","topic":"Topic A","relevant":true,"score":1.0,"reason":"a"},{"domain":"Mechanics","topic":"Topic B","relevant":true,"score":1.0,"reason":"b"},{"domain":"Mechanics","topic":"Topic C","relevant":true,"score":0.9,"reason":"c"}]}',
                    '{"clusters":[{"cluster_id":"cluster_0","relevant":true,"score":1.0,"reason":"a"}]}',
                    '{"clusters":[{"cluster_id":"cluster_1","relevant":true,"score":1.0,"reason":"b"}]}',
                    '{"clusters":[{"cluster_id":"cluster_2","relevant":true,"score":0.9,"reason":"c"}]}',
                    '{"rules":[{"rule_id":"r0_0","applicable":true,"score":1.0,"reason":"best"},{"rule_id":"r0_1","applicable":true,"score":0.95,"reason":"also"},{"rule_id":"r0_2","applicable":true,"score":0.9,"reason":"weaker"}]}',
                    '{"rules":[{"rule_id":"r1_0","applicable":true,"score":0.98,"reason":"best"},{"rule_id":"r1_1","applicable":true,"score":0.94,"reason":"also"},{"rule_id":"r1_2","applicable":true,"score":0.89,"reason":"weaker"}]}',
                    '{"rules":[{"rule_id":"r2_0","applicable":true,"score":0.97,"reason":"best"},{"rule_id":"r2_1","applicable":true,"score":0.93,"reason":"also"},{"rule_id":"r2_2","applicable":true,"score":0.88,"reason":"weaker"}]}',
                ]
            ),
        )

        result = matcher.select_tree_semantically(
            {"id": "wide", "question": "A multi-mechanism mechanics problem.", "prediction": "Solution.", "answer": ""},
            catalog,
        )

        self.assertEqual(len(result["selected_rules"]), 5)
        self.assertEqual(
            [item["rule_id"] for item in result["selected_rules"]],
            ["r0_0", "r1_0", "r2_0", "r0_1", "r1_1"],
        )


if __name__ == "__main__":
    unittest.main()
