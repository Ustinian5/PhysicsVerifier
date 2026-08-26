from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.rule_catalog_retrieval import (
    norm_text,
    ordered_unique,
    score_rule_candidate,
    topic_rule_leaves,
)

try:
    import httpx
except ImportError:  # pragma: no cover - environment-dependent
    httpx = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment-dependent
    OpenAI = None  # type: ignore[assignment]


UNIFIED_SEMANTIC_MATCHER_PROMPT_VERSION = "semantic-navigation-background-only-v1"


class SemanticSelectionError(RuntimeError):
    """Identify which semantic-tree stage failed without hiding the root error."""

    def __init__(
        self,
        stage: str,
        cause: Exception,
        *,
        trace: Dict[str, Any] | None = None,
        partial_result: Dict[str, Any] | None = None,
    ) -> None:
        self.stage = str(stage or "unknown")
        self.cause = cause
        self.trace = copy.deepcopy(trace or {})
        self.partial_result = copy.deepcopy(partial_result or {})
        super().__init__(f"{self.stage}: {type(cause).__name__}: {cause}")


class ProviderIdentityError(RuntimeError):
    """The semantic endpoint response is not attributable to the requested model."""


class UnifiedSemanticMatcher:
    MAX_SELECTED_DOMAINS = 2
    MAX_SELECTED_TOPICS = 3
    MAX_SELECTED_CLUSTERS = 4
    MAX_SELECTED_RULES = 5
    MAX_PROVISIONAL_RULES_PER_BATCH = 12
    MAX_RULE_CONFIRMATION_CANDIDATES = 18
    RULE_CONFIRMATION_DECISIONS = (
        "confirm",
        "reject_missing_background",
        "reject_different_configuration",
    )
    TOPIC_SHORTLIST_TRIGGER = 8
    MAX_TOPIC_SHORTLIST = 6
    RULE_CANDIDATE_BATCH_SIZE = 24
    RULE_CANDIDATE_BATCH_CHARS = 24_000
    MAX_JSON_RETRIES = 3
    MAX_RESPONSE_TOKENS = 1_024
    RETRY_RESPONSE_TOKENS = 512
    HARD_MAX_RESPONSE_TOKENS = 4_096
    RAW_TRACE_PREVIEW_CHARS = 240
    INPUT_POLICY = "background_navigation_prediction_rule_only"
    STRUCTURED_OUTPUT_ADAPTERS = {
        "openai_json_schema",
        "vllm_structured_outputs",
        "vllm_guided_json",
        "forced_tool_call",
    }
    BACKGROUND_LIST_FIELDS = (
        "objects",
        "processes",
        "conditions",
        "symbols_and_units",
        "missing_information",
        "inactive_context",
    )

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        trust_env: bool | None = None,
        max_selected_rules: int | None = None,
        rule_candidate_batch_size: int | None = None,
        rule_candidate_batch_chars: int | None = None,
        max_response_tokens: int | None = None,
        json_retries: int | None = None,
        request_timeout: float | None = None,
        allow_json_object_fallback: bool | None = None,
        structured_output_adapter: str | None = None,
        require_provider_identity: bool = False,
        expected_provider_model: str | None = None,
    ) -> None:
        self.model = norm_text(model)
        self.temperature = float(temperature)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or None
        env_trust = norm_text(os.getenv("UNIFIED_SEMANTIC_TRUST_ENV") or "")
        self.trust_env = bool(trust_env) if trust_env is not None else env_trust in {"1", "true", "yes", "on"}
        self.max_selected_rules = max(
            1,
            int(max_selected_rules if max_selected_rules is not None else self.MAX_SELECTED_RULES),
        )
        self.rule_candidate_batch_size = max(
            1,
            int(
                rule_candidate_batch_size
                if rule_candidate_batch_size is not None
                else self.RULE_CANDIDATE_BATCH_SIZE
            ),
        )
        self.rule_candidate_batch_chars = max(
            2_048,
            int(
                rule_candidate_batch_chars
                if rule_candidate_batch_chars is not None
                else self.RULE_CANDIDATE_BATCH_CHARS
            ),
        )
        self.max_response_tokens = min(
            self.HARD_MAX_RESPONSE_TOKENS,
            max(256, int(max_response_tokens or self.MAX_RESPONSE_TOKENS)),
        )
        self.json_retries = min(
            self.MAX_JSON_RETRIES,
            max(0, int(self.MAX_JSON_RETRIES if json_retries is None else json_retries)),
        )
        timeout_env = norm_text(
            os.getenv("UNIFIED_SEMANTIC_REQUEST_TIMEOUT_SEC")
            or os.getenv("PHYSICSVERIFIER_LLM_TIMEOUT_SEC")
            or ""
        )
        try:
            configured_timeout = (
                float(request_timeout)
                if request_timeout is not None
                else float(timeout_env or 120)
            )
        except (TypeError, ValueError):
            configured_timeout = 120.0
        self.request_timeout = max(1.0, configured_timeout)
        fallback_env = norm_text(
            os.getenv("UNIFIED_SEMANTIC_ALLOW_JSON_OBJECT_FALLBACK") or ""
        ).casefold()
        self.allow_json_object_fallback = (
            bool(allow_json_object_fallback)
            if allow_json_object_fallback is not None
            else fallback_env in {"1", "true", "yes", "on"}
        )
        configured_adapter = norm_text(
            structured_output_adapter
            or os.getenv("UNIFIED_SEMANTIC_OUTPUT_ADAPTER")
            or "openai_json_schema"
        ).casefold()
        if configured_adapter not in self.STRUCTURED_OUTPUT_ADAPTERS:
            allowed = ", ".join(sorted(self.STRUCTURED_OUTPUT_ADAPTERS))
            raise ValueError(
                f"structured_output_adapter must be one of: {allowed}"
            )
        self.structured_output_adapter = configured_adapter
        self.require_provider_identity = bool(require_provider_identity)
        self.expected_provider_model = norm_text(expected_provider_model or self.model)
        if self.require_provider_identity and not self.expected_provider_model:
            raise ValueError(
                "expected_provider_model or model is required when provider identity is enforced"
            )
        self._json_schema_supported: Optional[bool] = None
        self._client = client
        self._trace_run_active = False
        self._active_stage = ""
        self._background_analysis = self._empty_background_analysis()
        self.last_trace: Dict[str, Any] = {}
        self.last_partial_result: Dict[str, Any] = {}
        self._reset_trace()

    @classmethod
    def _empty_background_analysis(cls) -> Dict[str, Any]:
        return {
            "task_focus": "",
            "objects": [],
            "processes": [],
            "conditions": [],
            "target_quantity": "",
            "symbols_and_units": [],
            "missing_information": [],
            "inactive_context": [],
        }

    def _reset_trace(self) -> None:
        self._background_analysis = self._empty_background_analysis()
        self.last_partial_result = {}
        self.last_trace = {
            "input_policy": self.INPUT_POLICY,
            "model_config": {
                "model": self.model,
                "temperature": self.temperature,
                "thinking_disabled": self._thinking_disabled(),
                "max_response_tokens": self.max_response_tokens,
                "retry_response_tokens": (
                    self.max_response_tokens
                    if self.structured_output_adapter == "forced_tool_call"
                    else min(self.max_response_tokens, self.RETRY_RESPONSE_TOKENS)
                ),
                "json_attempts": self.json_retries + 1,
                "request_timeout_seconds": self.request_timeout,
                "sdk_max_retries": 0,
                "structured_output": (
                    "forced_tool_call_schema_validated"
                    if self.structured_output_adapter == "forced_tool_call"
                    else "strict_json_schema_required"
                ),
                "structured_output_adapter": self.structured_output_adapter,
                "allow_json_object_fallback": self.allow_json_object_fallback,
                "require_provider_identity": self.require_provider_identity,
                "expected_provider_model": self.expected_provider_model,
                "empty_navigation_recheck": self.json_retries > 0,
                "max_provisional_rules_per_batch": self.MAX_PROVISIONAL_RULES_PER_BATCH,
            },
            "background_analysis": copy.deepcopy(self._background_analysis),
            "stages": {},
            "partial_result": {},
            "terminal_stage": "",
            "empty_reason": "",
            "status": "running",
        }
        self._active_stage = ""

    def _stage_trace(self, stage: str) -> Dict[str, Any]:
        stages = self.last_trace.setdefault("stages", {})
        trace = stages.setdefault(
            stage,
            {
                "candidate_count": 0,
                "candidates": [],
                "not_selected": [],
                "model_response": None,
                "model_responses": [],
                "api_attempts": [],
                "accepted": [],
                "rejected": [],
                "empty_reason": "",
            },
        )
        return trace

    def _begin_stage(self, stage: str, *, candidate_count: int = 0, reset: bool = False) -> Dict[str, Any]:
        self._active_stage = stage
        if reset and stage in self.last_trace.get("stages", {}):
            del self.last_trace["stages"][stage]
        trace = self._stage_trace(stage)
        trace["candidate_count"] = int(trace.get("candidate_count") or 0) + max(0, int(candidate_count))
        return trace

    def _trace_accept(self, stage: str, item: Dict[str, Any]) -> None:
        self._stage_trace(stage)["accepted"].append(copy.deepcopy(item))

    def _trace_candidates(self, stage: str, items: Iterable[Dict[str, Any]]) -> None:
        trace = self._stage_trace(stage)
        trace["candidates"].extend(copy.deepcopy(list(items)))

    def _trace_not_selected(
        self,
        stage: str,
        items: Iterable[Dict[str, Any]],
        *,
        returned_ids: Iterable[str],
        id_key: str,
    ) -> None:
        returned = {norm_text(item) for item in returned_ids if norm_text(item)}
        trace = self._stage_trace(stage)
        for item in items:
            candidate_id = norm_text(item.get(id_key) or "")
            if candidate_id and candidate_id not in returned:
                trace["not_selected"].append(
                    {
                        **copy.deepcopy(item),
                        "reason": "not_returned_by_model",
                    }
                )

    def _trace_reject(self, stage: str, item: Any, reason: str, **context: Any) -> None:
        record: Dict[str, Any] = {"item": copy.deepcopy(item), "reason": norm_text(reason)}
        record.update({key: copy.deepcopy(value) for key, value in context.items()})
        self._stage_trace(stage)["rejected"].append(record)

    def _set_empty_reason(self, stage: str, reason: str) -> None:
        self._stage_trace(stage)["empty_reason"] = norm_text(reason)

    @staticmethod
    def _strict_bool(value: Any) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _thinking_disabled() -> bool:
        return norm_text(os.getenv("OPENAI_DISABLE_THINKING") or "") in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _safe_score(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if score != score or score in {float("inf"), float("-inf")}:
            return None
        return max(0.0, min(score, 1.0))

    @classmethod
    def _background_analysis_schema(cls) -> Dict[str, Any]:
        list_schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        return {
            "type": "object",
            "properties": {
                "task_focus": {"type": "string"},
                "objects": copy.deepcopy(list_schema),
                "processes": copy.deepcopy(list_schema),
                "conditions": copy.deepcopy(list_schema),
                "target_quantity": {"type": "string"},
                "symbols_and_units": copy.deepcopy(list_schema),
                "missing_information": copy.deepcopy(list_schema),
                "inactive_context": copy.deepcopy(list_schema),
            },
            "required": [
                "task_focus",
                "objects",
                "processes",
                "conditions",
                "target_quantity",
                "symbols_and_units",
                "missing_information",
                "inactive_context",
            ],
            "additionalProperties": False,
        }

    @classmethod
    def _selection_response_schema(
        cls,
        *,
        list_key: str,
        id_key: str,
        bool_key: str,
        allowed_ids: Iterable[str],
        max_items: int,
        include_background: bool = False,
    ) -> Dict[str, Any]:
        candidate_ids = ordered_unique(
            [norm_text(item) for item in allowed_ids if norm_text(item)]
        )
        if not candidate_ids:
            raise ValueError(
                f"Cannot build '{list_key}' selection schema without candidate IDs."
            )
        item_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                id_key: {"type": "string", "enum": candidate_ids},
                bool_key: {
                    "type": "boolean",
                    "enum": [True],
                    "description": "Only positive selections may be returned; omit all other candidates.",
                },
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
            },
            "required": [id_key, bool_key, "score", "reason"],
            "additionalProperties": False,
        }
        properties: Dict[str, Any] = {
            list_key: {
                "type": "array",
                "items": item_schema,
                "maxItems": max(0, int(max_items)),
                "description": "Return positive selections only; use an empty array when none apply.",
            }
        }
        required = [list_key]
        if include_background:
            properties["background_analysis"] = cls._background_analysis_schema()
            required.insert(0, "background_analysis")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @classmethod
    def _rule_selection_response_schema(
        cls,
        *,
        allowed_ids: Iterable[str],
        background_anchors: Iterable[str],
        claim_anchors: Iterable[str],
        max_items: int,
    ) -> Dict[str, Any]:
        candidate_ids = ordered_unique(
            [norm_text(item) for item in allowed_ids if norm_text(item)]
        )
        if not candidate_ids:
            raise ValueError("Cannot build rule selection schema without candidate IDs.")
        allowed_background_anchors = ordered_unique(
            [norm_text(item) for item in background_anchors if norm_text(item)]
        )
        allowed_claim_anchors = ordered_unique(
            [norm_text(item) for item in claim_anchors if norm_text(item)]
        )
        if not allowed_background_anchors or not allowed_claim_anchors:
            raise ValueError("Cannot build rule selection schema without source anchors.")
        return {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "maxItems": max(0, int(max_items)),
                    "description": (
                        "Selected applicable rules only. Omit every rejected rule; use an empty array "
                        "when no rule has both required source anchors."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_id": {"type": "string", "enum": candidate_ids},
                            "score": {"type": "number", "minimum": 0.8, "maximum": 1.0},
                            "background_anchor_index": {
                                "type": "integer",
                                "enum": list(range(len(allowed_background_anchors))),
                                "description": (
                                    "Zero-based index of the question/context quote that establishes "
                                    "this rule's trigger or precondition."
                                ),
                            },
                            "claim_anchor_index": {
                                "type": "integer",
                                "enum": list(range(len(allowed_claim_anchors))),
                                "description": (
                                    "Zero-based index of the student_solution quote identifying the claim "
                                    "or step this rule can audit."
                                ),
                            },
                        },
                        "required": [
                            "rule_id",
                            "score",
                            "background_anchor_index",
                            "claim_anchor_index",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["rules"],
            "additionalProperties": False,
        }

    @classmethod
    def _rule_confirmation_response_schema(
        cls,
        *,
        allowed_ids: Iterable[str],
        background_anchor_count: int,
    ) -> Dict[str, Any]:
        candidate_ids = ordered_unique(
            [norm_text(item) for item in allowed_ids if norm_text(item)]
        )
        if not candidate_ids:
            raise ValueError("Cannot build rule confirmation schema without candidate IDs.")
        if background_anchor_count <= 0:
            raise ValueError("Cannot build rule confirmation schema without source anchors.")
        background_indices = [-1, *range(background_anchor_count)]
        return {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": len(candidate_ids),
                    "maxItems": len(candidate_ids),
                    "description": "Exactly one compact decision for every preliminary rule.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_id": {"type": "string", "enum": candidate_ids},
                            "decision": {
                                "type": "string",
                                "enum": list(cls.RULE_CONFIRMATION_DECISIONS),
                            },
                            "background_anchor_index": {
                                "type": "integer",
                                "enum": background_indices,
                                "description": "Use -1 when the decision is a rejection.",
                            },
                        },
                        "required": [
                            "rule_id",
                            "decision",
                            "background_anchor_index",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        }

    @staticmethod
    def _anchor_is_exact_source_text(anchor: Any, source: Any, *, max_chars: int) -> bool:
        anchor_text = norm_text(anchor or "")
        source_text = norm_text(source or "")
        return bool(
            anchor_text
            and len(anchor_text) <= max_chars
            and anchor_text.casefold() in source_text.casefold()
        )

    @classmethod
    def _source_anchor_candidates(
        cls,
        source: Any,
        *,
        max_items: int,
        max_chars: int = 160,
        focus: Any = "",
    ) -> List[str]:
        text = norm_text(source or "")
        if not text:
            return []
        segments = re.split(r"(?<=[.!?。！？;；:：])\s+", text)
        raw_candidates: List[str] = []
        for segment in segments:
            remaining = norm_text(segment)
            while remaining:
                if len(remaining) <= max_chars:
                    raw_candidates.append(remaining)
                    break
                cut = remaining.rfind(" ", 0, max_chars + 1)
                if cut < max_chars // 2:
                    cut = max_chars
                raw_candidates.append(remaining[:cut].strip())
                remaining = remaining[cut:].strip()
        candidates = ordered_unique(raw_candidates)
        item_limit = max(0, int(max_items))
        if len(candidates) <= item_limit:
            return candidates
        if item_limit == 0:
            return []

        # Long olympiad statements often put the actual sub-question near the
        # end. Keep stable head/tail coverage, then spend the remaining budget
        # on source chunks related to the background analysis or rule context.
        head_count = min(2, item_limit)
        tail_count = min(3, max(0, item_limit - head_count))
        selected_indices = set(range(head_count))
        if tail_count:
            selected_indices.update(range(len(candidates) - tail_count, len(candidates)))

        focus_text = norm_text(focus or "").casefold()
        stopwords = {
            "and", "are", "for", "from", "into", "only", "that", "the", "this",
            "with", "when", "where", "which", "rule", "check", "problem", "question",
        }
        focus_terms = ordered_unique(
            term
            for term in re.findall(r"[a-z][a-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", focus_text)
            if term not in stopwords
        )
        scored: List[tuple[float, int]] = []
        for index, candidate in enumerate(candidates):
            if index in selected_indices:
                continue
            folded = candidate.casefold()
            score = sum(
                min(12, len(term)) * folded.count(term)
                for term in focus_terms
                if term in folded
            )
            scored.append((float(score), index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for score, index in scored:
            if len(selected_indices) >= item_limit or score <= 0:
                break
            selected_indices.add(index)

        # Fill any unused slots with evenly distributed chunks so an empty or
        # sparse focus never degenerates back to first-N truncation.
        if len(selected_indices) < item_limit:
            remaining_indices = [
                index for index in range(len(candidates)) if index not in selected_indices
            ]
            slots = item_limit - len(selected_indices)
            if slots >= len(remaining_indices):
                selected_indices.update(remaining_indices)
            elif slots > 0:
                for position in range(slots):
                    pick = round(position * (len(remaining_indices) - 1) / max(1, slots - 1))
                    selected_indices.add(remaining_indices[pick])

        return [candidates[index] for index in sorted(selected_indices)[:item_limit]]

    @classmethod
    def _validate_rule_selection_contract(
        cls,
        response: Dict[str, Any],
        *,
        resolve_id: Callable[[Dict[str, Any]], str],
        background_source: str,
        claim_source: str,
        allowed_background_anchors: Iterable[str],
        allowed_claim_anchors: Iterable[str],
        max_items: int,
    ) -> None:
        items = response.get("rules")
        if not isinstance(items, list):
            raise RuntimeError("'rules' must be a JSON array.")
        background_anchor_list = ordered_unique(
            [norm_text(item) for item in allowed_background_anchors if norm_text(item)]
        )
        claim_anchor_list = ordered_unique(
            [norm_text(item) for item in allowed_claim_anchors if norm_text(item)]
        )
        best_by_rule_id: Dict[str, tuple[float, Dict[str, Any]]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f"'rules[{index}]' must be a JSON object.")
            resolved_rule_id = norm_text(resolve_id(item))
            if not resolved_rule_id:
                raise RuntimeError(f"'rules[{index}]' references an unknown candidate.")
            score = cls._safe_score(item.get("score"))
            if score is None or score < 0.8:
                raise RuntimeError(f"'rules[{index}].score' must be a number from 0.8 to 1.")
            background_anchor_index = item.get("background_anchor_index")
            if (
                isinstance(background_anchor_index, bool)
                or not isinstance(background_anchor_index, int)
                or not 0 <= background_anchor_index < len(background_anchor_list)
            ):
                raise RuntimeError(
                    f"'rules[{index}].background_anchor_index' must reference an allowed question/context quote."
                )
            claim_anchor_index = item.get("claim_anchor_index")
            if (
                isinstance(claim_anchor_index, bool)
                or not isinstance(claim_anchor_index, int)
                or not 0 <= claim_anchor_index < len(claim_anchor_list)
            ):
                raise RuntimeError(
                    f"'rules[{index}].claim_anchor_index' must reference an allowed student_solution quote."
                )
            background_anchor = background_anchor_list[background_anchor_index]
            claim_anchor = claim_anchor_list[claim_anchor_index]
            if not cls._anchor_is_exact_source_text(
                background_anchor,
                background_source,
                max_chars=160,
            ) or not cls._anchor_is_exact_source_text(
                claim_anchor,
                claim_source,
                max_chars=160,
            ):
                raise RuntimeError(f"'rules[{index}]' references a non-source anchor.")

            current = best_by_rule_id.get(resolved_rule_id)
            if current is None or score > current[0]:
                canonical_item = dict(item)
                canonical_item["rule_id"] = resolved_rule_id
                best_by_rule_id[resolved_rule_id] = (score, canonical_item)

        unique_limit = max(0, int(max_items))
        if len(best_by_rule_id) > unique_limit:
            raise RuntimeError(f"'rules' must contain at most {max_items} unique selected items.")
        response["rules"] = [record[1] for record in best_by_rule_id.values()]

    @classmethod
    def _validate_rule_confirmation_contract(
        cls,
        response: Dict[str, Any],
        *,
        allowed_ids: Iterable[str],
        background_source: str,
        allowed_background_anchors: Iterable[str],
    ) -> None:
        candidate_ids = ordered_unique(
            [norm_text(item) for item in allowed_ids if norm_text(item)]
        )
        candidate_id_set = set(candidate_ids)
        decisions = response.get("decisions")
        if not isinstance(decisions, list):
            raise RuntimeError("'decisions' must be a JSON array.")
        if len(decisions) != len(candidate_ids):
            raise RuntimeError(
                "'decisions' must contain exactly one item for every preliminary rule."
            )
        background_anchor_list = ordered_unique(
            [norm_text(item) for item in allowed_background_anchors if norm_text(item)]
        )
        seen: set[str] = set()
        for index, item in enumerate(decisions):
            if not isinstance(item, dict):
                raise RuntimeError(f"'decisions[{index}]' must be a JSON object.")
            rule_id = norm_text(item.get("rule_id") or "")
            if rule_id not in candidate_id_set or rule_id in seen:
                raise RuntimeError(
                    f"'decisions[{index}].rule_id' must be a unique preliminary rule id."
                )
            seen.add(rule_id)
            decision = norm_text(item.get("decision") or "")
            if decision not in cls.RULE_CONFIRMATION_DECISIONS:
                raise RuntimeError(f"'decisions[{index}].decision' is invalid.")
            background_index = item.get("background_anchor_index")
            if (
                isinstance(background_index, bool)
                or not isinstance(background_index, int)
                or not -1 <= background_index < len(background_anchor_list)
            ):
                raise RuntimeError(
                    f"'decisions[{index}].background_anchor_index' is out of range."
                )
            if decision != "confirm":
                continue
            if background_index < 0:
                raise RuntimeError(
                    f"'decisions[{index}]' must provide a source anchor when confirmed."
                )
            if not cls._anchor_is_exact_source_text(
                background_anchor_list[background_index],
                background_source,
                max_chars=160,
            ):
                raise RuntimeError(f"'decisions[{index}]' references a non-source anchor.")
        if seen != candidate_id_set:
            raise RuntimeError("'decisions' does not cover every preliminary rule exactly once.")

    @classmethod
    def _validate_background_analysis_contract(cls, response: Dict[str, Any]) -> None:
        analysis = response.get("background_analysis")
        if not isinstance(analysis, dict):
            raise RuntimeError("'background_analysis' must be a JSON object.")
        task_focus = analysis.get("task_focus")
        if not isinstance(task_focus, str) or not norm_text(task_focus):
            raise RuntimeError("'background_analysis.task_focus' must be a non-empty string.")
        target_quantity = analysis.get("target_quantity")
        if not isinstance(target_quantity, str):
            raise RuntimeError("'background_analysis.target_quantity' must be a string.")
        for field in cls.BACKGROUND_LIST_FIELDS:
            values = analysis.get(field)
            if not isinstance(values, list):
                raise RuntimeError(f"'background_analysis.{field}' must be a JSON array.")
            if any(not isinstance(item, str) for item in values):
                raise RuntimeError(
                    f"'background_analysis.{field}' must contain only strings."
                )

    @classmethod
    def _validate_selection_contract(
        cls,
        response: Dict[str, Any],
        *,
        list_key: str,
        bool_key: str,
        resolve_id: Callable[[Dict[str, Any]], str],
    ) -> None:
        items = response.get(list_key)
        if not isinstance(items, list):
            raise RuntimeError(f"'{list_key}' must be a JSON array.")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f"'{list_key}[{index}]' must be a JSON object.")
            if not norm_text(resolve_id(item)):
                raise RuntimeError(f"'{list_key}[{index}]' references an unknown candidate.")
            selected = cls._strict_bool(item.get(bool_key))
            if selected is None:
                raise RuntimeError(f"'{list_key}[{index}].{bool_key}' must be a JSON boolean.")
            if not isinstance(item.get("reason"), str):
                raise RuntimeError(f"'{list_key}[{index}].reason' must be a string.")
            raw_score = item.get("score")
            if isinstance(raw_score, bool):
                raise RuntimeError(f"'{list_key}[{index}].score' must be a number from 0 to 1.")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = float("nan")
            if score != score or score in {float("inf"), float("-inf")} or not 0.0 <= score <= 1.0:
                raise RuntimeError(f"'{list_key}[{index}].score' must be a number from 0 to 1.")

    @staticmethod
    def _canonical_key(value: Any) -> str:
        text = norm_text(value or "").casefold().replace("&", " and ")
        return "".join(char for char in text if char.isalnum())

    @classmethod
    def _normalize_background_analysis(cls, value: Any) -> Dict[str, Any]:
        result = cls._empty_background_analysis()
        if not isinstance(value, dict):
            return result
        result["task_focus"] = norm_text(value.get("task_focus") or "")
        result["target_quantity"] = norm_text(value.get("target_quantity") or "")
        for field in cls.BACKGROUND_LIST_FIELDS:
            raw_items = value.get(field)
            if not isinstance(raw_items, list):
                raw_items = []
            result[field] = ordered_unique(
                [norm_text(item) for item in raw_items if isinstance(item, str) and norm_text(item)]
            )[:24]
        return result

    def _background_prompt_fields(
        self,
        sample: Dict[str, Any],
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "problem_background": self._problem_background(sample),
            "background_analysis": copy.deepcopy(
                background_analysis if background_analysis is not None else self._background_analysis
            ),
        }

    @staticmethod
    def _compact_judgment(item: Dict[str, Any]) -> Dict[str, Any]:
        excluded = {"topic_obj", "cluster_obj", "rule_obj", "topic_rules", "rule_groups"}
        return {key: copy.deepcopy(value) for key, value in item.items() if key not in excluded}

    def _update_partial_result(
        self,
        *,
        domain_result: Dict[str, Any],
        topic_result: Dict[str, Any] | None = None,
        cluster_result: Dict[str, Any] | None = None,
        rule_result: Dict[str, Any] | None = None,
    ) -> None:
        topics = topic_result or {"topic_judgments": [], "selected_topics": []}
        clusters = cluster_result or {"cluster_judgments": [], "selected_clusters": []}
        rules = rule_result or {"rule_judgments": [], "selected_rules": []}
        self.last_partial_result = {
            "input_policy": self.INPUT_POLICY,
            "background_analysis": copy.deepcopy(self._background_analysis),
            "domain_judgments": copy.deepcopy(domain_result.get("domain_judgments") or []),
            "topic_judgments": copy.deepcopy(topics.get("topic_judgments") or []),
            "cluster_judgments": copy.deepcopy(clusters.get("cluster_judgments") or []),
            "rule_judgments": copy.deepcopy(rules.get("rule_judgments") or []),
            "selected_domains": copy.deepcopy(domain_result.get("selected_domains") or []),
            "selected_topics": copy.deepcopy(topics.get("selected_topics") or []),
            "selected_clusters": copy.deepcopy(clusters.get("selected_clusters") or []),
            "selected_rules": copy.deepcopy(rules.get("selected_rules") or []),
        }
        self._sync_trace_partial_result()

    def _sync_trace_partial_result(self) -> None:
        compact: Dict[str, Any] = {}
        for key, value in self.last_partial_result.items():
            if key in {"topic_judgments", "selected_topics", "cluster_judgments", "selected_clusters", "rule_judgments", "selected_rules"}:
                compact[key] = [self._compact_judgment(item) for item in value if isinstance(item, dict)]
            else:
                compact[key] = copy.deepcopy(value)
        self.last_trace["partial_result"] = compact

    def _checkpoint_partial_items(self, stage: str, items: Iterable[Dict[str, Any]]) -> None:
        """Keep completed calls visible if a later call in the same stage fails."""
        keys = {
            "cluster": ("cluster_judgments", "selected_clusters"),
            "rule": ("rule_judgments", "selected_rules"),
        }
        if stage not in keys:
            return
        if not self.last_partial_result:
            self.last_partial_result = {
                "input_policy": self.INPUT_POLICY,
                "background_analysis": copy.deepcopy(self._background_analysis),
                "domain_judgments": [],
                "topic_judgments": [],
                "cluster_judgments": [],
                "rule_judgments": [],
                "selected_domains": [],
                "selected_topics": [],
                "selected_clusters": [],
                "selected_rules": [],
            }
        judgments_key, selected_key = keys[stage]
        checkpoint = copy.deepcopy(list(items))
        self.last_partial_result[judgments_key] = checkpoint
        self.last_partial_result[selected_key] = copy.deepcopy(checkpoint)
        self._sync_trace_partial_result()

    def _refresh_env_config(self) -> None:
        # SemanticRuleChecker may load .env before this matcher is first used.
        # Re-read only missing values so explicit constructor arguments still win.
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY") or None
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None

    @property
    def available(self) -> bool:
        if self._client is not None:
            return True
        self._refresh_env_config()
        return bool(OpenAI and self.model and self.api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._refresh_env_config()
        if not OpenAI:
            raise RuntimeError("OpenAI package is not available.")
        if not self.model:
            raise RuntimeError("Semantic matcher model is not configured.")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.request_timeout,
            "max_retries": 0,
        }
        if httpx is not None:
            client_kwargs["http_client"] = httpx.Client(trust_env=self.trust_env)
        self._client = OpenAI(**client_kwargs)
        return self._client

    @staticmethod
    def _extract_json_object(text: str, *, list_key: str | None = None) -> Dict[str, Any]:
        raw = norm_text(text)
        if not raw:
            raise RuntimeError("Semantic matcher returned empty content.")
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
            raw = re.sub(r"\s*```$", "", raw).strip()
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        else:
            if isinstance(parsed, dict):
                return parsed
            if list_key and isinstance(parsed, list):
                return {list_key: parsed}
            parsed_type = "null" if parsed is None else type(parsed).__name__
            raise RuntimeError(
                "Semantic matcher returned a top-level "
                f"{parsed_type}; a JSON object is required."
            )
        fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, flags=re.S | re.I)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
                if list_key and isinstance(parsed, list):
                    return {list_key: parsed}
            except Exception:
                pass
        loose = re.search(r"\{.*\}", raw, flags=re.S)
        if loose:
            try:
                parsed = json.loads(loose.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        if list_key:
            loose_list = re.search(r"\[.*\]", raw, flags=re.S)
            if loose_list:
                try:
                    parsed = json.loads(loose_list.group(0))
                    if isinstance(parsed, list):
                        return {list_key: parsed}
                except Exception:
                    pass
        hint = ""
        if raw.startswith("{") and not raw.endswith("}"):
            hint = " The response looks truncated."
        expected = f" or a JSON array for '{list_key}'" if list_key else ""
        raise RuntimeError(f"Semantic matcher must return a JSON object{expected}.{hint}")

    @staticmethod
    def _decode_forced_tool_object(raw: str) -> tuple[Dict[str, Any], str]:
        """Decode tool arguments, tolerating only surplus closing delimiters."""
        stripped = str(raw or "").strip()
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            try:
                parsed, end = json.JSONDecoder().raw_decode(stripped)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Structured semantic output must be one exact JSON object."
                ) from exc
            trailing = stripped[end:].strip()
            if not trailing or not re.fullmatch(r"[\]\}]+", trailing):
                raise RuntimeError(
                    "Structured semantic output must be one exact JSON object."
                )
            repair = "ignored_surplus_closing_delimiters"
        else:
            repair = ""
        if not isinstance(parsed, dict):
            raise RuntimeError("Structured semantic output must have a JSON object root.")
        return parsed, repair

    @classmethod
    def _raw_trace_fields(cls, raw: str) -> Dict[str, Any]:
        encoded = raw.encode("utf-8", errors="replace")
        return {
            "raw_preview": raw[: cls.RAW_TRACE_PREVIEW_CHARS],
            "raw_suffix": raw[-cls.RAW_TRACE_PREVIEW_CHARS :],
            "raw_length": len(raw),
            "raw_sha256": hashlib.sha256(encoded).hexdigest(),
            "raw_truncated": len(raw) > cls.RAW_TRACE_PREVIEW_CHARS,
        }

    @staticmethod
    def _format_violation_kind(raw: str, finish_reason: Any) -> str:
        stripped = str(raw or "").strip()
        numeric_stream = bool(
            len(stripped) >= 32
            and re.fullmatch(r"[+\-0-9.eE]+", stripped)
        )
        if numeric_stream:
            return "numeric_stream_degeneration"
        if norm_text(finish_reason).casefold() in {"length", "max_tokens"}:
            return "output_limit_reached"
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return "invalid_json"
        if not isinstance(parsed, dict):
            return "non_object_json_root"
        return ""

    @staticmethod
    def _compact_retry_user_prompt(user_prompt: str) -> str:
        try:
            payload = json.loads(user_prompt)
        except (TypeError, ValueError):
            return norm_text(user_prompt)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _build_retry_user_prompt(cls, user_prompt: str, correction: str) -> str:
        compact = cls._compact_retry_user_prompt(user_prompt)
        try:
            payload = json.loads(compact)
        except (TypeError, ValueError):
            return f"{compact}\n\n{correction}"
        if isinstance(payload, dict):
            payload["response_correction"] = correction
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"{compact}\n\n{correction}"

    @staticmethod
    def _json_schema_is_unsupported(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code not in {400, 422}:
            return False
        body = getattr(exc, "body", None)
        message = f"{exc} {body}".casefold()
        schema_markers = (
            "json_schema",
            "response_format",
            "structured output",
            "structured_output",
        )
        unsupported_markers = (
            "unsupported",
            "not support",
            "not implemented",
            "unknown",
            "invalid type",
            "not one of",
        )
        return any(marker in message for marker in schema_markers) and any(
            marker in message for marker in unsupported_markers
        )

    @staticmethod
    def _schema_name(value: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", norm_text(value))
        return (name or "semantic_selection")[:64]

    def _structured_output_request_fields(
        self,
        *,
        response_schema: Dict[str, Any] | None,
        schema_name: str,
    ) -> Dict[str, Any]:
        if response_schema is None:
            return {"response_format": {"type": "json_object"}}

        adapter = self.structured_output_adapter
        if adapter == "openai_json_schema":
            if self._json_schema_supported is False and self.allow_json_object_fallback:
                return {"response_format": {"type": "json_object"}}
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self._schema_name(schema_name),
                        "strict": True,
                        "schema": copy.deepcopy(response_schema),
                    },
                }
            }
        if adapter == "vllm_structured_outputs":
            return {
                "extra_body": {
                    "structured_outputs": {"json": copy.deepcopy(response_schema)}
                }
            }
        if adapter == "vllm_guided_json":
            return {"extra_body": {"guided_json": copy.deepcopy(response_schema)}}
        if adapter == "forced_tool_call":
            function_name = self._schema_name(f"physics_{schema_name}")
            return {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": (
                                "Return the verifier's semantic selection in the required structure."
                            ),
                            "parameters": copy.deepcopy(response_schema),
                            "strict": True,
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": function_name},
                },
                "parallel_tool_calls": False,
            }
        raise RuntimeError(f"Unsupported structured-output adapter: {adapter}")

    @staticmethod
    def _request_adapter_label(request: Dict[str, Any]) -> str:
        if request.get("tools"):
            return "forced_tool_call"
        extra_body = request.get("extra_body") if isinstance(request.get("extra_body"), dict) else {}
        if "structured_outputs" in extra_body:
            return "vllm_structured_outputs"
        if "guided_json" in extra_body:
            return "vllm_guided_json"
        response_format = request.get("response_format")
        if isinstance(response_format, dict):
            return norm_text(response_format.get("type") or "")
        return "prompt_only"

    def _validate_provider_response_identity(
        self, actual_model: str, response_id: str
    ) -> None:
        if not self.require_provider_identity:
            return
        actual = norm_text(actual_model)
        response = norm_text(response_id)
        if actual != self.expected_provider_model:
            raise ProviderIdentityError(
                "provider model mismatch: "
                f"expected {self.expected_provider_model!r}, received {actual or '<empty>'!r}"
            )
        if not response:
            raise ProviderIdentityError("provider response id is empty")

    def _chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        list_key: str | None = None,
        response_validator: Callable[[Dict[str, Any]], None] | None = None,
        response_schema: Dict[str, Any] | None = None,
        schema_name: str = "",
        contract_hint: str = "",
        retry_empty_selection: bool = False,
        selection_bool_key: str | None = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        stage = self._active_stage or "chat_json"
        stage_trace = self._stage_trace(stage)
        request_index = int(stage_trace.get("chat_call_count") or 0) + 1
        stage_trace["chat_call_count"] = request_index

        def build_request(prompt: str, *, max_tokens: int) -> Dict[str, Any]:
            request: Dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": int(max_tokens),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            request.update(
                self._structured_output_request_fields(
                    response_schema=response_schema,
                    schema_name=schema_name or f"{stage}_selection",
                )
            )
            if self._thinking_disabled():
                extra_body = request.setdefault("extra_body", {})
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            return request

        last_error: Exception | None = None
        active_request = build_request(user_prompt, max_tokens=self.max_response_tokens)
        empty_selection_rechecked = False
        for attempt in range(1, self.json_retries + 2):
            while True:
                response_format_type = self._request_adapter_label(active_request)
                try:
                    response = client.chat.completions.create(**active_request)
                except Exception as exc:
                    if (
                        response_format_type == "json_schema"
                        and self._json_schema_is_unsupported(exc)
                    ):
                        fallback_error = (
                            "json_schema_unsupported_fallback"
                            if self.allow_json_object_fallback
                            else "json_schema_required_but_unsupported"
                        )
                        stage_trace["api_attempts"].append(
                            {
                                "request_index": request_index,
                                "attempt": attempt,
                                "response_format": response_format_type,
                                **self._raw_trace_fields(""),
                                "finish_reason": None,
                                "error": fallback_error,
                            }
                        )
                        if not self.allow_json_object_fallback:
                            raise RuntimeError(
                                "The semantic endpoint does not support strict json_schema output. "
                                "Use a compatible endpoint/model; unsafe json_object fallback is disabled."
                            ) from exc
                        self._json_schema_supported = False
                        active_request = build_request(
                            str(active_request["messages"][-1]["content"]),
                            max_tokens=int(active_request["max_tokens"]),
                        )
                        continue
                    stage_trace["api_attempts"].append(
                        {
                            "request_index": request_index,
                            "attempt": attempt,
                            "response_format": response_format_type,
                            **self._raw_trace_fields(""),
                            "finish_reason": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise
                break
            choices = response.choices if getattr(response, "choices", None) else []
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None) if choice else None
            raw_content = getattr(message, "content", "") if message else ""
            raw = "" if raw_content is None else str(raw_content)
            payload_error = ""
            tool_call_count: int | None = None
            tool_call_names: List[str] = []
            if response_format_type == "forced_tool_call":
                tool_calls = list(getattr(message, "tool_calls", None) or []) if message else []
                expected_name = norm_text(
                    (((active_request.get("tool_choice") or {}).get("function") or {}).get("name") or "")
                )
                functions = [getattr(tool_call, "function", None) for tool_call in tool_calls]
                tool_call_names = [
                    norm_text(getattr(function, "name", "") or "")
                    for function in functions
                ]
                matching = [
                    function
                    for function in functions
                    if function is not None
                    and norm_text(getattr(function, "name", "") or "") == expected_name
                ]
                tool_call_count = len(tool_calls)
                if not tool_calls:
                    payload_error = "Forced semantic tool call was not returned by the endpoint."
                elif len(tool_calls) != 1 or len(matching) != 1:
                    payload_error = (
                        "Forced semantic tool call output must contain exactly one matching call; "
                        f"received count={len(tool_calls)}, names={tool_call_names}."
                    )
                else:
                    arguments = getattr(matching[0], "arguments", None)
                    if arguments is None:
                        payload_error = "Forced semantic tool call did not contain arguments."
                    elif isinstance(arguments, dict):
                        raw = json.dumps(arguments, ensure_ascii=False)
                    else:
                        raw = str(arguments)
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            attempt_trace: Dict[str, Any] = {
                "request_index": request_index,
                "attempt": attempt,
                "response_format": response_format_type,
                **self._raw_trace_fields(raw),
                "finish_reason": finish_reason,
                "actual_model": norm_text(getattr(response, "model", "") or ""),
                "response_id": norm_text(getattr(response, "id", "") or ""),
            }
            if tool_call_count is not None:
                attempt_trace["tool_call_count"] = tool_call_count
                attempt_trace["tool_call_names"] = tool_call_names
            try:
                self._validate_provider_response_identity(
                    attempt_trace["actual_model"], attempt_trace["response_id"]
                )
                if payload_error:
                    raise RuntimeError(payload_error)
                if norm_text(finish_reason).casefold() in {"length", "max_tokens"}:
                    raise RuntimeError(
                        "Semantic matcher response reached the output limit before completion."
                    )
                if response_schema is not None:
                    if response_format_type == "forced_tool_call":
                        parsed, structured_repair = self._decode_forced_tool_object(raw)
                        if structured_repair:
                            attempt_trace["structured_repair"] = structured_repair
                    else:
                        try:
                            parsed = json.loads(raw.strip())
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                "Structured semantic output must be one exact JSON object."
                            ) from exc
                        if not isinstance(parsed, dict):
                            raise RuntimeError(
                                "Structured semantic output must have a JSON object root."
                            )
                else:
                    parsed = self._extract_json_object(raw, list_key=list_key)
                if list_key and not isinstance(parsed.get(list_key), list):
                    raise RuntimeError(
                        "Semantic matcher JSON must contain a top-level "
                        f"'{list_key}' array."
                    )
                if response_validator is not None:
                    response_validator(parsed)
                selection_items = parsed.get(list_key) if list_key else None
                no_positive_selection = not selection_items
                if selection_bool_key and isinstance(selection_items, list):
                    no_positive_selection = not any(
                        isinstance(item, dict)
                        and self._strict_bool(item.get(selection_bool_key)) is True
                        for item in selection_items
                    )
                if (
                    retry_empty_selection
                    and list_key
                    and no_positive_selection
                    and not empty_selection_rechecked
                    and attempt < self.json_retries + 1
                ):
                    empty_selection_rechecked = True
                    raise RuntimeError(
                        f"'{list_key}' has no positive selection; reconsider the supplied candidates once "
                        "before confirming no match."
                    )
            except RuntimeError as exc:
                last_error = exc
                attempt_trace["error"] = str(exc)
                if isinstance(exc, ProviderIdentityError):
                    stage_trace["api_attempts"].append(attempt_trace)
                    raise
                violation_kind = (
                    self._format_violation_kind(raw, finish_reason)
                    if response_schema is not None
                    else ""
                )
                if violation_kind:
                    attempt_trace["format_violation_kind"] = violation_kind
                stage_trace["api_attempts"].append(attempt_trace)
                if violation_kind in {
                    "numeric_stream_degeneration",
                    "non_object_json_root",
                }:
                    if response_format_type == "json_schema":
                        self._json_schema_supported = False
                    raise RuntimeError(
                        f"Structured-output adapter '{response_format_type}' was not enforced "
                        f"by the endpoint ({violation_kind})."
                    ) from exc
                expected_shape = (
                    f"a top-level JSON object whose '{list_key}' field is an array"
                    if list_key
                    else "a top-level JSON object"
                )
                if contract_hint:
                    expected_shape = f"{expected_shape}; {contract_hint}"
                correction = (
                    "RESPONSE CORRECTION: Return "
                    f"{expected_shape} only. Follow the supplied output_schema exactly. "
                    "Do not return a scalar number, prose, analysis, or Markdown fencing."
                )
                if empty_selection_rechecked:
                    correction += (
                        " Reconsider all supplied candidates once; an empty selection is valid only "
                        "when none satisfies the stated boundary."
                    )
                retry_prompt = self._build_retry_user_prompt(user_prompt, correction)
                active_request = build_request(
                retry_prompt,
                    max_tokens=(
                        self.max_response_tokens
                        if response_format_type == "forced_tool_call"
                        else min(self.max_response_tokens, self.RETRY_RESPONSE_TOKENS)
                    ),
                )
                continue
            if response_format_type == "json_schema":
                self._json_schema_supported = True
            stage_trace["api_attempts"].append(attempt_trace)
            stage_trace["model_response"] = copy.deepcopy(parsed)
            stage_trace["model_responses"].append(copy.deepcopy(parsed))
            return parsed
        assert last_error is not None
        raise RuntimeError(
            f"Invalid semantic JSON after {self.json_retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _problem_background(sample: Dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Question:\n{norm_text(sample.get('question') or '')}",
                f"Context:\n{norm_text(sample.get('context') or '')}",
            ]
        ).strip()

    @staticmethod
    def _student_solution(sample: Dict[str, Any]) -> str:
        return norm_text(sample.get("prediction") or "")

    @classmethod
    def _build_domain_candidates(cls, catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for domain in catalog.get("domains", []) or []:
            if not isinstance(domain, dict):
                continue
            topics = [
                topic
                for topic in (domain.get("topics") or [])
                if isinstance(topic, dict) and cls._topic_rule_objects(topic)
            ]
            if not topics:
                continue
            topic_names = [norm_text(topic.get("name") or "") for topic in topics if norm_text(topic.get("name") or "")]
            domain_name = norm_text(domain.get("name") or "Unknown")
            domain_id = norm_text(domain.get("id") or domain.get("domain_id") or "")
            if not domain_id:
                domain_id = cls._canonical_key(domain_name) or "unknown_domain"
            out.append(
                {
                    "domain_id": domain_id,
                    "domain": domain_name,
                    "summary": norm_text(domain.get("summary") or ""),
                    "topic_count": len(topics),
                    "sample_topics": topic_names[:5],
                }
            )
        return out

    @classmethod
    def _compact_retrieval_hints(cls, topic: Dict[str, Any]) -> Dict[str, Any]:
        hints = topic.get("retrieval_hints") if isinstance(topic.get("retrieval_hints"), dict) else {}
        limits = {
            "scene_keywords": (5, 70),
            "topic_keywords": (6, 40),
            "required_symbols": (6, 25),
            "llm_discriminative_terms": (3, 70),
            "llm_problem_phrases": (1, 120),
        }
        compact: Dict[str, Any] = {}
        for field, (item_limit, char_limit) in limits.items():
            values = hints.get(field) if isinstance(hints.get(field), list) else []
            clipped = ordered_unique(
                [cls._clip_prompt_text(value, char_limit) for value in values if norm_text(value)]
            )[:item_limit]
            if clipped:
                compact[field] = clipped
        return compact

    @classmethod
    def _topic_cluster_previews(cls, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        previews: List[Dict[str, Any]] = []
        topic_rule_ids = {
            norm_text(rule.get("rule_id") or rule.get("id") or "")
            for rule in cls._topic_rule_objects(topic)
        }
        for cluster in topic.get("scenario_clusters", []) or []:
            if not isinstance(cluster, dict):
                continue
            cluster_rule_ids = {
                norm_text(rule_id)
                for rule_id in (cluster.get("rule_ids") or [])
                if norm_text(rule_id)
            }
            if not cluster_rule_ids.intersection(topic_rule_ids):
                continue
            activation_conditions = []
            for group in cluster.get("rule_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                condition = norm_text(group.get("activation_condition") or "")
                if condition:
                    activation_conditions.append(cls._clip_prompt_text(condition, 120))
            previews.append(
                {
                    "cluster_id": norm_text(cluster.get("id") or cluster.get("cluster_id") or ""),
                    "name": cls._clip_prompt_text(cluster.get("name") or "", 80),
                    "summary": cls._clip_prompt_text(cluster.get("summary") or "", 160),
                    "activation_conditions": ordered_unique(activation_conditions)[:1],
                }
            )
            if len(previews) >= 2:
                break
        return previews

    @classmethod
    def _topic_shortlist_candidate(cls, candidate: Dict[str, Any]) -> Dict[str, Any]:
        hints = candidate.get("retrieval_hints")
        hints = hints if isinstance(hints, dict) else {}
        anchors: List[str] = []
        for field in ("scene_keywords", "llm_discriminative_terms"):
            values = hints.get(field) if isinstance(hints.get(field), list) else []
            anchors.extend(
                cls._clip_prompt_text(item, 80)
                for item in values[:2]
                if norm_text(item)
            )
        return {
            "domain": candidate["domain"],
            "topic_id": candidate["topic_id"],
            "canonical_name": candidate["topic"],
            "summary": cls._clip_prompt_text(candidate.get("summary") or "", 220),
            "coarse_anchors": ordered_unique(anchors)[:3],
        }

    @classmethod
    def _build_topic_candidates(cls, catalog: Dict[str, Any], domains: Iterable[str]) -> List[Dict[str, Any]]:
        domain_filter = {norm_text(item) for item in domains if norm_text(item)}
        if not domain_filter:
            return []
        out: List[Dict[str, Any]] = []
        for domain in catalog.get("domains", []) or []:
            if not isinstance(domain, dict):
                continue
            domain_name = norm_text(domain.get("name") or "Unknown")
            if domain_filter and domain_name not in domain_filter:
                continue
            for topic in domain.get("topics", []) or []:
                if not isinstance(topic, dict):
                    continue
                executable_rules = cls._topic_rule_objects(topic)
                if not executable_rules:
                    continue
                topic_name = norm_text(topic.get("name") or "Unknown")
                topic_id = norm_text(topic.get("id") or topic.get("topic_id") or "")
                if not topic_id:
                    domain_key = cls._canonical_key(domain_name) or "unknown_domain"
                    topic_key = cls._canonical_key(topic_name) or "unknown_topic"
                    topic_id = f"{domain_key}.{topic_key}"
                out.append(
                    {
                        "domain": domain_name,
                        "topic_id": topic_id,
                        "topic": topic_name,
                        "summary": norm_text(topic.get("summary") or ""),
                        "rule_count": len(executable_rules),
                        "retrieval_hints": cls._compact_retrieval_hints(topic),
                        "cluster_previews": cls._topic_cluster_previews(topic),
                        "topic_obj": topic,
                    }
                )
        return out

    @staticmethod
    def _topic_rule_objects(topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            rule
            for rule in topic_rule_leaves(topic)
            if isinstance(rule, dict)
            and norm_text(rule.get("rule_id") or rule.get("id") or "")
        ]

    @classmethod
    def _build_rule_candidates(cls, topic_match: Dict[str, Any]) -> List[Dict[str, Any]]:
        topic_obj = topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {}
        out: List[Dict[str, Any]] = []
        for rule in cls._topic_rule_objects(topic_obj):
            out.append(
                {
                    "rule_id": norm_text(rule.get("rule_id") or ""),
                    "title": norm_text(rule.get("title") or ""),
                    "summary": norm_text(rule.get("summary") or ""),
                    "trigger": norm_text(rule.get("trigger") or ""),
                    "check_logic": norm_text(rule.get("check_logic") or ""),
                    "error_type": norm_text(rule.get("error_type") or "logic") or "logic",
                    "preconditions": [
                        norm_text(item) for item in (rule.get("preconditions") or []) if norm_text(item)
                    ],
                    "violation_signatures": [
                        norm_text(item) for item in (rule.get("violation_signatures") or []) if norm_text(item)
                    ],
                    "negative_conditions": [
                        norm_text(item) for item in (rule.get("negative_conditions") or []) if norm_text(item)
                    ],
                    "evidence_requirements": [
                        norm_text(item) for item in (rule.get("evidence_requirements") or []) if norm_text(item)
                    ],
                    "symbolic_hint": rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {},
                    "rule_obj": rule,
                }
            )
        return out

    @staticmethod
    def _cluster_navigation_role(cluster: Dict[str, Any]) -> str:
        explicit = norm_text(cluster.get("navigation_role") or "").casefold()
        if explicit in {"primary", "deferred_bucket", "general_fallback"}:
            return explicit
        cluster_id = norm_text(cluster.get("id") or cluster.get("cluster_id") or "").casefold()
        if cluster_id == "general_reasoning":
            return "general_fallback"
        if cluster_id.startswith(("embedding_cluster_", "residual_rules_")):
            return "deferred_bucket"
        return "primary"

    @classmethod
    def _cluster_representative_rules(
        cls,
        *,
        rule_ids: Iterable[str],
        topic_rules: Dict[str, Dict[str, Any]],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        representatives: List[Dict[str, Any]] = []
        for rule_id in rule_ids:
            rule = topic_rules.get(norm_text(rule_id))
            if not isinstance(rule, dict):
                continue
            symbolic_hint = (
                rule.get("symbolic_hint")
                if isinstance(rule.get("symbolic_hint"), dict)
                else {}
            )
            representatives.append(
                {
                    "rule_id": norm_text(rule.get("rule_id") or ""),
                    "title": cls._clip_prompt_text(rule.get("title") or "", 100),
                    "trigger": cls._clip_prompt_text(rule.get("trigger") or "", 140),
                    "canonical": cls._clip_prompt_text(
                        symbolic_hint.get("canonical") or "", 140
                    ),
                }
            )
            if len(representatives) >= max(1, int(limit)):
                break
        return representatives

    @classmethod
    def _build_cluster_candidates(cls, topic_match: Dict[str, Any]) -> List[Dict[str, Any]]:
        topic_obj = topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {}
        topic_rules = {
            str(rule.get("rule_id") or ""): rule
            for rule in cls._topic_rule_objects(topic_obj)
        }
        out: List[Dict[str, Any]] = []
        for cluster in topic_obj.get("scenario_clusters", []) or []:
            if not isinstance(cluster, dict):
                continue
            navigation_role = cls._cluster_navigation_role(cluster)
            cluster_rule_ids = ordered_unique(
                norm_text(item)
                for item in (cluster.get("rule_ids") or [])
                if norm_text(item) in topic_rules
            )
            if not cluster_rule_ids:
                continue
            rule_groups = []
            for group in cluster.get("rule_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                group_rule_ids = ordered_unique(
                    norm_text(item)
                    for item in (group.get("rule_ids") or [])
                    if norm_text(item) in cluster_rule_ids
                )
                if not group_rule_ids:
                    continue
                rule_groups.append(
                    {
                        "group_id": norm_text(group.get("id") or group.get("group_id") or ""),
                        "name": norm_text(group.get("name") or ""),
                        "summary": norm_text(group.get("summary") or ""),
                        "activation_condition": norm_text(group.get("activation_condition") or ""),
                        "rule_ids": group_rule_ids,
                    }
                )
            out.append(
                {
                    "domain": topic_match["domain"],
                    "topic": topic_match["topic"],
                    "cluster_id": norm_text(cluster.get("id") or cluster.get("cluster_id") or ""),
                    "cluster": norm_text(cluster.get("name") or "Unknown"),
                    "summary": norm_text(cluster.get("summary") or ""),
                    "navigation_role": navigation_role,
                    "rule_groups": rule_groups,
                    "rule_ids": cluster_rule_ids,
                    "representative_rules": cls._cluster_representative_rules(
                        rule_ids=cluster_rule_ids,
                        topic_rules=topic_rules,
                        limit=12 if navigation_role == "deferred_bucket" else 3,
                    ),
                    "topic_obj": topic_obj,
                    "cluster_obj": cluster,
                    "topic_rules": topic_rules,
                }
            )
        return out

    def select_domains_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
        if not self._trace_run_active:
            self._reset_trace()
        domain_candidates = self._build_domain_candidates(catalog)
        self._begin_stage("domain", candidate_count=len(domain_candidates), reset=True)
        domain_request_index = int(self._stage_trace("domain").get("chat_call_count") or 0) + 1
        domain_trace_candidates = [
            {
                "request_index": domain_request_index,
                "domain_id": item["domain_id"],
                "domain": item["domain"],
            }
            for item in domain_candidates
        ]
        self._trace_candidates("domain", domain_trace_candidates)
        if not domain_candidates:
            self._set_empty_reason("domain", "no_domain_candidates")
            return {
                "background_analysis": copy.deepcopy(self._background_analysis),
                "domain_judgments": [],
                "selected_domains": [],
            }
        prompt_payload = {
            **self._background_prompt_fields(sample, self._empty_background_analysis()),
            "candidate_domains": domain_candidates,
            "max_selected_domains": self.MAX_SELECTED_DOMAINS,
            "output_schema": {
                "background_analysis": self._empty_background_analysis(),
                "domains": [
                    {
                        "domain_id": "stable candidate domain_id",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        candidates_by_id = {item["domain_id"]: item for item in domain_candidates}
        candidates_by_name = {
            self._canonical_key(item["domain"]): item for item in domain_candidates
        }

        def resolve_domain_id(item: Dict[str, Any]) -> str:
            domain_id = norm_text(item.get("domain_id") or "")
            if domain_id in candidates_by_id:
                return domain_id
            candidate = candidates_by_name.get(
                self._canonical_key(item.get("domain") or item.get("canonical_name") or "")
            )
            return norm_text(candidate.get("domain_id") or "") if candidate else ""

        def validate_domain_response(payload: Dict[str, Any]) -> None:
            self._validate_background_analysis_contract(payload)
            self._validate_selection_contract(
                payload,
                list_key="domains",
                bool_key="relevant",
                resolve_id=resolve_domain_id,
            )

        response = self._chat_json(
            system_prompt=(
                "You are a physics rule navigator. First extract a structured background_analysis from the raw "
                "question and context: task_focus, objects, processes, conditions, target_quantity, "
                "symbols_and_units, missing_information, and inactive_context. Then select the minimum set of "
                "physics domains whose laws apply to that stated physical system. The task focus and active "
                "conditions must dominate historical, illustrative, or otherwise inactive context. Do not infer "
                "problem facts from any student solution. A domain is active only when it supplies an independent "
                "governing law or model needed for this task. The name of a requested quantity does not by itself "
                "activate a domain when another domain supplies the actual governing relation. Keep multiple domains only when "
                "each contributes a distinct physical law. Reject adjacent domains that match only by vocabulary. "
                "If the background is incomplete, explicitly list what is missing, but still select at least one "
                "broad domain when the active task is recognizably a physics problem. Copy the stable domain_id "
                "exactly; do not rewrite it. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            list_key="domains",
            response_validator=validate_domain_response,
            response_schema=self._selection_response_schema(
                list_key="domains",
                id_key="domain_id",
                bool_key="relevant",
                allowed_ids=candidates_by_id,
                max_items=self.MAX_SELECTED_DOMAINS,
                include_background=True,
            ),
            schema_name="semantic_domain_selection",
            retry_empty_selection=True,
            selection_bool_key="relevant",
            contract_hint=(
                "background_analysis must be an object with a non-empty task_focus; "
                "every domain item must use a known candidate id, a JSON boolean relevant, and a numeric score"
            ),
        )
        self._background_analysis = self._normalize_background_analysis(response.get("background_analysis"))
        self.last_trace["background_analysis"] = copy.deepcopy(self._background_analysis)
        response_items = response.get("domains")
        if not isinstance(response_items, list):
            self._trace_reject("domain", response_items, "domains_must_be_a_list")
            response_items = []
        judgments: List[Dict[str, Any]] = []
        returned_domain_ids = [
            resolve_domain_id(item) for item in response_items if isinstance(item, dict)
        ]
        self._trace_not_selected(
            "domain",
            domain_trace_candidates,
            returned_ids=returned_domain_ids,
            id_key="domain_id",
        )
        for item in response_items:
            if not isinstance(item, dict):
                self._trace_reject("domain", item, "domain_item_must_be_an_object")
                continue
            domain_id = norm_text(item.get("domain_id") or "")
            candidate = candidates_by_id.get(domain_id) if domain_id else None
            if candidate is None:
                candidate = candidates_by_name.get(
                    self._canonical_key(item.get("domain") or item.get("canonical_name") or "")
                )
            if candidate is None:
                self._trace_reject("domain", item, "unknown_domain")
                continue
            relevant = self._strict_bool(item.get("relevant"))
            if relevant is None:
                self._trace_reject("domain", item, "relevant_must_be_json_boolean")
                continue
            if not relevant:
                self._trace_reject("domain", item, "model_marked_irrelevant")
                continue
            score = self._safe_score(item.get("score"))
            if score is None:
                self._trace_reject("domain", item, "invalid_score")
                continue
            judgments.append(
                {
                    "domain_id": candidate["domain_id"],
                    "domain": candidate["domain"],
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                }
            )
        best_by_domain: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            current = best_by_domain.get(judgment["domain_id"])
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("domain", self._compact_judgment(current), "duplicate_lower_score")
                best_by_domain[judgment["domain_id"]] = judgment
            else:
                self._trace_reject("domain", self._compact_judgment(judgment), "duplicate_lower_score")
        judgments = list(best_by_domain.values())
        judgments.sort(key=lambda item: (-float(item["score"]), item["domain"]))
        for omitted in judgments[self.MAX_SELECTED_DOMAINS :]:
            self._trace_reject("domain", self._compact_judgment(omitted), "selection_limit")
        judgments = judgments[: self.MAX_SELECTED_DOMAINS]
        for judgment in judgments:
            self._trace_accept("domain", self._compact_judgment(judgment))
        if not judgments:
            reason = "model_selected_no_domains" if not response_items else "all_domain_items_rejected"
            self._set_empty_reason("domain", reason)
        return {
            "background_analysis": copy.deepcopy(self._background_analysis),
            "domain_judgments": judgments,
            "selected_domains": [item["domain"] for item in judgments],
        }

    def select_topics_semantically(
        self,
        sample: Dict[str, Any],
        catalog: Dict[str, Any],
        domains: Iterable[str],
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        selected_domain_names = [norm_text(item) for item in domains if norm_text(item)]
        if not selected_domain_names:
            self._begin_stage("topic", reset=True)
            self._set_empty_reason("topic", "no_selected_domains")
            return {"topic_judgments": [], "selected_topics": []}
        topic_candidates = self._build_topic_candidates(catalog, selected_domain_names)
        if not topic_candidates:
            self._begin_stage("topic", reset=True)
            self._set_empty_reason("topic", "no_topic_candidates")
            return {"topic_judgments": [], "selected_topics": []}
        if len(topic_candidates) > self.TOPIC_SHORTLIST_TRIGGER:
            self._begin_stage(
                "topic_shortlist",
                candidate_count=len(topic_candidates),
                reset=True,
            )
            shortlist_request_index = int(
                self._stage_trace("topic_shortlist").get("chat_call_count") or 0
            ) + 1
            shortlist_trace_candidates = [
                {
                    "request_index": shortlist_request_index,
                    "domain": item["domain"],
                    "topic_id": item["topic_id"],
                    "topic": item["topic"],
                }
                for item in topic_candidates
            ]
            self._trace_candidates("topic_shortlist", shortlist_trace_candidates)
            shortlist_index = {item["topic_id"]: item for item in topic_candidates}

            def resolve_shortlist_topic_id(item: Dict[str, Any]) -> str:
                topic_id = norm_text(item.get("topic_id") or "")
                return topic_id if topic_id in shortlist_index else ""

            def validate_shortlist_response(payload: Dict[str, Any]) -> None:
                self._validate_selection_contract(
                    payload,
                    list_key="topics",
                    bool_key="relevant",
                    resolve_id=resolve_shortlist_topic_id,
                )
                items = payload.get("topics")
                positive_domains = {
                    shortlist_index[topic_id]["domain"]
                    for item in (items if isinstance(items, list) else [])
                    if isinstance(item, dict)
                    and self._strict_bool(item.get("relevant")) is True
                    and (topic_id := resolve_shortlist_topic_id(item))
                }
                missing_domains = [
                    domain_name
                    for domain_name in selected_domain_names
                    if domain_name not in positive_domains
                ]
                if missing_domains:
                    raise RuntimeError(
                        "Topic shortlist must retain at least one relevant topic from every "
                        "selected domain."
                    )

            shortlist_payload = {
                **self._background_prompt_fields(sample, background_analysis),
                "selection_phase": "recall_first_topic_shortlist",
                "max_shortlist_topics": self.MAX_TOPIC_SHORTLIST,
                "required_domain_coverage": selected_domain_names,
                "candidate_topics": [
                    self._topic_shortlist_candidate(item) for item in topic_candidates
                ],
                "output_schema": {
                    "topics": [
                        {
                            "topic_id": "stable candidate topic_id",
                            "relevant": True,
                            "score": 0.0,
                            "reason": "short reason",
                        }
                    ]
                },
            }
            shortlist_response = self._chat_json(
                system_prompt=(
                    "You are the recall-first topic shortlist stage of a physics rule navigator. Use only the "
                    "problem background. Keep every topic that may contain a concrete auditing rule for the active "
                    "physical scenario, even when a broader neighboring topic also applies. Prefer the specific "
                    "stated mechanism over a topic supported only by a generic quantity name. Reject topics "
                    "supported only by inactive context. Every candidate domain has already been judged active, so "
                    "retain at least one plausible topic from each domain before using remaining shortlist slots. "
                    "Copy topic_id exactly. Return JSON only."
                ),
                user_prompt=json.dumps(shortlist_payload, ensure_ascii=False, separators=(",", ":")),
                list_key="topics",
                response_validator=validate_shortlist_response,
                response_schema=self._selection_response_schema(
                    list_key="topics",
                    id_key="topic_id",
                    bool_key="relevant",
                    allowed_ids=shortlist_index,
                    max_items=self.MAX_TOPIC_SHORTLIST,
                ),
                schema_name="semantic_topic_shortlist",
                retry_empty_selection=True,
                selection_bool_key="relevant",
                contract_hint=(
                    "every topic item must use a known candidate id, a JSON boolean relevant, and a numeric score; "
                    "include at least one relevant topic from every selected domain"
                ),
            )
            shortlist_items = shortlist_response.get("topics")
            shortlist_items = shortlist_items if isinstance(shortlist_items, list) else []
            returned_shortlist_ids = [
                resolve_shortlist_topic_id(item)
                for item in shortlist_items
                if isinstance(item, dict)
            ]
            self._trace_not_selected(
                "topic_shortlist",
                shortlist_trace_candidates,
                returned_ids=returned_shortlist_ids,
                id_key="topic_id",
            )
            shortlisted: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for item in shortlist_items:
                if not isinstance(item, dict):
                    self._trace_reject(
                        "topic_shortlist", item, "topic_item_must_be_an_object"
                    )
                    continue
                topic_id = resolve_shortlist_topic_id(item)
                relevant = self._strict_bool(item.get("relevant"))
                score = self._safe_score(item.get("score"))
                if not topic_id or relevant is not True or score is None:
                    self._trace_reject(
                        "topic_shortlist", item, "model_did_not_shortlist_topic"
                    )
                    continue
                shortlisted.append((score, shortlist_index[topic_id], item))
            shortlisted.sort(key=lambda value: (-value[0], value[1]["topic_id"]))
            domain_primaries: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            primary_topic_ids: set[str] = set()
            for domain_name in selected_domain_names:
                primary = next(
                    (
                        value
                        for value in shortlisted
                        if value[1]["domain"] == domain_name
                        and value[1]["topic_id"] not in primary_topic_ids
                    ),
                    None,
                )
                if primary is None:
                    continue
                domain_primaries.append(primary)
                primary_topic_ids.add(primary[1]["topic_id"])
            shortlisted = (
                domain_primaries
                + [
                    value
                    for value in shortlisted
                    if value[1]["topic_id"] not in primary_topic_ids
                ]
            )[: self.MAX_TOPIC_SHORTLIST]
            topic_candidates = [candidate for _score, candidate, _item in shortlisted]
            for score, candidate, item in shortlisted:
                self._trace_accept(
                    "topic_shortlist",
                    {
                        "domain": candidate["domain"],
                        "topic_id": candidate["topic_id"],
                        "topic": candidate["topic"],
                        "score": score,
                        "reason": norm_text(item.get("reason") or ""),
                    },
                )
            if not topic_candidates:
                self._set_empty_reason("topic_shortlist", "model_selected_no_topics")
                self._begin_stage("topic", reset=True)
                self._set_empty_reason("topic", "topic_shortlist_empty")
                return {"topic_judgments": [], "selected_topics": []}

        self._begin_stage("topic", candidate_count=len(topic_candidates), reset=True)
        topic_request_index = int(self._stage_trace("topic").get("chat_call_count") or 0) + 1
        topic_trace_candidates = [
            {
                "request_index": topic_request_index,
                "domain": item["domain"],
                "topic_id": item["topic_id"],
                "topic": item["topic"],
            }
            for item in topic_candidates
        ]
        self._trace_candidates("topic", topic_trace_candidates)
        prompt_payload = {
            **self._background_prompt_fields(sample, background_analysis),
            "max_selected_topics": self.MAX_SELECTED_TOPICS,
            "candidate_topics": [
                {
                    "domain": item["domain"],
                    "topic_id": item["topic_id"],
                    "canonical_name": item["topic"],
                    "summary": item["summary"],
                    "retrieval_hints": item["retrieval_hints"],
                    "cluster_previews": item["cluster_previews"],
                    "rule_count": item["rule_count"],
                }
                for item in topic_candidates
            ],
            "output_schema": {
                "topics": [
                    {
                        "topic_id": "stable candidate topic_id",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        candidate_by_id = {item["topic_id"]: item for item in topic_candidates}
        candidates_by_name: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for candidate in topic_candidates:
            key = (self._canonical_key(candidate["domain"]), self._canonical_key(candidate["topic"]))
            candidates_by_name.setdefault(key, []).append(candidate)

        def resolve_topic_id(item: Dict[str, Any]) -> str:
            topic_id = norm_text(item.get("topic_id") or "")
            if topic_id:
                return topic_id if topic_id in candidate_by_id else ""
            domain_key = self._canonical_key(item.get("domain") or "")
            topic_key = self._canonical_key(item.get("topic") or item.get("canonical_name") or "")
            matches = candidates_by_name.get((domain_key, topic_key), [])
            if not matches and topic_key:
                matches = [
                    value
                    for (candidate_domain, candidate_topic), values in candidates_by_name.items()
                    if candidate_topic == topic_key
                    for value in values
                ]
            return matches[0]["topic_id"] if len(matches) == 1 else ""

        response = self._chat_json(
            system_prompt=(
                "You are a physics rule navigator. Inside the selected domains, use only the problem background to "
                "choose the minimum set of topics that govern the physical mechanism. Prefer 1-2 topics. Reject "
                "neighboring concepts, prerequisite knowledge, downstream consequences, and shared-symbol matches. "
                "A topic must contribute its distinctive law or construction; the target quantity's name alone is "
                "not evidence. Do not infer problem facts from a student solution. Treat topic summaries as hard semantic "
                "boundaries. Use retrieval hints and cluster previews only to disambiguate those boundaries. Copy "
                "the stable topic_id exactly; do not rewrite it. If the background is incomplete, be conservative. "
                "Missing a figure or a fine-grained condition is not by itself a reason to omit the broad topic; "
                "record the missing information and select the closest supported physical mechanism. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            list_key="topics",
            response_validator=lambda payload: self._validate_selection_contract(
                payload,
                list_key="topics",
                bool_key="relevant",
                resolve_id=resolve_topic_id,
            ),
            response_schema=self._selection_response_schema(
                list_key="topics",
                id_key="topic_id",
                bool_key="relevant",
                allowed_ids=candidate_by_id,
                max_items=self.MAX_SELECTED_TOPICS,
            ),
            schema_name="semantic_topic_selection",
            retry_empty_selection=True,
            selection_bool_key="relevant",
            contract_hint=(
                "every topic item must use a known candidate id, a JSON boolean relevant, and a numeric score"
            ),
        )
        response_items = response.get("topics")
        if not isinstance(response_items, list):
            self._trace_reject("topic", response_items, "topics_must_be_a_list")
            response_items = []
        returned_topic_ids = [
            resolve_topic_id(item) for item in response_items if isinstance(item, dict)
        ]
        self._trace_not_selected(
            "topic",
            topic_trace_candidates,
            returned_ids=returned_topic_ids,
            id_key="topic_id",
        )
        judgments: List[Dict[str, Any]] = []
        for item in response_items:
            if not isinstance(item, dict):
                self._trace_reject("topic", item, "topic_item_must_be_an_object")
                continue
            topic_id = norm_text(item.get("topic_id") or "")
            candidate = candidate_by_id.get(topic_id) if topic_id else None
            if topic_id and candidate is None:
                self._trace_reject("topic", item, "unknown_topic_id")
                continue
            if candidate is None:
                domain_key = self._canonical_key(item.get("domain") or "")
                topic_key = self._canonical_key(item.get("topic") or item.get("canonical_name") or "")
                matches = candidates_by_name.get((domain_key, topic_key), [])
                if not matches and topic_key:
                    matches = [
                        value for (candidate_domain, candidate_topic), values in candidates_by_name.items()
                        if candidate_topic == topic_key
                        for value in values
                    ]
                if len(matches) != 1:
                    self._trace_reject("topic", item, "unknown_or_ambiguous_canonical_topic")
                    continue
                candidate = matches[0]
                topic_id = candidate["topic_id"]
            relevant = self._strict_bool(item.get("relevant"))
            if relevant is None:
                self._trace_reject("topic", item, "relevant_must_be_json_boolean")
                continue
            if not relevant:
                self._trace_reject("topic", item, "model_marked_irrelevant")
                continue
            score = self._safe_score(item.get("score"))
            if score is None:
                self._trace_reject("topic", item, "invalid_score")
                continue
            judgments.append(
                {
                    "domain": candidate["domain"],
                    "topic_id": topic_id,
                    "topic": candidate["topic"],
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                    "topic_obj": candidate["topic_obj"],
                }
            )
        best_by_topic: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            key = judgment["topic_id"]
            current = best_by_topic.get(key)
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("topic", self._compact_judgment(current), "duplicate_lower_score")
                best_by_topic[key] = judgment
            else:
                self._trace_reject("topic", self._compact_judgment(judgment), "duplicate_lower_score")
        judgments = list(best_by_topic.values())
        judgments.sort(key=lambda item: (-float(item["score"]), item["topic_id"]))
        for omitted in judgments[self.MAX_SELECTED_TOPICS :]:
            self._trace_reject("topic", self._compact_judgment(omitted), "selection_limit")
        judgments = judgments[: self.MAX_SELECTED_TOPICS]
        for judgment in judgments:
            self._trace_accept("topic", self._compact_judgment(judgment))
        if not judgments:
            reason = "model_selected_no_topics" if not response_items else "all_topic_items_rejected"
            self._set_empty_reason("topic", reason)
        return {"topic_judgments": judgments, "selected_topics": judgments}

    def select_clusters_semantically(
        self,
        sample: Dict[str, Any],
        selected_topics: Iterable[Dict[str, Any]],
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self._begin_stage("cluster", reset=True)
        all_judgments: List[Dict[str, Any]] = []
        selected_topic_list = list(selected_topics)
        for topic_match in selected_topic_list:
            all_cluster_candidates = self._build_cluster_candidates(topic_match)
            if not all_cluster_candidates:
                continue
            cluster_candidates = [
                item
                for item in all_cluster_candidates
                if item.get("navigation_role") == "primary"
            ]
            self._begin_stage("cluster", candidate_count=len(all_cluster_candidates))
            cluster_request_index = int(
                self._stage_trace("cluster").get("chat_call_count") or 0
            ) + 1
            cluster_trace_candidates = [
                {
                    "request_index": cluster_request_index,
                    "domain": item["domain"],
                    "topic_id": topic_match.get("topic_id") or "",
                    "topic": item["topic"],
                    "cluster_id": item["cluster_id"],
                    "cluster": item["cluster"],
                    "navigation_role": item["navigation_role"],
                }
                for item in all_cluster_candidates
            ]
            self._trace_candidates("cluster", cluster_trace_candidates)
            primary_trace_candidates = [
                item for item in cluster_trace_candidates if item["navigation_role"] == "primary"
            ]
            for item in cluster_trace_candidates:
                if item["navigation_role"] != "primary":
                    self._stage_trace("cluster")["not_selected"].append(
                        {
                            **copy.deepcopy(item),
                            "reason": "deferred_by_navigation_role",
                        }
                    )
            if not cluster_candidates:
                continue
            prompt_payload = {
                **self._background_prompt_fields(sample, background_analysis),
                "max_selected_clusters": self.MAX_SELECTED_CLUSTERS,
                "domain": topic_match["domain"],
                "topic_id": topic_match.get("topic_id") or "",
                "topic": topic_match["topic"],
                "topic_summary": norm_text(topic_match.get("topic_obj", {}).get("summary") or ""),
                "candidate_clusters": [
                    {
                        "cluster_id": item["cluster_id"],
                        "cluster": item["cluster"],
                        "summary": item["summary"],
                        "navigation_role": item["navigation_role"],
                        "representative_rules": item["representative_rules"],
                        "rule_group_summaries": [
                            {
                                "group_id": group["group_id"],
                                "name": group["name"],
                                "summary": group["summary"],
                                "activation_condition": group["activation_condition"],
                                "rule_count": len(group["rule_ids"]),
                            }
                            for group in item["rule_groups"]
                        ],
                        "rule_count": len(item["rule_ids"]),
                    }
                    for item in cluster_candidates
                ],
                "output_schema": {
                    "clusters": [
                        {
                            "cluster_id": "string",
                            "relevant": True,
                            "score": 0.0,
                            "reason": "short reason",
                        }
                    ]
                },
            }
            cluster_index = {item["cluster_id"]: item for item in cluster_candidates}

            def resolve_cluster_id(item: Dict[str, Any]) -> str:
                cluster_id = norm_text(item.get("cluster_id") or "")
                return cluster_id if cluster_id in cluster_index else ""

            response = self._chat_json(
                system_prompt=(
                    "You are a physics rule navigator. Inside the selected topic, use only the problem background to "
                    "choose the minimum set of scenario clusters that describe the concrete physical mechanism and "
                    "conditions. Prefer one cluster and add another only for a distinct applicable mechanism. Reject "
                    "generic approximations and neighboring derivation styles. Do not infer problem facts from a "
                    "student solution. Respect cluster summaries and activation conditions as hard boundaries. "
                    "Return JSON only."
                ),
                user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
                list_key="clusters",
                response_validator=lambda payload: self._validate_selection_contract(
                    payload,
                    list_key="clusters",
                    bool_key="relevant",
                    resolve_id=resolve_cluster_id,
                ),
                response_schema=self._selection_response_schema(
                    list_key="clusters",
                    id_key="cluster_id",
                    bool_key="relevant",
                    allowed_ids=cluster_index,
                    max_items=self.MAX_SELECTED_CLUSTERS,
                ),
                schema_name="semantic_cluster_selection",
                retry_empty_selection=True,
                selection_bool_key="relevant",
                contract_hint=(
                    "every cluster item must use a known candidate id, a JSON boolean relevant, and a numeric score"
                ),
            )
            topic_judgments: List[Dict[str, Any]] = []
            response_items = response.get("clusters")
            if not isinstance(response_items, list):
                self._trace_reject(
                    "cluster",
                    response_items,
                    "clusters_must_be_a_list",
                    topic_id=topic_match.get("topic_id") or "",
                )
                response_items = []
            returned_cluster_ids = [
                resolve_cluster_id(item) for item in response_items if isinstance(item, dict)
            ]
            self._trace_not_selected(
                "cluster",
                primary_trace_candidates,
                returned_ids=returned_cluster_ids,
                id_key="cluster_id",
            )
            for item in response_items:
                if not isinstance(item, dict):
                    self._trace_reject("cluster", item, "cluster_item_must_be_an_object")
                    continue
                cluster_id = norm_text(item.get("cluster_id") or "")
                if cluster_id not in cluster_index:
                    self._trace_reject("cluster", item, "unknown_cluster_id")
                    continue
                relevant = self._strict_bool(item.get("relevant"))
                if relevant is None:
                    self._trace_reject("cluster", item, "relevant_must_be_json_boolean")
                    continue
                if not relevant:
                    self._trace_reject("cluster", item, "model_marked_irrelevant")
                    continue
                score = self._safe_score(item.get("score"))
                if score is None:
                    self._trace_reject("cluster", item, "invalid_score")
                    continue
                candidate = cluster_index[cluster_id]
                topic_judgments.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster": candidate["cluster"],
                        "navigation_role": candidate["navigation_role"],
                        "candidate_source": "selected_cluster",
                        "domain": candidate["domain"],
                        "topic_id": topic_match.get("topic_id") or "",
                        "topic": candidate["topic"],
                        "relevant": True,
                        "score": score,
                        "reason": norm_text(item.get("reason") or ""),
                        "cluster_obj": candidate["cluster_obj"],
                        "topic_obj": candidate["topic_obj"],
                        "rule_groups": candidate["rule_groups"],
                        "rule_ids": candidate["rule_ids"],
                        "topic_rules": candidate["topic_rules"],
                    }
                )
            topic_judgments.sort(key=lambda item: (-float(item["score"]), item["cluster_id"]))
            all_judgments.extend(topic_judgments)
            self._checkpoint_partial_items("cluster", all_judgments)
        best_by_cluster: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for judgment in all_judgments:
            key = (judgment["domain"], judgment.get("topic_id") or judgment["topic"], judgment["cluster_id"])
            current = best_by_cluster.get(key)
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("cluster", self._compact_judgment(current), "duplicate_lower_score")
                best_by_cluster[key] = judgment
            else:
                self._trace_reject("cluster", self._compact_judgment(judgment), "duplicate_lower_score")
        all_judgments = list(best_by_cluster.values())
        all_judgments.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"], item["cluster_id"]))
        topic_order = {
            (item["domain"], item.get("topic_id") or item["topic"]): index
            for index, item in enumerate(selected_topic_list)
        }
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for judgment in all_judgments:
            key = (judgment["domain"], judgment.get("topic_id") or judgment["topic"])
            grouped.setdefault(key, []).append(judgment)
        primaries = [
            items[0]
            for _key, items in sorted(
                grouped.items(),
                key=lambda pair: topic_order.get(pair[0], len(topic_order)),
            )
            if items
        ]
        primary_ids = {id(item) for item in primaries}
        extras = [item for item in all_judgments if id(item) not in primary_ids]
        fair_order = primaries + extras
        for omitted in fair_order[self.MAX_SELECTED_CLUSTERS :]:
            self._trace_reject("cluster", self._compact_judgment(omitted), "selection_limit")
        all_judgments = fair_order[: self.MAX_SELECTED_CLUSTERS]
        for judgment in all_judgments:
            self._trace_accept("cluster", self._compact_judgment(judgment))
        stage_trace = self._stage_trace("cluster")
        if not all_judgments:
            reason = (
                "no_cluster_candidates"
                if not stage_trace["candidate_count"]
                else "model_selected_no_clusters_or_all_items_rejected"
            )
            self._set_empty_reason("cluster", reason)
        return {
            "cluster_judgments": list(all_judgments),
            "selected_clusters": list(all_judgments),
        }

    def _select_one_navigation_cluster(
        self,
        *,
        sample: Dict[str, Any],
        topic_match: Dict[str, Any],
        candidates: Iterable[Dict[str, Any]],
        background_analysis: Dict[str, Any] | None,
        stage: str,
        candidate_source: str,
    ) -> Dict[str, Any] | None:
        candidate_list = list(candidates)
        if not candidate_list:
            return None
        reset_stage = stage not in (self.last_trace.get("stages") or {})
        self._begin_stage(
            stage,
            candidate_count=len(candidate_list),
            reset=reset_stage,
        )
        request_index = int(self._stage_trace(stage).get("chat_call_count") or 0) + 1
        trace_candidates = [
            {
                "request_index": request_index,
                "domain": item["domain"],
                "topic_id": topic_match.get("topic_id") or "",
                "topic": item["topic"],
                "cluster_id": item["cluster_id"],
                "cluster": item["cluster"],
                "navigation_role": item["navigation_role"],
                "candidate_source": candidate_source,
            }
            for item in candidate_list
        ]
        self._trace_candidates(stage, trace_candidates)
        candidate_index = {item["cluster_id"]: item for item in candidate_list}

        def resolve_cluster_id(item: Dict[str, Any]) -> str:
            cluster_id = norm_text(item.get("cluster_id") or "")
            return cluster_id if cluster_id in candidate_index else ""

        prompt_payload = {
            **self._background_prompt_fields(sample, background_analysis),
            "selection_phase": candidate_source,
            "domain": topic_match["domain"],
            "topic_id": topic_match.get("topic_id") or "",
            "topic": topic_match["topic"],
            "candidate_clusters": [
                {
                    "cluster_id": item["cluster_id"],
                    "cluster": item["cluster"],
                    "summary": self._clip_prompt_text(item["summary"], 240),
                    "navigation_role": item["navigation_role"],
                    "representative_rules": item["representative_rules"],
                    "activation_conditions": ordered_unique(
                        [
                            self._clip_prompt_text(group["activation_condition"], 160)
                            for group in item["rule_groups"]
                            if norm_text(group["activation_condition"])
                        ]
                    )[:3],
                }
                for item in candidate_list
            ],
            "output_schema": {
                "clusters": [
                    {
                        "cluster_id": "stable candidate cluster_id",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        response = self._chat_json(
            system_prompt=(
                "You are performing one bounded in-tree navigation reconsideration. Use only the problem "
                "background and select at most one alternative cluster whose concrete scenario or representative "
                "rules directly match the active task. Generic storage-bucket names are not evidence; use their "
                "representative rules. Return an empty list when no candidate is supported. Do not use the student "
                "solution. Copy cluster_id exactly and return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
            list_key="clusters",
            response_validator=lambda payload: self._validate_selection_contract(
                payload,
                list_key="clusters",
                bool_key="relevant",
                resolve_id=resolve_cluster_id,
            ),
            response_schema=self._selection_response_schema(
                list_key="clusters",
                id_key="cluster_id",
                bool_key="relevant",
                allowed_ids=candidate_index,
                max_items=1,
            ),
            schema_name=f"semantic_{stage}",
            contract_hint=(
                "return at most one known cluster id with a JSON boolean relevant and a numeric score"
            ),
        )
        response_items = response.get("clusters")
        response_items = response_items if isinstance(response_items, list) else []
        returned_ids = [
            resolve_cluster_id(item) for item in response_items if isinstance(item, dict)
        ]
        self._trace_not_selected(
            stage,
            trace_candidates,
            returned_ids=returned_ids,
            id_key="cluster_id",
        )
        judgments: List[Dict[str, Any]] = []
        for item in response_items:
            if not isinstance(item, dict):
                self._trace_reject(stage, item, "cluster_item_must_be_an_object")
                continue
            cluster_id = resolve_cluster_id(item)
            relevant = self._strict_bool(item.get("relevant"))
            score = self._safe_score(item.get("score"))
            if not cluster_id or relevant is not True or score is None:
                self._trace_reject(stage, item, "alternative_cluster_not_selected")
                continue
            candidate = candidate_index[cluster_id]
            judgments.append(
                {
                    "cluster_id": cluster_id,
                    "cluster": candidate["cluster"],
                    "navigation_role": candidate["navigation_role"],
                    "candidate_source": candidate_source,
                    "domain": candidate["domain"],
                    "topic_id": topic_match.get("topic_id") or "",
                    "topic": candidate["topic"],
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                    "cluster_obj": candidate["cluster_obj"],
                    "topic_obj": candidate["topic_obj"],
                    "rule_groups": candidate["rule_groups"],
                    "rule_ids": candidate["rule_ids"],
                    "topic_rules": candidate["topic_rules"],
                }
            )
        judgments.sort(key=lambda item: (-float(item["score"]), item["cluster_id"]))
        if not judgments:
            self._set_empty_reason(stage, "no_supported_alternative_cluster")
            return None
        selected = judgments[0]
        self._trace_accept(stage, self._compact_judgment(selected))
        return selected

    @staticmethod
    def _clip_prompt_text(value: Any, max_chars: int) -> str:
        text = norm_text(value or "")
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return f"{text[: max_chars - 1]}…"

    @classmethod
    def _clip_prompt_items(cls, values: Iterable[Any], max_chars: int) -> List[str]:
        items = [norm_text(item) for item in values if norm_text(item)][:6]
        if not items:
            return []
        per_item = max(16, max_chars // len(items))
        return [cls._clip_prompt_text(item, per_item) for item in items]

    def _rule_prompt_candidate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "rule_id": item["rule_id"],
            "title": item["title"],
            "summary": item["summary"],
            "trigger": item["trigger"],
            "check_logic": item["check_logic"],
            "error_type": item["error_type"],
            "preconditions": item.get("preconditions") or [],
            "violation_signatures": item.get("violation_signatures") or [],
            "negative_conditions": item.get("negative_conditions") or [],
            "evidence_requirements": item.get("evidence_requirements") or [],
            "symbolic_hint": item["symbolic_hint"],
        }
        candidate_limit = self.rule_candidate_batch_chars - 2
        if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) <= candidate_limit:
            return payload

        available = max(256, candidate_limit - 512)
        symbolic_hint = item.get("symbolic_hint") if isinstance(item.get("symbolic_hint"), dict) else {}
        compact = {
            "rule_id": item["rule_id"],
            "title": self._clip_prompt_text(item["title"], max(32, int(available * 0.04))),
            "summary": self._clip_prompt_text(item["summary"], max(64, int(available * 0.10))),
            "trigger": self._clip_prompt_text(item["trigger"], max(64, int(available * 0.10))),
            "check_logic": self._clip_prompt_text(item["check_logic"], max(64, int(available * 0.15))),
            "error_type": self._clip_prompt_text(item["error_type"], 100),
            "preconditions": self._clip_prompt_items(
                item.get("preconditions") or [], max(64, int(available * 0.07))
            ),
            "violation_signatures": self._clip_prompt_items(
                item.get("violation_signatures") or [], max(64, int(available * 0.07))
            ),
            "negative_conditions": self._clip_prompt_items(
                item.get("negative_conditions") or [], max(64, int(available * 0.07))
            ),
            "evidence_requirements": self._clip_prompt_items(
                item.get("evidence_requirements") or [], max(64, int(available * 0.07))
            ),
            "symbolic_hint": {
                "primitive": self._clip_prompt_text(symbolic_hint.get("primitive") or "", 100),
                "canonical": self._clip_prompt_text(
                    symbolic_hint.get("canonical") or "", max(64, int(available * 0.05))
                ),
                "required_symbols": self._clip_prompt_items(
                    symbolic_hint.get("required_symbols") or [], max(64, int(available * 0.03))
                ),
            },
        }
        if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) <= candidate_limit:
            return compact

        for field in (
            "preconditions",
            "violation_signatures",
            "negative_conditions",
            "evidence_requirements",
        ):
            compact[field] = []
        compact["symbolic_hint"] = {}
        for field in ("title", "summary", "trigger", "check_logic"):
            compact[field] = self._clip_prompt_text(compact[field], 128)
        if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) <= candidate_limit:
            return compact
        return {
            "rule_id": item["rule_id"],
            "title": self._clip_prompt_text(item["title"], 128),
            "summary": self._clip_prompt_text(item["summary"], 256),
        }

    def _batch_rule_candidates(
        self,
        rule_candidates: List[Dict[str, Any]],
    ) -> List[List[tuple[Dict[str, Any], Dict[str, Any]]]]:
        batches: List[List[tuple[Dict[str, Any], Dict[str, Any]]]] = []
        current: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        current_chars = 2
        for candidate in rule_candidates:
            prompt_candidate = self._rule_prompt_candidate(candidate)
            candidate_chars = len(json.dumps(prompt_candidate, ensure_ascii=False, separators=(",", ":")))
            added_chars = candidate_chars + (1 if current else 0)
            if current and (
                len(current) >= self.rule_candidate_batch_size
                or current_chars + added_chars > self.rule_candidate_batch_chars
            ):
                batches.append(current)
                current = []
                current_chars = 2
                added_chars = candidate_chars
            current.append((candidate, prompt_candidate))
            current_chars += added_chars
        if current:
            batches.append(current)
        return batches

    def _provisional_rule_limit(self, batch_size: int) -> int:
        """Keep the first Rule pass recall-oriented while the final cap stays strict."""
        recall_target = min(
            self.MAX_PROVISIONAL_RULES_PER_BATCH,
            self.max_selected_rules * 2,
        )
        return min(
            max(0, int(batch_size)),
            max(self.max_selected_rules, recall_target),
        )

    @staticmethod
    def _rank_rule_candidates_by_background(
        rule_candidates: Iterable[Dict[str, Any]],
        background_text: str,
    ) -> List[Dict[str, Any]]:
        """Order semantic candidates by background evidence without filtering any rule."""
        ranked: List[tuple[float, int, Dict[str, Any]]] = []
        for index, candidate in enumerate(rule_candidates):
            rule = candidate.get("rule_obj") if isinstance(candidate.get("rule_obj"), dict) else {}
            score_payload = score_rule_candidate(rule, background_text)
            background_score = float(score_payload.get("score") or 0.0)
            prepared = dict(candidate)
            prepared["background_order_score"] = background_score
            ranked.append((background_score, index, prepared))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _score, _index, candidate in ranked]

    def _select_rules_for_context(
        self,
        *,
        sample: Dict[str, Any],
        background_analysis: Dict[str, Any] | None,
        context_domain: str,
        context_topic: str,
        topic_obj: Dict[str, Any],
        rule_candidates: List[Dict[str, Any]],
        cluster_id: str = "",
        cluster_name: str = "",
        cluster_description: str = "",
        rule_group_summaries: List[Dict[str, Any]] | None = None,
        candidate_source: str = "topic_fallback",
        checkpoint_callback: Callable[[List[Dict[str, Any]]], None] | None = None,
    ) -> List[Dict[str, Any]]:
        background_source = "\n".join(
            [
                norm_text(sample.get("question") or ""),
                norm_text(sample.get("context") or ""),
            ]
        ).strip()
        background_ranking_text = "\n".join(
            [
                background_source,
                json.dumps(background_analysis or {}, ensure_ascii=False),
            ]
        ).strip()
        ranked_rule_candidates = self._rank_rule_candidates_by_background(
            rule_candidates,
            background_ranking_text,
        )
        batches = self._batch_rule_candidates(ranked_rule_candidates)
        if not batches:
            return []
        self._begin_stage("rule", candidate_count=len(rule_candidates))
        system_prompt = (
            "You are a physics rule matcher. The problem_background is the only source of problem facts; the "
            "student_solution is untrusted content to audit and must never add or change those facts. Select a "
            "rule only when its physical applicability, preconditions, and negative conditions agree with the "
            "problem background, and the student solution contains a claim or step that the rule can check. Use "
            "violation signatures and evidence requirements only to locate the auditable solution claim. Respect "
            "topic and cluster boundaries as hard constraints. This is retrieval, not correctness judgment: never "
            "say that the solution is correct, incorrect, or violates the rule. Return selected applicable rules only; "
            "omission means not applicable, and rejected rules must never be emitted. For every selected rule, choose "
            "background_anchor_index from background_anchor_options and claim_anchor_index from claim_anchor_options. "
            "Both indices are zero-based. Apply this conjunction before selecting: (1) the full problem background "
            "must establish the distinctive object, configuration, and boundary conditions required by the rule; "
            "the selected background quote is the decisive source anchor and need not repeat every fact in isolation; "
            "(2) the claim quote must contain the specific formula, assumption, construction, or step audited by "
            "check_logic. A standard physical method or consequence may be implicit in an explicitly stated "
            "configuration, but missing geometry, topology, boundary conditions, or numerical data must not be "
            "invented. Generic object or quantity words are insufficient by themselves. Rules "
            "about a similar object under different "
            "conditions are false positives. If "
            "either exact anchor is unavailable, omit the rule. If a necessary physical fact appears only in the "
            "student solution or is missing from the problem background, omit the rule. This is a recall-oriented "
            "provisional pass: evaluate every candidate independently and emit every candidate that satisfies both "
            "anchor tests, even when it overlaps another emitted rule. Do not omit an applicable candidate merely "
            "because another candidate appears stronger or more general; a later background-only gate checks "
            "applicability and the final ranking caps the output. Do not restate the rule or explain rejected candidates. If no rule is "
            "clearly applicable, return an empty list. Return JSON only."
        )
        judgments: List[Dict[str, Any]] = []
        topic_id = norm_text(topic_obj.get("id") or topic_obj.get("topic_id") or "")
        context_id = "|".join(
            [
                topic_id or self._canonical_key(context_topic),
                cluster_id or "topic_scope",
                candidate_source,
            ]
        )
        claim_source = self._student_solution(sample)
        background_anchor_focus = "\n".join(
            [
                json.dumps(background_analysis or {}, ensure_ascii=False),
                context_domain,
                context_topic,
                cluster_name,
                cluster_description,
            ]
        )
        background_anchor_candidates = self._source_anchor_candidates(
            background_source,
            max_items=12,
            focus=background_anchor_focus,
        )
        claim_anchor_candidates = self._source_anchor_candidates(
            claim_source,
            max_items=32,
        )
        if not background_anchor_candidates or not claim_anchor_candidates:
            self._set_empty_reason("rule", "missing_source_anchor_candidates")
            return []
        for batch_index, batch in enumerate(batches, start=1):
            provisional_rule_limit = self._provisional_rule_limit(len(batch))
            batch_rule_index = {candidate["rule_id"]: candidate for candidate, _ in batch}
            request_index = int(self._stage_trace("rule").get("chat_call_count") or 0) + 1
            rule_trace_candidates = [
                {
                    "request_index": request_index,
                    "context_id": context_id,
                    "batch_index": batch_index,
                    "batch_total": len(batches),
                    "domain": context_domain,
                    "topic": context_topic,
                    "cluster_id": cluster_id,
                    "rule_id": candidate["rule_id"],
                    "title": candidate["title"],
                    "background_order_score": float(
                        candidate.get("background_order_score") or 0.0
                    ),
                    "candidate_source": candidate_source,
                }
                for candidate, _ in batch
            ]
            self._trace_candidates("rule", rule_trace_candidates)

            def resolve_rule_id(item: Dict[str, Any]) -> str:
                rule_id = norm_text(item.get("rule_id") or "")
                return rule_id if rule_id in batch_rule_index else ""

            prompt_payload = {
                **self._background_prompt_fields(sample, background_analysis),
                "student_solution": self._student_solution(sample),
                "domain": context_domain,
                "topic": context_topic,
                "topic_id": topic_id,
                "context_id": context_id,
                "candidate_source": candidate_source,
                "topic_summary": norm_text(topic_obj.get("summary") or ""),
                "cluster_id": cluster_id,
                "cluster": cluster_name,
                "cluster_summary": cluster_description,
                "rule_group_summaries": rule_group_summaries or [],
                "background_anchor_options": [
                    {"index": index, "text": anchor}
                    for index, anchor in enumerate(background_anchor_candidates)
                ],
                "claim_anchor_options": [
                    {"index": index, "text": anchor}
                    for index, anchor in enumerate(claim_anchor_candidates)
                ],
                "candidate_batch": {"index": batch_index, "total": len(batches)},
                "max_provisional_rules": provisional_rule_limit,
                "candidate_rules": [prompt_candidate for _, prompt_candidate in batch],
                "output_schema": {
                    "rules": [
                        {
                            "rule_id": "string",
                            "score": 0.8,
                            "background_anchor_index": 0,
                            "claim_anchor_index": 0,
                        }
                    ]
                },
            }
            response = self._chat_json(
                system_prompt=system_prompt,
                user_prompt=json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
                list_key="rules",
                response_validator=lambda payload: self._validate_rule_selection_contract(
                    payload,
                    resolve_id=resolve_rule_id,
                    background_source=background_source,
                    claim_source=claim_source,
                    allowed_background_anchors=background_anchor_candidates,
                    allowed_claim_anchors=claim_anchor_candidates,
                    max_items=provisional_rule_limit,
                ),
                response_schema=self._rule_selection_response_schema(
                    allowed_ids=batch_rule_index,
                    background_anchors=background_anchor_candidates,
                    claim_anchors=claim_anchor_candidates,
                    max_items=provisional_rule_limit,
                ),
                schema_name="semantic_rule_selection",
                contract_hint=(
                    "return every independently applicable provisional rule, up to max_provisional_rules; "
                    "do not deduplicate overlaps; every item must use an id from this batch, score 0.8-1, "
                    "and choose both zero-based indices from the supplied anchor option lists"
                ),
            )
            response_items = response.get("rules")
            if not isinstance(response_items, list):
                self._trace_reject(
                    "rule",
                    response_items,
                    "rules_must_be_a_list",
                    candidate_source=candidate_source,
                    batch_index=batch_index,
                )
                response_items = []
            returned_rule_ids = [
                resolve_rule_id(item) for item in response_items if isinstance(item, dict)
            ]
            self._trace_not_selected(
                "rule",
                rule_trace_candidates,
                returned_ids=returned_rule_ids,
                id_key="rule_id",
            )
            for item in response_items:
                if not isinstance(item, dict):
                    self._trace_reject("rule", item, "rule_item_must_be_an_object", candidate_source=candidate_source)
                    continue
                rule_id = norm_text(item.get("rule_id") or "")
                if rule_id not in batch_rule_index:
                    self._trace_reject("rule", item, "unknown_rule_id", candidate_source=candidate_source)
                    continue
                score = self._safe_score(item.get("score"))
                if score is None or score < 0.8:
                    self._trace_reject("rule", item, "invalid_score", candidate_source=candidate_source)
                    continue
                background_anchor_index = int(item["background_anchor_index"])
                claim_anchor_index = int(item["claim_anchor_index"])
                background_anchor = background_anchor_candidates[background_anchor_index]
                claim_anchor = claim_anchor_candidates[claim_anchor_index]
                candidate = batch_rule_index[rule_id]
                judgments.append(
                    {
                        "rule_id": rule_id,
                        "title": candidate["title"],
                        "domain": context_domain,
                        "topic": context_topic,
                        "topic_id": topic_id,
                        "context_id": context_id,
                        "cluster_id": cluster_id,
                        "cluster": cluster_name,
                        "applicable": True,
                        "score": score,
                        "background_order_score": float(
                            candidate.get("background_order_score") or 0.0
                        ),
                        "background_anchor": background_anchor,
                        "claim_anchor": claim_anchor,
                        "reason": (
                            f"Background anchor: {background_anchor} | Claim anchor: {claim_anchor}"
                        ),
                        "candidate_source": candidate_source,
                        "rule_obj": candidate["rule_obj"],
                    }
                )
            if checkpoint_callback is not None:
                checkpoint_callback(judgments)
        best_by_rule: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            current = best_by_rule.get(judgment["rule_id"])
            judgment_rank = (
                float(judgment["score"]),
                float(judgment.get("background_order_score") or 0.0),
            )
            current_rank = (
                float(current["score"]),
                float(current.get("background_order_score") or 0.0),
            ) if current is not None else (-1.0, -1.0)
            if current is None or judgment_rank > current_rank:
                if current is not None:
                    self._trace_reject("rule", self._compact_judgment(current), "duplicate_lower_score")
                best_by_rule[judgment["rule_id"]] = judgment
            else:
                self._trace_reject("rule", self._compact_judgment(judgment), "duplicate_lower_score")
        merged = list(best_by_rule.values())
        merged.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item.get("background_order_score") or 0.0),
                item["rule_id"],
            )
        )
        return merged

    @classmethod
    def _rule_confirmation_candidate(cls, judgment: Dict[str, Any]) -> Dict[str, Any]:
        rule = judgment.get("rule_obj") if isinstance(judgment.get("rule_obj"), dict) else {}

        def compact_items(field: str, max_chars: int) -> List[str]:
            values = rule.get(field)
            return cls._clip_prompt_items(values if isinstance(values, list) else [], max_chars)

        return {
            "rule_id": norm_text(judgment.get("rule_id") or ""),
            "domain": norm_text(judgment.get("domain") or ""),
            "topic": norm_text(judgment.get("topic") or ""),
            "cluster": norm_text(judgment.get("cluster") or ""),
            "title": cls._clip_prompt_text(rule.get("title") or judgment.get("title") or "", 180),
            "summary": cls._clip_prompt_text(rule.get("summary") or "", 320),
            "trigger": cls._clip_prompt_text(rule.get("trigger") or "", 420),
            "check_logic": cls._clip_prompt_text(rule.get("check_logic") or "", 640),
            "preconditions": compact_items("preconditions", 360),
            "negative_conditions": compact_items("negative_conditions", 280),
            "evidence_requirements": compact_items("evidence_requirements", 360),
            "provisional_score": float(judgment.get("score") or 0.0),
            "background_order_score": float(
                judgment.get("background_order_score") or 0.0
            ),
            "provisional_background_anchor": cls._clip_prompt_text(
                judgment.get("background_anchor") or "", 160
            ),
        }

    def _confirm_rule_judgments(
        self,
        *,
        sample: Dict[str, Any],
        background_analysis: Dict[str, Any] | None,
        judgments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not judgments:
            return []
        self._begin_stage(
            "rule_confirmation",
            candidate_count=len(judgments),
            reset="rule_confirmation" not in (self.last_trace.get("stages") or {}),
        )
        background_source = "\n".join(
            [
                norm_text(sample.get("question") or ""),
                norm_text(sample.get("context") or ""),
            ]
        ).strip()
        candidate_index = {
            norm_text(item.get("rule_id") or ""): item
            for item in judgments
            if norm_text(item.get("rule_id") or "")
        }
        confirmation_focus = "\n".join(
            [
                json.dumps(background_analysis or {}, ensure_ascii=False),
                json.dumps(
                    [self._rule_confirmation_candidate(item) for item in judgments],
                    ensure_ascii=False,
                ),
            ]
        )
        background_anchor_candidates = self._source_anchor_candidates(
            background_source,
            max_items=12,
            focus=confirmation_focus,
        )
        if not background_anchor_candidates:
            self._set_empty_reason("rule_confirmation", "missing_source_anchor_candidates")
            self._active_stage = "rule"
            return []

        request_index = int(
            self._stage_trace("rule_confirmation").get("chat_call_count") or 0
        ) + 1
        trace_candidates = [
            {
                "request_index": request_index,
                "rule_id": rule_id,
                "title": norm_text(item.get("title") or ""),
                "domain": norm_text(item.get("domain") or ""),
                "topic": norm_text(item.get("topic") or ""),
                "cluster_id": norm_text(item.get("cluster_id") or ""),
                "candidate_source": "global_precision_confirmation",
            }
            for rule_id, item in candidate_index.items()
        ]
        self._trace_candidates("rule_confirmation", trace_candidates)

        def resolve_rule_id(item: Dict[str, Any]) -> str:
            rule_id = norm_text(item.get("rule_id") or "")
            return rule_id if rule_id in candidate_index else ""

        prompt_payload = {
            **self._background_prompt_fields(sample, background_analysis),
            "selection_phase": "background_only_rule_precision_confirmation",
            "background_anchor_options": [
                {"index": index, "text": anchor}
                for index, anchor in enumerate(background_anchor_candidates)
            ],
            "preliminary_rules": [
                self._rule_confirmation_candidate(item) for item in judgments
            ],
            "output_schema": {
                "decisions": [
                    {
                        "rule_id": "stable candidate rule_id",
                        "decision": "confirm or one compact rejection code",
                        "background_anchor_index": "0 when confirmed, otherwise -1",
                    }
                ]
            },
        }
        response = self._chat_json(
            system_prompt=(
                "You are the background-only final precision gate for physics rule retrieval. The student solution "
                "is intentionally unavailable here and must not be inferred. A previous stage already established "
                "that each preliminary rule has a potentially auditable solution step; your only task is to decide "
                "whether the raw question/context actually establishes the rule's physical applicability. Every "
                "preliminary rule is only a hypothesis produced by an independent batch. Classify each rule "
                "independently; final ranking and capping happen later. Consider the full raw question/context and "
                "use one background option as the decisive source anchor; that anchor need not repeat every relevant "
                "fact in isolation. Confirm a rule when the source establishes its distinctive objects, configuration, "
                "boundary conditions, and target quantity. A standard physical method or consequence may be implicit "
                "in that explicit configuration and need not be named by the problem. The prior stage has already "
                "checked the solution-side auditable claim; do not repeat or infer that unavailable evidence. Shared "
                "domain, topic, symbols, or generic object words are never evidence for a missing distinctive "
                "precondition. Facts present only in an unseen solution must never be supplied to the problem. "
                "Omit similar-object rules whose required configuration is absent. Do not preserve "
                "one rule per domain or topic, and do not reject a rule merely because another rule overlaps it; "
                "the unseen claim may make either rule necessary. Classify every preliminary rule exactly once using "
                "one of these decisions: confirm, reject_missing_background, or reject_different_configuration. "
                "Reject whenever a distinctive source fact is absent, contradicted, or replaced by a different "
                "configuration; never invent missing geometry, topology, boundary conditions, or data. For a rejection set the background "
                "anchor index to -1; for confirmation use a valid zero-based index. This is "
                "applicability retrieval, not a correctness verdict. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
            list_key="decisions",
            response_validator=lambda payload: self._validate_rule_confirmation_contract(
                payload,
                allowed_ids=candidate_index,
                background_source=background_source,
                allowed_background_anchors=background_anchor_candidates,
            ),
            response_schema=self._rule_confirmation_response_schema(
                allowed_ids=candidate_index,
                background_anchor_count=len(background_anchor_candidates),
            ),
            schema_name="semantic_rule_precision_confirmation",
            contract_hint=(
                "return exactly one independent background-applicability decision for every preliminary rule; "
                "confirmed items require a valid zero-based background anchor and rejected items use -1"
            ),
        )
        response_items = response.get("decisions")
        response_items = response_items if isinstance(response_items, list) else []
        confirmed_rule_ids = [
            resolve_rule_id(item)
            for item in response_items
            if isinstance(item, dict) and item.get("decision") == "confirm"
        ]
        self._trace_not_selected(
            "rule_confirmation",
            trace_candidates,
            returned_ids=confirmed_rule_ids,
            id_key="rule_id",
        )

        confirmed: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in response_items:
            if not isinstance(item, dict):
                continue
            rule_id = resolve_rule_id(item)
            if not rule_id or rule_id in seen:
                continue
            seen.add(rule_id)
            decision = norm_text(item.get("decision") or "")
            if decision != "confirm":
                self._trace_reject(
                    "rule_confirmation",
                    self._compact_judgment(candidate_index[rule_id]),
                    decision or "global_precision_confirmation_rejected",
                )
                continue
            background_anchor = background_anchor_candidates[int(item["background_anchor_index"])]
            judgment = copy.deepcopy(candidate_index[rule_id])
            judgment["background_anchor"] = background_anchor
            judgment["reason"] = (
                f"Background anchor: {background_anchor} | Claim anchor: {judgment.get('claim_anchor') or ''}"
            )
            judgment["confirmation_status"] = "confirmed"
            judgment["confirmation_decision"] = decision
            confirmed.append(judgment)
            self._trace_accept("rule_confirmation", self._compact_judgment(judgment))
        if not confirmed:
            self._set_empty_reason("rule_confirmation", "model_confirmed_no_rules")
        else:
            self._stage_trace("rule_confirmation")["empty_reason"] = ""
        self._active_stage = "rule"
        return confirmed

    def select_rules_semantically(
        self,
        sample: Dict[str, Any],
        selected_topics: Iterable[Dict[str, Any]],
        selected_clusters: Iterable[Dict[str, Any]] | None = None,
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self._begin_stage("rule", reset=True)
        all_judgments: List[Dict[str, Any]] = []
        selected_topic_list = list(selected_topics)
        cluster_list = list(selected_clusters or [])
        topics_with_rule_hits: set[tuple[str, str]] = set()
        attempted_rule_ids: Dict[tuple[str, str], set[str]] = {}
        navigation_clusters: List[Dict[str, Any]] = []

        def evaluate_cluster(item: Dict[str, Any]) -> bool:
            topic_key = (item["domain"], item["topic"])
            topic_match = next(
                (topic for topic in selected_topic_list if (topic["domain"], topic["topic"]) == topic_key),
                item,
            )
            all_topic_candidates = self._build_rule_candidates(topic_match)
            allowed_ids = {norm_text(rule_id) for rule_id in item.get("rule_ids", []) or [] if norm_text(rule_id)}
            already_attempted = attempted_rule_ids.setdefault(topic_key, set())
            rule_candidates = [
                candidate
                for candidate in all_topic_candidates
                if candidate["rule_id"] in allowed_ids
                and candidate["rule_id"] not in already_attempted
            ]
            if not rule_candidates:
                return False
            already_attempted.update(candidate["rule_id"] for candidate in rule_candidates)
            topic_obj = item.get("topic_obj") if isinstance(item.get("topic_obj"), dict) else {}
            cluster_judgments = self._select_rules_for_context(
                sample=sample,
                background_analysis=background_analysis,
                context_domain=item["domain"],
                context_topic=item["topic"],
                topic_obj=topic_obj,
                rule_candidates=rule_candidates,
                cluster_id=item["cluster_id"],
                cluster_name=item["cluster"],
                cluster_description=norm_text(item.get("cluster_obj", {}).get("summary") or ""),
                rule_group_summaries=[
                    {
                        "group_id": group["group_id"],
                        "name": group["name"],
                        "summary": group["summary"],
                        "activation_condition": group["activation_condition"],
                        "rule_count": len(group["rule_ids"]),
                    }
                    for group in item.get("rule_groups", [])
                ],
                candidate_source=norm_text(item.get("candidate_source") or "selected_cluster"),
                checkpoint_callback=lambda current: self._checkpoint_partial_items(
                    "rule", [*all_judgments, *current]
                ),
            )
            if cluster_judgments:
                topics_with_rule_hits.add(topic_key)
            all_judgments.extend(cluster_judgments)
            return bool(cluster_judgments)

        for item in cluster_list:
            evaluate_cluster(item)

        # If the first cluster does not yield an applicable rule, reconsider one
        # unselected primary cluster, then at most one deferred storage bucket.
        # This remains bounded navigation inside the selected topic.
        for topic_match in selected_topic_list:
            topic_key = (topic_match["domain"], topic_match["topic"])
            if topic_key in topics_with_rule_hits:
                continue
            if not self._build_rule_candidates(topic_match):
                continue
            all_clusters = self._build_cluster_candidates(topic_match)
            selected_topic_clusters = [
                item
                for item in cluster_list
                if (item.get("domain"), item.get("topic")) == topic_key
            ]
            selected_ids = {
                norm_text(item.get("cluster_id") or "")
                for item in selected_topic_clusters
            }
            primary_alternatives = [
                item
                for item in all_clusters
                if item["navigation_role"] == "primary"
                and item["cluster_id"] not in selected_ids
            ]
            alternative = self._select_one_navigation_cluster(
                sample=sample,
                topic_match=topic_match,
                candidates=primary_alternatives,
                background_analysis=background_analysis,
                stage="cluster_backtrack",
                candidate_source="primary_cluster_backtrack",
            )
            if alternative is not None:
                navigation_clusters.append(alternative)
                if evaluate_cluster(alternative):
                    continue
            deferred_candidates = [
                item
                for item in all_clusters
                if item["navigation_role"] == "deferred_bucket"
            ]
            deferred = self._select_one_navigation_cluster(
                sample=sample,
                topic_match=topic_match,
                candidates=deferred_candidates,
                background_analysis=background_analysis,
                stage="cluster_fallback",
                candidate_source="deferred_cluster_fallback",
            )
            if deferred is not None:
                navigation_clusters.append(deferred)
                evaluate_cluster(deferred)

        # Only clusterless topics use a topic-wide rule pass. When scenario clusters
        # exist, their semantic boundary remains hard; broadening to hundreds of
        # unrelated topic rules would increase both false positives and API cost.
        # The small general-reasoning bucket is a bounded fallback only when
        # the selected scenario path produced no applicable rule.
        general_clusters: Dict[tuple[str, str], Dict[str, Any]] = {}
        for topic_match in selected_topic_list:
            topic_key = (topic_match["domain"], topic_match["topic"])
            topic_cluster_candidates = self._build_cluster_candidates(topic_match)
            general_cluster = next(
                (
                    cluster
                    for cluster in topic_cluster_candidates
                    if cluster["cluster_id"] == "general_reasoning"
                ),
                None,
            )
            if general_cluster:
                general_clusters[topic_key] = general_cluster
            if topic_key in topics_with_rule_hits:
                continue
            if any(cluster["cluster_id"] != "general_reasoning" for cluster in topic_cluster_candidates):
                continue
            already_attempted = attempted_rule_ids.setdefault(topic_key, set())
            general_rule_ids = set((general_cluster or {}).get("rule_ids") or [])
            rule_candidates = [
                candidate
                for candidate in self._build_rule_candidates(topic_match)
                if candidate["rule_id"] not in already_attempted
                and candidate["rule_id"] not in general_rule_ids
            ]
            if not rule_candidates:
                continue
            already_attempted.update(candidate["rule_id"] for candidate in rule_candidates)
            topic_judgments = self._select_rules_for_context(
                sample=sample,
                background_analysis=background_analysis,
                context_domain=topic_match["domain"],
                context_topic=topic_match["topic"],
                topic_obj=topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {},
                rule_candidates=rule_candidates,
                candidate_source="topic_fallback",
                checkpoint_callback=lambda current: self._checkpoint_partial_items(
                    "rule", [*all_judgments, *current]
                ),
            )
            if topic_judgments:
                topics_with_rule_hits.add(topic_key)
            all_judgments.extend(topic_judgments)

        for topic_match in selected_topic_list:
            topic_key = (topic_match["domain"], topic_match["topic"])
            had_specific_hits = topic_key in topics_with_rule_hits
            if had_specific_hits:
                continue
            general_cluster = general_clusters.get(topic_key)
            if not general_cluster:
                continue
            already_attempted = attempted_rule_ids.setdefault(topic_key, set())
            general_rule_ids = set(general_cluster.get("rule_ids") or [])
            rule_candidates = [
                candidate
                for candidate in self._build_rule_candidates(topic_match)
                if candidate["rule_id"] in general_rule_ids
                and candidate["rule_id"] not in already_attempted
            ]
            if not rule_candidates:
                continue
            already_attempted.update(candidate["rule_id"] for candidate in rule_candidates)
            general_candidate_source = "general_reasoning_fallback"
            general_judgments = self._select_rules_for_context(
                sample=sample,
                background_analysis=background_analysis,
                context_domain=topic_match["domain"],
                context_topic=topic_match["topic"],
                topic_obj=topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {},
                rule_candidates=rule_candidates,
                cluster_id="general_reasoning",
                cluster_name=general_cluster["cluster"],
                cluster_description=general_cluster["summary"],
                candidate_source=general_candidate_source,
                checkpoint_callback=lambda current: self._checkpoint_partial_items(
                    "rule", [*all_judgments, *current]
                ),
            )
            if general_judgments:
                topics_with_rule_hits.add(topic_key)
                navigation_clusters.append(
                    {
                        "cluster_id": general_cluster["cluster_id"],
                        "cluster": general_cluster["cluster"],
                        "navigation_role": general_cluster["navigation_role"],
                        "candidate_source": general_candidate_source,
                        "domain": general_cluster["domain"],
                        "topic_id": topic_match.get("topic_id") or "",
                        "topic": general_cluster["topic"],
                        "relevant": True,
                        "score": max(float(item["score"]) for item in general_judgments),
                        "reason": "A general fallback rule matched after specific clusters produced no rule.",
                        "cluster_obj": general_cluster["cluster_obj"],
                        "topic_obj": general_cluster["topic_obj"],
                        "rule_groups": general_cluster["rule_groups"],
                        "rule_ids": general_cluster["rule_ids"],
                        "topic_rules": general_cluster["topic_rules"],
                    }
                )
            all_judgments.extend(general_judgments)

        best_by_rule: Dict[str, Dict[str, Any]] = {}
        for judgment in all_judgments:
            rule_id = judgment["rule_id"]
            current = best_by_rule.get(rule_id)
            judgment_rank = (
                float(judgment["score"]),
                float(judgment.get("background_order_score") or 0.0),
            )
            current_rank = (
                float(current["score"]),
                float(current.get("background_order_score") or 0.0),
            ) if current is not None else (-1.0, -1.0)
            if current is None or judgment_rank > current_rank:
                if current is not None:
                    self._trace_reject("rule", self._compact_judgment(current), "duplicate_lower_score")
                best_by_rule[rule_id] = judgment
            else:
                self._trace_reject("rule", self._compact_judgment(judgment), "duplicate_lower_score")
        merged = list(best_by_rule.values())
        merged.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item.get("background_order_score") or 0.0),
                item["domain"],
                item["topic"],
                item["rule_id"],
            )
        )
        if merged:
            topic_primaries: List[Dict[str, Any]] = []
            primary_ids: set[str] = set()
            for judgment in merged:
                topic_key = (judgment["domain"], judgment["topic"])
                if any(
                    (item["domain"], item["topic"]) == topic_key
                    for item in topic_primaries
                ):
                    continue
                topic_primaries.append(judgment)
                primary_ids.add(judgment["rule_id"])
            confirmation_order = topic_primaries + [
                item for item in merged if item["rule_id"] not in primary_ids
            ]
            confirmation_limit = max(
                self.max_selected_rules,
                min(
                    self.MAX_RULE_CONFIRMATION_CANDIDATES,
                    self.max_selected_rules * 3,
                ),
            )
            confirmation_pool = confirmation_order[:confirmation_limit]
            confirmation_omitted = confirmation_order[confirmation_limit:]
            self._checkpoint_partial_items("rule", confirmation_pool)
            merged = self._confirm_rule_judgments(
                sample=sample,
                background_analysis=background_analysis,
                judgments=confirmation_pool,
            )
            for omitted in confirmation_omitted:
                self._trace_reject(
                    "rule_confirmation",
                    self._compact_judgment(omitted),
                    "confirmation_candidate_limit",
                )
            merged.sort(
                key=lambda item: (
                    -float(item["score"]),
                    -float(item.get("background_order_score") or 0.0),
                    item["domain"],
                    item["topic"],
                    item["rule_id"],
                )
            )

        # A provisional rule is not a successful navigation result until the
        # background-only gate confirms it. If every provisional rule for a
        # topic is rejected, perform one bounded in-tree recovery: one
        # untried primary cluster followed by one deferred bucket. This avoids
        # the former dead end where a false provisional hit prevented access
        # to a relevant coarse leaf.
        confirmed_topic_keys = {
            (item["domain"], item["topic"])
            for item in merged
        }
        provisional_topic_keys = {
            (item["domain"], item["topic"])
            for item in all_judgments
        }
        for topic_match in selected_topic_list:
            topic_key = (topic_match["domain"], topic_match["topic"])
            if (
                topic_key in confirmed_topic_keys
                or topic_key not in provisional_topic_keys
            ):
                continue
            all_clusters = self._build_cluster_candidates(topic_match)
            attempted_ids = attempted_rule_ids.setdefault(topic_key, set())

            def has_untried_rules(cluster: Dict[str, Any]) -> bool:
                return any(
                    norm_text(rule_id) not in attempted_ids
                    for rule_id in (cluster.get("rule_ids") or [])
                    if norm_text(rule_id)
                )

            recovery_groups = (
                (
                    "primary",
                    "cluster_confirmation_backtrack",
                    "confirmation_primary_backtrack",
                ),
                (
                    "deferred_bucket",
                    "cluster_confirmation_fallback",
                    "confirmation_deferred_fallback",
                ),
            )
            for navigation_role, stage, candidate_source in recovery_groups:
                candidates = [
                    cluster
                    for cluster in all_clusters
                    if cluster["navigation_role"] == navigation_role
                    and has_untried_rules(cluster)
                ]
                recovered_cluster = self._select_one_navigation_cluster(
                    sample=sample,
                    topic_match=topic_match,
                    candidates=candidates,
                    background_analysis=background_analysis,
                    stage=stage,
                    candidate_source=candidate_source,
                )
                if recovered_cluster is None:
                    continue
                navigation_clusters.append(recovered_cluster)
                judgment_start = len(all_judgments)
                evaluate_cluster(recovered_cluster)
                recovery_judgments = all_judgments[judgment_start:]
                if not recovery_judgments:
                    continue
                recovery_pool = recovery_judgments[:confirmation_limit]
                self._checkpoint_partial_items(
                    "rule",
                    [*merged, *recovery_pool],
                )
                recovered_rules = self._confirm_rule_judgments(
                    sample=sample,
                    background_analysis=background_analysis,
                    judgments=recovery_pool,
                )
                if recovered_rules:
                    merged.extend(recovered_rules)
                    confirmed_topic_keys.add(topic_key)
                    break
            merged.sort(
                key=lambda item: (
                    -float(item["score"]),
                    -float(item.get("background_order_score") or 0.0),
                    item["domain"],
                    item["topic"],
                    item["rule_id"],
                )
            )
        clustered_topic_keys = {(item["domain"], item["topic"]) for item in cluster_list}
        clusterless_kept: Dict[tuple[str, str], int] = {}
        capped: List[Dict[str, Any]] = []
        for judgment in merged:
            topic_key = (judgment["domain"], judgment["topic"])
            if judgment["cluster_id"] or topic_key in clustered_topic_keys:
                capped.append(judgment)
                continue
            kept = clusterless_kept.get(topic_key, 0)
            if kept >= 2:
                self._trace_reject("rule", self._compact_judgment(judgment), "clusterless_topic_limit")
                continue
            clusterless_kept[topic_key] = kept + 1
            capped.append(judgment)
        topic_order = {
            (item["domain"], item["topic"]): index
            for index, item in enumerate(selected_topic_list)
        }
        grouped_rules: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for judgment in capped:
            grouped_rules.setdefault((judgment["domain"], judgment["topic"]), []).append(judgment)
        primary_rules = [
            items[0]
            for key, items in sorted(
                grouped_rules.items(),
                key=lambda pair: topic_order.get(pair[0], len(topic_order)),
            )
            if items
        ]
        primary_ids = {id(item) for item in primary_rules}
        fair_order = primary_rules + [item for item in capped if id(item) not in primary_ids]
        for omitted in fair_order[self.max_selected_rules :]:
            self._trace_reject("rule", self._compact_judgment(omitted), "global_selection_limit")
        globally_capped = fair_order[: self.max_selected_rules]
        for judgment in globally_capped:
            self._trace_accept("rule", self._compact_judgment(judgment))
        if not globally_capped:
            reason = (
                "no_rule_candidates"
                if not self._stage_trace("rule")["candidate_count"]
                else "model_selected_no_rules_or_all_items_rejected"
            )
            self._set_empty_reason("rule", reason)
        unique_navigation_clusters: List[Dict[str, Any]] = []
        seen_navigation_clusters: set[tuple[str, str, str]] = set()
        for item in navigation_clusters:
            key = (
                norm_text(item.get("domain") or ""),
                norm_text(item.get("topic_id") or item.get("topic") or ""),
                norm_text(item.get("cluster_id") or ""),
            )
            if not key[2] or key in seen_navigation_clusters:
                continue
            seen_navigation_clusters.add(key)
            unique_navigation_clusters.append(item)
        return {
            "rule_judgments": globally_capped,
            "selected_rules": globally_capped,
            "navigation_clusters": unique_navigation_clusters,
        }

    def _tree_result(
        self,
        *,
        domain_result: Dict[str, Any],
        topic_result: Dict[str, Any] | None = None,
        cluster_result: Dict[str, Any] | None = None,
        rule_result: Dict[str, Any] | None = None,
        terminal_stage: str,
        empty_reason: str = "",
    ) -> Dict[str, Any]:
        self._update_partial_result(
            domain_result=domain_result,
            topic_result=topic_result,
            cluster_result=cluster_result,
            rule_result=rule_result,
        )
        self.last_trace["background_analysis"] = copy.deepcopy(self._background_analysis)
        self.last_trace["terminal_stage"] = norm_text(terminal_stage)
        self.last_trace["empty_reason"] = norm_text(empty_reason)
        self.last_trace["status"] = "empty" if empty_reason else "complete"
        result = copy.deepcopy(self.last_partial_result)
        result["terminal_stage"] = norm_text(terminal_stage)
        result["empty_reason"] = norm_text(empty_reason)
        result["navigation_trace"] = copy.deepcopy(self.last_trace)
        return result

    @staticmethod
    def _trace_candidate_identity(stage: str, item: Any) -> tuple[str, ...]:
        if not isinstance(item, dict):
            return ()
        if stage == "domain":
            value = norm_text(item.get("domain_id") or item.get("domain") or "")
            return (value,) if value else ()
        if stage in {"topic", "topic_shortlist"}:
            value = norm_text(item.get("topic_id") or item.get("topic") or "")
            return (value,) if value else ()
        if stage in {"cluster", "cluster_backtrack", "cluster_fallback"}:
            cluster_id = norm_text(item.get("cluster_id") or "")
            topic_id = norm_text(item.get("topic_id") or item.get("topic") or "")
            return (topic_id, cluster_id) if cluster_id else ()
        if stage == "rule":
            rule_id = norm_text(item.get("rule_id") or "")
            request_index = str(item.get("request_index") or "")
            return (request_index, rule_id) if rule_id else ()
        return ()

    def _close_failed_stage_candidates(self, stage: str) -> None:
        trace = self._stage_trace(stage)
        attempts = [item for item in (trace.get("api_attempts") or []) if isinstance(item, dict)]
        failed_attempts = [item for item in attempts if norm_text(item.get("error") or "")]
        terminal_attempt = failed_attempts[-1] if failed_attempts else (attempts[-1] if attempts else {})
        failed_request_index = int(terminal_attempt.get("request_index") or 0)
        accounted: set[tuple[str, ...]] = set()
        for key in ("accepted", "not_selected"):
            for item in trace.get(key, []) or []:
                identity = self._trace_candidate_identity(stage, item)
                if identity:
                    accounted.add(identity)
        for record in trace.get("rejected", []) or []:
            if not isinstance(record, dict):
                continue
            item = record.get("item")
            if isinstance(item, dict):
                merged = {**record, **item}
                identity = self._trace_candidate_identity(stage, merged)
                if identity:
                    accounted.add(identity)
        for candidate in trace.get("candidates", []) or []:
            candidate_request_index = int(candidate.get("request_index") or 0)
            if (
                failed_request_index
                and candidate_request_index
                and candidate_request_index != failed_request_index
            ):
                continue
            identity = self._trace_candidate_identity(stage, candidate)
            if not identity or identity in accounted:
                continue
            trace["not_selected"].append(
                {
                    **copy.deepcopy(candidate),
                    "reason": "stage_failed_before_classification",
                    "failed_stage": stage,
                }
            )
            accounted.add(identity)

    def _raise_stage_error(self, stage: str, exc: Exception) -> None:
        failure_stage = self._active_stage or stage
        self._close_failed_stage_candidates(failure_stage)
        self.last_trace["terminal_stage"] = failure_stage
        self.last_trace["empty_reason"] = "selection_error"
        self.last_trace["status"] = "failed"
        self.last_trace["error"] = f"{type(exc).__name__}: {exc}"
        raise SemanticSelectionError(
            failure_stage,
            exc,
            trace=self.last_trace,
            partial_result=self.last_partial_result,
        ) from exc

    def select_tree_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
        self._reset_trace()
        self._trace_run_active = True
        domain_result: Dict[str, Any] = {"domain_judgments": [], "selected_domains": []}
        topic_result: Dict[str, Any] = {"topic_judgments": [], "selected_topics": []}
        cluster_result: Dict[str, Any] = {"cluster_judgments": [], "selected_clusters": []}
        try:
            try:
                domain_result = self.select_domains_semantically(sample, catalog)
                self._update_partial_result(domain_result=domain_result)
            except Exception as exc:
                self._raise_stage_error("domain", exc)
            if not domain_result["selected_domains"]:
                empty_reason = self._stage_trace("domain")["empty_reason"] or "no_selected_domains"
                return self._tree_result(
                    domain_result=domain_result,
                    terminal_stage="domain",
                    empty_reason=empty_reason,
                )
            try:
                topic_result = self.select_topics_semantically(
                    sample,
                    catalog,
                    domain_result["selected_domains"],
                    domain_result.get("background_analysis"),
                )
                self._update_partial_result(domain_result=domain_result, topic_result=topic_result)
            except Exception as exc:
                self._raise_stage_error("topic", exc)
            if not topic_result["selected_topics"]:
                empty_reason = self._stage_trace("topic")["empty_reason"] or "no_selected_topics"
                return self._tree_result(
                    domain_result=domain_result,
                    topic_result=topic_result,
                    terminal_stage="topic",
                    empty_reason=empty_reason,
                )
            try:
                cluster_result = self.select_clusters_semantically(
                    sample,
                    topic_result["selected_topics"],
                    domain_result.get("background_analysis"),
                )
                self._update_partial_result(
                    domain_result=domain_result,
                    topic_result=topic_result,
                    cluster_result=cluster_result,
                )
            except Exception as exc:
                self._raise_stage_error("cluster", exc)
            try:
                rule_result = self.select_rules_semantically(
                    sample,
                    topic_result["selected_topics"],
                    cluster_result["selected_clusters"],
                    domain_result.get("background_analysis"),
                )
            except Exception as exc:
                self._raise_stage_error("rule", exc)
            existing_cluster_keys = {
                (
                    norm_text(item.get("domain") or ""),
                    norm_text(item.get("topic_id") or item.get("topic") or ""),
                    norm_text(item.get("cluster_id") or ""),
                )
                for item in cluster_result.get("selected_clusters", [])
                if isinstance(item, dict)
            }
            for item in rule_result.get("navigation_clusters", []) or []:
                if not isinstance(item, dict):
                    continue
                key = (
                    norm_text(item.get("domain") or ""),
                    norm_text(item.get("topic_id") or item.get("topic") or ""),
                    norm_text(item.get("cluster_id") or ""),
                )
                if not key[2] or key in existing_cluster_keys:
                    continue
                existing_cluster_keys.add(key)
                cluster_result.setdefault("cluster_judgments", []).append(item)
                cluster_result.setdefault("selected_clusters", []).append(item)
            empty_reason = ""
            if not rule_result["selected_rules"]:
                empty_reason = self._stage_trace("rule")["empty_reason"] or "no_selected_rules"
            return self._tree_result(
                domain_result=domain_result,
                topic_result=topic_result,
                cluster_result=cluster_result,
                rule_result=rule_result,
                terminal_stage="rule",
                empty_reason=empty_reason,
            )
        finally:
            self._trace_run_active = False
            self._active_stage = ""
