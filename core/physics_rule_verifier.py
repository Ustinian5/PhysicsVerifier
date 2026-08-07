"""层次化规则检查主流程：规则匹配 → 语义检查 → 符号核查 → 结果合并。

实现类 `PhysicsRuleVerifier` 串联 `core/rule_catalog_retrieval.py`（候选主题/规则检索）、
`core/semantic_rule_checker.py`（LLM+SRD 语义检查）、`rules/symbolic_checks.py` 与
`symbolic/`（符号执行与目录）。"""
import json
import datetime
import inspect
import os
import re
import math
import time
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from pathlib import Path

from core.rule_catalog_retrieval import (
    apply_topic_symbol_overlap_boost,
    build_signal_document_frequency,
    build_topic_candidates,
    build_unified_topic_retrieval_text,
    extract_prediction_symbol_set,
    norm_text,
    ordered_unique,
    rule_topic_context,
    rule_sort_key,
    select_rules_with_topic_priority,
    score_rule_candidate,
    score_topic_candidate,
    topic_rule_leaves,
    topic_sort_key,
)
from core.semantic_rule_checker import SemanticRuleChecker
from core.unified_semantic_matcher import UnifiedSemanticMatcher
from symbolic.experience_code_engine import ExperienceCodeEngine

class PhysicsRuleVerifier:
    UNIFIED_V2_MIN_DIAGNOSTIC_RULE_SCORE = 4.0
    STRICT_RELEASE_INCONCLUSIVE_SCORE_BONUS = 2.0
    # Per-sample / per-paragraph caps default to 0 (disabled). They can be
    # enabled via CLI when over-diagnosis appears to dominate precision losses.
    DEFAULT_MAX_DIAGNOSTICS_PER_SAMPLE = 0
    DEFAULT_MAX_DIAGNOSTICS_PER_PARAGRAPH = 0
    # Quote-level required-symbol overlap defaults to 0 so that the historically
    # noisy "missing required symbol" signal does not cause unintended recall
    # regressions; users can opt-in to a stricter threshold via CLI.
    DEFAULT_QUOTE_REQUIRED_SYMBOL_RATIO = 0.0

    def __init__(
        self,
        rules_catalog_path: str = "catalogs/rules_catalog_top_down.json",
        llm_model: str = "qwen3-30b-a3b",
        log_dir: str = "logs",
        results_dir: str = "results",
        enable_symbolic_check: bool = True,
        enable_llm_cache: bool = True,
        unified_rules_path: Optional[str] = None,
        experience_code_manifest_path: str = "results/experience_symbolic_program_manifest_v2_unified.json",
        experience_code_module: str = "symbolic.generated_experience_checks_v2_unified",
        symbolic_topic_check_limit: int = 40,
        precision_mode: str = "strict",
        min_diagnostic_rule_score: Optional[float] = None,
        max_diagnostics_per_sample: Optional[int] = None,
        max_diagnostics_per_paragraph: Optional[int] = None,
        quote_required_symbol_ratio: Optional[float] = None,
        unified_rule_top_n: Optional[int] = None,
        unified_retrieval_mode: Optional[str] = None,
        semantic_min_publish_score: Optional[float] = None,
        semantic_matcher: Optional[Any] = None,
        semantic_json_attempts: Optional[int] = None,
        semantic_output_adapter: Optional[str] = None,
        # Legacy kwargs (accepted for backward compatibility, ignored).
        enable_agentic_postcheck: Optional[bool] = None,
        agentic_max_checks_per_sample: Optional[int] = None,
        enable_experience_pipeline: Optional[bool] = None,
        experience_rules_path: Optional[str] = None,
    ):
        # Legacy kwargs are accepted but ignored: the primitive+spec / agentic
        # LLM paths and the keyword-trigger experience pipeline have been
        # removed. The single symbolic verification path now is the
        # generated experience-code engine, which is on by default.
        _ = (
            enable_agentic_postcheck,
            agentic_max_checks_per_sample,
            enable_experience_pipeline,
            experience_rules_path,
        )
        self.llm_model = llm_model
        self.enable_llm_cache = bool(enable_llm_cache)
        self.precision_mode = str(precision_mode or "strict").strip().lower()
        if self.precision_mode not in {"strict", "balanced", "score_only"}:
            self.precision_mode = "strict"
        self.min_diagnostic_rule_score = (
            float(min_diagnostic_rule_score)
            if min_diagnostic_rule_score is not None
            else float(self.UNIFIED_V2_MIN_DIAGNOSTIC_RULE_SCORE)
        )
        self.max_diagnostics_per_sample = int(
            max_diagnostics_per_sample
            if max_diagnostics_per_sample is not None
            else self.DEFAULT_MAX_DIAGNOSTICS_PER_SAMPLE
        )
        self.max_diagnostics_per_paragraph = int(
            max_diagnostics_per_paragraph
            if max_diagnostics_per_paragraph is not None
            else self.DEFAULT_MAX_DIAGNOSTICS_PER_PARAGRAPH
        )
        self.quote_required_symbol_ratio = float(
            quote_required_symbol_ratio
            if quote_required_symbol_ratio is not None
            else self.DEFAULT_QUOTE_REQUIRED_SYMBOL_RATIO
        )
        _env_top = str(os.getenv("PHYSICSVERIFIER_UNIFIED_RULE_TOP_N", "")).strip()
        if unified_rule_top_n is not None:
            self.unified_rule_top_n = max(1, int(unified_rule_top_n))
        elif _env_top.isdigit():
            self.unified_rule_top_n = max(1, int(_env_top))
        else:
            self.unified_rule_top_n = 6
        retrieval_mode = str(
            unified_retrieval_mode
            or os.getenv("PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE")
            or "semantic"
        ).strip().lower()
        if retrieval_mode not in {"semantic", "lexical"}:
            raise ValueError("unified_retrieval_mode must be 'semantic' or 'lexical'")
        self.unified_retrieval_mode = retrieval_mode
        _env_semantic_min = str(os.getenv("PHYSICSVERIFIER_SEMANTIC_MIN_PUBLISH_SCORE", "")).strip()
        if semantic_min_publish_score is not None:
            self.semantic_min_publish_score = float(semantic_min_publish_score)
        elif _env_semantic_min:
            self.semantic_min_publish_score = float(_env_semantic_min)
        else:
            self.semantic_min_publish_score = 0.0
        if not 0.0 <= self.semantic_min_publish_score <= 1.0:
            raise ValueError("semantic_min_publish_score must be between 0.0 and 1.0")
        self.semantic_matcher: Optional[Any] = semantic_matcher
        self.semantic_output_adapter = (
            str(semantic_output_adapter).strip().lower()
            if semantic_output_adapter is not None
            else None
        )
        self.semantic_json_attempts: Optional[int] = None
        if semantic_json_attempts is not None:
            self.semantic_json_attempts = int(semantic_json_attempts)
            if self.semantic_json_attempts < 1:
                raise ValueError("semantic_json_attempts must be at least 1")
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # An explicitly requested unified catalog must never silently fall
        # back to the legacy top-down catalog.
        self._unified_mode = False
        if unified_rules_path:
            if not Path(unified_rules_path).is_file():
                raise FileNotFoundError(f"Unified rules catalog does not exist: {unified_rules_path}")
            self.rules_catalog_path = unified_rules_path
            self._unified_mode = True
        else:
            self.rules_catalog_path = rules_catalog_path

        self.catalog = self._load_catalog()
        meta = self.catalog.get("metadata") if isinstance(self.catalog, dict) else {}
        self._retrieval_tuning: Dict[str, Any] = {}
        if isinstance(meta, dict):
            rt = meta.get("retrieval_tuning")
            if isinstance(rt, dict):
                self._retrieval_tuning = dict(rt)
        if str(os.getenv("PHYSICSVERIFIER_TOPIC_SKIP_PREDICTION", "")).strip().lower() in {"1", "true", "yes"}:
            self._retrieval_tuning["topic_include_prediction"] = False
        self._unified_v2_mode = bool(
            self._unified_mode and isinstance(meta, dict) and meta.get("catalog_type") == "unified_rules_v2"
        )
        self.topics = self._flatten_topics()
        self._unified_v2_topic_candidates = build_topic_candidates(self.catalog) if self._unified_v2_mode else []
        self._unified_v2_signal_df = (
            build_signal_document_frequency(self._unified_v2_topic_candidates) if self._unified_v2_topic_candidates else {}
        )
        
        # Initialize the base verifiers
        # We will dynamically update rules for the rule-based verifier
        self.semantic_checker = SemanticRuleChecker(
            llm_model=self.llm_model,
            rule_mode='srd', # We will inject SRDs dynamically
            rule_translations_path="rule_translations.json", # Dummy path, we'll overwrite
            enable_cache=self.enable_llm_cache,
        )
        # Clear initial translations as we will set them per request
        self.semantic_checker.rule_translations = {} 

        # SemanticRuleChecker loads .env during initialization. Construct the
        # tree matcher afterwards so server-side API settings are visible.
        if (
            self._unified_v2_mode
            and self.unified_retrieval_mode == "semantic"
            and self.semantic_matcher is None
        ):
            matcher_kwargs: Dict[str, Any] = {
                "model": str(self.llm_model or ""),
                "max_selected_rules": self.unified_rule_top_n,
            }
            if self.semantic_output_adapter:
                matcher_kwargs["structured_output_adapter"] = self.semantic_output_adapter
            # CLI exposes total attempts, while the matcher counts retries after
            # the initial request. Signature inspection keeps this verifier
            # compatible with older matcher implementations during migration.
            if self.semantic_json_attempts is not None:
                try:
                    matcher_parameters = inspect.signature(UnifiedSemanticMatcher).parameters
                except (TypeError, ValueError):
                    matcher_parameters = {}
                if "json_retries" in matcher_parameters:
                    matcher_kwargs["json_retries"] = self.semantic_json_attempts - 1
            self.semantic_matcher = UnifiedSemanticMatcher(**matcher_kwargs)

        if self.semantic_matcher is not None and self.semantic_json_attempts is not None:
            if hasattr(self.semantic_matcher, "json_retries"):
                retry_count = self.semantic_json_attempts - 1
                matcher_retry_cap = getattr(self.semantic_matcher, "MAX_JSON_RETRIES", None)
                if isinstance(matcher_retry_cap, int):
                    retry_count = min(retry_count, max(0, matcher_retry_cap))
                setattr(self.semantic_matcher, "json_retries", retry_count)
        
        self.enable_symbolic_check = bool(enable_symbolic_check)
        self.experience_code_manifest_path = str(experience_code_manifest_path)
        self.experience_code_module = str(experience_code_module)
        self.symbolic_topic_check_limit = max(0, int(symbolic_topic_check_limit))
        self.experience_code_engine = ExperienceCodeEngine(
            manifest_path=self.experience_code_manifest_path,
            module_name=self.experience_code_module,
        )

        self.error_experiences = []

    def _filter_low_confidence_unified_diagnostics(
        self,
        diagnostics: List[Dict[str, Any]],
        selected_rule_records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Suppress diagnostics emitted by weakly matched rules.

        Top-ranked topics can admit broad rules with low raw match scores. If an
        LLM still emits a finding for those rules, it is usually a precision risk.
        """
        record_by_rule = {
            str((item.get("rule") or {}).get("id") or ""): item
            for item in selected_rule_records
            if isinstance(item, dict)
        }
        kept: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        min_score = self.min_diagnostic_rule_score
        for d in diagnostics or []:
            if not isinstance(d, dict):
                kept.append(d)
                continue
            rid = str(d.get("rule") or "")
            record = record_by_rule.get(rid) if isinstance(record_by_rule.get(rid), dict) else {}
            if str(record.get("retrieval_strategy") or "") == "semantic_tree_selection":
                # The semantic tree has already made the applicability decision.
                # Final quote/metadata/symbolic gates still run below.
                kept.append(d)
                continue
            rule_score = float(record.get("score") or 0.0)
            if rule_score < min_score:
                suppressed.append(
                    {
                        "reason": "low_rule_match_score",
                        "rule_id": rid,
                        "rule_score": rule_score,
                        "min_rule_score": min_score,
                        "original_diagnostic": d,
                    }
                )
                continue
            kept.append(d)
        return kept, suppressed

    def _quote_required_symbols(self, rule: Dict[str, Any]) -> List[str]:
        """Aggregate symbols a quote should mention for a rule diagnostic to be credible."""
        if not isinstance(rule, dict):
            return []
        out: List[str] = []
        sh = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
        out.extend(str(s) for s in (sh.get("required_symbols") or []) if str(s).strip())
        mf = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
        out.extend(str(s) for s in (mf.get("required_symbols") or []) if str(s).strip())
        out.extend(str(s) for s in (rule.get("required_symbols") or []) if str(s).strip())
        return ordered_unique([s for s in out if s])

    def _quote_symbol_overlap(
        self,
        quote: str,
        required_symbols: List[str],
    ) -> Tuple[List[str], float]:
        if not required_symbols or not quote:
            return [], 0.0
        from rules.symbolic_checks import _normalize_text_for_match, _token_present  # local import to avoid cycles

        text_lower = quote.lower()
        text_norm = _normalize_text_for_match(quote)
        hits = [s for s in required_symbols if _token_present(s, text_lower, text_norm)]
        ratio = len(hits) / max(1, len(required_symbols))
        return hits, ratio

    def _diagnostic_release_gate(
        self,
        diagnostic: Dict[str, Any],
        rule_record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        severity = str(diagnostic.get("severity") or "").strip().lower()
        allowed_severities = {"error", "warning"} if self.precision_mode == "balanced" else {"error"}
        if severity not in allowed_severities:
            reasons.append("severity_not_error")
        if self.semantic_checker._is_negative_or_uncertain_diagnostic(diagnostic):
            reasons.append("negative_or_uncertain_diagnostic")

        evidence = diagnostic.get("evidence") if isinstance(diagnostic.get("evidence"), dict) else {}
        quote = str(evidence.get("quote") or "").strip()
        loc = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}
        if not quote:
            reasons.append("missing_quote")
        if not bool(loc.get("locatable_valid")):
            reasons.append("unlocatable_quote")

        publish_gate = rule_record.get("publish_gate") if isinstance(rule_record, dict) else None
        if isinstance(publish_gate, dict):
            if not bool(publish_gate.get("publishable")):
                reasons.extend(str(r) for r in (publish_gate.get("reasons") or []))
        else:
            reasons.append("missing_rule_publish_gate")

        rule = rule_record.get("rule") if isinstance(rule_record, dict) and isinstance(rule_record.get("rule"), dict) else {}
        precision = self._rule_precision_metadata(rule)
        evidence_requirement_hits = self._match_text_list(precision["evidence_requirements"], quote)
        if precision["evidence_requirements"] and not evidence_requirement_hits:
            reasons.append("missing_required_evidence")
        if self._match_text_list(precision["negative_conditions"], quote):
            reasons.append("quote_hits_negative_condition")

        recon = diagnostic.get("symbolic_reconciliation") if isinstance(diagnostic.get("symbolic_reconciliation"), dict) else {}
        symbolic_status = str(recon.get("status") or "").strip().lower()
        if isinstance(publish_gate, dict) and "min_publish_score" in publish_gate:
            min_score = float(publish_gate.get("min_publish_score") or 0.0)
        else:
            min_score = float(self.min_diagnostic_rule_score)
        score = float((rule_record or {}).get("score") or 0.0)
        topic_rank = int((rule_record or {}).get("topic_rank") or 0)
        semantic_scale = str((publish_gate or {}).get("score_kind") or "") == "semantic_0_1"
        inconclusive_bonus = 0.15 if semantic_scale else self.STRICT_RELEASE_INCONCLUSIVE_SCORE_BONUS
        secondary_topic_bonus = 0.10 if semantic_scale else 1.0
        if symbolic_status in {"supported", "quote_overlap"}:
            # Either the canonical-missing primitive triggered (fail-as-supported)
            # or the diagnostic's quote sits on top of the canonical pattern;
            # both indicate the LLM critique is plausibly grounded.
            pass
        elif symbolic_status == "inconclusive":
            if precision["symbolic_policy"] in {"require_fail", "suppress_on_inconclusive"}:
                reasons.append("symbolic_inconclusive_suppressed")
            elif self.precision_mode == "strict" and score < (min_score + inconclusive_bonus):
                reasons.append("symbolic_inconclusive_below_strict_score")
            elif self.precision_mode == "strict" and topic_rank >= 1 and score < (
                min_score + inconclusive_bonus + secondary_topic_bonus
            ):
                reasons.append("symbolic_inconclusive_secondary_topic_below_score")
        elif precision["symbolic_policy"] == "require_fail" and diagnostic.get("symbolic_cross_checks"):
            reasons.append("symbolic_fail_required")

        # Quote-level required-symbol overlap: if the rule's symbolic hint specifies which
        # symbols a violation should reference, the diagnostic's quote must mention them.
        quote_required_symbols = self._quote_required_symbols(rule)
        quote_symbol_hits: List[str] = []
        quote_symbol_ratio = 0.0
        if quote_required_symbols and quote:
            quote_symbol_hits, quote_symbol_ratio = self._quote_symbol_overlap(quote, quote_required_symbols)
            if (
                self.precision_mode == "strict"
                and len(quote_required_symbols) >= 2
                and symbolic_status != "supported"
                and quote_symbol_ratio < self.quote_required_symbol_ratio
            ):
                reasons.append("quote_missing_required_symbols")

        return {
            "publishable": not reasons,
            "reasons": ordered_unique(reasons),
            "rule_score": score,
            "min_publish_score": min_score,
            "score_kind": str((publish_gate or {}).get("score_kind") or "lexical"),
            "symbolic_status": symbolic_status or "none",
            "evidence_requirement_hits": evidence_requirement_hits,
            "quote_symbol_hits": quote_symbol_hits,
            "quote_symbol_ratio": round(quote_symbol_ratio, 4),
            "quote_symbol_required_count": len(quote_required_symbols),
        }

    def _dedupe_final_diagnostics(self, diagnostics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_location: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for d in diagnostics:
            evidence = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
            quote = str(evidence.get("quote") or "").strip()
            loc = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}
            para = str(loc.get("paragraph_index") or "")
            key = (quote.casefold(), para)
            gate = d.get("release_gate") if isinstance(d.get("release_gate"), dict) else {}
            candidate_score = float(gate.get("rule_score") or 0.0)
            current = best_by_location.get(key)
            if current is None:
                best_by_location[key] = d
                continue
            current_gate = current.get("release_gate") if isinstance(current.get("release_gate"), dict) else {}
            current_score = float(current_gate.get("rule_score") or 0.0)
            if candidate_score > current_score:
                best_by_location[key] = d
        return list(best_by_location.values())

    def _apply_diagnostic_release_gate(
        self,
        diagnostics: List[Dict[str, Any]],
        selected_rule_records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        record_by_rule = {
            str((item.get("rule") or {}).get("id") or item.get("rule_id") or ""): item
            for item in selected_rule_records
            if isinstance(item, dict)
        }
        kept: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        for d in diagnostics or []:
            if not isinstance(d, dict):
                continue
            rid = str(d.get("rule") or "")
            gate = self._diagnostic_release_gate(d, record_by_rule.get(rid))
            enriched = dict(d)
            enriched["release_gate"] = gate
            if gate["publishable"]:
                if rid in record_by_rule:
                    rec = record_by_rule[rid]
                    enriched["rule_match"] = {
                        "score": float(rec.get("score") or 0.0),
                        "score_kind": str(
                            (rec.get("publish_gate") or {}).get("score_kind") or "lexical"
                        ),
                        "semantic_score": rec.get("semantic_score"),
                        "grounding_score": rec.get("grounding_score"),
                        "retrieval_strategy": str(rec.get("retrieval_strategy") or "lexical"),
                        "min_score": float(rec.get("min_score") or 0.0),
                        "topic_gap": float(rec.get("topic_gap") or 0.0),
                        "topic_rank": int(rec.get("topic_rank") or 0),
                        "publish_gate": rec.get("publish_gate") or {},
                    }
                kept.append(enriched)
            else:
                suppressed.append(
                    {
                        "reason": "diagnostic_release_gate",
                        "rule_id": rid,
                        "release_gate": gate,
                        "original_diagnostic": d,
                    }
                )
        deduped = self._dedupe_final_diagnostics(kept)
        deduped_ids = {id(d) for d in deduped}
        for d in kept:
            if id(d) not in deduped_ids:
                suppressed.append(
                    {
                        "reason": "duplicate_location_or_quote",
                        "rule_id": str(d.get("rule") or ""),
                        "original_diagnostic": d,
                    }
                )

        capped, cap_suppressed = self._apply_diagnostic_caps(deduped)
        suppressed.extend(cap_suppressed)
        return capped, suppressed

    def _apply_diagnostic_caps(
        self,
        diagnostics: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Enforce per-paragraph and per-sample caps on published diagnostics.

        Diagnostics are ranked by (symbolic_status priority, rule_score). For
        each paragraph we keep at most ``max_diagnostics_per_paragraph``; the
        global list is then truncated to ``max_diagnostics_per_sample``. This
        eliminates "over-diagnosis" cases where multiple rules emit findings on
        the same passage or where a single sample collects many low-confidence
        signals.
        """

        def _priority(d: Dict[str, Any]) -> Tuple[int, float, float]:
            gate = d.get("release_gate") if isinstance(d.get("release_gate"), dict) else {}
            status = str(gate.get("symbolic_status") or "none").lower()
            status_rank = {"supported": 0, "none": 1, "inconclusive": 2}.get(status, 3)
            score = float(gate.get("rule_score") or 0.0)
            quote_ratio = float(gate.get("quote_symbol_ratio") or 0.0)
            return (status_rank, -score, -quote_ratio)

        if self.max_diagnostics_per_paragraph <= 0 and self.max_diagnostics_per_sample <= 0:
            return list(diagnostics), []

        ordered = sorted(diagnostics, key=_priority)
        per_paragraph_keep: Dict[str, int] = {}
        kept: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        for d in ordered:
            evidence = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
            loc = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}
            para = str(loc.get("paragraph_index") or "<none>")
            paragraph_quota = self.max_diagnostics_per_paragraph
            if paragraph_quota and per_paragraph_keep.get(para, 0) >= paragraph_quota:
                suppressed.append(
                    {
                        "reason": "over_paragraph_cap",
                        "rule_id": str(d.get("rule") or ""),
                        "original_diagnostic": d,
                    }
                )
                continue
            sample_quota = self.max_diagnostics_per_sample
            if sample_quota and len(kept) >= sample_quota:
                suppressed.append(
                    {
                        "reason": "over_sample_cap",
                        "rule_id": str(d.get("rule") or ""),
                        "original_diagnostic": d,
                    }
                )
                continue
            per_paragraph_keep[para] = per_paragraph_keep.get(para, 0) + 1
            kept.append(d)
        return kept, suppressed

    def _normalize_topic_key(self, domain: str, topic: str) -> str:
        d = str(domain or "Unknown").strip().lower()
        t_raw = str(topic or "Unknown").strip()
        # Distilled experience topic may look like "Domain / Topic"; keep the last segment for robust matching.
        t = t_raw.split("/")[-1].strip().lower() if "/" in t_raw else t_raw.lower()
        return f"{d}::{t}"

    def _load_catalog(self) -> Dict:
        with open(self.rules_catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_topics(self) -> List[Dict]:
        topics = []
        for domain in self.catalog.get("domains", []):
            domain_name = domain.get("name")
            for topic in domain.get("topics", []):
                t = topic.copy()
                t["domain"] = domain_name
                topics.append(t)
        return topics

    @staticmethod
    def _ordered_unique(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            item = str(value or "").strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @staticmethod
    def _match_phrase_or_symbol(needle: str, haystack: str) -> bool:
        target = str(needle or "").strip()
        text = str(haystack or "")
        if not target or not text:
            return False
        if len(target) == 1 and re.fullmatch(r"[A-Za-z]", target):
            pat = re.compile(rf"(^|[^A-Za-z0-9_]){re.escape(target)}([^A-Za-z0-9_]|$)", re.I)
            return bool(pat.search(text))
        return target.casefold() in text.casefold()

    @staticmethod
    def _as_text_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, dict):
            out: List[str] = []
            for item in value.values():
                out.extend(PhysicsRuleVerifier._as_text_list(item))
            return ordered_unique(out)
        if isinstance(value, list):
            out = []
            for item in value:
                out.extend(PhysicsRuleVerifier._as_text_list(item))
            return ordered_unique(out)
        text = str(value).strip()
        return [text] if text else []

    def _rule_precision_metadata(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        precision = rule.get("precision") if isinstance(rule.get("precision"), dict) else {}
        publishable_raw = rule.get("publishable", precision.get("publishable", True))
        if isinstance(publishable_raw, str):
            publishable = publishable_raw.strip().lower() not in {"0", "false", "no", "off"}
        else:
            publishable = bool(publishable_raw)
        out = {
            "precision_profile": str(
                rule.get("precision_profile") or precision.get("precision_profile") or precision.get("profile") or "strict"
            ).strip().lower(),
            "publishable": publishable,
            "preconditions": self._as_text_list(rule.get("preconditions") or precision.get("preconditions")),
            "violation_signatures": self._as_text_list(
                rule.get("violation_signatures") or precision.get("violation_signatures")
            ),
            "negative_conditions": self._as_text_list(
                rule.get("negative_conditions") or precision.get("negative_conditions")
            ),
            "evidence_requirements": self._as_text_list(
                rule.get("evidence_requirements") or precision.get("evidence_requirements")
            ),
            "symbolic_policy": str(
                rule.get("symbolic_policy") or precision.get("symbolic_policy") or "suppress_on_pass"
            ).strip().lower(),
        }
        if out["precision_profile"] not in {"strict", "balanced", "recall"}:
            out["precision_profile"] = "strict"
        return out

    def _match_text_list(self, items: List[str], text: str) -> List[str]:
        return [item for item in items if self._match_phrase_or_symbol(item, text)]

    def _build_rule_publish_gate(
        self,
        *,
        rule: Dict[str, Any],
        score_payload: Dict[str, Any],
        topic_ctx: Dict[str, Any],
        topic_rank: int,
        text_for_rule: str,
    ) -> Dict[str, Any]:
        evidence = score_payload.get("evidence") if isinstance(score_payload.get("evidence"), dict) else {}
        precision = self._rule_precision_metadata(rule)
        score = float(score_payload.get("score") or 0.0)
        min_publish_score = max(
            float(self.min_diagnostic_rule_score),
            float(topic_ctx.get("min_score") or 0.0),
        )
        if str(score_payload.get("scope") or "domain") == "meta":
            min_publish_score += 0.5
        if topic_rank > 0:
            min_publish_score += 1.0
        if precision["precision_profile"] == "balanced":
            min_publish_score += 0.5
        elif precision["precision_profile"] == "recall":
            min_publish_score += 2.0

        precondition_hits = self._match_text_list(precision["preconditions"], text_for_rule)
        violation_hits = self._match_text_list(precision["violation_signatures"], text_for_rule)
        negative_hits = self._match_text_list(precision["negative_conditions"], text_for_rule)
        evidence_requirement_hits = self._match_text_list(precision["evidence_requirements"], text_for_rule)

        trigger_hits = list(evidence.get("trigger_hits") or [])
        object_hits = list(evidence.get("object_hits") or [])
        symbol_hits = list(evidence.get("required_symbol_hits") or [])
        precondition_hits = ordered_unique(precondition_hits + list(evidence.get("precondition_hits") or []))
        violation_hits = ordered_unique(violation_hits + list(evidence.get("violation_signature_hits") or []))
        negative_hits = ordered_unique(negative_hits + list(evidence.get("negative_keyword_hits") or []))
        evidence_requirement_hits = ordered_unique(
            evidence_requirement_hits + list(evidence.get("evidence_requirement_hits") or [])
        )
        llm_phrase_hits = list(evidence.get("llm_phrase_hits") or [])
        llm_term_hits = list(evidence.get("llm_term_hits") or [])
        strong_anchor_hits = ordered_unique(trigger_hits + symbol_hits + precondition_hits)
        llm_only = bool(evidence.get("llm_only_soft_hit"))
        if not llm_only:
            llm_only = bool(llm_phrase_hits or llm_term_hits) and not bool(trigger_hits or object_hits or symbol_hits)

        reasons: List[str] = []
        if self.precision_mode == "score_only":
            if score < self.min_diagnostic_rule_score:
                reasons.append("below_score_only_threshold")
            return {
                "publishable": not reasons,
                "reasons": reasons,
                "score": score,
                "min_publish_score": round(float(self.min_diagnostic_rule_score), 4),
                "precision_profile": precision["precision_profile"],
                "symbolic_policy": precision["symbolic_policy"],
                "strong_anchor_hits": strong_anchor_hits,
                "precondition_hits": precondition_hits,
                "violation_signature_hits": violation_hits,
                "negative_condition_hits": negative_hits,
                "evidence_requirement_hits": evidence_requirement_hits,
                "llm_hint_only": llm_only,
            }
        if precision["publishable"] is False:
            reasons.append("rule_marked_unpublishable")
        if precision["precision_profile"] == "recall":
            reasons.append("recall_profile_not_publishable_in_strict_mode")
        if bool(evidence.get("generic_signal_only")):
            reasons.append("generic_signal_only")
        if llm_only:
            reasons.append("llm_hint_only")
        if precision["preconditions"] and not precondition_hits:
            reasons.append("missing_precondition_evidence")
        if precision["violation_signatures"] and not violation_hits:
            reasons.append("missing_violation_signature")
        if precision["evidence_requirements"] and not evidence_requirement_hits:
            reasons.append("missing_evidence_requirement")
        if negative_hits:
            reasons.append("negative_condition_hit")
        if score < min_publish_score:
            reasons.append("below_dynamic_publish_score")

        return {
            "publishable": not reasons,
            "reasons": reasons,
            "score": score,
            "min_publish_score": round(min_publish_score, 4),
            "precision_profile": precision["precision_profile"],
            "symbolic_policy": precision["symbolic_policy"],
            "strong_anchor_hits": strong_anchor_hits,
            "precondition_hits": precondition_hits,
            "violation_signature_hits": violation_hits,
            "negative_condition_hits": negative_hits,
            "evidence_requirement_hits": evidence_requirement_hits,
            "llm_hint_only": llm_only,
        }

    def _prepare_unified_v2_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(rule)
        rid = str(rule.get("rule_id") or rule.get("id") or "").strip()
        prepared["id"] = rid
        if not prepared.get("description"):
            parts = []
            if rule.get("trigger"):
                parts.append(f"Trigger: {rule.get('trigger')}")
            if rule.get("check_logic"):
                parts.append(f"Check Logic: {rule.get('check_logic')}")
            prepared["description"] = "\n".join(parts)
        return prepared

    def _score_unified_v2_topic(self, topic: Dict[str, Any], text_for_topic: str) -> Dict[str, Any]:
        candidate = {
            "domain": str(topic.get("domain") or "Unknown"),
            "topic": str(topic.get("name") or "Unknown"),
            "topic_obj": topic,
            "aliases": [],
            "scene_keywords": [],
            "topic_keywords": [],
            "knowledge_keywords": [],
            "required_symbols": [],
        }
        for item in self._unified_v2_topic_candidates:
            if item.get("topic_obj") is topic:
                candidate = item
                break

        payload = score_topic_candidate(candidate, text_for_topic, signal_df=self._unified_v2_signal_df)
        return {
            "domain": payload["domain"],
            "name": payload["topic"],
            "score": payload["score"],
            "evidence": payload["evidence"],
            "topic": payload["topic_obj"],
        }

    def _prediction_symbol_set(self, sample: Dict[str, Any]) -> Set[str]:
        return extract_prediction_symbol_set(str(sample.get("prediction") or ""))

    def _build_semantic_rule_publish_gate(
        self,
        *,
        rule: Dict[str, Any],
        semantic_score: float,
    ) -> Dict[str, Any]:
        """Keep semantic applicability separate from legacy lexical thresholds."""
        precision = self._rule_precision_metadata(rule)
        reasons: List[str] = []
        if float(semantic_score) < self.semantic_min_publish_score:
            reasons.append("below_semantic_publish_score")
        if self.precision_mode != "score_only":
            if precision["publishable"] is False:
                reasons.append("rule_marked_unpublishable")
            if precision["precision_profile"] == "recall":
                reasons.append("recall_profile_not_publishable_in_strict_mode")
        return {
            "publishable": not reasons,
            "reasons": reasons,
            "score": round(float(semantic_score), 4),
            "semantic_score": round(float(semantic_score), 4),
            "score_kind": "semantic_0_1",
            "min_publish_score": round(float(self.semantic_min_publish_score), 4),
            "precision_profile": precision["precision_profile"],
            "symbolic_policy": precision["symbolic_policy"],
            "selection_strategy": "semantic_tree_selection",
            "strong_anchor_hits": [],
            "precondition_hits": [],
            "violation_signature_hits": [],
            "negative_condition_hits": [],
            "evidence_requirement_hits": [],
            "llm_hint_only": False,
        }

    @staticmethod
    def _partial_semantic_result_from_trace(trace: Any) -> Dict[str, Any]:
        """Recover completed tree levels from a matcher's failure trace.

        New matchers expose a navigation trace with one ``accepted`` list per
        stage. During a rolling upgrade, some versions may instead attach a
        partial tree result directly. This adapter accepts both shapes so an
        error at Cluster/Rule does not erase successful Domain/Topic work.
        """
        if not isinstance(trace, dict):
            return {}

        result: Dict[str, Any] = {}
        for key in ("partial_result", "tree_result", "result"):
            candidate = trace.get(key)
            if isinstance(candidate, dict):
                result.update(candidate)
                break

        tree_keys = {
            "domain_judgments",
            "topic_judgments",
            "cluster_judgments",
            "rule_judgments",
            "selected_domains",
            "selected_topics",
            "selected_clusters",
            "selected_rules",
        }
        for key in tree_keys:
            if key in trace and key not in result:
                result[key] = trace.get(key)

        stages = trace.get("stages") if isinstance(trace.get("stages"), dict) else {}

        def _accepted(stage_name: str) -> List[Any]:
            stage = stages.get(stage_name) if isinstance(stages.get(stage_name), dict) else {}
            accepted = stage.get("accepted")
            return list(accepted) if isinstance(accepted, list) else []

        domain_items = _accepted("domain")
        topic_items = _accepted("topic")
        cluster_items = _accepted("cluster")
        rule_items = _accepted("rule")

        if "domain_judgments" not in result:
            result["domain_judgments"] = [item for item in domain_items if isinstance(item, dict)]
        if "selected_domains" not in result:
            result["selected_domains"] = [
                str(item.get("domain") or item.get("name") or "")
                if isinstance(item, dict)
                else str(item or "")
                for item in domain_items
                if (isinstance(item, dict) and (item.get("domain") or item.get("name")))
                or (not isinstance(item, dict) and str(item or "").strip())
            ]
        result.setdefault("topic_judgments", [item for item in topic_items if isinstance(item, dict)])
        result.setdefault("selected_topics", [item for item in topic_items if isinstance(item, dict)])
        result.setdefault("cluster_judgments", [item for item in cluster_items if isinstance(item, dict)])
        result.setdefault("selected_clusters", [item for item in cluster_items if isinstance(item, dict)])
        result.setdefault("rule_judgments", [item for item in rule_items if isinstance(item, dict)])
        result.setdefault("selected_rules", [item for item in rule_items if isinstance(item, dict)])

        navigation_trace = trace.get("navigation_trace")
        result.setdefault(
            "navigation_trace",
            navigation_trace if isinstance(navigation_trace, dict) else dict(trace),
        )
        result.setdefault("background_analysis", trace.get("background_analysis") or {})
        result.setdefault("terminal_stage", str(trace.get("terminal_stage") or ""))
        result.setdefault("empty_reason", str(trace.get("empty_reason") or ""))
        result.setdefault("input_policy", str(trace.get("input_policy") or ""))
        return result

    def _retrieve_unified_v2_semantic_tree(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run the authoritative API tree and adapt its trace to the current verifier."""
        input_policy = UnifiedSemanticMatcher.INPUT_POLICY
        empty: Dict[str, Any] = {
            "selection_strategy": "semantic_unavailable",
            "retrieval_score_kind": "semantic_0_1",
            "semantic_selection_error": "",
            "semantic_failed_stage": "",
            "semantic_input_policy": input_policy,
            "background_analysis": {},
            "navigation_trace": {},
            "terminal_stage": "",
            "empty_reason": "",
            "retrieved_domains": [],
            "topic_matches": [],
            "retrieved_topics": [],
            "retrieved_clusters": [],
            "selected_rule_records": [],
            "retrieved_rules": [],
        }
        matcher = self.semantic_matcher
        if matcher is None or not bool(getattr(matcher, "available", False)):
            empty["semantic_selection_error"] = "Semantic matcher is not available."
            empty["semantic_failed_stage"] = "initialization"
            empty["terminal_stage"] = "initialization"
            empty["empty_reason"] = "semantic_matcher_unavailable"
            return empty

        semantic_selection_error = ""
        semantic_failed_stage = ""
        matcher_sample = {
            key: sample.get(key)
            for key in ("id", "question", "context", "prediction")
            if key in sample
        }
        try:
            semantic_result = matcher.select_tree_semantically(matcher_sample, self.catalog)
        except Exception as exc:
            semantic_selection_error = f"{type(exc).__name__}: {exc}"
            semantic_failed_stage = str(getattr(exc, "stage", "tree") or "tree")
            failure_trace = getattr(exc, "trace", None)
            if not isinstance(failure_trace, dict) or not failure_trace:
                failure_trace = getattr(matcher, "last_trace", None)
            partial_result = getattr(exc, "partial_result", None)
            if not isinstance(partial_result, dict) or not partial_result:
                partial_result = getattr(matcher, "last_partial_result", None)
            semantic_result = self._partial_semantic_result_from_trace(failure_trace)
            if isinstance(partial_result, dict):
                # Use the full partial result for adaptation (it still contains
                # topic_obj/rule_obj), while leaving navigation_trace compact.
                for key, value in partial_result.items():
                    if key != "navigation_trace":
                        semantic_result[key] = value

        if not isinstance(semantic_result, dict):
            semantic_selection_error = "Semantic matcher returned a non-object result."
            semantic_failed_stage = "tree"
            semantic_result = {}

        input_policy = str(semantic_result.get("input_policy") or input_policy)
        domain_judgments = [
            item for item in (semantic_result.get("domain_judgments") or []) if isinstance(item, dict)
        ]
        retrieved_domains = [
            {
                "domain_id": str(item.get("domain_id") or ""),
                "domain": str(item.get("domain") or "Unknown"),
                "score": float(item.get("score") or 0.0),
                "score_kind": "semantic_0_1",
                "reason": str(item.get("reason") or ""),
            }
            for item in domain_judgments
        ]

        selected_topics = [
            item for item in (semantic_result.get("selected_topics") or []) if isinstance(item, dict)
        ]
        topic_matches: List[Dict[str, Any]] = []
        for item in selected_topics:
            topic_obj = item.get("topic_obj") if isinstance(item.get("topic_obj"), dict) else {}
            topic_obj = dict(topic_obj)
            topic_obj.setdefault("domain", str(item.get("domain") or "Unknown"))
            topic_matches.append(
                {
                    "domain": str(item.get("domain") or "Unknown"),
                    "topic_id": str(item.get("topic_id") or topic_obj.get("id") or ""),
                    "name": str(item.get("topic") or "Unknown"),
                    "score": float(item.get("score") or 0.0),
                    "evidence": {
                        "reason": str(item.get("reason") or ""),
                        "input_policy": input_policy,
                    },
                    "topic": topic_obj,
                }
            )
        retrieved_topics = [
            {
                "domain": item["domain"],
                "topic_id": item["topic_id"],
                "topic": item["name"],
                "score": float(item["score"]),
                "score_kind": "semantic_0_1",
                "evidence": item["evidence"],
            }
            for item in topic_matches
        ]

        retrieved_clusters = [
            {
                "domain": str(item.get("domain") or "Unknown"),
                "topic_id": str(item.get("topic_id") or ""),
                "topic": str(item.get("topic") or "Unknown"),
                "cluster_id": str(item.get("cluster_id") or ""),
                "cluster": str(item.get("cluster") or ""),
                "score": float(item.get("score") or 0.0),
                "score_kind": "semantic_0_1",
                "reason": str(item.get("reason") or ""),
                "navigation_role": str(item.get("navigation_role") or "primary"),
                "candidate_source": str(item.get("candidate_source") or ""),
            }
            for item in (semantic_result.get("selected_clusters") or [])
            if isinstance(item, dict)
        ]

        topic_index = {
            (str(item["domain"]), str(item["name"])): item
            for item in topic_matches
        }
        topic_rank = {key: rank for rank, key in enumerate(topic_index)}
        top_topic_score = float(topic_matches[0]["score"] or 0.0) if topic_matches else 0.0
        text_for_grounding = "\n".join(
            [
                str(sample.get("question") or ""),
                str(sample.get("context") or ""),
                str(sample.get("prediction") or ""),
            ]
        )

        selected_rule_records: List[Dict[str, Any]] = []
        for item in list(semantic_result.get("selected_rules") or [])[: self.unified_rule_top_n]:
            if not isinstance(item, dict):
                continue
            raw_rule = item.get("rule_obj") if isinstance(item.get("rule_obj"), dict) else None
            if not raw_rule:
                continue
            rule = self._prepare_unified_v2_rule(raw_rule)
            semantic_score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
            grounding = self._score_unified_v2_rule(rule, text_for_grounding)
            key = (str(item.get("domain") or "Unknown"), str(item.get("topic") or "Unknown"))
            matched_topic = topic_index.get(key) or {}
            topic_obj = matched_topic.get("topic") if isinstance(matched_topic.get("topic"), dict) else {}
            rank = int(topic_rank.get(key, 0))
            matched_topic_score = float(matched_topic.get("score") or 0.0)
            semantic_reason = str(item.get("reason") or "")
            evidence = {
                "semantic_reason": semantic_reason,
                "semantic_score": semantic_score,
                "grounding_score": float(grounding.get("score") or 0.0),
                "grounding_evidence": grounding.get("evidence") or {},
                "input_policy": input_policy,
            }
            selected_rule_records.append(
                {
                    "domain": key[0],
                    "topic_id": str(item.get("topic_id") or topic_obj.get("id") or ""),
                    "topic_name": key[1],
                    "topic": topic_obj,
                    "topic_rank": rank,
                    "cluster_id": str(item.get("cluster_id") or ""),
                    "cluster": str(item.get("cluster") or ""),
                    "rule": rule,
                    "score": semantic_score,
                    "score_kind": "semantic_0_1",
                    "semantic_score": semantic_score,
                    "grounding_score": float(grounding.get("score") or 0.0),
                    "adjusted_score": semantic_score,
                    "topic_gap": max(0.0, top_topic_score - matched_topic_score),
                    "min_score": float(self.semantic_min_publish_score),
                    "scope": str(rule.get("scope") or grounding.get("scope") or "domain"),
                    "evidence": evidence,
                    "publish_gate": self._build_semantic_rule_publish_gate(
                        rule=rule,
                        semantic_score=semantic_score,
                    ),
                    "manual_override_reason": "",
                    "retrieval_strategy": "semantic_tree_selection",
                }
            )

        retrieved_rules = [
            {
                "rule_id": str(item["rule"].get("id") or ""),
                "domain": str(item.get("domain") or "Unknown"),
                "topic_id": str(item.get("topic_id") or ""),
                "topic": str(item.get("topic_name") or "Unknown"),
                "cluster_id": str(item.get("cluster_id") or ""),
                "cluster": str(item.get("cluster") or ""),
                "title": str(item["rule"].get("title") or ""),
                "scope": str(item.get("scope") or item["rule"].get("scope") or "domain"),
                "score": float(item.get("score") or 0.0),
                "score_kind": "semantic_0_1",
                "semantic_score": float(item.get("semantic_score") or 0.0),
                "grounding_score": float(item.get("grounding_score") or 0.0),
                "publish_gate": item.get("publish_gate") or {},
                "manual_override_reason": "",
                "evidence": item.get("evidence") or {},
            }
            for item in selected_rule_records
        ]
        if semantic_selection_error:
            # Partial rules are valuable for debugging but have not passed the
            # complete dedupe/ranking/cap pipeline. Keep them visible only and
            # never execute them in the downstream verifier.
            for item in retrieved_rules:
                item["partial"] = True
                item["executable"] = False
            selected_rule_records = []
        terminal_stage = str(semantic_result.get("terminal_stage") or semantic_failed_stage or "")
        empty_reason = str(semantic_result.get("empty_reason") or "")
        if semantic_selection_error and not empty_reason:
            empty_reason = "semantic_retrieval_error"
        elif not semantic_selection_error and not selected_rule_records and not empty_reason:
            if not retrieved_domains:
                empty_reason = "no_domain_selected"
            elif not retrieved_topics:
                empty_reason = "no_topic_selected"
            else:
                empty_reason = "no_rule_selected"

        if semantic_selection_error:
            selection_strategy = "semantic_error"
        elif selected_rule_records:
            selection_strategy = "semantic_tree_selection"
        else:
            selection_strategy = "semantic_tree_empty"

        return {
            "selection_strategy": selection_strategy,
            "retrieval_score_kind": "semantic_0_1",
            "semantic_selection_error": semantic_selection_error,
            "semantic_failed_stage": semantic_failed_stage,
            "semantic_input_policy": input_policy,
            "background_analysis": semantic_result.get("background_analysis") or {},
            "navigation_trace": semantic_result.get("navigation_trace") or {},
            "terminal_stage": terminal_stage,
            "empty_reason": empty_reason,
            "retrieved_domains": retrieved_domains,
            "topic_matches": topic_matches,
            "retrieved_topics": retrieved_topics,
            "retrieved_clusters": retrieved_clusters,
            "selected_rule_records": selected_rule_records,
            "retrieved_rules": retrieved_rules,
        }

    def retrieve_unified_semantic_tree(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return the canonical unified-v2 semantic retrieval trace only.

        This is a public stage boundary for inspecting Domain → Topic →
        ScenarioCluster → Rule navigation. It deliberately does not invoke the
        semantic checker, generated experience code, or diagnostic gates.
        """
        if not self._unified_v2_mode:
            raise ValueError("retrieve_unified_semantic_tree requires a unified_rules_v2 catalog")
        if self.unified_retrieval_mode != "semantic":
            raise ValueError("retrieve_unified_semantic_tree requires unified_retrieval_mode='semantic'")

        trace = self._retrieve_unified_v2_semantic_tree(sample)
        retrieved_topics = list(trace.get("retrieved_topics") or [])
        primary_topic = str(retrieved_topics[0].get("topic") or "") if retrieved_topics else None
        return {
            "id": sample.get("id"),
            "topic": primary_topic,
            "verifier": "unified_v2_semantic_retrieval_only",
            "unified_mode": self._unified_mode,
            "unified_retrieval_mode": self.unified_retrieval_mode,
            "selection_strategy": str(trace.get("selection_strategy") or "semantic_error"),
            "retrieval_score_kind": str(trace.get("retrieval_score_kind") or "semantic_0_1"),
            "semantic_min_publish_score": float(self.semantic_min_publish_score),
            "semantic_selection_error": str(trace.get("semantic_selection_error") or ""),
            "semantic_failed_stage": str(trace.get("semantic_failed_stage") or ""),
            "semantic_input_policy": str(trace.get("semantic_input_policy") or ""),
            "background_analysis": trace.get("background_analysis") or {},
            "navigation_trace": trace.get("navigation_trace") or {},
            "terminal_stage": str(trace.get("terminal_stage") or ""),
            "empty_reason": str(trace.get("empty_reason") or ""),
            "retrieved_domains": list(trace.get("retrieved_domains") or []),
            "retrieved_topics": retrieved_topics,
            "retrieved_clusters": list(trace.get("retrieved_clusters") or []),
            "retrieved_rules": list(trace.get("retrieved_rules") or []),
        }

    def _maybe_llm_rerank_topics(
        self,
        scored: List[Dict[str, Any]],
        sample: Dict[str, Any],
        text_for_topic: str,
    ) -> List[Dict[str, Any]]:
        if not self._retrieval_tuning.get("llm_topic_rerank_enabled"):
            return scored
        if not self.semantic_checker._llm_available():
            return scored
        take = min(int(self._retrieval_tuning.get("llm_topic_rerank_pool", 8) or 8), len(scored))
        pool = scored[:take]
        lines = []
        for i, item in enumerate(pool):
            lines.append(
                f"{i}. domain={item.get('domain')!s} topic={item.get('name')!s} score={float(item.get('score') or 0.0):.3f}"
            )
        user = (
            "You reorder physics catalog TOPICS by relevance to the problem text.\n"
            "Return JSON only: {\"order\": [indices as integers]} — a permutation of 0..n-1, most relevant first.\n\n"
            f"Problem text (truncated):\n{text_for_topic[:6000]}\n\n"
            "Candidates:\n" + "\n".join(lines)
        )
        try:
            resp = self.semantic_checker._llm_json(
                system_prompt="You are a precise router. Output JSON only.",
                user_prompt=user,
            )
        except Exception:
            return scored
        order = []
        if isinstance(resp, dict):
            order = resp.get("order") or resp.get("indices") or []
        if not isinstance(order, list) or len(order) < 2:
            return scored
        reordered: List[Dict[str, Any]] = []
        seen = set()
        for idx in order:
            try:
                j = int(idx)
            except Exception:
                continue
            if j < 0 or j >= len(pool) or j in seen:
                continue
            seen.add(j)
            reordered.append(pool[j])
        for i, item in enumerate(pool):
            if i not in seen:
                reordered.append(item)
        tail = scored[take:]
        return reordered + tail

    def _sidecar_rule_ids(self) -> List[str]:
        path = str(self._retrieval_tuning.get("vector_sidecar_rule_ids_path") or "").strip()
        if not path or not Path(path).exists():
            return []
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError:
            return []
        out: List[str] = []
        for line in raw.splitlines():
            rid = line.strip().split("#", 1)[0].strip()
            if rid:
                out.append(rid)
        return ordered_unique(out)

    def _find_rule_across_catalog(self, rule_id: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        for domain in self.catalog.get("domains") or []:
            if not isinstance(domain, dict):
                continue
            domain_name = str(domain.get("name") or "Unknown")
            for topic in domain.get("topics") or []:
                if not isinstance(topic, dict):
                    continue
                topic_name = str(topic.get("name") or "Unknown")
                for raw in topic_rule_leaves(topic):
                    if not isinstance(raw, dict):
                        continue
                    rid = str(raw.get("rule_id") or raw.get("id") or "").strip()
                    if rid == rule_id:
                        topic_wrap = dict(topic)
                        topic_wrap.setdefault("domain", domain_name)
                        return topic_wrap, raw
        return None

    def _retrieve_unified_v2_topics(self, sample: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        text_for_topic = build_unified_topic_retrieval_text(sample, tuning=self._retrieval_tuning)
        scored = [
            {
                "domain": payload["domain"],
                "name": payload["topic"],
                "score": payload["score"],
                "evidence": payload["evidence"],
                "topic": payload["topic_obj"],
            }
            for payload in (
                score_topic_candidate(candidate, text_for_topic, signal_df=self._unified_v2_signal_df)
                for candidate in self._unified_v2_topic_candidates
            )
        ]
        pred_syms = self._prediction_symbol_set(sample)
        apply_topic_symbol_overlap_boost(scored, pred_syms, tuning=self._retrieval_tuning)
        scored.sort(key=lambda item: topic_sort_key({"score": item["score"], "domain": item["domain"], "topic": item["name"]}))
        scored = self._maybe_llm_rerank_topics(scored, sample, text_for_topic)
        return scored[:top_k]

    def _score_unified_v2_rule(self, rule: Dict[str, Any], text_for_rule: str) -> Dict[str, Any]:
        payload = score_rule_candidate(rule, text_for_rule)
        return {
            "rule_id": payload["rule_id"],
            "title": payload["title"],
            "score": payload["score"],
            "scope": payload["scope"],
            "evidence": payload["evidence"],
        }

    def _maybe_llm_rerank_rules(self, scored: List[Dict[str, Any]], text_for_rule: str) -> None:
        if not self._retrieval_tuning.get("llm_rule_rerank_enabled"):
            return
        if not self.semantic_checker._llm_available() or not scored:
            return
        pool_size = min(int(self._retrieval_tuning.get("llm_rule_rerank_pool", 18) or 18), len(scored))
        pool = sorted(
            scored,
            key=lambda x: -float(x.get("adjusted_score", x.get("score") or 0.0)),
        )[:pool_size]
        lines: List[str] = []
        short_ids: List[str] = []
        for item in pool:
            rid = str(item["rule"].get("id") or "")
            if not rid:
                continue
            short_ids.append(rid)
            lines.append(f"- {rid}: {norm_text(item['rule'].get('title') or '')[:120]}")
        if len(short_ids) < 2:
            return
        user = (
            "Rank these verification rule ids by relevance to the student's combined problem+solution text. "
            "Return JSON only: {\"order\": [\"rule_id\", ...]} including each id exactly once.\n\n"
            f"Text (truncated):\n{text_for_rule[:5000]}\n\nCandidates:\n" + "\n".join(lines)
        )
        try:
            resp = self.semantic_checker._llm_json(
                system_prompt="You are a precise retrieval ranker. Output JSON only.",
                user_prompt=user,
            )
        except Exception:
            return
        order: List[str] = []
        if isinstance(resp, dict):
            order = [str(x) for x in (resp.get("order") or resp.get("rule_ids") or []) if str(x)]
        if len(order) < 2:
            return
        rank_by_id = {rid: i for i, rid in enumerate(order) if rid in set(short_ids)}
        if not rank_by_id:
            return
        for item in scored:
            rid = str(item["rule"].get("id") or "")
            if rid not in rank_by_id:
                continue
            r = rank_by_id[rid]
            mult = 1.0 + max(0.0, 0.28 - 0.015 * float(r))
            item["score"] = float(item.get("score") or 0.0) * mult
            item["adjusted_score"] = float(item.get("adjusted_score") or 0.0) * mult
            ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            ev = dict(ev)
            ev["llm_rerank_multiplier"] = round(mult, 4)
            item["evidence"] = ev

    def _retrieve_unified_v2_rules(self, topic_matches: List[Dict[str, Any]], sample: Dict[str, Any], top_n: int = 6) -> List[Dict[str, Any]]:
        text_for_rule = "\n".join(
            [
                str(sample.get("question") or ""),
                str(sample.get("context") or ""),
                str(sample.get("prediction") or ""),
            ]
        )
        scored: List[Dict[str, Any]] = []
        top1_score = float(topic_matches[0]["score"]) if topic_matches else 0.0
        top1_margin = top1_score - float(topic_matches[1]["score"]) if len(topic_matches) > 1 else top1_score
        top1_key = None
        if topic_matches:
            top1_key = (str(topic_matches[0].get("domain") or ""), str(topic_matches[0].get("name") or ""))

        for topic_rank, item in enumerate(topic_matches):
            topic = item.get("topic") if isinstance(item.get("topic"), dict) else {}
            domain_name = str(item.get("domain") or topic.get("domain") or "Unknown")
            topic_name = str(item.get("name") or topic.get("name") or "Unknown")
            for raw_rule in topic_rule_leaves(topic):
                if not isinstance(raw_rule, dict):
                    continue
                prepared_rule = self._prepare_unified_v2_rule(raw_rule)
                score_payload = self._score_unified_v2_rule(prepared_rule, text_for_rule)
                topic_ctx = rule_topic_context(
                    raw_score=float(score_payload["score"] or 0.0),
                    topic_rank=topic_rank,
                    topic_score=float(item.get("score") or 0.0),
                    top1_topic_score=top1_score,
                    scope=str(score_payload.get("scope") or "domain"),
                    rule_evidence=score_payload.get("evidence") or {},
                    topic_evidence=item.get("evidence") or {},
                    retrieval_tuning=self._retrieval_tuning,
                )
                publish_gate = self._build_rule_publish_gate(
                    rule=prepared_rule,
                    score_payload=score_payload,
                    topic_ctx=topic_ctx,
                    topic_rank=topic_rank,
                    text_for_rule=text_for_rule,
                )
                scored.append(
                    {
                        "domain": domain_name,
                        "topic_name": topic_name,
                        "topic": topic,
                        "topic_rank": topic_rank,
                        "rule": prepared_rule,
                        "score": score_payload["score"],
                        "adjusted_score": topic_ctx["adjusted_score"],
                        "topic_gap": topic_ctx["topic_gap"],
                        "min_score": topic_ctx["min_score"],
                        "scope": score_payload.get("scope") or "domain",
                        "evidence": score_payload["evidence"],
                        "publish_gate": publish_gate,
                        "manual_override_reason": (score_payload.get("evidence") or {}).get("manual_override_reason") or "",
                    }
                )

        seen_rule_ids = {str(item["rule"].get("id") or "") for item in scored if item.get("rule")}
        for rid in self._sidecar_rule_ids():
            if not rid or rid in seen_rule_ids:
                continue
            found = self._find_rule_across_catalog(rid)
            if not found:
                continue
            topic_wrap, raw_rule = found
            prepared_rule = self._prepare_unified_v2_rule(raw_rule)
            score_payload = self._score_unified_v2_rule(prepared_rule, text_for_rule)
            anchor_topic = topic_matches[0] if topic_matches else {"score": 0.0, "evidence": {}}
            topic_ctx = rule_topic_context(
                raw_score=float(score_payload["score"] or 0.0),
                topic_rank=0,
                topic_score=float(anchor_topic.get("score") or 0.0),
                top1_topic_score=top1_score,
                scope=str(score_payload.get("scope") or "domain"),
                rule_evidence=score_payload.get("evidence") or {},
                topic_evidence=anchor_topic.get("evidence") or {},
                retrieval_tuning=self._retrieval_tuning,
            )
            publish_gate = self._build_rule_publish_gate(
                rule=prepared_rule,
                score_payload=score_payload,
                topic_ctx=topic_ctx,
                topic_rank=0,
                text_for_rule=text_for_rule,
            )
            domain_name = str(topic_wrap.get("domain") or "Unknown")
            topic_name = str(topic_wrap.get("name") or "Unknown")
            scored.append(
                {
                    "domain": domain_name,
                    "topic_name": topic_name,
                    "topic": topic_wrap,
                    "topic_rank": 0,
                    "rule": prepared_rule,
                    "score": score_payload["score"],
                    "adjusted_score": topic_ctx["adjusted_score"],
                    "topic_gap": topic_ctx["topic_gap"],
                    "min_score": topic_ctx["min_score"],
                    "scope": score_payload.get("scope") or "domain",
                    "evidence": score_payload["evidence"],
                    "publish_gate": publish_gate,
                    "manual_override_reason": (score_payload.get("evidence") or {}).get("manual_override_reason") or "",
                    "sidecar_injected": True,
                }
            )
            seen_rule_ids.add(rid)

        self._maybe_llm_rerank_rules(scored, text_for_rule)

        scored.sort(
            key=lambda item: rule_sort_key(
                {
                    "scope": item.get("scope") or "domain",
                    "score": item["score"],
                    "domain": item["domain"],
                    "topic": item["topic_name"],
                    "rule_id": str(item["rule"].get("id") or ""),
                }
            )
        )
        return select_rules_with_topic_priority(
            [
                {
                    **item,
                    "topic": item["topic_name"],
                    "rule_id": str(item["rule"].get("id") or ""),
                }
                for item in scored
                if item["score"] > 0
            ],
            top_n=top_n,
            top1_key=top1_key,
            top1_margin=float(top1_margin or 0.0),
        )

    def _build_unified_v2_topic_for_synthesis(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        adapted = dict(topic)
        adapted["rules"] = [self._prepare_unified_v2_rule(rule) for rule in topic_rule_leaves(topic) if isinstance(rule, dict)]
        return adapted

    # ---- Unified-mode SRD helpers ----

    @staticmethod
    def _build_srd_for_rule(r: Dict[str, Any]) -> str:
        """Construct an SRD prompt string from a rule dict, adapting to its source type."""
        source = r.get("source", "knowledge")
        if source == "experience_tagged":
            # Tagged experience rules have very detailed descriptions that serve as full SRDs
            return r.get("description", "")
        elif source == "experience":
            # Distilled experience: use trigger + check_logic
            parts = []
            if r.get("title"):
                parts.append(f"Title: {r['title']}")
            if r.get("trigger"):
                parts.append(f"Trigger: {r['trigger']}")
            if r.get("check_logic"):
                parts.append(f"Check Logic: {r['check_logic']}")
            if r.get("preconditions"):
                parts.append("Preconditions: " + "; ".join(str(x) for x in (r.get("preconditions") or []) if str(x).strip()))
            if r.get("violation_signatures"):
                parts.append(
                    "Violation Signatures: "
                    + "; ".join(str(x) for x in (r.get("violation_signatures") or []) if str(x).strip())
                )
            if r.get("negative_conditions"):
                parts.append(
                    "Do Not Trigger When: "
                    + "; ".join(str(x) for x in (r.get("negative_conditions") or []) if str(x).strip())
                )
            if r.get("evidence_requirements"):
                parts.append(
                    "Evidence Requirements: "
                    + "; ".join(str(x) for x in (r.get("evidence_requirements") or []) if str(x).strip())
                )
            return "\n".join(parts)
        elif r.get("rule_id") or (r.get("id") and not r.get("source")):
            parts = []
            if r.get("title"):
                parts.append(f"Title: {r['title']}")
            if r.get("trigger"):
                parts.append(f"Trigger: {r['trigger']}")
            if r.get("check_logic"):
                parts.append(f"Check Logic: {r['check_logic']}")
            if r.get("preconditions"):
                parts.append("Preconditions: " + "; ".join(str(x) for x in (r.get("preconditions") or []) if str(x).strip()))
            if r.get("violation_signatures"):
                parts.append(
                    "Violation Signatures: "
                    + "; ".join(str(x) for x in (r.get("violation_signatures") or []) if str(x).strip())
                )
            if r.get("negative_conditions"):
                parts.append(
                    "Do Not Trigger When: "
                    + "; ".join(str(x) for x in (r.get("negative_conditions") or []) if str(x).strip())
                )
            if r.get("evidence_requirements"):
                parts.append(
                    "Evidence Requirements: "
                    + "; ".join(str(x) for x in (r.get("evidence_requirements") or []) if str(x).strip())
                )
            if r.get("description") and not parts:
                parts.append(str(r.get("description") or ""))
            return "\n".join(parts)
        else:
            # Knowledge rules: classic Title + Description + Check Logic
            return f"Title: {r.get('title')}\nDescription: {r.get('description')}\nCheck Logic: {r.get('check_logic')}"

    def classify_topic(self, question: str) -> Optional[Dict]:
        # Use LLM to classify the question into one of the topics
        topic_list_str = "\n".join([f"- {t['domain']}: {t['name']}" for t in self.topics])
        
        prompt = f"""
You are a physics expert. Classify the following physics problem into one of the provided domains and topics.
Return ONLY the JSON object with "domain" and "topic" keys.

Problem:
{question[:1000]}...

Available Topics:
{topic_list_str}

JSON Output:
"""
        # We can use the semantic checker's LLM method for this
        response = self.semantic_checker._llm_json(
            system_prompt="You are a classifier.",
            user_prompt=prompt
        )
        
        if isinstance(response, dict):
            domain = response.get("domain")
            topic_name = response.get("topic")
            
            # Find the matching topic object
            for t in self.topics:
                if t["name"] == topic_name and (not domain or t["domain"] == domain):
                    return t
            
            # Fallback: try fuzzy match or just return None
            # For now, strict match on topic name
            for t in self.topics:
                if t["name"] == topic_name:
                    return t
                    
        return None

    def verify(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        verification_sample = dict(sample)
        # The reference answer is evaluation-only data. Letting it enter either
        # the LLM checker or generated checks can suppress genuine reasoning errors.
        verification_sample.pop("answer", None)

        topic: Optional[Dict[str, Any]] = None
        diagnostics: List[Dict[str, Any]] = []
        used_rules: List[str] = []
        verifier_used = "top_down_rule_based"
        symbolic_actions: List[Dict[str, Any]] = []
        suppressed_diagnostics: List[Dict[str, Any]] = []
        experience_code_post_diagnostics: List[Dict[str, Any]] = []
        experience_post_diagnostics: List[Dict[str, Any]] = []
        candidate_diagnostics: List[Dict[str, Any]] = []

        selected_rule_records: List[Dict[str, Any]] = []
        semantic_rule_records: List[Dict[str, Any]] = []
        retrieved_domains_payload: List[Dict[str, Any]] = []
        retrieved_topics_payload: List[Dict[str, Any]] = []
        retrieved_clusters_payload: List[Dict[str, Any]] = []
        retrieved_rules_payload: List[Dict[str, Any]] = []
        selection_strategy = "not_applicable"
        retrieval_score_kind = "not_applicable"
        semantic_selection_error = ""
        semantic_failed_stage = ""
        semantic_input_policy = ""
        background_analysis: Dict[str, Any] = {}
        navigation_trace: Dict[str, Any] = {}
        terminal_stage = ""
        empty_reason = ""

        if self._unified_v2_mode:
            topic_matches: List[Dict[str, Any]] = []
            if self.unified_retrieval_mode == "semantic":
                semantic_trace = self._retrieve_unified_v2_semantic_tree(sample)
                selection_strategy = str(semantic_trace.get("selection_strategy") or "semantic_error")
                retrieval_score_kind = str(semantic_trace.get("retrieval_score_kind") or "semantic_0_1")
                semantic_selection_error = str(semantic_trace.get("semantic_selection_error") or "")
                semantic_failed_stage = str(semantic_trace.get("semantic_failed_stage") or "")
                semantic_input_policy = str(semantic_trace.get("semantic_input_policy") or "")
                background_analysis = (
                    semantic_trace.get("background_analysis")
                    if isinstance(semantic_trace.get("background_analysis"), dict)
                    else {}
                )
                navigation_trace = (
                    semantic_trace.get("navigation_trace")
                    if isinstance(semantic_trace.get("navigation_trace"), dict)
                    else {}
                )
                terminal_stage = str(semantic_trace.get("terminal_stage") or "")
                empty_reason = str(semantic_trace.get("empty_reason") or "")
                retrieved_domains_payload = list(semantic_trace.get("retrieved_domains") or [])
                topic_matches = list(semantic_trace.get("topic_matches") or [])
                retrieved_topics_payload = list(semantic_trace.get("retrieved_topics") or [])
                retrieved_clusters_payload = list(semantic_trace.get("retrieved_clusters") or [])
                selected_rule_records = list(semantic_trace.get("selected_rule_records") or [])
                retrieved_rules_payload = list(semantic_trace.get("retrieved_rules") or [])
                verifier_used = "unified_v2_semantic_rule_based"
            else:
                selection_strategy = "lexical_tree_selection"
                retrieval_score_kind = "lexical"
                topic_matches = self._retrieve_unified_v2_topics(sample, top_k=3)
                retrieved_topics_payload = [
                    {
                        "domain": str(item.get("domain") or "Unknown"),
                        "topic": str(item.get("name") or "Unknown"),
                        "score": float(item.get("score") or 0.0),
                        "score_kind": "lexical",
                        "evidence": item.get("evidence") or {},
                    }
                    for item in topic_matches
                ]
                selected_rule_records = self._retrieve_unified_v2_rules(
                    topic_matches, sample, top_n=int(self.unified_rule_top_n)
                )
                retrieved_rules_payload = [
                    {
                        "rule_id": str(item["rule"].get("id") or ""),
                        "domain": str(item.get("domain") or "Unknown"),
                        "topic": str(item.get("topic_name") or "Unknown"),
                        "title": str(item["rule"].get("title") or ""),
                        "scope": str(item.get("scope") or item["rule"].get("scope") or "domain"),
                        "score": float(item.get("score") or 0.0),
                        "score_kind": "lexical",
                        "publish_gate": item.get("publish_gate") or {},
                        "manual_override_reason": str(item.get("manual_override_reason") or ""),
                        "evidence": item.get("evidence") or {},
                    }
                    for item in selected_rule_records
                ]
                verifier_used = "unified_v2_rule_based"

            if topic_matches:
                topic = topic_matches[0].get("topic") if isinstance(topic_matches[0].get("topic"), dict) else None
                if topic:
                    print(f"Retrieved primary topic: {topic['domain']} - {topic['name']}")

            semantic_rule_records = [
                item for item in selected_rule_records
                if bool((item.get("publish_gate") or {}).get("publishable"))
            ]
            for item in selected_rule_records:
                gate = item.get("publish_gate") if isinstance(item.get("publish_gate"), dict) else {}
                if gate and not bool(gate.get("publishable")):
                    suppressed_diagnostics.append(
                        {
                            "reason": "rule_publish_gate_precheck",
                            "rule_id": str((item.get("rule") or {}).get("id") or ""),
                            "publish_gate": gate,
                        }
                    )

            current_translations: Dict[str, Dict[str, str]] = {}
            rule_ids: List[str] = []
            for item in semantic_rule_records:
                rule = item["rule"]
                rid = str(rule.get("id") or "").strip()
                if not rid:
                    continue
                rule_ids.append(rid)
                current_translations[rid] = {"srd": self._build_srd_for_rule(rule)}

            if rule_ids:
                self.semantic_checker.rules_to_check = rule_ids
                self.semantic_checker.rule_translations = current_translations
                print(f"Running unified v2 rule check with {len(rule_ids)} rules...")
                result = self.semantic_checker.analyze(verification_sample)
                diagnostics = result.get("diagnostics", [])
                candidate_diagnostics = list(diagnostics)
                diagnostics, low_conf_suppressed = self._filter_low_confidence_unified_diagnostics(
                    diagnostics,
                    semantic_rule_records,
                )
                suppressed_diagnostics.extend(low_conf_suppressed)
                used_rules = rule_ids
            else:
                self.semantic_checker.rules_to_check = []
                self.semantic_checker.rule_translations = {}
        else:
            # 1. Classify
            topic = self.classify_topic(question)

            if topic:
                print(f"Classified into: {topic['domain']} - {topic['name']}")
                rules = topic.get("rules", [])

                if rules:
                    # 2. Prepare Rule Verifier
                    # Convert catalog rules to the format expected by SemanticRuleChecker
                    # We use source-aware SRD construction
                    current_translations = {}
                    rule_ids = []
                    for r in rules:
                        rid = r.get("id")
                        if not rid:
                            continue
                        rule_ids.append(rid)
                        current_translations[rid] = {
                            "srd": self._build_srd_for_rule(r)
                        }

                    self.semantic_checker.rules_to_check = rule_ids
                    self.semantic_checker.rule_translations = current_translations

                    # 3. Run Rule Check
                    print(f"Running rule check with {len(rule_ids)} rules...")
                    result = self.semantic_checker.analyze(verification_sample)
                    diagnostics = result.get("diagnostics", [])
                    used_rules = rule_ids
                    verifier_used = "top_down_rule_based" if not self._unified_mode else "unified_rule_based"
            else:
                print("Could not classify topic or no topic found.")

        # Build the lookup from rule id -> rule dict + topic for downstream
        # release-gate metadata. The symbolic check now runs deterministic
        # generated experience code keyed by ``rule_id``; no LLM, no
        # primitive+spec catalog, no experience bank. It is on by default.
        _rule_by_id: Dict[str, Dict[str, Any]] = {}
        _rule_topic_by_id: Dict[str, Dict[str, Any]] = {}
        if self._unified_v2_mode:
            for item in selected_rule_records:
                rule = item.get("rule") if isinstance(item.get("rule"), dict) else None
                if not rule:
                    continue
                rid = rule.get("id")
                if not rid:
                    continue
                _rule_by_id[str(rid)] = rule
                _rule_topic_by_id[str(rid)] = {
                    "domain": str(item.get("domain") or "Unknown"),
                    "name": str(item.get("topic_name") or "Unknown"),
                }
        elif topic:
            for _r in (topic.get("rules") or []):
                _rid = _r.get("id")
                if _rid:
                    _rule_by_id[_rid] = _r
                    _rule_topic_by_id[_rid] = {
                        "domain": str(topic.get("domain") or "Unknown"),
                        "name": str(topic.get("name") or "Unknown"),
                    }

        symbolic_enabled = bool(self.enable_symbolic_check) and self.experience_code_engine.available
        sample_for_check: Dict[str, Any] = {
            "question": str(sample.get("question") or ""),
            "prediction": str(sample.get("prediction") or ""),
            "context": str(sample.get("context") or ""),
            "id": sample.get("id"),
        }

        # 4. Top-down: run experience-rule code for every LLM diagnostic whose
        #    rule has a generated check function. Results are tagged with a
        #    ``exp_code::<rule_id>`` spec id and feed into the reconciliation
        #    step below; the legacy ``symbolic_cross_checks`` /
        #    ``symbolic_reconciliation`` fields are preserved so existing
        #    release-gate logic keeps working unchanged.
        #
        #    When the LLM diagnostic's rule_id is not directly covered by the
        #    manifest (rule_id taxonomies can drift between catalog versions),
        #    fall back to a topic bridge: run every manifest check that lives
        #    under the same (domain, topic) pair. ``fail`` results from such
        #    bridge checks are accepted only as corroboration (mark supported);
        #    ``pass`` is treated as ``inconclusive`` because the underlying
        #    rule statement may not match the LLM diagnostic's assertion.
        triggered_rule_ids: Set[str] = set()
        if symbolic_enabled and diagnostics:
            for d in diagnostics:
                if not isinstance(d, dict):
                    continue
                rid = str(d.get("rule") or "").strip()
                if not rid:
                    continue
                triggered_rule_ids.add(rid)
                if self.experience_code_engine.has_rule(rid):
                    res = self.experience_code_engine.run_rule(rid, sample_for_check)
                    if res is None:
                        continue
                    spec_id = f"exp_code::{rid}"
                    payload = {
                        "spec_id": spec_id,
                        "rule": f"experience_code::{rid}",
                        "rule_id": rid,
                        "primitive": "experience_code",
                        "title": f"Experience code check {rid}",
                        "result": res.get("result", "inconclusive"),
                        "symbolic_result": res.get("result", "inconclusive"),
                        "message": str(res.get("message") or ""),
                        "evidence": str(res.get("evidence") or ""),
                        "source": "experience_code_top_down",
                    }
                    experience_code_post_diagnostics.append(payload)
                    d.setdefault("symbolic_cross_checks", []).append(spec_id)
                    symbolic_actions.append(
                        {
                            "diagnostic_rule": rid,
                            "spec_ids": [spec_id],
                            "source": "experience_code_top_down",
                            "result": payload["result"],
                        }
                    )
                else:
                    # Topic bridge: try every manifest check in the same topic.
                    topic_pair = _rule_topic_by_id.get(rid) or {}
                    domain_name = str(topic_pair.get("domain") or "Unknown")
                    topic_name = str(topic_pair.get("name") or "Unknown")
                    if domain_name == "Unknown" and topic_name == "Unknown":
                        continue
                    bridge_rule_ids = self.experience_code_engine.list_topic_rule_ids(domain_name, topic_name)
                    bridge_limit = max(1, int(self.symbolic_topic_check_limit or 0))
                    bridge_rule_ids = bridge_rule_ids[:bridge_limit]
                    for bridge_rid in bridge_rule_ids:
                        triggered_rule_ids.add(bridge_rid)
                        bres = self.experience_code_engine.run_rule(bridge_rid, sample_for_check)
                        if bres is None:
                            continue
                        bridge_spec_id = f"exp_code_bridge::{rid}::{bridge_rid}"
                        bridge_result = bres.get("result", "inconclusive")
                        # Topic-bridge ``pass`` is downgraded to ``inconclusive``
                        # because a different rule passing does not refute the
                        # LLM's specific assertion. Only ``fail`` corroborates.
                        sym_result = bridge_result if bridge_result == "fail" else "inconclusive"
                        bridge_payload = {
                            "spec_id": bridge_spec_id,
                            "rule": f"experience_code::{bridge_rid}",
                            "rule_id": bridge_rid,
                            "bridge_for_rule_id": rid,
                            "primitive": "experience_code",
                            "title": f"Experience code topic-bridge {bridge_rid}",
                            "result": bridge_result,
                            "symbolic_result": sym_result,
                            "message": str(bres.get("message") or ""),
                            "evidence": str(bres.get("evidence") or ""),
                            "source": "experience_code_topic_bridge",
                        }
                        experience_code_post_diagnostics.append(bridge_payload)
                        # Only attach bridge spec to the diagnostic when it
                        # corroborates (fail). Avoid polluting the spec list
                        # with neutral inconclusive entries.
                        if sym_result == "fail":
                            d.setdefault("symbolic_cross_checks", []).append(bridge_spec_id)
                            symbolic_actions.append(
                                {
                                    "diagnostic_rule": rid,
                                    "spec_ids": [bridge_spec_id],
                                    "source": "experience_code_topic_bridge",
                                    "result": bridge_result,
                                }
                            )

        # 5. Reconcile LLM diagnostics against deterministic experience code
        #    outcomes. ``fail`` reinforces the diagnostic (mark supported);
        #    ``pass`` refutes it (suppress); ``inconclusive`` keeps it tagged.
        if symbolic_enabled and diagnostics and experience_code_post_diagnostics:
            spec_failed: Set[str] = set()
            spec_passed: Set[str] = set()
            spec_inconclusive: Set[str] = set()
            for sd in experience_code_post_diagnostics:
                spec_id = str(sd.get("spec_id") or "")
                if not spec_id:
                    continue
                flag = sd.get("symbolic_result")
                if flag == "fail":
                    spec_failed.add(spec_id)
                elif flag == "pass":
                    spec_passed.add(spec_id)
                elif flag == "inconclusive":
                    spec_inconclusive.add(spec_id)

            reconciled: List[Dict[str, Any]] = []
            for d in diagnostics:
                if not isinstance(d, dict):
                    reconciled.append(d)
                    continue
                spec_ids = [str(s) for s in (d.get("symbolic_cross_checks") or [])]
                if not spec_ids:
                    reconciled.append(d)
                    continue
                any_fail = any(s in spec_failed for s in spec_ids)
                any_pass = any(s in spec_passed for s in spec_ids)
                any_inconclusive = any(s in spec_inconclusive for s in spec_ids)

                # Quote-aware safety: when ``pass`` would refute the
                # diagnostic, only suppress if the diagnostic's quote does not
                # already overlap the rule's required symbols. Otherwise we
                # fall back to ``quote_overlap`` which keeps the diagnostic.
                quote_overlap_blocks_pass = False
                if any_pass:
                    rule_obj = _rule_by_id.get(str(d.get("rule")))
                    if isinstance(rule_obj, dict):
                        required_quote_symbols = self._quote_required_symbols(rule_obj)
                        evidence = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
                        quote = str(evidence.get("quote") or "").strip()
                        if quote and required_quote_symbols:
                            _, ratio = self._quote_symbol_overlap(quote, required_quote_symbols)
                            if ratio >= 0.5:
                                quote_overlap_blocks_pass = True

                if any_fail:
                    d["symbolic_reconciliation"] = {"status": "supported", "spec_ids": spec_ids}
                    reconciled.append(d)
                elif any_pass and not quote_overlap_blocks_pass:
                    suppressed_diagnostics.append(
                        {
                            "reason": "symbolic_check_refuted_original",
                            "spec_ids": spec_ids,
                            "original_diagnostic": d,
                        }
                    )
                elif quote_overlap_blocks_pass:
                    d["symbolic_reconciliation"] = {"status": "quote_overlap", "spec_ids": spec_ids}
                    reconciled.append(d)
                else:
                    # All specs returned ``inconclusive`` (or had no signal).
                    # Leave the diagnostic untagged so the release gate treats
                    # it as neutral ``none`` rather than down-weighted
                    # ``inconclusive``. The catalog's experience-code coverage
                    # is uneven and many functions conservatively return
                    # ``inconclusive``; treating that as a negative signal
                    # would prematurely suppress otherwise valid diagnostics.
                    reconciled.append(d)
            diagnostics = reconciled

        if self._unified_v2_mode and diagnostics:
            diagnostics, release_suppressed = self._apply_diagnostic_release_gate(
                diagnostics,
                semantic_rule_records or selected_rule_records,
            )
            suppressed_diagnostics.extend(release_suppressed)

        # 6. Bottom-up experience-code pass. In semantic-tree mode, the API
        #    Rule selection is authoritative: only selected rules that passed
        #    the static publish gate may run bottom-up. Lexical mode preserves
        #    the historical retrieved-topic sweep. Any rule whose code returns
        #    ``fail`` becomes a new bottom-up diagnostic only when its evidence
        #    string can be located in the prediction text.
        prediction_text = str(sample.get("prediction") or "")
        paragraphs_for_location: List[Dict[str, Any]] = []
        if prediction_text and symbolic_enabled:
            paragraphs_for_location = self.semantic_checker._paragraph_ranges(prediction_text)

        def _build_location_for_evidence(evidence_text: str) -> Optional[Dict[str, Any]]:
            ev = str(evidence_text or "").strip()
            if not ev or not prediction_text:
                return None
            # Try the evidence string as-is, then progressively shorter prefixes
            # (common shape: "<short snippet>: <explanation>"). Stop on the
            # first locatable substring of length >= 6.
            candidates: List[str] = [ev]
            if ":" in ev:
                head = ev.split(":", 1)[0].strip()
                if head and head not in candidates:
                    candidates.append(head)
            if len(ev) > 80:
                candidates.append(ev[:80].strip())
            seen: Set[str] = set()
            for cand in candidates:
                if not cand or cand in seen:
                    continue
                seen.add(cand)
                if len(cand) < 6:
                    continue
                located = self.semantic_checker._locate_quote_span(prediction_text, cand)
                if not bool(located.get("span_valid")):
                    continue
                start = int(located.get("start_char") or -1)
                end = int(located.get("end_char") or -1)
                if start < 0 or end <= start:
                    continue
                paragraph = self.semantic_checker._paragraph_from_offset(
                    paragraphs_for_location, start
                )
                paragraph_index = int(paragraph.get("paragraph_index") or -1) if paragraph else -1
                paragraph_start = int(paragraph.get("start_char") or -1) if paragraph else -1
                paragraph_end = int(paragraph.get("end_char") or -1) if paragraph else -1
                return {
                    "quote": cand,
                    "location": {
                        "start_char": start,
                        "end_char": end,
                        "line_index": int(located.get("line_index") or -1),
                        "span_valid": True,
                        "locate_method": f"experience_code_{located.get('locate_method') or 'quote_match'}",
                        "locate_confidence": float(located.get("locate_confidence") or 0.6),
                        "paragraph_index": paragraph_index,
                        "paragraph_start_char": paragraph_start,
                        "paragraph_end_char": paragraph_end,
                        "paragraph_valid": paragraph_index >= 1,
                        "paragraph_source": "experience_code_anchor",
                        "locatable_valid": True,
                    },
                }
            return None

        if symbolic_enabled:
            bottom_up_rule_candidates: List[Tuple[str, str, str]] = []
            if self._unified_v2_mode and self.unified_retrieval_mode == "semantic":
                for item in semantic_rule_records:
                    rule = item.get("rule") if isinstance(item.get("rule"), dict) else {}
                    rid = str(rule.get("id") or "").strip()
                    if not rid:
                        continue
                    bottom_up_rule_candidates.append(
                        (
                            rid,
                            str(item.get("domain") or "Unknown"),
                            str(item.get("topic_name") or "Unknown"),
                        )
                    )
            else:
                topic_pairs: List[Tuple[str, str]] = []
                seen_pairs: Set[Tuple[str, str]] = set()
                if self._unified_v2_mode:
                    for tp in retrieved_topics_payload[:3]:
                        pair = (str(tp.get("domain") or "Unknown"), str(tp.get("topic") or "Unknown"))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        topic_pairs.append(pair)
                elif topic:
                    pair = (str(topic.get("domain") or "Unknown"), str(topic.get("name") or "Unknown"))
                    if pair not in seen_pairs:
                        topic_pairs.append(pair)

                for domain_name, topic_name in topic_pairs:
                    rule_ids_for_topic = self.experience_code_engine.list_topic_rule_ids(domain_name, topic_name)
                    if self.symbolic_topic_check_limit > 0:
                        rule_ids_for_topic = rule_ids_for_topic[: self.symbolic_topic_check_limit]
                    bottom_up_rule_candidates.extend(
                        (str(rid), domain_name, topic_name) for rid in rule_ids_for_topic if str(rid).strip()
                    )

            for rid, domain_name, topic_name in bottom_up_rule_candidates:
                if rid in triggered_rule_ids:
                    continue
                triggered_rule_ids.add(rid)
                res = self.experience_code_engine.run_rule(rid, sample_for_check)
                if res is None:
                    continue
                spec_id = f"exp_code::{rid}"
                payload = {
                    "spec_id": spec_id,
                    "rule": f"experience_code::{rid}",
                    "rule_id": rid,
                    "primitive": "experience_code",
                    "title": f"Experience code check {rid}",
                    "result": res.get("result", "inconclusive"),
                    "symbolic_result": res.get("result", "inconclusive"),
                    "message": str(res.get("message") or ""),
                    "evidence": str(res.get("evidence") or ""),
                    "source": "experience_code_bottom_up",
                }
                experience_code_post_diagnostics.append(payload)
                symbolic_actions.append(
                    {
                        "diagnostic_rule": rid,
                        "spec_ids": [spec_id],
                        "source": "experience_code_bottom_up",
                        "result": payload["result"],
                    }
                )
                if payload["result"] != "fail":
                    continue
                located_evidence = _build_location_for_evidence(payload["evidence"])
                if located_evidence is None:
                    # Try the message text as a last-resort anchor (some
                    # functions return only a message, not a quote).
                    located_evidence = _build_location_for_evidence(payload["message"])
                if located_evidence is None:
                    # Last resort: try the catalog rule's own
                    # ``trigger_keywords`` / ``object_keywords`` /
                    # ``required_symbols``. These are precisely the tokens
                    # the rule was designed to fire on, so anchoring there
                    # keeps the bottom-up finding semantically aligned
                    # with the rule even if the generated function did
                    # not surface a usable quote.
                    rule_obj = _rule_by_id.get(rid) if isinstance(_rule_by_id.get(rid), dict) else None
                    if rule_obj:
                        mf = rule_obj.get("match_features") if isinstance(rule_obj.get("match_features"), dict) else {}
                        anchor_candidates: List[str] = []
                        for key in ("trigger_keywords", "object_keywords", "required_symbols"):
                            vals = mf.get(key) or []
                            if isinstance(vals, list):
                                for v in vals:
                                    if isinstance(v, str) and len(v.strip()) >= 3:
                                        anchor_candidates.append(v.strip())
                        for cand in anchor_candidates[:6]:
                            located_evidence = _build_location_for_evidence(cand)
                            if located_evidence is not None:
                                break
                if located_evidence is None:
                    # Cannot locate in prediction; skip publishing to
                    # protect precision but keep the audit record above.
                    payload["publish_skipped"] = "non_locatable"
                    continue
                experience_post_diagnostics.append(
                    {
                        "severity": "warning",
                        "rule": f"experience_code::{rid}",
                        "symbol": None,
                        "message": f"Experience code check failed: {payload['message']}"[:300],
                        "evidence": located_evidence,
                        "experience_code": {
                            "rule_id": rid,
                            "spec_id": spec_id,
                            "domain": domain_name,
                            "topic": topic_name,
                        },
                        "symbolic_reconciliation": {
                            "status": "supported",
                            "spec_ids": [spec_id],
                        },
                    }
                )

            if experience_post_diagnostics:
                diagnostics.extend(experience_post_diagnostics)

        return {
            "id": sample.get("id"),
            "topic": topic.get("name") if topic else None,
            "verifier": verifier_used,
            "unified_mode": self._unified_mode,
            "unified_retrieval_mode": self.unified_retrieval_mode if self._unified_v2_mode else None,
            "semantic_min_publish_score": (
                self.semantic_min_publish_score
                if self._unified_v2_mode and self.unified_retrieval_mode == "semantic"
                else None
            ),
            "selection_strategy": selection_strategy,
            "retrieval_score_kind": retrieval_score_kind,
            "semantic_selection_error": semantic_selection_error,
            "semantic_failed_stage": semantic_failed_stage,
            "semantic_input_policy": semantic_input_policy,
            "background_analysis": background_analysis,
            "navigation_trace": navigation_trace,
            "terminal_stage": terminal_stage,
            "empty_reason": empty_reason,
            "retrieved_domains": retrieved_domains_payload,
            "retrieved_topics": retrieved_topics_payload,
            "retrieved_clusters": retrieved_clusters_payload,
            "retrieved_rules": retrieved_rules_payload,
            "candidate_diagnostics": candidate_diagnostics,
            "diagnostics": diagnostics,
            "symbolic_post_diagnostics": list(experience_code_post_diagnostics),
            "experience_post_diagnostics": experience_post_diagnostics,
            "experience_code_post_diagnostics": experience_code_post_diagnostics,
            "experience_symbolic_post_diagnostics": [],
            "symbolic_check": {
                "enabled": symbolic_enabled,
                "manifest": self.experience_code_manifest_path,
                "module": self.experience_code_module,
                "actions": symbolic_actions,
                "suppressed_diagnostics": suppressed_diagnostics,
            },
            # Backwards-compatible aliases for downstream tooling that still
            # reads ``agentic`` / ``experience_pipeline`` keys.
            "agentic": {
                "enabled": False,
                "actions": [],
                "generated_specs": [],
                "suppressed_diagnostics": suppressed_diagnostics,
            },
            "experience_pipeline": {
                "enabled": symbolic_enabled,
                "actions": symbolic_actions,
                "manifest": self.experience_code_manifest_path,
            },
            "score": -1.0 * len(diagnostics),
        }

    def _record_error_experience(self, sample: Dict, topic: Optional[Dict], errors: List[Dict]):
        experience = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_id": sample.get("id"),
            "topic": topic.get("name") if topic else "Unknown",
            "question_snippet": sample.get("question", "")[:200],
            "errors_found": errors
        }
        self.error_experiences.append(experience)
        
        # Append to file immediately
        exp_file = self.results_dir / "error_experiences.json"
        
        all_exps = []
        if exp_file.exists():
            try:
                with open(exp_file, 'r') as f:
                    all_exps = json.load(f)
            except:
                pass
        
        all_exps.append(experience)
        
        with open(exp_file, 'w') as f:
            json.dump(all_exps, f, indent=2, ensure_ascii=False)

    def run_batch(
        self,
        samples: List[Dict],
        *,
        progress_interval: int = 10,
        verbose_per_sample: bool = False,
        fail_fast_on_semantic_error: bool = False,
        on_result: Optional[Callable[[List[Dict]], None]] = None,
    ) -> List[Dict]:
        """Run verifier on each sample; optionally log throughput milestones.

        progress_interval: emit one summary line every N completed samples (and at the end).
            Set to 0 to disable milestone logs.
        verbose_per_sample: if True, print a line before each sample (very noisy).
        fail_fast_on_semantic_error: stop after the first failed API-tree sample and return its trace.
        on_result: optional callback invoked after each completed sample with all results so far.
        """
        results: List[Dict] = []
        total = len(samples)
        t0 = time.perf_counter()

        def _elapsed_s() -> float:
            return time.perf_counter() - t0

        def _fmt_duration(sec: float) -> str:
            if sec < 60.0:
                return f"{sec:.1f}s"
            sec_i = int(sec)
            m, s = divmod(sec_i, 60)
            if m < 60:
                return f"{m}m{s}s"
            h, m = divmod(m, 60)
            return f"{h}h{m}m{s}s"

        for idx, s in enumerate(samples, start=1):
            sid = s.get("id")
            if verbose_per_sample:
                print(f"Verifying sample {sid}...", flush=True)
            res = self.verify(s)
            results.append(res)
            if on_result is not None:
                on_result(results)
            if fail_fast_on_semantic_error and str(res.get("selection_strategy") or "") in {
                "semantic_error",
                "semantic_unavailable",
            }:
                stage = str(res.get("semantic_failed_stage") or "unknown")
                error = str(res.get("semantic_selection_error") or "unknown semantic retrieval error")
                print(
                    f"[PhysicsVerifier] stopping after semantic retrieval failure for sample "
                    f"{sid!r} at stage {stage}: {error}",
                    flush=True,
                )
                break

            done = idx
            milestone = done == total
            if progress_interval > 0 and (done % progress_interval == 0 or milestone):
                elapsed = _elapsed_s()
                rate = elapsed / done if done else 0.0
                print(
                    f"[PhysicsVerifier] progress {done}/{total} samples | "
                    f"elapsed {_fmt_duration(elapsed)} | "
                    f"avg {rate:.2f}s/sample | last_id={sid!r}",
                    flush=True,
                )

        return results

if __name__ == "__main__":
    # Test run
    import sys
    
    # Load sample data
    data_path = "data/evaluation_sample_30.json"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
    try:
        with open(data_path, 'r') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"File {data_path} not found.")
        sys.exit(1)

    # Limit to first few for testing if needed, or run all
    # samples = samples[:1] 

    verifier = PhysicsRuleVerifier()
    results = verifier.run_batch(samples)
    
    # Save results
    with open("results/verifier_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Results saved to results/verifier_results.json")
