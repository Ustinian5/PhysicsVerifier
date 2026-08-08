"""PhysicsVerifier: 基于 LLM 的语义规则检查（SRD）引擎。

由 `physics_rule_verifier.py`（`PhysicsRuleVerifier`）从层次化规则库中选出规则并注入 SRD，
再调用本模块的 `SemanticRuleChecker` 对单条规则逐条做语义级检查。

`SemanticRuleChecker` 的职责：
- 从作答中抽取符号/公式并构建 `SymbolGraph`（可选）
- 将结构化摘要 + 规则文本（SRD）组合成 prompt
- 调用 LLM 输出结构化 diagnostics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import ast
import re
import json
import hashlib
import math
import os
import tempfile
import datetime
import importlib
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

# 使用 OpenAI API
try:
    import openai
except ImportError:
    print("OpenAI package not found. Please run 'pip install openai'")
    openai = None


SEMANTIC_RULE_CHECKER_PROMPT_VERSION = "semantic-rule-checker-dual-evidence-v1"


# ------------------------- 符号节点网络 (保持不变) -------------------------
@dataclass
class SymbolNode:
    name: str
    kind: str = "unknown"
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    defined_by: List[str] = field(default_factory=list)
    used_in: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormulaNode:
    fid: str
    raw: str
    relation: str
    lhs: Optional[str]
    rhs: Optional[str]
    symbols: List[str]
    line_index: int
    # 新增：用于精确的自引用检查
    lhs_symbols: Set[str] = field(default_factory=set)
    rhs_symbols: Set[str] = field(default_factory=set)


class SymbolGraph:
    def __init__(self) -> None:
        self.symbols: Dict[str, SymbolNode] = {}
        self.formulas: Dict[str, FormulaNode] = {}
        self.edges: List[Dict[str, Any]] = []

    def sym(self, name: str) -> SymbolNode:
        if name not in self.symbols:
            self.symbols[name] = SymbolNode(name=name)
        return self.symbols[name]

    def add_occurrence(self, name: str, line: int, context: str):
        node = self.sym(name)
        node.occurrences.append({"line": line, "context": context})

    def add_formula(self, fid: str, node: FormulaNode):
        self.formulas[fid] = node
        for s in node.symbols:
            self.sym(s).used_in.append(fid)
            self.edges.append({"type": "use", "symbol": s, "fid": fid})
        if node.relation in {"=", "≈", "~"} and node.lhs:
            m = re.search(r'([A-Za-z][A-Za-z0-9_]*)', node.lhs)
            if m:
                lhs_sym = m.group(1)
                if lhs_sym in node.symbols:
                    self.sym(lhs_sym).defined_by.append(fid)
                    self.edges.append({"type": "define", "symbol": lhs_sym, "fid": fid})


# ------------------------- 规则插件导入 (保持不变) -------------------------
try:
    try:
        import sys as _sys
        _CURR_DIR = str(Path(__file__).resolve().parent)
        if _CURR_DIR not in _sys.path:
            _sys.path.insert(0, _CURR_DIR)
    except Exception:
        pass
    from rules.base import RulePlugin, RuleContext, RuleRuntime
except Exception:
    from rules.base import RulePlugin, RuleContext, RuleRuntime

_BUILTIN_RULES_MAP = {
    "graph_consistency": "rules.graph_consistency:GraphConsistencyRule",
    "var_const_consistency": "rules.llm_rules:VarConstConsistencyRule",
    "formula_correctness": "rules.llm_rules:FormulaCorrectnessRule",
    "precondition_consistency": "rules.llm_rules:PreconditionConsistencyRule",
    "dimensional_homogeneity": "rules.llm_rules:DimensionalHomogeneityRule",
    "small_angle_approx": "rules.llm_rules:SmallAngleApproxRule",
    "energy_conservation_context": "rules.llm_rules:EnergyConservationContextRule",
    "momentum_conservation_context": "rules.llm_rules:MomentumConservationContextRule",
    "given_data_use": "rules.llm_rules:GivenDataUseRule",
    "non_empty_solution": "rules.llm_rules:NonEmptySolutionRule",
    "order_of_magnitude": "rules.llm_rules:OrderOfMagnitudeRule",
    "safe_divide": "rules.llm_rules:SafeDivideRule",
    "function_domain_guard": "rules.llm_rules:FunctionDomainRule",
}


def _openai_disable_thinking_kwargs() -> Dict[str, Any]:
    """Qwen3 local vLLM: disable chain-of-thought template so JSON outputs parse."""
    flag = str(os.getenv("OPENAI_DISABLE_THINKING", "")).strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def _load_env_file_fallback() -> None:
    """Load simple KEY=VALUE pairs from .env when python-dotenv is unavailable."""
    if load_dotenv:
        load_dotenv()
        return
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
    except Exception:
        return

def _load_rule_class(spec: str):
    module_name, class_name = None, None
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    elif "." in spec:
        module_name, class_name = spec.rsplit(".", 1)
    
    if not module_name or not class_name:
        raise ImportError(f"Invalid rule spec: {spec}")
    
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except ImportError:
        raise


# ------------------------- 主检查器实现 (重构) -------------------------
class SemanticRuleChecker:
    CHECKER_MODE_LEGACY = "legacy"
    CHECKER_MODE_DUAL_EVIDENCE = "dual_evidence"
    CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY = "dual_evidence_consistency"
    CHECKER_MODES = frozenset(
        {
            CHECKER_MODE_LEGACY,
            CHECKER_MODE_DUAL_EVIDENCE,
            CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY,
        }
    )
    DUAL_EVIDENCE_SCHEMA_VERSION = "semantic-rule-checker-dual-evidence-v1"
    _DUAL_ROOT_STATUSES = frozenset({"violation", "no_violation", "abstain"})
    _DUAL_SEVERITIES = frozenset({"error", "warning"})
    _DUAL_EVIDENCE_SOURCES = frozenset({"question", "context", "prediction"})
    _DUAL_CONSISTENCY_STATUSES = frozenset(
        {
            "confirmed_violation",
            "self_corrected",
            "equivalent_or_alternative",
            "uncertain",
        }
    )
    _NEGATIVE_DIAGNOSTIC_PATTERNS = (
        r"\bno violation\b",
        r"\bnot triggered\b",
        r"\bnot trigger(?:ed|ing)?\b",
        r"\btrigger condition (?:is )?not met\b",
        r"\bcondition (?:is )?not met\b",
        r"\bdoes not apply\b",
        r"\bnot applicable\b",
        r"\bunrelated to\b",
        r"\bnot relevant\b",
        r"\bcomplies with\b",
        r"\bcompliant\b",
        r"\bno direct contradiction\b",
        r"\bno violation (?:exists|is present|can be assessed)\b",
        r"\bno .* violation .* found\b",
        r"\btherefore,? .* no violation\b",
        r"未(?:发现|构成|触发|违反)",
        r"没有(?:发现)?(?:明显)?(?:违规|违反|错误)",
        r"不(?:适用|相关|触发|违反)",
        r"无法(?:判断|确认|确定)",
        r"证据不足",
        r"条件不足",
    )
    # These patterns are intentionally consistency-only. Adding them to the
    # historical negative-diagnostic filter would silently change the legacy
    # and dual-evidence arms, and scanning the student's quote would invert a
    # real error such as an unsupported claim that a derivation is correct.
    _CONSISTENCY_REASON_REJECTION_PATTERNS = (
        r"\b(?:the\s+)?(?:derivation|expression|reasoning|solution|method|formulation)\s+"
        r"(?:is|was|remains)\s+(?:physically\s+)?(?:correct|valid)\b",
        r"\b(?:the\s+)?(?:quoted\s+)?(?:claim|derivation|expression|reasoning|solution|method|formulation)\s+"
        r"(?:is|was|remains)\s+(?:physically\s+)?equivalent\b",
        r"\b(?:the\s+)?(?:derivation|expression|reasoning|solution|method|formulation)\s+"
        r"(?:is|was)\s+(?:a\s+)?valid alternative\b",
        r"\b(?:the\s+)?(?:student|submission)\s+"
        r"(?:later|subsequently|then|explicitly)\s+"
        r"(?:corrected|withdrew|retracted)\b",
        r"\b(?:the\s+)?(?:quoted\s+)?(?:claim|error|statement|assertion)\s+"
        r"(?:was|has been)\s+(?:already\s+)?(?:corrected|withdrawn|retracted)\b",
        r"(?:该|此)?(?:推导|表达|解法|方法)(?:是|为)(?:物理上)?(?:正确|等价)",
        r"(?:该|此)?(?:推导|表达|解法|方法)(?:是|为)?有效的?替代(?:方案|解法|方法)?",
        r"(?:学生|作答)(?:随后|之后|后来)(?:已经|已)?(?:修正|撤回|更正)",
        r"(?:该|此)?(?:说法|错误|陈述)(?:已经|已)(?:明确)?(?:修正|撤回|更正)",
    )
    _CONSISTENCY_REASON_REFUTATION_SUFFIX = re.compile(
        r"^\s*(?:itself\s+)?(?:is|was|would be|remains)?\s*"
        r"(?:false|wrong|incorrect|rejected|not true|unsupported)\b",
        flags=re.I,
    )

    def __init__(self, llm_model: Optional[str] = None, max_llm_calls: int = 0, logger=None,
                 enable_cache: bool = True, llm_temperature: float = 0.1,
                 llm_max_output_tokens: int = 2048,
                 rules: Optional[List[str]] = None,
                 rule_translations_path: str = "rule_translations.json",
                 llm_symbol_extraction: bool = False,
                 rule_mode: str = 'srd',
                 use_symbol_graph: bool = True,
                 checker_mode: str = "legacy",
                 checker_json_attempts: int = 2,
                 checker_min_confidence: float = 0.8) -> None:
        self.llm_model = llm_model
        self.max_llm_calls = int(max_llm_calls)
        self.logger = logger
        self._llm = None
        self._llm_calls_used = 0
        self.llm_temperature = float(llm_temperature)
        self.llm_max_output_tokens = int(llm_max_output_tokens)
        self.llm_symbol_extraction = bool(llm_symbol_extraction)
        self.use_symbol_graph = bool(use_symbol_graph)
        self.rule_mode = rule_mode
        self.checker_mode = str(checker_mode or self.CHECKER_MODE_LEGACY).strip().lower()
        if self.checker_mode not in self.CHECKER_MODES:
            raise ValueError(
                "checker_mode must be one of: " + ", ".join(sorted(self.CHECKER_MODES))
            )
        if isinstance(checker_json_attempts, bool):
            raise ValueError("checker_json_attempts must be an integer between 1 and 5")
        self.checker_json_attempts = int(checker_json_attempts)
        if not 1 <= self.checker_json_attempts <= 5:
            raise ValueError("checker_json_attempts must be an integer between 1 and 5")
        if isinstance(checker_min_confidence, bool):
            raise ValueError("checker_min_confidence must be between 0.0 and 1.0")
        self.checker_min_confidence = float(checker_min_confidence)
        if not math.isfinite(self.checker_min_confidence) or not 0.0 <= self.checker_min_confidence <= 1.0:
            raise ValueError("checker_min_confidence must be between 0.0 and 1.0")
        self.llm_trace_path = str(os.getenv("PHYSICSVERIFIER_LLM_TRACE_PATH") or "").strip()
        self.llm_trace_include_prompts = str(os.getenv("PHYSICSVERIFIER_LLM_TRACE_INCLUDE_PROMPTS") or "").strip().lower() in {"1", "true", "yes"}
        self._http_llm_enabled = False
        _timeout_env = str(os.getenv("PHYSICSVERIFIER_LLM_TIMEOUT_SEC") or "").strip()
        try:
            self.llm_timeout_sec = float(_timeout_env) if _timeout_env else None
        except Exception:
            self.llm_timeout_sec = None
        _retries_env = str(os.getenv("PHYSICSVERIFIER_LLM_MAX_RETRIES") or "").strip()
        try:
            self.llm_max_retries = int(_retries_env) if _retries_env else None
        except Exception:
            self.llm_max_retries = None
        
        if rules is None:
            self.rules_to_check = list(_BUILTIN_RULES_MAP.keys())
        else:
            self.rules_to_check = list(rules)
            
        self.rule_translations = {}
        if self.rule_mode == 'srd':
            self.rule_translations = self._load_rule_translations(rule_translations_path)
        else:
            self.rule_translations = self._load_direct_rule_descriptions()
            self._log("Running in 'direct' rule mode. Using raw rule descriptions as prompts.")

        self.enable_cache = bool(enable_cache)
        self._cache: Dict[str, Any] = {}
        try:
            base_dir = Path(__file__).parent
        except Exception:
            base_dir = Path(".").resolve()
        self._cache_path = (base_dir / ".cache" / "semantic_llm_cache.json").resolve()
        if self.enable_cache:
            self._load_cache()

        if self.llm_model and openai:
            _load_env_file_fallback()
            try:
                base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
                client_kwargs: Dict[str, Any] = {"base_url": base_url}
                if self.llm_max_retries is not None:
                    client_kwargs["max_retries"] = self.llm_max_retries
                self._llm = openai.OpenAI(**client_kwargs)
                if not getattr(self._llm, "api_key", None):
                    raise ValueError("OPENAI_API_KEY is not set")
                self._log(f"OpenAI client enabled for model: {self.llm_model}")
            except Exception as e:
                self._llm = None
                print(f"[Verifier] Failed to initialize OpenAI client: {e}")
        elif self.llm_model:
            _load_env_file_fallback()
            if os.getenv("OPENAI_API_KEY"):
                self._http_llm_enabled = True
                self._log(f"HTTP LLM fallback enabled for model: {self.llm_model}")

    def _load_rule_translations(self, path: str) -> dict:
        trans_path = Path(path)
        if not trans_path.exists():
            self._log(f"Warning: Rule translations file not found at '{path}'. LLM checks will be skipped.")
            return {}
        with trans_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_direct_rule_descriptions(self) -> dict:
        descriptions = {}
        for rule_id, spec in _BUILTIN_RULES_MAP.items():
            try:
                rule_class = _load_rule_class(spec)
                rule_instance = rule_class()
                descriptions[rule_id] = {
                    "title": getattr(rule_instance, "title", rule_id),
                    "description": getattr(rule_instance, "description", ""),
                    "srd": getattr(rule_instance, "description", ""),
                }
            except Exception as exc:
                self._log(f"Failed to load rule '{rule_id}' for direct mode: {exc}")
        return descriptions

    # ------------------------- 日志/缓存/LLM 工具 (简化) -------------------------
    def _log(self, *args):
        if self.logger:
            self.logger.info(" ".join(map(str, args)))

    def _load_cache(self):
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            if self._cache_path.exists():
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            self._cache = {}

    def _cache_key(self, payload: Any) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, namespace: str, payload: Any) -> Optional[Any]:
        if not self.enable_cache: return None
        key = f"{namespace}:{self._cache_key(payload)}"
        return self._cache.get(key)

    def _cache_set(self, namespace: str, payload: Any, value: Any):
        if not self.enable_cache: return
        key = f"{namespace}:{self._cache_key(payload)}"
        self._cache[key] = value
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=self._cache_path.parent, encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=None)
            os.replace(f.name, str(self._cache_path))
        except Exception:
            pass

    def _cache_delete(self, namespace: str, payload: Any) -> None:
        if not self.enable_cache:
            return
        key = f"{namespace}:{self._cache_key(payload)}"
        if key not in self._cache:
            return
        self._cache.pop(key, None)
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, dir=self._cache_path.parent, encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=None)
            os.replace(f.name, str(self._cache_path))
        except Exception:
            pass

    def _llm_available(self) -> bool:
        if self._llm is None and not self._http_llm_enabled:
            return False
        if self.max_llm_calls <= 0:
            return True
        return self._llm_calls_used < self.max_llm_calls

    def _llm_json_http(self, messages: List[Dict[str, str]]) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        payload.update(_openai_disable_thinking_kwargs())
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.llm_timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"] or "")

    def _append_llm_trace(self, record: Dict[str, Any]) -> None:
        if not self.llm_trace_path:
            return
        try:
            trace_path = Path(self.llm_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback=None,
        trace_meta: Optional[Dict[str, Any]] = None,
        return_meta: bool = False,
        json_validator: Optional[Any] = None,
    ) -> Any:
        fallback_value = fallback if fallback is not None else []

        def _result(
            *,
            ok: bool,
            status: str,
            data: Any,
            errors: Optional[List[str]] = None,
            cache_hit: bool = False,
            attempt_count: int = 1,
        ) -> Any:
            if not return_meta:
                return data
            return {
                "ok": bool(ok),
                "status": status,
                "data": data,
                "errors": list(errors or []),
                "cache_hit": bool(cache_hit),
                "attempt_count": int(attempt_count),
            }

        if not self._llm_available():
            return _result(
                ok=False,
                status="transport_failure",
                data=fallback_value,
                errors=["llm_unavailable_or_budget_exhausted"],
                attempt_count=0,
            )

        payload = {
            "prompt_version": SEMANTIC_RULE_CHECKER_PROMPT_VERSION,
            "checker_mode": self.checker_mode,
            "system": system_prompt,
            "user": user_prompt,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_output_tokens": self.llm_max_output_tokens,
            "retry_policy": "generic_error_category_v1",
        }
        cached = self._cache_get("llm_json", payload)
        if cached is not None:
            cache_validation_errors = (
                list(json_validator(cached) or []) if json_validator is not None else []
            )
            if not cache_validation_errors:
                return _result(
                    ok=True,
                    status="valid_json",
                    data=cached,
                    cache_hit=True,
                    attempt_count=0,
                )
            self._cache_delete("llm_json", payload)

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            if self._llm is not None:
                response = self._llm.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_output_tokens,
                    # Request JSON output from models that support it
                    response_format={"type": "json_object"},
                    timeout=self.llm_timeout_sec,
                    **_openai_disable_thinking_kwargs(),
                )
                resp = response.choices[0].message.content
            else:
                resp = self._llm_json_http(messages)
            self._llm_calls_used += 1

            trace_record = {
                "ts": datetime.datetime.now().isoformat(),
                "model": self.llm_model,
                "checker_mode": self.checker_mode,
                "trace_meta": trace_meta or {},
                "raw_response": resp,
                "raw_len": len(str(resp or "")),
            }
            if self.llm_trace_include_prompts:
                trace_record["system_prompt"] = system_prompt
                trace_record["user_prompt"] = user_prompt
            
            # The model should return a JSON string. We'll try to parse it directly.
            # A regex search is kept as a fallback for models that might wrap the JSON in text.
            try:
                data = json.loads(resp)
                validation_errors = list(json_validator(data) or []) if json_validator is not None else []
                if validation_errors:
                    trace_record["parse_status"] = "schema_failure"
                    trace_record["schema_errors"] = validation_errors
                    self._append_llm_trace(trace_record)
                    return _result(
                        ok=False,
                        status="schema_failure",
                        data=fallback_value,
                        errors=validation_errors,
                    )
                trace_record["parse_status"] = "json.loads_ok"
                self._append_llm_trace(trace_record)
                self._cache_set("llm_json", payload, data)
                return _result(ok=True, status="valid_json", data=data)
            except (json.JSONDecodeError, TypeError, ValueError):
                match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", str(resp or ""))
                if match:
                    try:
                        data = json.loads(match.group(1))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        data = None
                    else:
                        validation_errors = (
                            list(json_validator(data) or []) if json_validator is not None else []
                        )
                        if validation_errors:
                            trace_record["parse_status"] = "schema_failure"
                            trace_record["schema_errors"] = validation_errors
                            self._append_llm_trace(trace_record)
                            return _result(
                                ok=False,
                                status="schema_failure",
                                data=fallback_value,
                                errors=validation_errors,
                            )
                        trace_record["parse_status"] = "regex_extract_ok"
                        self._append_llm_trace(trace_record)
                        self._cache_set("llm_json", payload, data)
                        return _result(ok=True, status="valid_json", data=data)
            
            trace_record["parse_status"] = "parse_failed"
            self._append_llm_trace(trace_record)
            self._log(f"LLM response could not be parsed as JSON: {resp}")
            return _result(
                ok=False,
                status="parse_failure",
                data=fallback_value,
                errors=["response_is_not_valid_json"],
            )
        except Exception as e:
            self._append_llm_trace(
                {
                    "ts": datetime.datetime.now().isoformat(),
                    "model": self.llm_model,
                    "checker_mode": self.checker_mode,
                    "trace_meta": trace_meta or {},
                    "parse_status": "exception",
                    "exception": f"{type(e).__name__}: {e}",
                }
            )
            self._log(f"LLM call failed: {e}")
            return _result(
                ok=False,
                status="transport_failure",
                data=fallback_value,
                errors=[f"{type(e).__name__}: {e}"],
            )

    def _request_json_object_text(self, system_prompt: str, user_prompt: str) -> str:
        """Request one raw JSON-object response for the strict checker modes.

        This deliberately does not share the legacy parser: a transport failure,
        malformed JSON, and a schema failure must remain distinguishable from a
        valid response whose ``diagnostics`` list is empty.
        """
        if not self._llm_available():
            raise RuntimeError("LLM is unavailable or the call budget is exhausted")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Count attempted calls, including failed transports, so a failing endpoint
        # cannot bypass ``max_llm_calls`` through retries.
        self._llm_calls_used += 1
        if self._llm is not None:
            response = self._llm.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_output_tokens,
                response_format={"type": "json_object"},
                timeout=self.llm_timeout_sec,
                **_openai_disable_thinking_kwargs(),
            )
            return str(response.choices[0].message.content or "")
        return self._llm_json_http(messages)

    @staticmethod
    def _strict_json_object_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    @staticmethod
    def _reject_nonfinite_json_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number: {value}")

    @staticmethod
    def _is_strict_number(value: Any) -> bool:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            return False

    @staticmethod
    def _exact_key_errors(value: Dict[str, Any], expected: Set[str], path: str) -> List[str]:
        actual = set(value.keys())
        errors: List[str] = []
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{path}:missing_keys:{','.join(missing)}")
        if extra:
            errors.append(f"{path}:unexpected_keys:{','.join(extra)}")
        return errors

    def _validate_dual_evidence_schema(
        self,
        payload: Any,
        *,
        expected_rule_id: str,
    ) -> List[str]:
        """Validate the exact, single-object dual-evidence wire schema."""
        if not isinstance(payload, dict):
            return ["root:expected_object"]

        errors = self._exact_key_errors(
            payload,
            {"schema_version", "rule_id", "status", "applicability", "diagnostics"},
            "root",
        )
        if payload.get("schema_version") != self.DUAL_EVIDENCE_SCHEMA_VERSION:
            errors.append("root:invalid_schema_version")
        if not isinstance(payload.get("rule_id"), str) or payload.get("rule_id") != expected_rule_id:
            errors.append("root:rule_id_mismatch")
        status = payload.get("status")
        if not isinstance(status, str) or status not in self._DUAL_ROOT_STATUSES:
            errors.append("root:invalid_status")

        def _validate_confidence(value: Any, path: str) -> None:
            if not self._is_strict_number(value) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{path}:invalid_confidence")

        def _validate_evidence_list(
            value: Any,
            path: str,
            *,
            allowed_sources: Set[str],
        ) -> None:
            if not isinstance(value, list):
                errors.append(f"{path}:expected_list")
                return
            for index, evidence in enumerate(value):
                item_path = f"{path}[{index}]"
                if not isinstance(evidence, dict):
                    errors.append(f"{item_path}:expected_object")
                    continue
                errors.extend(
                    self._exact_key_errors(evidence, {"source", "quote", "location"}, item_path)
                )
                source = evidence.get("source")
                if not isinstance(source, str) or source not in self._DUAL_EVIDENCE_SOURCES:
                    errors.append(f"{item_path}:invalid_source")
                elif source not in allowed_sources:
                    errors.append(f"{item_path}:wrong_source_for_role")
                quote = evidence.get("quote")
                if not isinstance(quote, str) or not quote.strip():
                    errors.append(f"{item_path}:invalid_quote")
                location = evidence.get("location")
                if not isinstance(location, dict):
                    errors.append(f"{item_path}.location:expected_object")
                    continue
                errors.extend(
                    self._exact_key_errors(location, {"start_char", "end_char"}, f"{item_path}.location")
                )
                for field_name in ("start_char", "end_char"):
                    field_value = location.get(field_name)
                    if not isinstance(field_value, int) or isinstance(field_value, bool):
                        errors.append(f"{item_path}.location:{field_name}_must_be_integer")

        applicability = payload.get("applicability")
        if not isinstance(applicability, dict):
            errors.append("applicability:expected_object")
        else:
            errors.extend(
                self._exact_key_errors(
                    applicability,
                    {"applies", "confidence", "evidence"},
                    "applicability",
                )
            )
            if type(applicability.get("applies")) is not bool:
                errors.append("applicability:applies_must_be_boolean")
            _validate_confidence(applicability.get("confidence"), "applicability")
            _validate_evidence_list(
                applicability.get("evidence"),
                "applicability.evidence",
                allowed_sources={"question", "context"},
            )

        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, list):
            errors.append("diagnostics:expected_list")
            diagnostics = []
        elif status in {"no_violation", "abstain"} and diagnostics:
            errors.append("diagnostics:must_be_empty_for_non_violation_status")
        elif status == "violation" and not diagnostics:
            errors.append("diagnostics:required_for_violation_status")

        diagnostic_keys = {"severity", "symbol", "message", "violation"}
        consistency_required = self.checker_mode == self.CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY
        if consistency_required:
            diagnostic_keys.add("consistency")

        for index, diagnostic in enumerate(diagnostics):
            path = f"diagnostics[{index}]"
            if not isinstance(diagnostic, dict):
                errors.append(f"{path}:expected_object")
                continue
            errors.extend(self._exact_key_errors(diagnostic, diagnostic_keys, path))
            severity = diagnostic.get("severity")
            if not isinstance(severity, str) or severity not in self._DUAL_SEVERITIES:
                errors.append(f"{path}:invalid_severity")
            symbol = diagnostic.get("symbol")
            if symbol is not None and not isinstance(symbol, str):
                errors.append(f"{path}:symbol_must_be_string_or_null")
            message = diagnostic.get("message")
            if not isinstance(message, str) or not message.strip():
                errors.append(f"{path}:invalid_message")

            violation = diagnostic.get("violation")
            if not isinstance(violation, dict):
                errors.append(f"{path}.violation:expected_object")
            else:
                errors.extend(
                    self._exact_key_errors(
                        violation,
                        {"present", "confidence", "evidence"},
                        f"{path}.violation",
                    )
                )
                if type(violation.get("present")) is not bool:
                    errors.append(f"{path}.violation:present_must_be_boolean")
                _validate_confidence(violation.get("confidence"), f"{path}.violation")
                _validate_evidence_list(
                    violation.get("evidence"),
                    f"{path}.violation.evidence",
                    allowed_sources={"prediction"},
                )

            if consistency_required:
                consistency = diagnostic.get("consistency")
                if not isinstance(consistency, dict):
                    errors.append(f"{path}.consistency:expected_object")
                else:
                    errors.extend(
                        self._exact_key_errors(
                            consistency,
                            {"status", "confidence", "reason"},
                            f"{path}.consistency",
                        )
                    )
                    consistency_status = consistency.get("status")
                    if (
                        not isinstance(consistency_status, str)
                        or consistency_status not in self._DUAL_CONSISTENCY_STATUSES
                    ):
                        errors.append(f"{path}.consistency:invalid_status")
                    _validate_confidence(consistency.get("confidence"), f"{path}.consistency")
                    reason = consistency.get("reason")
                    if not isinstance(reason, str) or not reason.strip():
                        errors.append(f"{path}.consistency:reason_must_be_nonempty_string")
        return errors

    def _call_dual_evidence_json_object(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        expected_rule_id: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a validated object or an explicit transport/parse/schema failure."""
        cache_payload = {
            "prompt_version": SEMANTIC_RULE_CHECKER_PROMPT_VERSION,
            "schema_version": self.DUAL_EVIDENCE_SCHEMA_VERSION,
            "checker_mode": self.checker_mode,
            "system": system_prompt,
            "user": user_prompt,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_output_tokens": self.llm_max_output_tokens,
            "retry_policy": "generic_error_category_v1",
        }
        cached = self._cache_get("dual_evidence_json_object", cache_payload)
        if cached is not None:
            cache_errors = self._validate_dual_evidence_schema(
                cached,
                expected_rule_id=expected_rule_id,
            )
            if not cache_errors:
                return {
                    "ok": True,
                    "status": "valid_object",
                    "data": cached,
                    "attempt_count": 0,
                    "attempts": [{"attempt": 0, "status": "cache_hit"}],
                    "cache_hit": True,
                }
            self._cache_delete("dual_evidence_json_object", cache_payload)

        attempts: List[Dict[str, Any]] = []
        failure_status = "transport_failure"
        failure_errors: List[str] = []
        for attempt_index in range(1, self.checker_json_attempts + 1):
            retry_category = ""
            attempt_user_prompt = user_prompt
            if attempt_index > 1:
                previous_status = str((attempts[-1] if attempts else {}).get("status") or "")
                if previous_status == "parse_failure":
                    retry_category = "parse"
                elif previous_status == "schema_failure":
                    retry_category = "schema"
                else:
                    retry_category = "response"
                attempt_user_prompt = (
                    user_prompt
                    + "\n\nRETRY INSTRUCTION: The previous response had a generic "
                    + retry_category
                    + " error. Re-evaluate independently and return exactly one JSON object "
                    "matching the stated schema. Do not add prose or keys."
                )
            trace_record: Dict[str, Any] = {
                "ts": datetime.datetime.now().isoformat(),
                "model": self.llm_model,
                "checker_mode": self.checker_mode,
                "trace_meta": {**(trace_meta or {}), "attempt": attempt_index},
                "retry_category": retry_category,
            }
            if self.llm_trace_include_prompts:
                trace_record["system_prompt"] = system_prompt
                trace_record["user_prompt"] = attempt_user_prompt
            try:
                raw_response = self._request_json_object_text(system_prompt, attempt_user_prompt)
            except Exception as exc:
                failure_status = "transport_failure"
                failure_errors = [f"{type(exc).__name__}: {exc}"]
                attempt_record = {
                    "attempt": attempt_index,
                    "status": failure_status,
                    "errors": list(failure_errors),
                    "retry_category": retry_category,
                }
                attempts.append(attempt_record)
                trace_record.update(
                    {
                        "parse_status": failure_status,
                        "exception": failure_errors[0],
                    }
                )
                self._append_llm_trace(trace_record)
                continue

            trace_record["raw_response"] = raw_response
            trace_record["raw_len"] = len(raw_response)
            try:
                parsed = json.loads(
                    raw_response,
                    object_pairs_hook=self._strict_json_object_pairs,
                    parse_constant=self._reject_nonfinite_json_constant,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                failure_status = "parse_failure"
                failure_errors = [f"{type(exc).__name__}: {exc}"]
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "status": failure_status,
                        "errors": list(failure_errors),
                        "retry_category": retry_category,
                    }
                )
                trace_record["parse_status"] = failure_status
                trace_record["parse_errors"] = list(failure_errors)
                self._append_llm_trace(trace_record)
                continue

            schema_errors = self._validate_dual_evidence_schema(
                parsed,
                expected_rule_id=expected_rule_id,
            )
            if schema_errors:
                failure_status = "schema_failure"
                failure_errors = list(schema_errors)
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "status": failure_status,
                        "errors": list(schema_errors),
                        "retry_category": retry_category,
                    }
                )
                trace_record["parse_status"] = failure_status
                trace_record["schema_errors"] = list(schema_errors)
                self._append_llm_trace(trace_record)
                continue

            attempts.append(
                {
                    "attempt": attempt_index,
                    "status": "valid_object",
                    "retry_category": retry_category,
                }
            )
            trace_record["parse_status"] = "valid_object"
            self._append_llm_trace(trace_record)
            self._cache_set("dual_evidence_json_object", cache_payload, parsed)
            return {
                "ok": True,
                "status": "valid_object",
                "data": parsed,
                "attempt_count": attempt_index,
                "attempts": attempts,
                "cache_hit": False,
            }

        return {
            "ok": False,
            "status": failure_status,
            "data": None,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "errors": failure_errors,
            "cache_hit": False,
        }

    # ------------------------- 符号/公式提取 (简化和增强) -------------------------
    _symbol_regex = re.compile(r"\\?([a-zA-Z][a-zA-Z0-9_]*)")
    _stop_words = {"the", "of", "a", "an", "is", "in", "on", "for", "with", "and", "or", "not", "sin", "cos", "tan", "log", "ln", "exp"}
    _answer_whitespace_re = re.compile(r"\s+")
    _answer_text_command_re = re.compile(r"\\text\{([^}]*)\}")

    def _looks_like_symbol(self, tok: str) -> bool:
        if not tok or len(tok) > 20: return False
        low = tok.lower()
        if low in self._stop_words: return False
        if re.fullmatch(r"[a-z]+", tok) and len(tok) >= 4: return False
        return True

    def _extract_symbols_and_formulas(self, text: str) -> Dict[str, Any]:
        lines = (text or "").splitlines()
        symbols, symbol_set = [], set()
        formulas = []
        relation_re = re.compile(
            r"([A-Za-z0-9_\\{}()^+\-*/'., ]{0,100}"
            r"(?:=|<=|>=|<|>|≈|~|∝|≤|≥|\\leq|\\geq|\\le|\\ge)"
            r"[A-Za-z0-9_\\{}()^+\-*/'., ]{1,120})"
        )
        for line in lines:
            line = line.strip()
            if re.search(r'(=|≈|~|∝|<=|>=|<|>|≤|≥|\\leq|\\geq|\\le|\\ge)', line):
                extracted_any = False
                for m in relation_re.finditer(line):
                    candidate = m.group(1).strip(" .,;:")
                    if candidate:
                        formulas.append(candidate)
                        extracted_any = True
                if not extracted_any:
                    formulas.append(line)
            for m in self._symbol_regex.finditer(line):
                sym = m.group(1)
                if self._looks_like_symbol(sym) and sym not in symbol_set:
                    symbol_set.add(sym)
                    symbols.append(sym)
        return {"symbols": symbols, "formulas": formulas, "lines": lines}

    def _parse_formula_line(self, raw: str, line_idx: int) -> FormulaNode:
        raw = raw.strip()
        m = re.search(r'(=|≈|~|∝)', raw)
        relation, lhs, rhs = (m.group(1), *[p.strip() for p in raw.split(m.group(1), 1)]) if m else ("unknown", None, raw)

        def get_symbols(text: Optional[str]) -> Set[str]:
            if not text: return set()
            return {s for s in self._symbol_regex.findall(text) if self._looks_like_symbol(s)}

        lhs_symbols = get_symbols(lhs)
        rhs_symbols = get_symbols(rhs)
        all_symbols = list(lhs_symbols | rhs_symbols)

        return FormulaNode(
            fid=f"F{line_idx:04d}", raw=raw, relation=relation, lhs=lhs, rhs=rhs,
            symbols=all_symbols, line_index=line_idx,
            lhs_symbols=lhs_symbols, rhs_symbols=rhs_symbols
        )

    def _build_symbol_graph(self, lines: List[str], symbols: List[str], formulas: List[str]) -> SymbolGraph:
        graph = SymbolGraph()
        for i, raw in enumerate(lines):
            for s in symbols:
                if re.search(rf'\b{re.escape(s)}\b', raw):
                    graph.add_occurrence(s, i, raw)
        for i, f_raw in enumerate(formulas):
            fn = self._parse_formula_line(f_raw, i)
            graph.add_formula(fn.fid, fn)
        return graph

    def _create_context_summary(self, graph: SymbolGraph, text_all: str) -> str:
        """为LLM检查创建上下文摘要 (结构化信息)"""
        symbol_overview = []
        for name, node in list(graph.symbols.items())[:50]:
            symbol_overview.append({
                "name": name,
                "defined_count": len(node.defined_by),
                "used_in": node.used_in[:5],
                "occurrence_count": len(node.occurrences),
            })

        formula_overview = []
        for f in list(graph.formulas.values())[:20]:
            f_dict = {
                "fid": f.fid,
                "raw": f.raw,
                "relation": f.relation,
                "lhs": f.lhs,
                "rhs": f.rhs,
                "symbols": f.symbols,
                "line_index": f.line_index,
            }
            formula_overview.append(f_dict)

        summary = {
            "symbol_overview": symbol_overview,
            "formula_overview": formula_overview,
            "text_preview": text_all[:1200],
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    def _parse_expected_answers(self, sample: Dict[str, Any]) -> List[str]:
        answer_field = sample.get("answer")
        if answer_field in (None, ""):
            return []

        candidates: Any = answer_field
        if isinstance(answer_field, str):
            try:
                parsed = ast.literal_eval(answer_field)
                candidates = parsed
            except (ValueError, SyntaxError, TypeError):
                candidates = answer_field

        if isinstance(candidates, (list, tuple, set)):
            return [str(item) for item in candidates if item not in (None, "")]
        return [str(candidates)]

    def _normalize_answer_text(self, text: Any) -> str:
        if text in (None, ""):
            return ""
        cleaned = str(text)
        replacements = ["\\boxed", "\\left", "\\right", "$", "\\mathrm", "\\operatorname", "\\textbf", "\\mathit"]
        for token in replacements:
            cleaned = cleaned.replace(token, "")
        cleaned = self._answer_text_command_re.sub(r"\1", cleaned)
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = cleaned.lower()
        cleaned = self._answer_whitespace_re.sub("", cleaned)
        return cleaned

    def _answer_matches(self, sample: Dict[str, Any]) -> bool:
        expected_answers = [self._normalize_answer_text(a) for a in self._parse_expected_answers(sample)]
        expected_answers = [a for a in expected_answers if a]
        if not expected_answers:
            return False
        normalized_prediction = self._normalize_answer_text(sample.get("prediction", ""))
        if not normalized_prediction:
            return False
        return any(ans in normalized_prediction for ans in expected_answers)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _collapse_text_for_match(text: str) -> tuple[str, List[int]]:
        src = str(text or "")
        out: List[str] = []
        mapping: List[int] = []
        prev_space = False
        for i, ch in enumerate(src):
            c = ch
            if c in "{}[]()$`":
                continue
            if c == "\\":
                continue
            if c.isspace():
                if out and (not prev_space):
                    out.append(" ")
                    mapping.append(i)
                    prev_space = True
                continue
            out.append(c.lower())
            mapping.append(i)
            prev_space = False

        while out and out[0] == " ":
            out.pop(0)
            mapping.pop(0)
        while out and out[-1] == " ":
            out.pop()
            mapping.pop()
        return "".join(out), mapping

    def _locate_quote_span(self, answer_text: str, quote: str) -> Dict[str, Any]:
        src = str(answer_text or "")
        q = str(quote or "").strip()
        if not src or not q:
            return {
                "start_char": -1,
                "end_char": -1,
                "line_index": -1,
                "span_valid": False,
                "locate_method": "missing_quote",
                "locate_confidence": 0.0,
            }

        def _pack(start: int, end: int, method: str, confidence: float, ambiguous: bool) -> Dict[str, Any]:
            return {
                "start_char": int(start),
                "end_char": int(end),
                "line_index": int(src.count("\n", 0, start) + 1),
                "span_valid": True,
                "locate_method": method,
                "locate_confidence": float(confidence),
                "span_ambiguous": bool(ambiguous),
            }

        exact = list(re.finditer(re.escape(q), src))
        if exact:
            m0 = exact[0]
            return _pack(m0.start(), m0.end(), "exact", 1.0, len(exact) > 1)

        ci = list(re.finditer(re.escape(q), src, flags=re.I))
        if ci:
            m0 = ci[0]
            return _pack(m0.start(), m0.end(), "case_insensitive", 0.9, len(ci) > 1)

        parts = [re.escape(x) for x in re.split(r"\s+", q) if x]
        if parts:
            pat = r"\s+".join(parts)
            ws = list(re.finditer(pat, src, flags=re.I))
            if ws:
                m0 = ws[0]
                return _pack(m0.start(), m0.end(), "whitespace_fuzzy", 0.75, len(ws) > 1)

        src_norm, src_map = self._collapse_text_for_match(src)
        q_norm, _ = self._collapse_text_for_match(q)
        if src_norm and q_norm:
            normalized_positions: List[int] = []
            search_from = 0
            while True:
                k = src_norm.find(q_norm, search_from)
                if k < 0:
                    break
                normalized_positions.append(k)
                search_from = k + max(1, len(q_norm))
            if normalized_positions:
                k = normalized_positions[0]
                s = src_map[k]
                e = src_map[min(len(src_map) - 1, k + len(q_norm) - 1)] + 1
                return _pack(
                    s,
                    e,
                    "normalized_substring",
                    0.7,
                    len(normalized_positions) > 1,
                )

        return {
            "start_char": -1,
            "end_char": -1,
            "line_index": -1,
            "span_valid": False,
            "locate_method": "not_found",
            "locate_confidence": 0.0,
        }

    def _paragraph_ranges(self, answer_text: str) -> List[Dict[str, Any]]:
        src = str(answer_text or "")
        if not src:
            return []

        target_len = 220
        min_len = 120
        max_len = 360
        n = len(src)
        boundary_set = {0, n}
        for m in re.finditer(r"[。！？!?；;](?:\s+|$)|\n+", src):
            boundary_set.add(m.end())
        boundaries = sorted(boundary_set)

        out: List[Dict[str, Any]] = []
        start = 0
        para_idx = 0
        while start < n:
            if n - start <= max_len:
                end = n
            else:
                low = min(n, start + min_len)
                high = min(n, start + max_len)
                desired = min(n, start + target_len)
                candidates = [b for b in boundaries if low <= b <= high]
                if candidates:
                    end = min(candidates, key=lambda b: abs(b - desired))
                else:
                    end = high

            s = start
            e = max(start, end)
            while s < e and src[s].isspace():
                s += 1
            while e > s and src[e - 1].isspace():
                e -= 1

            if e > s:
                para_idx += 1
                out.append({"paragraph_index": para_idx, "start_char": s, "end_char": e})
            start = end if end > start else start + 1

        if not out and src.strip():
            out.append({"paragraph_index": 1, "start_char": 0, "end_char": len(src)})
        return out

    def _expand_span_to_context_window(
        self,
        answer_text: str,
        start_char: int,
        end_char: int,
        *,
        left_context: int = 90,
        right_context: int = 120,
        max_window: int = 320,
    ) -> Dict[str, int]:
        src = str(answer_text or "")
        n = len(src)
        if n <= 0 or start_char < 0 or end_char <= start_char:
            return {"start_char": -1, "end_char": -1}

        s = max(0, int(start_char) - left_context)
        e = min(n, int(end_char) + right_context)
        while s > 0 and (not src[s - 1].isspace()) and (int(start_char) - s) < (left_context + 50):
            s -= 1
        while e < n and (not src[e].isspace()) and (e - int(end_char)) < (right_context + 50):
            e += 1

        if e - s > max_window:
            mid = (int(start_char) + int(end_char)) // 2
            half = max_window // 2
            s = max(0, mid - half)
            e = min(n, s + max_window)

        while s < e and src[s].isspace():
            s += 1
        while e > s and src[e - 1].isspace():
            e -= 1
        return {"start_char": s if e > s else -1, "end_char": e if e > s else -1}

    def _paragraph_from_offset(self, paragraphs: List[Dict[str, Any]], offset: int) -> Optional[Dict[str, Any]]:
        if offset < 0:
            return None
        for p in paragraphs:
            s = int(p.get("start_char") or -1)
            e = int(p.get("end_char") or -1)
            if s <= offset < e:
                return p
        return None

    def _paragraph_by_index(self, paragraphs: List[Dict[str, Any]], paragraph_index: int) -> Optional[Dict[str, Any]]:
        if paragraph_index <= 0:
            return None
        for p in paragraphs:
            if int(p.get("paragraph_index") or -1) == paragraph_index:
                return p
        return None

    def _normalize_diagnostic_location(self, diagnostic: Dict[str, Any], answer_text: str) -> Dict[str, Any]:
        out = dict(diagnostic)
        paragraphs = self._paragraph_ranges(answer_text)

        ev_raw = out.get("evidence")
        if isinstance(ev_raw, dict):
            evidence = dict(ev_raw)
        elif isinstance(ev_raw, str) and ev_raw.strip():
            evidence = {"quote": ev_raw.strip()}
        else:
            evidence = {}

        quote = str(evidence.get("quote") or "").strip()
        loc_raw = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}

        start = self._safe_int(loc_raw.get("start_char"))
        end = self._safe_int(loc_raw.get("end_char"))
        line_index = self._safe_int(loc_raw.get("line_index"))
        source_text = str(answer_text or "")

        def _equivalent_quote_slice(candidate: str, expected: str) -> bool:
            if candidate == expected:
                return True
            candidate_norm, _ = self._collapse_text_for_match(candidate)
            expected_norm, _ = self._collapse_text_for_match(expected)
            return bool(candidate_norm and expected_norm and candidate_norm == expected_norm)

        span_valid = bool(
            quote
            and start is not None
            and end is not None
            and start >= 0
            and end > start
            and end <= len(source_text)
            and _equivalent_quote_slice(source_text[start:end], quote)
        )

        loc_obj: Dict[str, Any] = {
            "start_char": int(start) if start is not None else -1,
            "end_char": int(end) if end is not None else -1,
            "line_index": int(line_index) if line_index is not None else -1,
            "span_valid": span_valid,
            "span_ambiguous": False,
            "span_repaired": False,
            "locate_method": "model_span_verified" if span_valid else "model_span_invalid",
            "locate_confidence": float(loc_raw.get("locate_confidence") or (1.0 if span_valid else 0.0)),
            "paragraph_index": -1,
            "paragraph_start_char": -1,
            "paragraph_end_char": -1,
            "paragraph_valid": False,
            "paragraph_source": "",
        }

        if quote and not span_valid:
            fallback = self._locate_quote_span(answer_text, quote)
            if bool(fallback.get("span_valid")):
                loc_obj = {
                    "start_char": int(fallback.get("start_char", -1)),
                    "end_char": int(fallback.get("end_char", -1)),
                    "line_index": int(fallback.get("line_index", -1)),
                    "span_valid": True,
                    "span_ambiguous": bool(fallback.get("span_ambiguous")),
                    "span_repaired": True,
                    "locate_method": f"fallback_{fallback.get('locate_method') or 'quote_match'}",
                    "locate_confidence": float(fallback.get("locate_confidence") or 0.75),
                }

        if bool(loc_obj.get("span_valid")) and int(loc_obj.get("line_index") or -1) <= 0:
            s_value = self._safe_int(loc_obj.get("start_char"))
            s = int(s_value) if s_value is not None else -1
            if s >= 0:
                loc_obj["line_index"] = int(str(answer_text).count("\n", 0, s) + 1)

        if bool(loc_obj.get("span_valid")):
            span_start_value = self._safe_int(loc_obj.get("start_char"))
            span_end_value = self._safe_int(loc_obj.get("end_char"))
            span_start = int(span_start_value) if span_start_value is not None else -1
            span_end = int(span_end_value) if span_end_value is not None else -1
            p = self._paragraph_from_offset(paragraphs, span_start)
            if p is not None:
                ctx = self._expand_span_to_context_window(
                    answer_text,
                    span_start,
                    span_end,
                )
                loc_obj["paragraph_index"] = int(p.get("paragraph_index") or -1)
                ctx_start = self._safe_int(ctx.get("start_char"))
                ctx_end = self._safe_int(ctx.get("end_char"))
                p_start = self._safe_int(p.get("start_char"))
                p_end = self._safe_int(p.get("end_char"))
                loc_obj["paragraph_start_char"] = int(
                    ctx_start if ctx_start is not None and ctx_start >= 0 else (p_start if p_start is not None else -1)
                )
                loc_obj["paragraph_end_char"] = int(
                    ctx_end if ctx_end is not None and ctx_end >= 0 else (p_end if p_end is not None else -1)
                )
                loc_obj["paragraph_valid"] = True
                loc_obj["paragraph_source"] = "from_span_context"

        # A model-declared paragraph is never grounding by itself. Only a verified
        # quote span can make a legacy diagnostic locatable, and an ambiguous
        # fallback stays visible in trace while failing closed for publication.
        loc_obj["locatable_valid"] = bool(
            loc_obj.get("span_valid") and not loc_obj.get("span_ambiguous")
        )

        evidence["quote"] = quote
        evidence["location"] = loc_obj
        out["evidence"] = evidence
        return out

    @staticmethod
    def _ordered_unique_strings(items: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for item in items:
            value = str(item or "").strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def _normalize_strict_source_evidence(
        self,
        evidence: Dict[str, Any],
        *,
        source_texts: Dict[str, str],
        allowed_sources: Set[str],
        role: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Verify a quote against exactly one declared source.

        Model-provided offsets are accepted only when the exact source slice equals
        the quote. Otherwise, offsets are repaired only for one unique exact match.
        Multiple matches without a valid disambiguating span fail closed.
        """
        source = str(evidence.get("source") or "")
        if source not in allowed_sources:
            return None, f"{role}_evidence_wrong_source"
        source_text = str(source_texts.get(source) or "")
        quote = str(evidence.get("quote") or "").strip()
        if not quote:
            return None, f"{role}_evidence_missing_quote"

        location = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}
        start = location.get("start_char")
        end = location.get("end_char")
        span_verified = bool(
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(source_text)
            and source_text[start:end] == quote
        )
        repaired = False
        if not span_verified:
            matches = list(re.finditer(re.escape(quote), source_text))
            if not matches:
                return None, f"{role}_evidence_quote_not_found"
            if len(matches) != 1:
                return None, f"{role}_evidence_quote_ambiguous"
            start = matches[0].start()
            end = matches[0].end()
            repaired = True

        assert isinstance(start, int) and isinstance(end, int)
        paragraphs = self._paragraph_ranges(source_text)
        paragraph = self._paragraph_from_offset(paragraphs, start)
        paragraph_index = int(paragraph.get("paragraph_index") or -1) if paragraph else -1
        paragraph_start_value = self._safe_int(paragraph.get("start_char")) if paragraph else None
        paragraph_end_value = self._safe_int(paragraph.get("end_char")) if paragraph else None
        paragraph_start = int(paragraph_start_value) if paragraph_start_value is not None else -1
        paragraph_end = int(paragraph_end_value) if paragraph_end_value is not None else -1
        normalized = {
            "source": source,
            "quote": quote,
            "location": {
                "start_char": start,
                "end_char": end,
                "line_index": int(source_text.count("\n", 0, start) + 1),
                "span_valid": True,
                "span_ambiguous": False,
                "span_repaired": repaired,
                "locate_method": "unique_exact_repair" if repaired else "model_span_verified",
                "locate_confidence": 1.0,
                "paragraph_index": paragraph_index,
                "paragraph_start_char": paragraph_start,
                "paragraph_end_char": paragraph_end,
                "paragraph_valid": paragraph is not None,
                "paragraph_source": f"strict_{source}_span",
                "locatable_valid": True,
                "source": source,
            },
        }
        return normalized, None

    def _normalize_dual_evidence_response(
        self,
        payload: Dict[str, Any],
        *,
        expected_rule_id: str,
        source_texts: Dict[str, str],
    ) -> Dict[str, Any]:
        applicability_raw = payload["applicability"]
        valid_applicability_evidence: List[Dict[str, Any]] = []
        applicability_evidence_errors: List[str] = []
        for evidence in applicability_raw["evidence"]:
            normalized, error = self._normalize_strict_source_evidence(
                evidence,
                source_texts=source_texts,
                allowed_sources={"question", "context"},
                role="applicability",
            )
            if normalized is not None:
                valid_applicability_evidence.append(normalized)
            if error:
                applicability_evidence_errors.append(error)

        normalized_applicability = {
            "applies": applicability_raw["applies"],
            "confidence": float(applicability_raw["confidence"]),
            "evidence": valid_applicability_evidence,
        }
        base_gate_reasons = list(applicability_evidence_errors)
        if applicability_raw["applies"] is not True:
            base_gate_reasons.append("applicability_not_true")
        if float(applicability_raw["confidence"]) < self.checker_min_confidence:
            base_gate_reasons.append("applicability_below_confidence")
        if not valid_applicability_evidence:
            base_gate_reasons.append("missing_valid_applicability_evidence")

        emitted: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        for diagnostic_raw in payload["diagnostics"]:
            violation_raw = diagnostic_raw["violation"]
            valid_violation_evidence: List[Dict[str, Any]] = []
            violation_evidence_errors: List[str] = []
            for evidence in violation_raw["evidence"]:
                normalized, error = self._normalize_strict_source_evidence(
                    evidence,
                    source_texts=source_texts,
                    allowed_sources={"prediction"},
                    role="violation",
                )
                if normalized is not None:
                    valid_violation_evidence.append(normalized)
                if error:
                    violation_evidence_errors.append(error)

            reasons = list(base_gate_reasons) + violation_evidence_errors
            if payload["status"] != "violation":
                reasons.append("response_status_not_violation")
            if violation_raw["present"] is not True:
                reasons.append("violation_not_true")
            if float(violation_raw["confidence"]) < self.checker_min_confidence:
                reasons.append("violation_below_confidence")
            if not valid_violation_evidence:
                reasons.append("missing_valid_violation_evidence")

            consistency = diagnostic_raw.get("consistency")
            if self.checker_mode == self.CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY:
                consistency_status = str((consistency or {}).get("status") or "")
                consistency_confidence = float((consistency or {}).get("confidence") or 0.0)
                if consistency_status != "confirmed_violation":
                    reasons.append("consistency_not_confirmed")
                if consistency_confidence < self.checker_min_confidence:
                    reasons.append("consistency_below_confidence")
                if self._consistency_reason_rejects_confirmation(
                    str((consistency or {}).get("reason") or "")
                ):
                    reasons.append("consistency_reason_contradicts_status")

            normalized_violation = {
                "present": violation_raw["present"],
                "confidence": float(violation_raw["confidence"]),
                "evidence": valid_violation_evidence,
            }
            legacy_evidence = valid_violation_evidence[0] if valid_violation_evidence else {}
            candidate: Dict[str, Any] = {
                "severity": diagnostic_raw["severity"],
                "rule": expected_rule_id,
                "symbol": diagnostic_raw["symbol"],
                "message": diagnostic_raw["message"].strip(),
                "evidence": legacy_evidence,
                "applicability": normalized_applicability,
                "violation": normalized_violation,
                "checker_schema_version": self.DUAL_EVIDENCE_SCHEMA_VERSION,
                "checker_gate_mode": self.checker_mode,
            }
            if consistency is not None:
                candidate["consistency"] = {
                    "status": consistency["status"],
                    "confidence": float(consistency["confidence"]),
                    "reason": consistency["reason"],
                }
            gate_reasons = self._ordered_unique_strings(reasons)
            evidence_gate = {
                "passed": bool(not gate_reasons),
                "reasons": gate_reasons,
                "applicability_evidence_count": len(valid_applicability_evidence),
                "violation_evidence_count": len(valid_violation_evidence),
                "applicability_confidence": float(applicability_raw["confidence"]),
                "violation_confidence": float(violation_raw["confidence"]),
                "min_confidence": self.checker_min_confidence,
            }
            candidate["checker_evidence_gate"] = evidence_gate
            if evidence_gate["passed"] is True:
                emitted.append(candidate)
            else:
                suppressed.append(
                    {
                        "reason": "checker_evidence_gate",
                        "rule_id": expected_rule_id,
                        "checker_gate_mode": self.checker_mode,
                        "checker_evidence_gate": evidence_gate,
                        "original_diagnostic": candidate,
                    }
                )

        return {
            "diagnostics": emitted,
            "suppressed": suppressed,
            "applicability": normalized_applicability,
            "response_status": payload["status"],
        }

    @classmethod
    def _consistency_reason_rejects_confirmation(cls, reason: str) -> bool:
        text = str(reason or "").strip().lower()
        if not text:
            return False
        for pattern in cls._CONSISTENCY_REASON_REJECTION_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.I):
                # Do not suppress when the apparent affirmative phrase is the
                # proposition being explicitly refuted (for example,
                # "the derivation is correct is false").
                suffix = text[match.end(): match.end() + 40]
                if cls._CONSISTENCY_REASON_REFUTATION_SUFFIX.search(suffix):
                    continue
                return True
        return False

    @classmethod
    def _is_negative_or_uncertain_diagnostic(cls, diagnostic: Dict[str, Any]) -> bool:
        """Filter model explanations that explicitly say the rule was not violated.

        Some LLMs answer with a diagnostic-shaped object even when they conclude
        that a rule is irrelevant or untriggered. Those objects should not become
        user-facing error findings.
        """
        if not isinstance(diagnostic, dict):
            return False
        severity = str(diagnostic.get("severity") or "").strip().lower()
        if severity == "info":
            return True

        text_parts = [
            str(diagnostic.get("message") or ""),
            str(diagnostic.get("symbol") or ""),
        ]
        evidence = diagnostic.get("evidence")
        if isinstance(evidence, dict):
            text_parts.append(str(evidence.get("quote") or ""))
        elif isinstance(evidence, str):
            text_parts.append(evidence)
        text = " ".join(text_parts).strip().lower()
        if not text:
            return True
        return any(re.search(pattern, text, flags=re.I) for pattern in cls._NEGATIVE_DIAGNOSTIC_PATTERNS)

    # ------------------------- 新的LLM驱动的规则检查 -------------------------
    def _get_check_prompt(
        self,
        srd: str,
        raw_answer: str,
        problem_text: str,
        context_summary: str,
        rule_id: str,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are an expert physics grader. Your task is to check a student's answer "
            "against a specific, formal rule and report any violations in a structured JSON format. "
            "You must be conservative: it is better to miss a subtle issue than to incorrectly mark a correct solution as wrong. "
            "Output ONLY a valid JSON object (for a single violation) or a JSON array (for multiple violations). "
            "If no violations are found, output an empty array `[]`."
        )
        trimmed_answer = raw_answer.strip()
        trimmed_problem = problem_text.strip()
        # 适当放宽截断上限，保留更多原始作答内容
        max_chars = 12000
        if len(trimmed_answer) > max_chars:
            trimmed_answer = trimmed_answer[:max_chars] + "\n...[truncated]"
        if len(trimmed_problem) > max_chars:
            trimmed_problem = trimmed_problem[:max_chars] + "\n...[truncated]"
        if self.rule_mode == "direct":
            rule_block = f"Rule Description:\n{srd.strip()}"
        else:
            rule_block = f"You must enforce the following Symbolic Rule Definition (SRD):\n```\n{srd.strip()}\n```"

        user_prompt = f"""
{rule_block}

The physics problem being answered (verbatim text):
---
{trimmed_problem}
---

The student's submission (verbatim text):
---
{trimmed_answer}
---

Structured extraction summary (JSON helpers, may be incomplete):
{context_summary}

Instructions:
1. First rely on the raw text to understand the student's reasoning.
2. Use the structured summary only as a helper to locate symbols, equations, and counts; it may be incomplete or noisy.
3. Treat the SRD as a conditional diagnostic aid, not as a mandatory solution method or grading rubric:
    - First verify that the rule's physical scenario and preconditions apply to this exact problem.
    - Do NOT penalize an answer merely for omitting a derivation, method, quantity, or topic mentioned by the SRD.
    - An alternative derivation is acceptable when it answers what the problem asks and is physically sound.
    - Do not mention "SRD", "catalog", or "rule requirement" in the diagnostic message.
4. Only flag a violation if all of the following are true:
    - You can quote at least one concrete sentence or formula from the student's text that clearly contradicts the rule.
    - That quote cannot be reasonably interpreted as correct, harmless, or unrelated to this rule.
    - You are at least 80% confident that a real violation exists.
5. Distinguish severity:
    - Use "error" ONLY for clear, undeniable violations with strong direct evidence.
    - Use "warning" ONLY when there is strong indication of a problem but some minor uncertainty remains.
    - If you are not sure (for example, the text is ambiguous, the context is missing, or the rule's preconditions may not hold), you MUST treat the solution as compliant for this rule and return [].
6. Do NOT output explanatory "no violation", "not triggered", "not applicable", or "unrelated" diagnostics. In all such cases return [].
7. It is acceptable to miss some subtle issues. It is NOT acceptable to invent problems or penalize a solution that could reasonably be correct.

JSON Output Schema:
[
    {{
        "severity": "error" | "warning" | "info",
        "rule": "{rule_id}",
        "symbol": "symbol_or_equation_identifier",
        "message": "Short human-readable explanation",
        "evidence": {{
            "quote": "direct quote or formula from student's text",
            "location": {{
                "start_char": 0,
                "end_char": 10,
                "line_index": 1,
                "paragraph_index": 1
            }}
        }}
    }}
]

Location requirement:
- start_char/end_char are 0-based offsets in the student's submission text above.
- If exact offsets are uncertain, still provide quote; system will fallback-locate by quote.
- paragraph_index is 1-based approximate paragraph index. If unsure, use -1.

Respond with only the JSON output (array or empty array).
"""
        return system_prompt, user_prompt

    def _get_dual_evidence_prompt(
        self,
        *,
        srd: str,
        raw_answer: str,
        question_text: str,
        context_text: str,
        rule_id: str,
    ) -> Tuple[str, str]:
        """Build the source-separated prompt used by both strict checker modes."""
        max_chars = 12000

        def _trim(value: str) -> str:
            # Preserve source coordinates exactly. Stripping leading whitespace
            # would make model-provided offsets disagree with validation text.
            text = str(value or "")
            if len(text) > max_chars:
                return text[:max_chars] + "\n...[truncated]"
            return text

        question = _trim(question_text)
        context = _trim(context_text)
        prediction = _trim(raw_answer)
        if self.rule_mode == "direct":
            rule_block = f"Rule Description:\n{srd.strip()}"
        else:
            rule_block = f"Conditional Rule Definition:\n---\n{srd.strip()}\n---"

        diagnostic_example: Dict[str, Any] = {
            "severity": "error",
            "symbol": None,
            "message": "Concise explanation of the concrete contradiction.",
            "violation": {
                "present": True,
                "confidence": 0.95,
                "evidence": [
                    {
                        "source": "prediction",
                        "quote": "Exact quote from the student's submission.",
                        "location": {"start_char": -1, "end_char": -1},
                    }
                ],
            },
        }
        consistency_instructions = ""
        if self.checker_mode == self.CHECKER_MODE_DUAL_EVIDENCE_CONSISTENCY:
            diagnostic_example["consistency"] = {
                "status": "confirmed_violation",
                "confidence": 0.95,
                "reason": "The quoted claim remains asserted and is not corrected or equivalent.",
            }
            consistency_instructions = """
Consistency requirement for every diagnostic:
- Use `confirmed_violation` only when the quoted claim is the student's current conclusion.
- Use `self_corrected` when the student later withdraws or corrects it.
- Use `equivalent_or_alternative` when the reasoning is physically equivalent or a valid alternative.
- Use `uncertain` for a hypothetical, rejected example, ambiguity, or insufficient context.
- Only `confirmed_violation` is publishable.
"""

        schema_example = {
            "schema_version": self.DUAL_EVIDENCE_SCHEMA_VERSION,
            "rule_id": rule_id,
            "status": "violation",
            "applicability": {
                "applies": True,
                "confidence": 0.95,
                "evidence": [
                    {
                        "source": "question",
                        "quote": "Exact quote from QUESTION or CONTEXT establishing applicability.",
                        "location": {"start_char": -1, "end_char": -1},
                    }
                ],
            },
            "diagnostics": [diagnostic_example],
        }
        schema_json = json.dumps(schema_example, ensure_ascii=False, indent=2)
        confidence_percent = int(round(self.checker_min_confidence * 100))
        system_prompt = (
            "You are a conservative physics rule checker. Return exactly one JSON object "
            "matching the supplied schema. A bare array, prose, Markdown, unknown key, omitted "
            "key, wrong type, or different rule_id is invalid. Never use a reference answer or "
            "assume that the supplied rule is applicable."
        )
        user_prompt = f"""
{rule_block}

RULE_ID:
{rule_id}

QUESTION source (applicability evidence may quote only this field):
---
{question}
---

CONTEXT source (applicability evidence may quote only this field):
---
{context}
---

PREDICTION source (violation evidence may quote only this field):
---
{prediction}
---

Decision procedure:
1. Determine whether this rule's physical scenario and preconditions apply to the QUESTION/CONTEXT.
2. Applicability evidence must be an exact quote from QUESTION or CONTEXT. Never use PREDICTION as applicability evidence.
3. A violation must be a concrete contradiction still asserted in PREDICTION. Never use QUESTION or CONTEXT as violation evidence.
4. Omission of the rule's preferred method, a sound alternative derivation, or an equivalent formulation is not a violation.
5. If applicability or violation confidence is below {confidence_percent}%, abstain or return no violation.
6. If there is no publishable violation, set status to `no_violation` or `abstain` and diagnostics to an empty list.
7. Evidence offsets are 0-based within the single declared source. Use -1 for both offsets when uncertain; the system repairs only a unique exact quote.
8. Every listed key is required. Do not add keys. Use JSON booleans and numbers, not strings.
{consistency_instructions}
Exact JSON schema example:
{schema_json}

Return exactly one JSON object and nothing else.
"""
        return system_prompt, user_prompt

    @staticmethod
    def _validate_legacy_diagnostics_payload(
        payload: Any,
        *,
        expected_rule_id: str,
    ) -> Tuple[Optional[List[Any]], List[str]]:
        """Apply the smallest safe contract without changing the legacy wire shape."""
        if isinstance(payload, (dict, str)):
            diagnostics: List[Any] = [payload]
        elif isinstance(payload, list):
            diagnostics = list(payload)
        else:
            return None, ["legacy_root:expected_object_array_or_string"]

        errors: List[str] = []
        for index, diagnostic in enumerate(diagnostics):
            if not isinstance(diagnostic, (dict, str)):
                errors.append(f"legacy_diagnostics[{index}]:expected_object_or_string")
                continue
            if isinstance(diagnostic, dict):
                rule_id = diagnostic.get("rule")
                if not isinstance(rule_id, str) or rule_id != expected_rule_id:
                    errors.append(f"legacy_diagnostics[{index}]:rule_id_mismatch")
        if errors:
            return None, errors
        return diagnostics, []

    def analyze(self, sample: Dict[str, Any], dataset_key: Optional[str] = None, export_graph: bool = False) -> Dict[str, Any]:
        question_text = str(sample.get("question") or "")
        context_text = str(sample.get("context") or "")
        answer_text = str(sample.get("prediction") or "")
        text_all = "\n".join([question_text, context_text, answer_text])
        dual_mode = self.checker_mode != self.CHECKER_MODE_LEGACY

        # Strict modes must be independently safe when called outside
        # PhysicsRuleVerifier: they never inspect or use the reference ``answer``.
        answer_correct = False if dual_mode else self._answer_matches(sample)

        graph: Optional[SymbolGraph] = None
        context_summary: Optional[str] = None
        if self.use_symbol_graph and not answer_correct:
            parsed = self._extract_symbols_and_formulas(text_all)
            graph = self._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
            # The mixed summary remains legacy-only. Strict prompts receive the
            # three source fields separately and never receive this summary.
            if not dual_mode:
                context_summary = self._create_context_summary(graph, text_all)

        all_diagnostics: List[Dict[str, Any]] = []
        checker_decisions: List[Dict[str, Any]] = []
        checker_failures: List[Dict[str, Any]] = []
        checker_suppressed: List[Dict[str, Any]] = []
        legacy_ran = False

        if not answer_correct:
            for rule_id in self.rules_to_check:
                rule_info = self.rule_translations.get(rule_id)
                if not isinstance(rule_info, dict):
                    failure = {
                        "rule_id": str(rule_id or ""),
                        "checker_gate_mode": self.checker_mode,
                        "status": "configuration_failure",
                        "errors": ["missing_or_invalid_rule_translation"],
                        "attempt_count": 0,
                        "attempts": [],
                        "cache_hit": False,
                    }
                    if dual_mode:
                        failure["checker_schema_version"] = self.DUAL_EVIDENCE_SCHEMA_VERSION
                    checker_decisions.append(failure)
                    checker_failures.append(failure)
                    continue
                srd = rule_info.get("srd")
                if not isinstance(srd, str) or not srd.strip():
                    failure = {
                        "rule_id": str(rule_id or ""),
                        "checker_gate_mode": self.checker_mode,
                        "status": "configuration_failure",
                        "errors": ["missing_or_empty_rule_definition"],
                        "attempt_count": 0,
                        "attempts": [],
                        "cache_hit": False,
                    }
                    if dual_mode:
                        failure["checker_schema_version"] = self.DUAL_EVIDENCE_SCHEMA_VERSION
                    checker_decisions.append(failure)
                    checker_failures.append(failure)
                    continue

                if not dual_mode:
                    legacy_ran = True
                    system_prompt, user_prompt = self._get_check_prompt(
                        srd=srd,
                        raw_answer=answer_text,
                        problem_text="\n".join([question_text, context_text]),
                        context_summary=context_summary if context_summary is not None else "{}",
                        rule_id=rule_id,
                    )
                    call_result = self._llm_json(
                        system_prompt,
                        user_prompt,
                        fallback=[],
                        trace_meta={"sample_id": sample.get("id"), "rule_id": rule_id},
                        return_meta=True,
                        json_validator=lambda payload, expected=rule_id: (
                            self._validate_legacy_diagnostics_payload(
                                payload,
                                expected_rule_id=expected,
                            )[1]
                        ),
                    )
                    common_legacy_decision = {
                        "rule_id": rule_id,
                        "checker_gate_mode": self.CHECKER_MODE_LEGACY,
                        "attempt_count": int(call_result.get("attempt_count") or 0),
                        "attempts": [
                            {
                                "attempt": int(call_result.get("attempt_count") or 0),
                                "status": str(call_result.get("status") or "transport_failure"),
                            }
                        ],
                        "cache_hit": bool(call_result.get("cache_hit")),
                    }
                    if call_result.get("ok") is not True:
                        failure = {
                            **common_legacy_decision,
                            "status": str(call_result.get("status") or "transport_failure"),
                            "errors": list(call_result.get("errors") or []),
                        }
                        checker_decisions.append(failure)
                        checker_failures.append(failure)
                        continue

                    diagnostics, legacy_schema_errors = self._validate_legacy_diagnostics_payload(
                        call_result.get("data"),
                        expected_rule_id=rule_id,
                    )
                    if legacy_schema_errors:
                        failure = {
                            **common_legacy_decision,
                            "status": "schema_failure",
                            "errors": legacy_schema_errors,
                        }
                        checker_decisions.append(failure)
                        checker_failures.append(failure)
                        continue
                    assert diagnostics is not None

                    normalized_diag: List[Any] = []
                    for diagnostic in diagnostics:
                        if isinstance(diagnostic, dict):
                            if self._is_negative_or_uncertain_diagnostic(diagnostic):
                                checker_suppressed.append(
                                    {
                                        "reason": "legacy_negative_or_uncertain_diagnostic",
                                        "rule_id": rule_id,
                                        "checker_gate_mode": self.CHECKER_MODE_LEGACY,
                                        "original_diagnostic": diagnostic,
                                    }
                                )
                                continue
                            normalized_diag.append(
                                self._normalize_diagnostic_location(diagnostic, answer_text)
                            )
                        else:
                            normalized_diag.append(diagnostic)
                    all_diagnostics.extend(normalized_diag)
                    checker_decisions.append(
                        {
                            **common_legacy_decision,
                            "status": "valid_with_diagnostics" if normalized_diag else "valid_empty",
                            "published_diagnostic_count": len(normalized_diag),
                        }
                    )
                    continue

                system_prompt, user_prompt = self._get_dual_evidence_prompt(
                    srd=srd,
                    raw_answer=answer_text,
                    question_text=question_text,
                    context_text=context_text,
                    rule_id=rule_id,
                )
                call_result = self._call_dual_evidence_json_object(
                    system_prompt,
                    user_prompt,
                    expected_rule_id=rule_id,
                    trace_meta={"sample_id": sample.get("id"), "rule_id": rule_id},
                )
                common_decision = {
                    "rule_id": rule_id,
                    "checker_gate_mode": self.checker_mode,
                    "checker_schema_version": self.DUAL_EVIDENCE_SCHEMA_VERSION,
                    "attempt_count": int(call_result.get("attempt_count") or 0),
                    "attempts": list(call_result.get("attempts") or []),
                    "cache_hit": bool(call_result.get("cache_hit")),
                }
                if call_result.get("ok") is not True:
                    failure = {
                        **common_decision,
                        "status": str(call_result.get("status") or "transport_failure"),
                        "errors": list(call_result.get("errors") or []),
                    }
                    checker_decisions.append(failure)
                    checker_failures.append(failure)
                    continue

                normalized = self._normalize_dual_evidence_response(
                    call_result["data"],
                    expected_rule_id=rule_id,
                    source_texts={
                        "question": question_text,
                        "context": context_text,
                        "prediction": answer_text,
                    },
                )
                emitted = list(normalized["diagnostics"])
                suppressed = list(normalized["suppressed"])
                all_diagnostics.extend(emitted)
                checker_suppressed.extend(suppressed)
                checker_decisions.append(
                    {
                        **common_decision,
                        "status": "valid_with_diagnostics" if emitted else "valid_empty",
                        "response_status": normalized["response_status"],
                        "applicability": normalized["applicability"],
                        "published_diagnostic_count": len(emitted),
                        "suppressed_diagnostic_count": len(suppressed),
                    }
                )

        seen = set()
        unique_diagnostics: List[Dict[str, Any]] = []
        for diagnostic in all_diagnostics:
            if isinstance(diagnostic, str):
                key = (None, None, diagnostic)
                payload = {
                    "severity": "info",
                    "rule": None,
                    "symbol": None,
                    "message": diagnostic,
                }
            else:
                key = (
                    diagnostic.get("rule"),
                    diagnostic.get("symbol"),
                    diagnostic.get("message"),
                )
                payload = diagnostic
            if key in seen:
                if dual_mode:
                    checker_suppressed.append(
                        {
                            "reason": "duplicate_checker_diagnostic",
                            "rule_id": str(payload.get("rule") or ""),
                            "checker_gate_mode": self.checker_mode,
                            "original_diagnostic": payload,
                        }
                    )
                continue
            unique_diagnostics.append(payload)
            seen.add(key)

        score = sum(
            -1.0 if diagnostic.get("severity") == "error" else -0.5
            for diagnostic in unique_diagnostics
            if diagnostic.get("severity") in ["error", "warning"]
        )

        if dual_mode:
            successful_decisions = [
                decision
                for decision in checker_decisions
                if str(decision.get("status") or "").startswith("valid_")
            ]
            if checker_failures and successful_decisions:
                checker_status = "partial_failure"
            elif checker_failures:
                checker_status = "failed"
            elif not successful_decisions:
                checker_status = "not_run"
            elif unique_diagnostics:
                checker_status = "valid_with_diagnostics"
            else:
                checker_status = "valid_empty"
        elif legacy_ran or checker_decisions:
            successful_decisions = [
                decision
                for decision in checker_decisions
                if str(decision.get("status") or "").startswith("valid_")
            ]
            if checker_failures and successful_decisions:
                checker_status = "partial_failure"
            elif checker_failures:
                checker_status = "failed"
            else:
                checker_status = "valid_with_diagnostics" if unique_diagnostics else "valid_empty"
        else:
            checker_status = "not_run"

        out = {
            "id": sample.get("id"),
            "dataset": dataset_key,
            "diagnostics": unique_diagnostics,
            "score": score,
            "answer_correct": answer_correct,
            "checker_mode": self.checker_mode,
            "checker_status": checker_status,
            "checker_decisions": checker_decisions,
            "checker_failures": checker_failures,
            "checker_suppressed": checker_suppressed,
        }
        if export_graph and graph is not None:
            out["symbol_nodes"] = {k: vars(v) for k, v in graph.symbols.items()}
            out["formula_nodes"] = {k: vars(v) for k, v in graph.formulas.items()}
            out["graph_edges"] = graph.edges
        return out

    def analyze_batch(self, samples: List[Dict[str, Any]], dataset_key: Optional[str] = None, export_graph: bool = False) -> Dict[str, Any]:
        results = [self.analyze(s, dataset_key=dataset_key, export_graph=export_graph) for s in samples or []]
        total_score = sum(r.get("score", 0.0) for r in results)
        return {
            "summary": {
                "dataset": dataset_key,
                "num_samples": len(results),
                "total_score": total_score,
                "avg_score": (total_score / len(results)) if results else 0.0,
                "created_at": datetime.datetime.now().isoformat(),
            },
            "results": results,
        }


# ------------------------- 脚本运行 (重构为独立CLI) -------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Physics semantic (LLM+SRD) rule checker.")
    parser.add_argument("--input", "-i", type=str, default="data/evaluation_input.json",
                        help="Path to the input JSON file containing samples to verify.")
    parser.add_argument("--output", "-o", type=str, default="results/rule_check_report.json",
                        help="Path to save the output report JSON file.")
    parser.add_argument("--rules", nargs="+", default=None,
                        help=f"A list of rules to check. Defaults to all built-in rules: {list(_BUILTIN_RULES_MAP.keys())}")
    parser.add_argument("--llm-model", type=str, default="gpt-4o",
                        help="The LLM model to use for checking (e.g., 'gpt-4o', 'gpt-3.5-turbo').")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM-based checks entirely.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching for LLM calls.")
    parser.add_argument("--export-graph", action="store_true",
                        help="Export symbol and formula graphs in the output.")
    parser.add_argument("--output-mode", type=str, choices=['full_report', 'errors_only'], default='full_report',
                        help="Output mode: 'full_report' (default) or 'errors_only'.")
    parser.add_argument("--rule-mode", type=str, choices=['direct', 'srd'], default='srd',
                        help="Rule checking mode: 'srd' for symbolic rule definitions, 'direct' for raw descriptions.")
    parser.add_argument("--max-llm-calls", type=int, default=0,
                        help="Maximum total LLM calls (0 means unlimited).")
    
    if len(sys.argv) == 1:
        print("No arguments provided, running a simple demonstration.")
        # 示例：演示在无LLM或无翻译文件时如何优雅降级
        verifier = SemanticRuleChecker(llm_model=None, rules=["var_const_consistency"])
        sample = {
            "id": "demo1",
            "prediction": "Let v = 5. Later, v = 10. This is a self-reference v=v+1."
        }
        result = verifier.analyze(sample, export_graph=True)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        args = parser.parse_args()

        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            print(f"Error: Input file not found at '{input_path}'")
            sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with input_path.open("r", encoding="utf-8") as f:
                samples_to_check = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{input_path}'")
            sys.exit(1)

        print(f"Initializing verifier...")
        print(f"  - LLM Model: {'Disabled' if args.no_llm else args.llm_model}")
        print(f"  - Rules: {args.rules or 'All'}")
        print(f"  - Cache: {'Disabled' if args.no_cache else 'Enabled'}")

        verifier = SemanticRuleChecker(
            llm_model=None if args.no_llm else args.llm_model,
            rules=args.rules,
            enable_cache=not args.no_cache,
            max_llm_calls=args.max_llm_calls,
            rule_mode=args.rule_mode,
        )

        print(f"Analyzing {len(samples_to_check)} samples from '{input_path}'...")
        report = verifier.analyze_batch(
            samples_to_check, 
            dataset_key=input_path.stem, 
            export_graph=args.export_graph
        )

        # 根据输出模式决定最终要保存的内容
        if args.output_mode == 'errors_only':
            print("Filtering for samples with errors...")
            errors_found = []
            for i, result in enumerate(report['results']):
                if result.get('diagnostics'):  # 如果 diagnostics 列表不为空
                    original_sample = samples_to_check[i]
                    error_item = {
                        "id": original_sample.get('id'),
                        "question": original_sample.get('question'),
                        "prediction": original_sample.get('prediction'),
                        "answer": original_sample.get('answer'),
                        "diagnostics": result['diagnostics']
                    }
                    errors_found.append(error_item)
            
            output_data = errors_found
            print(f"Found {len(errors_found)} samples with errors.")
        else:
            output_data = report

        print(f"Analysis complete. Saving report to '{output_path}'...")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print("Done.")
