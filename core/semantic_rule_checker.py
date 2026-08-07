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


SEMANTIC_RULE_CHECKER_PROMPT_VERSION = "semantic-rule-checker-applicability-v1"


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

    def __init__(self, llm_model: Optional[str] = None, max_llm_calls: int = 0, logger=None,
                 enable_cache: bool = True, llm_temperature: float = 0.1,
                 llm_max_output_tokens: int = 2048,
                 rules: Optional[List[str]] = None,
                 rule_translations_path: str = "rule_translations.json",
                 llm_symbol_extraction: bool = False,
                 rule_mode: str = 'srd',
                 use_symbol_graph: bool = True) -> None:
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

    def _llm_json(self, system_prompt: str, user_prompt: str, fallback=None, trace_meta: Optional[Dict[str, Any]] = None) -> Any:
        if not self._llm_available():
            return fallback if fallback is not None else []
        
        payload = {"system": system_prompt, "user": user_prompt, "model": self.llm_model}
        cached = self._cache_get("llm_json", payload)
        if cached is not None:
            return cached

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
                trace_record["parse_status"] = "json.loads_ok"
                self._append_llm_trace(trace_record)
                self._cache_set("llm_json", payload, data)
                return data
            except json.JSONDecodeError:
                match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", resp)
                if match:
                    data = json.loads(match.group(1))
                    trace_record["parse_status"] = "regex_extract_ok"
                    self._append_llm_trace(trace_record)
                    self._cache_set("llm_json", payload, data)
                    return data
            
            trace_record["parse_status"] = "parse_failed"
            self._append_llm_trace(trace_record)
            self._log(f"LLM response could not be parsed as JSON: {resp}")
            return fallback if fallback is not None else []
        except Exception as e:
            self._append_llm_trace(
                {
                    "ts": datetime.datetime.now().isoformat(),
                    "model": self.llm_model,
                    "trace_meta": trace_meta or {},
                    "parse_status": "exception",
                    "exception": f"{type(e).__name__}: {e}",
                }
            )
            self._log(f"LLM call failed: {e}")
            return fallback if fallback is not None else []

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
            k = src_norm.find(q_norm)
            if k >= 0:
                s = src_map[k]
                e = src_map[min(len(src_map) - 1, k + len(q_norm) - 1)] + 1
                return _pack(s, e, "normalized_substring", 0.7, False)

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
        paragraph_index_raw = self._safe_int(loc_raw.get("paragraph_index"))
        span_valid = bool(start is not None and end is not None and start >= 0 and end > start)

        loc_obj: Dict[str, Any] = {
            "start_char": int(start) if start is not None else -1,
            "end_char": int(end) if end is not None else -1,
            "line_index": int(line_index) if line_index is not None else -1,
            "span_valid": span_valid,
            "locate_method": str(loc_raw.get("locate_method") or "model_provided"),
            "locate_confidence": float(loc_raw.get("locate_confidence") or (1.0 if span_valid else 0.0)),
            "paragraph_index": int(paragraph_index_raw) if paragraph_index_raw is not None else -1,
            "paragraph_start_char": int(self._safe_int(loc_raw.get("paragraph_start_char")) or -1),
            "paragraph_end_char": int(self._safe_int(loc_raw.get("paragraph_end_char")) or -1),
            "paragraph_valid": bool(paragraph_index_raw is not None and paragraph_index_raw >= 1),
            "paragraph_source": str(loc_raw.get("paragraph_source") or "model_provided"),
        }

        if quote and not span_valid:
            fallback = self._locate_quote_span(answer_text, quote)
            if bool(fallback.get("span_valid")):
                loc_obj = {
                    "start_char": int(fallback.get("start_char", -1)),
                    "end_char": int(fallback.get("end_char", -1)),
                    "line_index": int(fallback.get("line_index", -1)),
                    "span_valid": True,
                    "locate_method": f"fallback_{fallback.get('locate_method') or 'quote_match'}",
                    "locate_confidence": float(fallback.get("locate_confidence") or 0.75),
                }

        if bool(loc_obj.get("span_valid")) and int(loc_obj.get("line_index") or -1) <= 0:
            s = int(loc_obj.get("start_char") or -1)
            if s >= 0:
                loc_obj["line_index"] = int(str(answer_text).count("\n", 0, s) + 1)

        if bool(loc_obj.get("span_valid")) and (not bool(loc_obj.get("paragraph_valid"))):
            p = self._paragraph_from_offset(paragraphs, int(loc_obj.get("start_char") or -1))
            if p is not None:
                ctx = self._expand_span_to_context_window(
                    answer_text,
                    int(loc_obj.get("start_char") or -1),
                    int(loc_obj.get("end_char") or -1),
                )
                loc_obj["paragraph_index"] = int(p.get("paragraph_index") or -1)
                loc_obj["paragraph_start_char"] = int(ctx.get("start_char") or p.get("start_char") or -1)
                loc_obj["paragraph_end_char"] = int(ctx.get("end_char") or p.get("end_char") or -1)
                loc_obj["paragraph_valid"] = True
                loc_obj["paragraph_source"] = "from_span_context"

        if not bool(loc_obj.get("paragraph_valid")):
            pidx = int(loc_obj.get("paragraph_index") or -1)
            p2 = self._paragraph_by_index(paragraphs, pidx)
            if p2 is not None:
                loc_obj["paragraph_start_char"] = int(p2.get("start_char") or -1)
                loc_obj["paragraph_end_char"] = int(p2.get("end_char") or -1)
                loc_obj["paragraph_valid"] = True
                if not str(loc_obj.get("paragraph_source") or "").strip():
                    loc_obj["paragraph_source"] = "model_declared"

        loc_obj["locatable_valid"] = bool(loc_obj.get("span_valid") or loc_obj.get("paragraph_valid"))

        evidence["quote"] = quote
        evidence["location"] = loc_obj
        out["evidence"] = evidence
        return out

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

    def analyze(self, sample: Dict[str, Any], dataset_key: Optional[str] = None, export_graph: bool = False) -> Dict[str, Any]:
        # 使用完整回答，不再截断，让 LLM 看到全部作答
        answer_text = str(sample.get("prediction", ""))
        text_all = "\n".join([
            sample.get("question", ""),
            sample.get("context", ""),
            sample.get("prediction", ""),
        ])

        answer_correct = self._answer_matches(sample)

        graph: Optional[SymbolGraph] = None
        context_summary: Optional[str] = None

        if self.use_symbol_graph and not answer_correct:
            parsed = self._extract_symbols_and_formulas(text_all)
            graph = self._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
            context_summary = self._create_context_summary(graph, text_all)
        
        all_diagnostics: List[Dict[str, Any]] = []

        if (not answer_correct) and self._llm_available() and self.rule_translations:
            for rule_id in self.rules_to_check:
                rule_info = self.rule_translations.get(rule_id)
                if not rule_info or not rule_info.get("srd") or "failed" in rule_info["srd"].lower():
                    continue

                srd = rule_info["srd"]

                # 若不使用符号图，则只提供原始回答文本，不提供结构化 JSON
                if self.use_symbol_graph and context_summary is not None:
                    system_prompt, user_prompt = self._get_check_prompt(
                        srd=srd,
                        raw_answer=answer_text,
                        problem_text="\n".join(
                            [
                                str(sample.get("question") or ""),
                                str(sample.get("context") or ""),
                            ]
                        ),
                        context_summary=context_summary,
                        rule_id=rule_id,
                    )
                else:
                    system_prompt, user_prompt = self._get_check_prompt(
                        srd=srd,
                        raw_answer=answer_text,
                        problem_text="\n".join(
                            [
                                str(sample.get("question") or ""),
                                str(sample.get("context") or ""),
                            ]
                        ),
                        context_summary="{}",
                        rule_id=rule_id,
                    )
                
                diagnostics = self._llm_json(
                    system_prompt,
                    user_prompt,
                    fallback=[],
                    trace_meta={
                        "sample_id": sample.get("id"),
                        "rule_id": rule_id,
                    },
                )
                if isinstance(diagnostics, dict):
                    diagnostics = [diagnostics]
                elif isinstance(diagnostics, str):
                    diagnostics = [diagnostics]
                elif diagnostics is None:
                    diagnostics = []

                normalized_diag: List[Any] = []
                for d in diagnostics:
                    if isinstance(d, dict):
                        if self._is_negative_or_uncertain_diagnostic(d):
                            continue
                        normalized_diag.append(self._normalize_diagnostic_location(d, answer_text))
                    else:
                        normalized_diag.append(d)
                all_diagnostics.extend(normalized_diag)

        # 去重和计分
        seen = set()
        unique_diagnostics = []
        for d in all_diagnostics:
            if isinstance(d, str):
                key = (None, None, d)
                payload = {"severity": "info", "rule": None, "symbol": None, "message": d}
            else:
                key = (d.get("rule"), d.get("symbol"), d.get("message"))
                payload = d
            if key not in seen:
                unique_diagnostics.append(payload)
                seen.add(key)
        
        score = sum(-1.0 if d.get("severity") == "error" else -0.5 for d in unique_diagnostics if d.get("severity") in ["error", "warning"])

        out = {
            "id": sample.get("id"),
            "dataset": dataset_key,
            "diagnostics": unique_diagnostics,
            "score": score,
            "answer_correct": answer_correct,
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
