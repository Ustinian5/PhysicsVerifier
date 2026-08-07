"""从统一/层次化规则目录中检索候选主题与规则（规则匹配阶段）。

供 `PhysicsRuleVerifier`（`core/physics_rule_verifier.py`）及离线脚本
`scripts/analyze_rule_matching.py`、`scripts/manage_rule_library.py` 使用。"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "check",
    "checks",
    "correct",
    "derive",
    "description",
    "determine",
    "domain",
    "equation",
    "error",
    "errors",
    "for",
    "form",
    "from",
    "given",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "logic",
    "must",
    "of",
    "on",
    "or",
    "physics",
    "problem",
    "process",
    "result",
    "rule",
    "rules",
    "should",
    "solution",
    "such",
    "that",
    "the",
    "their",
    "them",
    "then",
    "this",
    "those",
    "through",
    "to",
    "topic",
    "under",
    "use",
    "using",
    "verify",
    "when",
    "where",
    "which",
    "with",
    "without",
    "题目",
    "公式",
    "关系",
    "分析",
    "出现",
    "判断",
    "前提",
    "利用",
    "可能",
    "必须",
    "条件",
    "检查",
    "根据",
    "正确",
    "注意",
    "涉及",
    "过程",
    "逻辑",
    "结果",
    "要求",
    "计算",
    "证明",
    "验证",
    "解答",
    "说明",
    "需要",
}

GENERIC_SCENE_TERMS = {
    "application",
    "applications",
    "body",
    "bodies",
    "charge",
    "charges",
    "current",
    "currents",
    "dynamics",
    "electric",
    "electromagnetism",
    "energy",
    "equation",
    "equations",
    "field",
    "fields",
    "force",
    "forces",
    "function",
    "functions",
    "induction",
    "interaction",
    "interactions",
    "law",
    "laws",
    "light",
    "magnetic",
    "mass",
    "mechanics",
    "motion",
    "optics",
    "particle",
    "particles",
    "physics",
    "potential",
    "radiation",
    "role",
    "statistical",
    "system",
    "systems",
    "temperature",
    "thermodynamics",
    "time",
    "transfer",
    "wave",
    "waves",
    "电学",
    "力学",
    "场",
    "定律",
    "方程",
    "热学",
    "物理",
    "电磁",
    "粒子",
    "能量",
    "运动",
}

LOW_SIGNAL_KEYWORDS = {
    "about",
    "all",
    "area",
    "being",
    "body",
    "center",
    "com",
    "constant",
    "current",
    "dynamics",
    "energy",
    "equal",
    "field",
    "fields",
    "function",
    "initial",
    "first",
    "length",
    "mass",
    "model",
    "models",
    "motion",
    "number",
    "numbers",
    "object",
    "objects",
    "per",
    "physical",
    "results",
    "surface",
    "sum",
    "system",
    "systems",
    "term",
    "terms",
    "theory",
    "time",
    "total",
    "use",
    "used",
    "vec",
    "all",
    "no",
    "relations",
    "relation",
    "applications",
    "principles",
    "second",
}

META_RULE_HINTS = (
    "chart",
    "data extraction",
    "diagram",
    "dimensional",
    "distinguish given",
    "figure",
    "graph",
    "known/unknown",
    "notation",
    "read graph",
    "semantic",
    "table",
    "unit consistency",
    "区分给定",
    "区分已知",
    "区分给定演化规律",
    "区分待求",
    "图区",
    "图像",
    "图表",
    "图表数据",
    "已知",
    "待求",
    "数据提取",
    "文字描述",
    "标注",
    "量纲",
    "量纲一致",
    "表格",
    "读图",
    "语义",
)

MANUAL_RULE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "exp_8f7c1ad1fb477295": {
        "scope": "meta",
        "reason": "generic_trig_substitution_rule",
        "drop_trigger_keywords": ["sin", "cos"],
        "drop_object_keywords": ["sin", "cos", "tan", "统一化为", "的表达形式"],
        "prepend_trigger_keywords": ["根式", "tanθ"],
        "prepend_object_keywords": ["根式方程", "三角代换"],
    }
}

MANUAL_TOPIC_HINT_OVERRIDES: Dict[Tuple[str, str], Dict[str, Any]] = {
    (
        "Experimental Physics",
        "Measurement Techniques (Length, Time, Mass, etc.)",
    ): {
        "drop_scene_keywords": ["Length", "Time", "Mass", "etc."],
        "drop_topic_keywords": ["Length", "Time", "Mass", "physical", "results", "used", "being", "etc"],
        "prepend_scene_keywords": ["measurement instrument", "calibration setup", "experimental measurement"],
        "prepend_topic_keywords": ["measurement", "instrument", "calibration", "uncertainty", "apparatus"],
    },
    (
        "Modern Physics",
        "Special Relativity (Time Dilation, Length Contraction)",
    ): {
        "prepend_scene_keywords": [
            "pinhole camera",
            "rod in motion",
            "moving rod",
            "simultaneous measurement",
            "observer frame",
        ],
        "prepend_topic_keywords": [
            "pinhole",
            "camera",
            "rod",
            "observer",
            "simultaneous",
            "contraction",
        ],
    },
    (
        "Optics",
        "Laser Principles and Applications",
    ): {
        "prepend_scene_keywords": ["ring resonator", "laser cavity", "sagnac effect"],
        "prepend_topic_keywords": ["resonator", "cavity", "sagnac"],
    },
    (
        "Optics",
        "Optical Coherence and Interferometers",
    ): {
        "prepend_scene_keywords": ["ring interferometer", "sagnac interferometer", "path difference"],
        "prepend_topic_keywords": ["interferometer", "sagnac", "coherence length"],
    },
    (
        "Electromagnetism",
        "Current, Resistance, and Ohm's Law",
    ): {
        "drop_scene_keywords": ["Resistance"],
        "drop_topic_keywords": ["Resistance", "Circuit", "circuits", "currents", "Direction", "all", "sum", "total"],
        "prepend_scene_keywords": ["ohmic resistor", "voltage current relation", "resistive wire loss"],
        "prepend_topic_keywords": ["ohm", "resistor", "voltage", "current", "resistive", "wire loss"],
    },
    (
        "Experimental Physics",
        "Use of Multimeters and Circuit Analysis Tools",
    ): {
        "drop_topic_keywords": ["Circuit", "resistance", "connected", "set", "component", "current"],
        "prepend_scene_keywords": ["multimeter probe", "ammeter connection", "voltmeter reading", "measurement mode"],
        "prepend_topic_keywords": ["multimeter", "ammeter", "voltmeter", "ohmmeter", "probe", "measurement mode"],
    },
    (
        "Electromagnetism",
        "Electromagnetic Induction and Faraday's Law",
    ): {
        "prepend_scene_keywords": [
            "eddy current",
            "foucault current",
            "magnetic braking",
            "rotating disc in magnetic field",
            "induced emf",
        ],
        "prepend_topic_keywords": [
            "eddy",
            "foucault",
            "induced emf",
            "flux change",
            "magnetic braking",
            "faraday",
        ],
    },
}

GENERIC_RULE_SIGNAL_TERMS = {
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "sqrt",
    "根式",
    "代换",
    "恒等式",
    "三角",
    "三角代换",
    "根式方程",
}

GENERIC_SCENE_PARTS = {
    "applications",
    "first",
    "law",
    "laws",
    "models",
    "no",
    "not",
    "principles",
    "relation",
    "relations",
    "second",
    "simple",
    "theory",
    "use",
}

GENERIC_SYMBOLS = {
    "a",
    "A",
    "b",
    "B",
    "c",
    "C",
    "d",
    "D",
    "e",
    "E",
    "f",
    "F",
    "g",
    "G",
    "h",
    "H",
    "i",
    "I",
    "k",
    "K",
    "l",
    "L",
    "m",
    "M",
    "n",
    "N",
    "p",
    "P",
    "q",
    "Q",
    "r",
    "R",
    "s",
    "S",
    "t",
    "T",
    "u",
    "U",
    "v",
    "V",
    "w",
    "W",
    "x",
    "X",
    "y",
    "Y",
    "z",
    "Z",
}


def norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = norm_text(value)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def iter_rule_leaves(topic: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield executable rule leaves from either flat topic.rules or rule_tree.

    Newer rule catalogs may keep a variable-depth tree for maintenance while
    preserving topic.rules as the verifier-facing flat view. This helper is the
    single read path for runtime and analysis code.
    """
    if not isinstance(topic, dict):
        return

    rules = [rule for rule in (topic.get("rules") or []) if isinstance(rule, dict)]
    if rules:
        for rule in rules:
            yield rule
        return

    by_id: Dict[str, Dict[str, Any]] = {}
    for rule in rules:
        rid = norm_text(rule.get("rule_id") or rule.get("id") or "")
        if rid:
            by_id[rid] = rule

    def walk(node: Any) -> Iterable[Dict[str, Any]]:
        if not isinstance(node, dict):
            return
        if isinstance(node.get("rule"), dict):
            yield node["rule"]
        for rid in node.get("rule_ids") or []:
            rule = by_id.get(norm_text(rid))
            if rule:
                yield rule
        for child in node.get("children") or []:
            yield from walk(child)

    yield from walk(topic.get("rule_tree") or {})


def topic_rule_leaves(topic: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(iter_rule_leaves(topic))


def match_phrase_or_symbol(needle: str, haystack: str) -> bool:
    target = norm_text(needle)
    text = norm_text(haystack)
    if not target or not text:
        return False

    if len(target) == 1 and re.fullmatch(r"[A-Za-z]", target):
        pat = re.compile(rf"(^|[^A-Za-z0-9_]){re.escape(target)}([^A-Za-z0-9_]|$)", re.I)
        return bool(pat.search(text))

    return target.casefold() in text.casefold()


def keep_token(token: str) -> bool:
    if not token or token.isdigit():
        return False
    if len(token) == 1 and re.fullmatch(r"[A-Za-z0-9_]", token):
        return False
    if token.casefold() in STOPWORDS:
        return False
    return True


def extract_keywords(texts: Iterable[str], *, max_keywords: int) -> List[str]:
    counter: Counter[str] = Counter()
    first_seen: Dict[str, str] = {}
    order: Dict[str, int] = {}

    for text in texts:
        for raw in TOKEN_RE.findall(norm_text(text)):
            token = raw.strip()
            if not keep_token(token):
                continue
            key = token.casefold()
            counter[key] += 1
            first_seen.setdefault(key, token)
            order.setdefault(key, len(order))

    ranked = sorted(counter.items(), key=lambda item: (-item[1], order[item[0]], item[0]))
    return [first_seen[key] for key, _ in ranked[:max_keywords]]


def get_manual_rule_override(rule_id: Any) -> Dict[str, Any]:
    return dict(MANUAL_RULE_OVERRIDES.get(norm_text(rule_id), {}))


def _remove_keywords(values: Iterable[str], blocked: Iterable[str]) -> List[str]:
    blocked_keys = {norm_text(item).casefold() for item in blocked if norm_text(item)}
    return [item for item in ordered_unique(values) if norm_text(item).casefold() not in blocked_keys]


def get_manual_topic_hint_override(domain: Any, topic: Any) -> Dict[str, Any]:
    key = (norm_text(domain), norm_text(topic))
    return dict(MANUAL_TOPIC_HINT_OVERRIDES.get(key, {}))


def apply_manual_topic_hint_override(
    *,
    domain: Any,
    topic: Any,
    scene_keywords: Iterable[str],
    topic_keywords: Iterable[str],
) -> Tuple[List[str], List[str]]:
    override = get_manual_topic_hint_override(domain, topic)
    if not override:
        return ordered_unique(scene_keywords), ordered_unique(topic_keywords)

    scene = _remove_keywords(scene_keywords, override.get("drop_scene_keywords") or [])
    keywords = _remove_keywords(topic_keywords, override.get("drop_topic_keywords") or [])
    scene = ordered_unique(list(override.get("prepend_scene_keywords") or []) + scene)
    keywords = ordered_unique(list(override.get("prepend_topic_keywords") or []) + keywords)
    return scene, keywords


def apply_manual_rule_override(rule: Dict[str, Any]) -> Dict[str, Any]:
    override = get_manual_rule_override(rule.get("rule_id") or rule.get("id"))
    if not override:
        return rule

    patched = dict(rule)
    if override.get("scope"):
        patched["scope"] = norm_text(override["scope"]) or patched.get("scope") or "domain"
    if override.get("reason"):
        patched["manual_override_reason"] = norm_text(override["reason"])

    match_features = dict(patched.get("match_features") or {})
    if match_features:
        trigger_keywords = _remove_keywords(
            match_features.get("trigger_keywords") or [],
            override.get("drop_trigger_keywords") or [],
        )
        object_keywords = _remove_keywords(
            match_features.get("object_keywords") or [],
            override.get("drop_object_keywords") or [],
        )
        match_features["trigger_keywords"] = ordered_unique(
            list(override.get("prepend_trigger_keywords") or []) + trigger_keywords
        )
        match_features["object_keywords"] = ordered_unique(
            list(override.get("prepend_object_keywords") or []) + object_keywords
        )
        patched["match_features"] = match_features

    return patched


def classify_rule_scope(*, title: str, trigger: str, check_logic: str, rule_id: Any = "") -> str:
    override = get_manual_rule_override(rule_id)
    if override.get("scope"):
        return norm_text(override["scope"]) or "domain"
    text = " ".join([norm_text(title), norm_text(trigger), norm_text(check_logic)]).casefold()
    if not text:
        return "domain"
    for hint in META_RULE_HINTS:
        if hint.casefold() in text:
            return "meta"
    return "domain"


def build_scene_keywords(
    *,
    topic_name: str,
    tagged_titles: Iterable[str],
    tagged_aliases: Iterable[str],
    rule_texts: Iterable[str] = (),
) -> List[str]:
    def keep_scene_part(text: str, *, allow_single: bool = False) -> bool:
        item = norm_text(text)
        if not item:
            return False
        lowered = item.casefold()
        if lowered in GENERIC_SCENE_TERMS or lowered in GENERIC_SCENE_PARTS or lowered in LOW_SIGNAL_KEYWORDS:
            return False
        if item.isalpha() and item.upper() == item and len(item) <= 2:
            return False
        if not allow_single and " " not in item and len(item) < 4:
            return False
        return True

    phrases: List[str] = []
    tagged_texts = [norm_text(item) for item in tagged_titles]
    alias_texts = [norm_text(item) for item in tagged_aliases]
    rule_phrase_texts = [norm_text(item) for item in rule_texts]

    for raw_text in [topic_name, *tagged_texts, *alias_texts, *rule_phrase_texts]:
        text = norm_text(raw_text)
        if not text:
            continue
        allow_single = raw_text == topic_name
        if keep_scene_part(text, allow_single=allow_single):
            phrases.append(text)
        for part in re.split(r"[()/,;:，；]+| and | of | with | - ", text, flags=re.I):
            item = norm_text(part)
            if keep_scene_part(item, allow_single=False):
                phrases.append(item)
    return ordered_unique(phrases)[:16]


def build_topic_required_symbols(rules: Iterable[Dict[str, Any]]) -> List[str]:
    symbols: List[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if norm_text(rule.get("scope") or "domain") == "meta":
            continue
        features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
        symbols.extend(str(item) for item in (features.get("required_symbols") or []))
    return ordered_unique(symbols)


def _match_llm_phrase(phrase: str, haystack: str, *, min_word_hits: int = 2, ratio: float = 0.45) -> bool:
    """Return True when a sufficient fraction of meaningful words in *phrase* appear in *haystack*.

    This provides fuzzy phrase-level matching for LLM-generated scenario sentences and
    problem phrases stored in ``llm_hints.match_phrases`` / ``retrieval_hints.llm_problem_phrases``.
    """
    tokens = [w for w in TOKEN_RE.findall(norm_text(phrase)) if keep_token(w) and len(w) >= 4]
    if not tokens:
        return False
    text_lower = norm_text(haystack).lower()
    hits = sum(1 for w in tokens if w.lower() in text_lower)
    threshold = max(min_word_hits, int(len(tokens) * ratio + 0.5))
    return hits >= threshold


def build_topic_candidates(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for domain in catalog.get("domains", []) or []:
        domain_name = norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            executable_rules = [
                rule
                for rule in topic_rule_leaves(topic)
                if isinstance(rule, dict)
                and norm_text(rule.get("rule_id") or rule.get("id") or "")
            ]
            if not executable_rules:
                continue
            topic_name = norm_text(topic.get("name") or "Unknown")
            topic_obj = dict(topic)
            topic_obj.setdefault("domain", domain_name)
            tagged_reference = topic.get("tagged_reference") if isinstance(topic.get("tagged_reference"), dict) else {}
            retrieval_hints = topic.get("retrieval_hints") if isinstance(topic.get("retrieval_hints"), dict) else {}
            knowledge_reference = topic.get("knowledge_reference") if isinstance(topic.get("knowledge_reference"), dict) else {}
            includes = ordered_unique(
                list(topic.get("includes") or [])
                + list(retrieval_hints.get("includes") or [])
                + list(retrieval_hints.get("positive_contexts") or [])
            )
            excludes = ordered_unique(
                list(topic.get("excludes") or [])
                + list(retrieval_hints.get("excludes") or [])
                + list(retrieval_hints.get("negative_contexts") or [])
            )
            out.append(
                {
                    "domain": domain_name,
                    "topic": topic_name,
                    "topic_obj": topic_obj,
                    "aliases": ordered_unique(tagged_reference.get("aliases") or []),
                    "scene_keywords": ordered_unique(retrieval_hints.get("scene_keywords") or []),
                    "topic_keywords": ordered_unique(retrieval_hints.get("topic_keywords") or []),
                    "knowledge_keywords": ordered_unique(knowledge_reference.get("keywords") or []),
                    "required_symbols": ordered_unique(retrieval_hints.get("required_symbols") or []),
                    # LLM-enhanced signals (present only in enhanced catalogs)
                    "llm_problem_phrases": ordered_unique(retrieval_hints.get("llm_problem_phrases") or []),
                    "llm_discriminative_terms": ordered_unique(retrieval_hints.get("llm_discriminative_terms") or []),
                    "includes": includes,
                    "excludes": excludes,
                }
            )
    return out


def build_signal_document_frequency(topic_candidates: List[Dict[str, Any]]) -> Dict[str, Counter[str]]:
    keyword_df: Counter[str] = Counter()
    scene_df: Counter[str] = Counter()
    symbol_df: Counter[str] = Counter()

    for candidate in topic_candidates:
        for item in ordered_unique(
            list(candidate.get("knowledge_keywords") or []) + list(candidate.get("topic_keywords") or [])
        ):
            keyword_df[item.casefold()] += 1
        for item in ordered_unique(candidate.get("scene_keywords") or []):
            scene_df[item.casefold()] += 1
        for item in ordered_unique(candidate.get("required_symbols") or []):
            symbol_df[item.casefold()] += 1

    return {
        "keyword_df": keyword_df,
        "scene_df": scene_df,
        "symbol_df": symbol_df,
    }


def _df_weight(value: str, df_counter: Counter[str], *, strong: float, weak: float) -> float:
    df = int(df_counter.get(norm_text(value).casefold(), 0) or 1)
    if df <= 1:
        return strong
    if df <= 3:
        return strong * 0.75
    if df <= 8:
        return (strong + weak) / 2
    return weak


def _symbol_df_weight(symbol: str, df_counter: Counter[str]) -> float:
    normalized = norm_text(symbol)
    if not normalized:
        return 0.0
    df = int(df_counter.get(normalized.casefold(), 0) or 1)
    if len(normalized) == 1 and normalized in GENERIC_SYMBOLS and df > 6:
        return 0.0
    if df <= 2:
        return 1.0
    if df <= 6:
        return 0.5
    return 0.25


def score_topic_candidate(
    candidate: Dict[str, Any],
    text_for_topic: str,
    *,
    signal_df: Dict[str, Counter[str]],
) -> Dict[str, Any]:
    phrase_hits: List[str] = []
    if match_phrase_or_symbol(candidate["topic"], text_for_topic):
        phrase_hits.append(candidate["topic"])
    for alias in candidate.get("aliases") or []:
        if match_phrase_or_symbol(alias, text_for_topic):
            phrase_hits.append(alias)

    scene_hits = [
        kw for kw in ordered_unique(candidate.get("scene_keywords") or []) if match_phrase_or_symbol(kw, text_for_topic)
    ]
    keyword_pool = ordered_unique(list(candidate.get("knowledge_keywords") or []) + list(candidate.get("topic_keywords") or []))
    keyword_hits = []
    for kw in keyword_pool:
        normalized = norm_text(kw).casefold()
        if normalized.isalpha() and len(normalized) <= 2:
            continue
        if normalized in LOW_SIGNAL_KEYWORDS:
            continue
        if signal_df["keyword_df"].get(normalized, 0) > 18 and len(normalized) <= 8:
            continue
        if match_phrase_or_symbol(kw, text_for_topic):
            keyword_hits.append(kw)

    symbol_gate_open = bool(phrase_hits or scene_hits or len(keyword_hits) >= 2)
    symbol_hits = []
    if symbol_gate_open:
        symbol_hits = [
            sym for sym in ordered_unique(candidate.get("required_symbols") or []) if match_phrase_or_symbol(sym, text_for_topic)
        ]
    include_hits = [
        item for item in ordered_unique(candidate.get("includes") or []) if match_phrase_or_symbol(item, text_for_topic)
    ]
    exclude_hits = [
        item for item in ordered_unique(candidate.get("excludes") or []) if match_phrase_or_symbol(item, text_for_topic)
    ]

    # LLM-enhanced signals (present only in LLM-enhanced catalogs)
    llm_phrase_hits: List[str] = []
    llm_term_hits: List[str] = []
    llm_score = 0.0
    llm_problem_phrases = candidate.get("llm_problem_phrases") or []
    llm_discriminative_terms = candidate.get("llm_discriminative_terms") or []
    if llm_problem_phrases or llm_discriminative_terms:
        llm_phrase_hits = [
            p for p in ordered_unique(llm_problem_phrases)
            if _match_llm_phrase(p, text_for_topic)
        ]
        llm_term_hits = [
            t for t in ordered_unique(llm_discriminative_terms)
            if match_phrase_or_symbol(t, text_for_topic)
        ]
        # LLM signals: phrase hits are high-confidence (cap 6), term hits moderate (cap 5)
        llm_score = min(len(llm_phrase_hits) * 3.0, 6.0) + min(len(llm_term_hits) * 0.8, 5.0)

    phrase_score = 6.0 if phrase_hits else 0.0
    scene_score = min(
        sum(_df_weight(hit, signal_df["scene_df"], strong=3.5, weak=0.75) for hit in scene_hits),
        12.0,
    )
    keyword_score = min(
        sum(_df_weight(hit, signal_df["keyword_df"], strong=2.0, weak=0.35) for hit in keyword_hits),
        8.0,
    )
    symbol_score = min(sum(_symbol_df_weight(hit, signal_df["symbol_df"]) for hit in symbol_hits), 3.0)
    include_score = min(len(include_hits) * 1.5, 4.5)
    exclude_penalty = min(len(exclude_hits) * 3.0, 9.0)
    score = max(0.0, float(phrase_score + scene_score + keyword_score + symbol_score + llm_score + include_score - exclude_penalty))

    return {
        "domain": candidate["domain"],
        "topic": candidate["topic"],
        "score": score,
        "evidence": {
            "name_or_alias_hits": ordered_unique(phrase_hits),
            "scene_keyword_hits": scene_hits,
            "keyword_hits": keyword_hits,
            "required_symbol_hits": symbol_hits,
            "symbol_gate_open": symbol_gate_open,
            "llm_phrase_hits": llm_phrase_hits,
            "llm_term_hits": llm_term_hits,
            "include_hits": include_hits,
            "exclude_hits": exclude_hits,
            "exclude_penalty": round(exclude_penalty, 4),
        },
        "topic_obj": candidate["topic_obj"],
    }


def _student_text_surfaces(text_for_rule: str) -> Tuple[str, str]:
    """Return raw haystack plus a lightly normalised variant for keyword matching."""
    raw = str(text_for_rule or "")
    norm = norm_text(raw)
    for pat, repl in (
        (r"\\mathrm\{([^}]*)\}", r"\1"),
        (r"\\text\{([^}]*)\}", r"\1"),
        (r"\\,|\~", " "),
        (r"\s+", " "),
    ):
        norm = re.sub(pat, repl, norm, flags=re.I)
    norm = norm.strip()
    return raw, norm


def build_unified_topic_retrieval_text(sample: Dict[str, Any], *, tuning: Optional[Dict[str, Any]] = None) -> str:
    """Topic-stage retrieval text (defaults: question + context + truncated prediction)."""
    tuning = tuning or {}
    parts = [norm_text(sample.get("question") or ""), norm_text(sample.get("context") or "")]
    if tuning.get("topic_include_prediction", True):
        pred = norm_text(sample.get("prediction") or "")
        max_c = int(tuning.get("topic_prediction_max_chars") or 4000)
        if max_c > 0 and len(pred) > max_c:
            pred = pred[:max_c] + "\n...[truncated]"
        if pred:
            parts.append(pred)
    return "\n".join(p for p in parts if p)


_SYMBOL_TOKEN_RE = re.compile(r"\\?([a-zA-Z][a-zA-Z0-9_]*)")
_STOP_WORDS_SYM = {
    "the", "of", "a", "an", "is", "in", "on", "for", "with", "and", "or", "not", "sin", "cos", "tan", "log", "ln", "exp",
}


def _looks_like_symbol_token(tok: str) -> bool:
    if not tok or len(tok) > 20:
        return False
    low = tok.lower()
    if low in _STOP_WORDS_SYM:
        return False
    if re.fullmatch(r"[a-z]+", tok) and len(tok) >= 4:
        return False
    return True


def extract_prediction_symbol_set(prediction_text: str) -> Set[str]:
    """Lightweight symbol tokens from student text (aligns with semantic checker heuristics)."""
    out: Set[str] = set()
    for line in (prediction_text or "").splitlines():
        for m in _SYMBOL_TOKEN_RE.finditer(line):
            sym = m.group(1)
            if _looks_like_symbol_token(sym):
                out.add(sym.casefold())
    return out


def topic_retrieval_required_symbols(topic_obj: Dict[str, Any]) -> Set[str]:
    hints = topic_obj.get("retrieval_hints") if isinstance(topic_obj.get("retrieval_hints"), dict) else {}
    return {norm_text(s).casefold() for s in (hints.get("required_symbols") or []) if norm_text(s)}


def apply_topic_symbol_overlap_boost(
    scored: List[Dict[str, Any]],
    pred_syms: Set[str],
    *,
    tuning: Optional[Dict[str, Any]] = None,
) -> None:
    """In-place: boost topic scores when prediction symbols overlap topic required_symbols (unified v2)."""
    if not pred_syms:
        return
    tuning = tuning or {}
    bonus_per_hit = float(tuning.get("topic_symbol_overlap_bonus", 0.45) or 0.45)
    max_bonus = float(tuning.get("topic_symbol_overlap_max_bonus", 2.2) or 2.2)
    for item in scored:
        topic_obj = item.get("topic") if isinstance(item.get("topic"), dict) else item.get("topic_obj")
        if not isinstance(topic_obj, dict):
            continue
        req = topic_retrieval_required_symbols(topic_obj)
        if not req:
            continue
        overlap = len(pred_syms.intersection(req))
        if overlap <= 0:
            continue
        add = min(max_bonus, overlap * bonus_per_hit)
        item["score"] = float(item.get("score") or 0.0) + add
        ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        ev = dict(ev)
        ev["prediction_symbol_overlap"] = overlap
        ev["prediction_symbol_overlap_bonus"] = round(add, 4)
        item["evidence"] = ev


def score_rule_candidate(rule: Dict[str, Any], text_for_rule: str) -> Dict[str, Any]:
    match_features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
    hay_raw, hay_norm = _student_text_surfaces(text_for_rule)

    def _hits_in_student(keywords: List[str]) -> List[str]:
        out: List[str] = []
        for kw in ordered_unique(keywords):
            if match_phrase_or_symbol(kw, hay_raw) or (hay_norm and match_phrase_or_symbol(kw, hay_norm)):
                out.append(kw)
        return out

    trigger_hits = _hits_in_student(list(match_features.get("trigger_keywords") or []))
    object_hits = _hits_in_student(list(match_features.get("object_keywords") or []))
    symbol_hits = [
        sym
        for sym in ordered_unique(match_features.get("required_symbols") or [])
        if match_phrase_or_symbol(sym, hay_raw) or (hay_norm and match_phrase_or_symbol(sym, hay_norm))
    ]
    precision_negative = ordered_unique(rule.get("negative_conditions") or [])
    precision_preconditions = ordered_unique(rule.get("preconditions") or [])
    precision_violations = ordered_unique(rule.get("violation_signatures") or [])
    precision_evidence = ordered_unique(rule.get("evidence_requirements") or [])

    negative_hits = _hits_in_student(
        list(ordered_unique(list(match_features.get("negative_keywords") or []) + precision_negative))
    )
    precondition_hits = _hits_in_student(precision_preconditions)
    violation_signature_hits = _hits_in_student(precision_violations)
    evidence_requirement_hits = _hits_in_student(precision_evidence)
    lexical_hits = len(trigger_hits) + len(object_hits) + len(symbol_hits)

    # LLM-enhanced signals (present only in enhanced catalogs)
    llm_hints = rule.get("llm_hints") if isinstance(rule.get("llm_hints"), dict) else {}
    llm_phrase_hits: List[str] = []
    llm_term_hits: List[str] = []
    llm_rule_score = 0.0
    llm_haystack = hay_raw if not hay_norm else (hay_raw + "\n" + hay_norm)
    if llm_hints:
        llm_phrase_hits = [
            p for p in ordered_unique(llm_hints.get("match_phrases") or []) if _match_llm_phrase(p, llm_haystack)
        ]
        llm_term_hits = [
            t
            for t in ordered_unique(llm_hints.get("discriminative_terms") or [])
            if match_phrase_or_symbol(t, hay_raw) or (hay_norm and match_phrase_or_symbol(t, hay_norm))
        ]
        # Phrase hits are strong evidence; term hits are moderate
        llm_rule_score = min(len(llm_phrase_hits) * 2.5, 7.5) + min(len(llm_term_hits) * 1.0, 4.0)

    llm_only = bool((llm_phrase_hits or llm_term_hits) and lexical_hits == 0)
    if llm_only:
        llm_rule_score *= 0.35

    support = rule.get("support") if isinstance(rule.get("support"), dict) else {}
    count = int(support.get("count") or 0)
    has_strong_anchor = bool(trigger_hits or symbol_hits)
    support_eligible = bool(count > 0 and (has_strong_anchor or (lexical_hits >= 2 and bool(object_hits))))
    support_prior = min(math.log2(count + 1), 3.0) if support_eligible else 0.0
    scope = norm_text(rule.get("scope") or "domain") or "domain"
    manual_override_reason = norm_text(rule.get("manual_override_reason") or "")

    def _is_generic_signal(value: str) -> bool:
        normalized = norm_text(value).casefold()
        if not normalized:
            return False
        if normalized in GENERIC_RULE_SIGNAL_TERMS:
            return True
        return any(term in normalized for term in GENERIC_RULE_SIGNAL_TERMS)

    generic_trigger_hits = [hit for hit in trigger_hits if _is_generic_signal(hit)]
    generic_object_hits = [hit for hit in object_hits if _is_generic_signal(hit)]
    generic_signal_only = bool(lexical_hits > 0 and not set(trigger_hits).difference(generic_trigger_hits) and not set(object_hits).difference(generic_object_hits))

    score = (
        min(len(trigger_hits) * 3, 9)
        + min(len(object_hits) * 2, 6)
        + min(len(symbol_hits) * 2, 4)
        + min(len(precondition_hits) * 1.0, 3.0)
        + min(len(violation_signature_hits) * 1.5, 4.5)
        + min(len(evidence_requirement_hits) * 0.8, 2.4)
        + support_prior
        + llm_rule_score
    )
    neg_penalty = min(len(negative_hits) * 2.0, 8.0)
    score = max(0.0, float(score) - neg_penalty)
    if scope == "meta":
        score = max(score - 1.5, 0.0)
        if not trigger_hits and not object_hits:
            score = max(score - 0.5, 0.0)

    return {
        "rule_id": norm_text(rule.get("rule_id") or ""),
        "title": norm_text(rule.get("title") or ""),
        "score": float(score),
        "scope": scope,
        "evidence": {
            "trigger_hits": trigger_hits,
            "object_hits": object_hits,
            "required_symbol_hits": symbol_hits,
            "generic_trigger_hits": generic_trigger_hits,
            "generic_object_hits": generic_object_hits,
            "generic_signal_only": generic_signal_only,
            "support_count": count,
            "support_prior": round(support_prior, 4),
            "manual_override_reason": manual_override_reason,
            "llm_phrase_hits": llm_phrase_hits,
            "llm_term_hits": llm_term_hits,
            "llm_only_soft_hit": llm_only,
            "negative_keyword_hits": negative_hits,
            "precondition_hits": precondition_hits,
            "violation_signature_hits": violation_signature_hits,
            "evidence_requirement_hits": evidence_requirement_hits,
            "negative_keyword_penalty": round(neg_penalty, 4),
            "score_components": {
                "anchor_score": min(len(trigger_hits) * 3, 9) + min(len(symbol_hits) * 2, 4),
                "object_score": min(len(object_hits) * 2, 6),
                "precondition_score": min(len(precondition_hits) * 1.0, 3.0),
                "violation_signature_score": min(len(violation_signature_hits) * 1.5, 4.5),
                "evidence_requirement_score": min(len(evidence_requirement_hits) * 0.8, 2.4),
                "support_prior": round(support_prior, 4),
                "llm_hint_score": round(llm_rule_score, 4),
            },
        },
    }


def topic_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (-float(item.get("score") or 0.0), str(item.get("domain") or ""), str(item.get("topic") or ""))


def rule_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    scope_rank = 1 if norm_text(item.get("scope") or "domain") == "meta" else 0
    return (
        scope_rank,
        -float(item.get("adjusted_score", item.get("score") or 0.0)),
        str(item.get("domain") or ""),
        str(item.get("topic") or ""),
        str(item.get("rule_id") or ""),
    )


def rule_topic_context(
    *,
    raw_score: float,
    topic_rank: int,
    topic_score: float,
    top1_topic_score: float,
    scope: str,
    rule_evidence: Dict[str, Any] | None = None,
    topic_evidence: Dict[str, Any] | None = None,
    retrieval_tuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    gap = max(0.0, float(top1_topic_score or 0.0) - float(topic_score or 0.0))
    evidence = rule_evidence if isinstance(rule_evidence, dict) else {}
    topic_hits = topic_evidence if isinstance(topic_evidence, dict) else {}
    generic_signal_only = bool(evidence.get("generic_signal_only"))
    has_topic_anchor = bool(topic_hits.get("name_or_alias_hits") or topic_hits.get("scene_keyword_hits"))
    normalized_scope = norm_text(scope or "domain") or "domain"
    tuning = retrieval_tuning if isinstance(retrieval_tuning, dict) else {}

    if topic_rank <= 0:
        min_score = 1.0
        bonus = 1.5
    elif gap <= 0.5:
        min_score = 4.0
        bonus = 0.0
    elif gap <= 1.5:
        min_score = 6.0
        bonus = -0.75
    elif gap <= 3.0:
        min_score = 7.0
        bonus = -1.75
    else:
        min_score = 8.0
        bonus = -2.5

    if normalized_scope == "meta":
        min_score += 0.5
        bonus -= 0.75

    if generic_signal_only:
        min_score += 1.0
        bonus -= 1.0
        if not has_topic_anchor:
            min_score += 2.0
            bonus -= 1.5

    min_score += float(tuning.get("min_score_delta", 0.0) or 0.0)
    bonus += float(tuning.get("bonus_delta", 0.0) or 0.0)

    return {
        "topic_gap": gap,
        "min_score": min_score,
        "adjusted_score": float(raw_score + bonus),
    }


def non_top1_rule_quota(*, top1_margin: float, top_n: int) -> int:
    if top1_margin >= 3.0:
        return 0
    if top1_margin > 1.5:
        return min(1, top_n)
    if top1_margin > 0.5:
        return min(2, top_n)
    return top_n


def select_rules_with_topic_priority(
    scored: List[Dict[str, Any]],
    *,
    top_n: int,
    top1_key: Tuple[str, str] | None,
    top1_margin: float,
) -> List[Dict[str, Any]]:
    eligible = [
        item
        for item in scored
        if float(item.get("score") or 0.0) >= float(item.get("min_score") or 0.0)
        and float(item.get("adjusted_score", item.get("score") or 0.0)) > 0.0
    ]
    if not top1_key:
        return eligible[:top_n]

    if top1_margin >= 3.0:
        top1_only = [
            item for item in eligible if (str(item.get("domain") or ""), str(item.get("topic") or "")) == top1_key
        ]
        if top1_only:
            return top1_only[:top_n]
        return eligible[: min(2, top_n)]

    eligible_top1_count = sum(
        1
        for item in eligible
        if (str(item.get("domain") or ""), str(item.get("topic") or "")) == top1_key
    )
    quota = non_top1_rule_quota(top1_margin=top1_margin, top_n=top_n)

    selected: List[Dict[str, Any]] = []
    non_top1_count = 0
    for item in eligible:
        item_key = (str(item.get("domain") or ""), str(item.get("topic") or ""))
        is_top1 = item_key == top1_key
        if not is_top1 and non_top1_count >= quota and eligible_top1_count >= max(top_n - quota, 0):
            continue
        selected.append(item)
        if not is_top1:
            non_top1_count += 1
        if len(selected) >= top_n:
            break

    return selected
