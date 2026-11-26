"""Build unified_rules v2 from knowledge skeleton + distilled experience rules.

Inputs:
1. catalogs/rules_catalog_top_down.json
2. catalogs/rules_300_tagged.json
3. catalogs/semantic_experience_distilled_300.json

Output:
- catalogs/rules_unified.json

The persisted v2 catalog is a compact semantic navigation tree:
- Domain -> Topic -> Scenario Cluster -> Rule
- rule_groups are optional inside scenario clusters
- old retrieval/debug fields are used only while building and are not stored
- hand-written includes/excludes are not stored in the runtime navigation tree
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.rule_catalog_retrieval import (
    build_scene_keywords,
    build_topic_required_symbols,
    classify_rule_scope,
    extract_keywords,
    norm_text,
    ordered_unique,
)
from scripts.catalog_build_retrieval_helpers import (
    PHYSICAL_CONTEXT_HINTS,
    is_strong_symbol,
    normalize_rule_for_retrieval,
    refine_topic_hints,
)

DOMAIN_SEMANTIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "Mechanics": {
        "description": "Covers motion, forces, energy, momentum, gravity, fluids, and mechanical systems under classical mechanics assumptions.",
        "includes": ["motion and trajectory", "force balance", "energy and momentum", "orbital and rotational dynamics"],
        "excludes": ["field propagation as the primary object", "microscopic quantum-state reasoning"],
    },
    "Electromagnetism": {
        "description": "Covers electric and magnetic fields, circuits, induction, charge transport, and electromagnetic interactions.",
        "includes": ["circuits and current", "electric or magnetic fields", "induction and flux change", "Lorentz-force effects"],
        "excludes": ["pure geometric optics without field dynamics", "thermodynamic state reasoning as the main mechanism"],
    },
    "Optics": {
        "description": "Covers propagation of light, refraction, reflection, interference, coherence, and optical measurement settings.",
        "includes": ["light propagation", "refraction and reflection", "interference and coherence", "optical imaging"],
        "excludes": ["general electromagnetic circuit behavior", "matter-wave or particle-physics reasoning"],
    },
    "Modern Physics": {
        "description": "Covers relativity, quantum ideas, particle and nuclear physics, and modern spacetime or microscopic frameworks.",
        "includes": ["special or general relativity", "quantum and particle effects", "modern spacetime models"],
        "excludes": ["purely classical mechanics unless used as comparison baseline", "standard lab measurement workflow only"],
    },
    "Thermodynamics & Statistical Physics": {
        "description": "Covers heat, thermal transport, thermodynamic processes, gases, entropy, and statistical macroscopic behavior.",
        "includes": ["heat transfer", "state variables", "gases and kinetic theory", "entropy and equilibrium"],
        "excludes": ["field or circuit dynamics as the dominant mechanism", "purely geometric optical paths"],
    },
    "Experimental Physics": {
        "description": "Covers measurement, uncertainty, graph reading, instrument usage, and data interpretation in physics experiments.",
        "includes": ["uncertainty and significant figures", "graph or table interpretation", "instrument reading and setup"],
        "excludes": ["domain-specific physical mechanism derivation unless explicitly experimental"],
    },
}

FOCUS_TOPIC_STRUCTURE: Dict[str, Dict[str, Any]] = {
    "mechanics::kinematics in 1d/2d/3d": {
        "description": "Classical kinematics expressed through position, velocity, acceleration, timing, trajectory constraints, and geometric motion relations.",
        "includes": ["timing and displacement relations", "trajectory or projection geometry", "average or instantaneous speed checks", "constraint-based kinematics"],
        "excludes": ["force-balance-first modeling", "energy conservation as the main organizing principle"],
        "related_topics": [
            "Mechanics::Relative Motion",
            "Mechanics::Newton's Laws and Free Body Diagrams",
        ],
    },
    "mechanics::newton's laws and free body diagrams": {
        "description": "Force decomposition, equation-of-motion setup, drag or contact modeling, and free-body consistency checks.",
        "includes": ["force balance", "free-body diagrams", "drag-force equations", "parameter simplification in Newtonian setups"],
        "excludes": ["pure kinematic timing relations", "energy-only reasoning without force modeling"],
        "related_topics": [
            "Mechanics::Friction and Contact Forces",
            "Mechanics::Kinematics in 1D/2D/3D",
        ],
    },
    "mechanics::friction and contact forces": {
        "description": "Friction-state judgment, contact constraints, rolling conditions, and anisotropic or path-dependent work from contact forces.",
        "includes": ["rolling and slipping states", "friction projection", "contact-work consistency", "anisotropic friction modeling"],
        "excludes": ["force-free kinematics", "global energy bookkeeping without contact specifics"],
        "related_topics": [
            "Mechanics::Newton's Laws and Free Body Diagrams",
            "Mechanics::Work-Energy Theorem and Conservation of Energy",
        ],
    },
    "mechanics::work-energy theorem and conservation of energy": {
        "description": "Energy baselines, work-sign conventions, constrained motion energy transfer, and kinetic-energy stitching across piecewise fields.",
        "includes": ["energy conservation", "work sign consistency", "constrained-surface kinetic checks", "piecewise-field energy matching"],
        "excludes": ["force-diagram-first derivations", "pure timing or projection kinematics", "orbital decay or celestial mechanics already governed by a more specific gravity topic"],
        "related_topics": [
            "Mechanics::Friction and Contact Forces",
            "Mechanics::Gravitation and Kepler's Laws",
        ],
    },
    "mechanics::oscillations and simple harmonic motion": {
        "description": "Simple harmonic motion, damped or forced oscillations, effective stiffness and mass reduction, and mode-identification in oscillatory systems.",
        "includes": ["SHM parameterization", "forced vs free vibration", "damping and stopping criteria", "equivalent stiffness or mass"],
        "excludes": ["general wave optics without oscillator dynamics", "multi-body mode structure as the only focus"],
        "related_topics": [
            "Mechanics::Normal Modes and Coupled Oscillations",
            "Mechanics::Waves on a String and Sound Waves",
        ],
    },
    "mechanics::normal modes and coupled oscillations": {
        "description": "Coupled oscillators, normal modes, boundary-coupling conditions, and reduced-coordinate derivations for multi-body vibration systems.",
        "includes": ["normal-mode frequency structure", "coupled-system periods", "boundary-force coupling", "small-angle reduced-coordinate derivations"],
        "excludes": ["single-oscillator SHM only", "generic rigid-body rotation without mode structure"],
        "related_topics": [
            "Mechanics::Oscillations and Simple Harmonic Motion",
            "Mechanics::Rotational Kinematics and Dynamics",
        ],
    },
    "electromagnetism::current, resistance, and ohm's law": {
        "description": "Ohmic conduction, resistance construction, microscopic transport relations, and boundary-conditioned current-distribution reasoning.",
        "includes": ["distributed resistance", "microscopic carrier relations", "superconducting or decay-current modeling", "boundary-driven current paths", "ohmic parameter identification"],
        "excludes": ["induction-dominated emf generation", "wave propagation without circuit transport", "Kirchhoff loop solving as the primary task", "reactance-dominated AC behavior"],
        "related_topics": [
            "Electromagnetism::RL, RC, and RLC Circuits",
            "Electromagnetism::Self-Inductance and Mutual Inductance",
            "Electromagnetism::DC Circuits and Kirchhoff's Laws",
        ],
    },
    "electromagnetism::self-inductance and mutual inductance": {
        "description": "Inductive coupling, self-inductance, mutual-inductance constraints, and circuit-state reasoning where inductive terms are the dominant mechanism.",
        "includes": ["self inductance", "mutual inductance", "open-circuit inductive voltage", "coupled-coil transient reasoning"],
        "excludes": ["static resistance-only geometry", "pure Faraday flux construction without inductive coupling", "generic RLC bookkeeping when inductive coupling is not central"],
        "related_topics": [
            "Electromagnetism::Electromagnetic Induction and Faraday's Law",
            "Electromagnetism::RL, RC, and RLC Circuits",
            "Electromagnetism::Current, Resistance, and Ohm's Law",
        ],
    },
    "electromagnetism::electric potential and potential energy": {
        "description": "Electrostatic potential, potential difference, reference choice, path independence, and field-potential relations in conservative electric systems.",
        "includes": ["potential difference", "potential superposition", "zero-reference choice", "electric field from potential"],
        "excludes": ["dipole far-field multipole expansion as the primary object", "Coulomb-force vector balance without potential reasoning"],
        "related_topics": [
            "Electromagnetism::Coulomb's Law and Electric Fields",
            "Electromagnetism::Electric Dipoles and Multipole Expansion",
        ],
    },
    "modern physics::relativistic energy and momentum": {
        "description": "Relativistic energy-momentum relations, invariant-mass reasoning, threshold kinematics, and regime checks between classical and relativistic limits.",
        "includes": ["energy-momentum invariant", "threshold conditions", "relativistic kinetic energy", "classical-vs-relativistic regime checks"],
        "excludes": ["pure observation geometry under length contraction", "classical collision bookkeeping without relativistic quantities"],
        "related_topics": [
            "Mechanics::Linear Momentum and Collisions",
            "Modern Physics::Special Relativity (Time Dilation, Length Contraction)",
        ],
    },
    "thermodynamics & statistical physics::heat transfer (conduction, convection, radiation)": {
        "description": "Thermal transport by conduction, convection, or radiation, including flux balance, thermal resistance, and heating/cooling evolution models.",
        "includes": ["dominant heat-transfer mode", "heat flux balance", "thermal resistance", "heating or cooling time evolution"],
        "excludes": ["state-equation-only gas-process classification", "first-law bookkeeping when transport mode is not central"],
        "related_topics": [
            "Thermodynamics & Statistical Physics::First Law of Thermodynamics",
            "Thermodynamics & Statistical Physics::Ideal Gas Law and Real Gas Behavior",
            "Thermodynamics & Statistical Physics::Specific Heat and Heat Capacities",
        ],
    },
    "thermodynamics & statistical physics::ideal gas law and real gas behavior": {
        "description": "Gas state equations, process identification, parameter mapping, and boundaries between idealized gas models and neighboring thermal mechanisms.",
        "includes": ["ideal-gas state equation", "process classification", "gas parameter mapping", "model-boundary checks"],
        "excludes": ["transport-mode-dominated heat transfer", "pure calorimetry without gas-state relations"],
        "related_topics": [
            "Thermodynamics & Statistical Physics::First Law of Thermodynamics",
            "Thermodynamics & Statistical Physics::Heat Transfer (Conduction, Convection, Radiation)",
            "Thermodynamics & Statistical Physics::Specific Heat and Heat Capacities",
        ],
    },
    "thermodynamics & statistical physics::first law of thermodynamics": {
        "description": "Energy bookkeeping for thermodynamic systems, with explicit heat-work-internal-energy sign conventions and system-boundary validity checks.",
        "includes": ["delta U = Q - W style bookkeeping", "sign convention", "closed-system boundary", "heat-work balance"],
        "excludes": ["transport-mode-first heat-transfer reasoning", "gas-state-equation classification when bookkeeping is secondary"],
        "related_topics": [
            "Thermodynamics & Statistical Physics::Heat Transfer (Conduction, Convection, Radiation)",
            "Thermodynamics & Statistical Physics::Ideal Gas Law and Real Gas Behavior",
            "Thermodynamics & Statistical Physics::Specific Heat and Heat Capacities",
        ],
    },
    "mechanics::circular motion and centripetal force": {
        "description": "Circular-motion constraints, centripetal-force balance, contact-force support conditions, and threshold-speed reasoning in curved trajectories.",
        "includes": ["centripetal-force balance", "normal-force conditions", "critical speed", "banked or constrained circular motion"],
        "excludes": ["generic rotational-energy bookkeeping", "Lorentz-force motion when magnetic dynamics are primary"],
        "related_topics": [
            "Mechanics::Newton's Laws and Free Body Diagrams",
            "Mechanics::Friction and Contact Forces",
            "Electromagnetism::Magnetic Fields and Lorentz Force",
        ],
    },
    "mechanics::rotational kinematics and dynamics": {
        "description": "Rigid-body rotational motion, torque balance, rolling rotation, and moment-relationship modeling where angular acceleration or torque transmission is central.",
        "includes": ["torque balance", "angular acceleration", "rolling rotation", "rotational inertia relations"],
        "excludes": ["pure orbit-like circular motion threshold checks", "angular-momentum conservation without torque modeling"],
        "related_topics": [
            "Mechanics::Angular Momentum Conservation",
            "Mechanics::Circular Motion and Centripetal Force",
            "Mechanics::Newton's Laws and Free Body Diagrams",
        ],
    },
    "mechanics::angular momentum conservation": {
        "description": "Angular-momentum conservation, external-torque validity checks, and impulse-transfer reasoning in rotational or collision-like systems.",
        "includes": ["angular momentum conservation", "external torque validity", "rotational collision transfer", "impulse-to-angular relation"],
        "excludes": ["continuous torque-balance derivations where equations of motion are primary", "pure moment-of-inertia bookkeeping without conservation logic"],
        "related_topics": [
            "Mechanics::Rotational Kinematics and Dynamics",
            "Mechanics::Linear Momentum and Collisions",
        ],
    },
    "modern physics::cosmology and general relativity (basics)": {
        "description": "Introductory spacetime and cosmology reasoning, including proper-vs-coordinate time, horizons, FRW evolution, and relativistic correction baselines.",
        "includes": ["proper and coordinate time", "horizon behavior", "Friedmann evolution", "GR correction signs"],
        "excludes": ["pure special-relativistic observation geometry", "classical orbital mechanics without spacetime correction"],
        "related_topics": [
            "Modern Physics::Special Relativity (Time Dilation, Length Contraction)",
            "Mechanics::Gravitation and Kepler's Laws",
        ],
    },
    "mechanics::gravitation and kepler's laws": {
        "description": "Classical orbital motion, gravity-governed trajectories, escape conditions, and geometric relations in celestial mechanics.",
        "includes": ["Kepler-style orbital geometry", "orbital energy and radius relations", "escape speed and orbital perturbation"],
        "excludes": ["full GR-only metric derivations", "electromagnetic drag mechanisms as primary topic", "binary-specific modeling unless explicitly stated"],
        "related_topics": [
            "Modern Physics::Cosmology and General Relativity (Basics)",
            "Mechanics::Circular Motion and Centripetal Force",
        ],
    },
    "modern physics::special relativity (time dilation, length contraction)": {
        "description": "Relativistic kinematics, simultaneity, length and time measurements, non-inertial extensions, and observation-dependent relativistic effects.",
        "includes": ["length contraction", "time dilation", "frame-dependent observation", "accelerated or rotating relativistic frames"],
        "excludes": ["general GR metric derivations as the main topic", "ordinary classical optical imaging without relativistic timing"],
        "related_topics": [
            "Optics::Laser Principles and Applications",
            "Modern Physics::Cosmology and General Relativity (Basics)",
        ],
    },
    "electromagnetism::electromagnetic induction and faraday's law": {
        "description": "Induced emf, flux change, moving conductors, rotating conductors, eddy currents, and induction-circuit coupling under Faraday-type reasoning.",
        "includes": ["flux change and induced emf", "motional emf", "rotating conductor induction", "induction-circuit coupling"],
        "excludes": ["static circuit law only", "pure wave propagation without induction"],
        "related_topics": [
            "Electromagnetism::Self-Inductance and Mutual Inductance",
            "Electromagnetism::Current, Resistance, and Ohm's Law",
        ],
    },
    "electromagnetism::dc circuits and kirchhoff's laws": {
        "description": "Static circuit solving with Kirchhoff constraints, branch-current decomposition, equivalent reduction, and loop/node consistency.",
        "includes": ["Kirchhoff loop law", "Kirchhoff current law", "branch current solving", "equivalent circuit reduction"],
        "excludes": ["inductive coupling as the dominant mechanism", "reactive AC impedance or resonance", "distributed resistance geometry without circuit solving"],
        "related_topics": [
            "Electromagnetism::Current, Resistance, and Ohm's Law",
            "Electromagnetism::RL, RC, and RLC Circuits",
            "Electromagnetism::Self-Inductance and Mutual Inductance",
        ],
    },
    "electromagnetism::biot-savart law and ampere's law": {
        "description": "Magnetic-field construction from steady currents using symmetry, line integration, and right-hand-rule direction logic.",
        "includes": ["Ampere loop symmetry", "Biot-Savart field element integration", "steady-current condition", "right-hand-rule direction"],
        "excludes": ["time-varying induction as the dominant mechanism", "charged-particle trajectory dynamics without field construction"],
        "related_topics": [
            "Electromagnetism::Magnetic Fields and Lorentz Force",
            "Electromagnetism::Electromagnetic Induction and Faraday's Law",
        ],
    },
    "electromagnetism::rl, rc, and rlc circuits": {
        "description": "Transient or frequency-dependent circuit response involving reactive elements, time constants, and impedance-based reasoning.",
        "includes": ["transient response", "time constant", "reactive impedance", "RLC resonance or damping"],
        "excludes": ["static DC branch solving without dynamics", "mutual inductive coupling as the dominant mechanism", "resistance-only geometry construction"],
        "related_topics": [
            "Electromagnetism::DC Circuits and Kirchhoff's Laws",
            "Electromagnetism::Self-Inductance and Mutual Inductance",
            "Electromagnetism::Current, Resistance, and Ohm's Law",
        ],
    },
    "optics::snell's law and critical angle": {
        "description": "Refraction, critical-angle behavior, and optical-path reasoning in media with sharp or continuous refractive-index variation.",
        "includes": ["Snell-like path bending", "critical angle", "gradient index medium", "mirage-like refractive paths"],
        "excludes": ["generic circuit-wave problems", "laser cavity mode selection"],
        "related_topics": [
            "Optics::Interference (Young's Double Slit, Thin Films)",
            "Mechanics::Fluid Statics (Buoyancy, Pressure in Fluids)",
        ],
    },
    "optics::laser principles and applications": {
        "description": "Laser operation, resonator conditions, coherent propagation, and optical devices where cavity structure or coherent source behavior is central.",
        "includes": ["laser cavity", "ring resonator", "mode selection", "coherent optical source behavior"],
        "excludes": ["generic interference without active cavity", "plain refraction problems"],
        "related_topics": [
            "Optics::Interference (Young's Double Slit, Thin Films)",
            "Modern Physics::Special Relativity (Time Dilation, Length Contraction)",
        ],
    },
}

DEFAULT_SCENARIO_CLUSTER_BLUEPRINTS_PATH = REPO_ROOT / "catalogs/scenario_cluster_blueprints.json"


def _load_scenario_cluster_blueprints(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario cluster blueprints must be a dict: {path}")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for topic_key, cluster_defs in payload.items():
        norm_topic_key = norm_text(topic_key).casefold()
        if not norm_topic_key:
            continue
        if not isinstance(cluster_defs, list):
            raise ValueError(f"Scenario cluster blueprints for {topic_key!r} must be a list")
        cleaned_clusters: List[Dict[str, Any]] = []
        for cluster_def in cluster_defs:
            if isinstance(cluster_def, dict):
                cleaned_clusters.append(cluster_def)
        out[norm_topic_key] = cleaned_clusters
    return out


def merge_scenario_cluster_blueprints(*sources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for topic_key, cluster_defs in source.items():
            norm_topic_key = norm_text(topic_key).casefold()
            if not norm_topic_key:
                continue
            cleaned_clusters = [cluster for cluster in (cluster_defs or []) if isinstance(cluster, dict)]
            merged.setdefault(norm_topic_key, []).extend(cleaned_clusters)
    return merged


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_key(value: Any) -> str:
    return norm_text(value).lower()


def _slug(value: Any) -> str:
    text = norm_text(value).casefold()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _normalize_topic(domain: str, topic: str) -> str:
    norm_domain = norm_text(domain)
    norm_topic = norm_text(topic)
    if "/" in norm_topic:
        left, right = [part.strip() for part in norm_topic.split("/", 1)]
        if left.casefold() == norm_domain.casefold():
            norm_topic = right
    return norm_topic


def _topic_key(domain: str, topic: str) -> str:
    return f"{_norm_key(domain)}::{_norm_key(topic)}"


DISTILLED_TOPIC_ALIASES: Dict[Tuple[str, str], Tuple[str, str]] = {
    (
        "thermodynamics & statistical physics",
        "thermodynamics / thermodynamic processes (isothermal, adiabatic, etc.)",
    ): (
        "Thermodynamics & Statistical Physics",
        "Thermodynamic Processes (Isothermal, Adiabatic, etc.)",
    ),
    (
        "electromagnetism",
        "poynting vector and energy transport",
    ): (
        "Electromagnetism",
        "Poynting Vector and Radiation Pressure",
    ),
    (
        "mechanics",
        "dimensional analysis and scaling",
    ): (
        "Experimental Physics",
        "Dimensional Analysis and Scaling",
    ),
    (
        "mechanics",
        "central forces and orbital motion",
    ): (
        "Mechanics",
        "Gravitation and Kepler's Laws",
    ),
}


def _resolve_distilled_topic(domain: str, topic: str) -> Tuple[str, str]:
    norm_domain = norm_text(domain) or "Unknown"
    norm_topic = _normalize_topic(norm_domain, topic)
    alias = DISTILLED_TOPIC_ALIASES.get((_norm_key(norm_domain), _norm_key(norm_topic)))
    if alias:
        return alias
    alias = DISTILLED_TOPIC_ALIASES.get((_norm_key(norm_domain), _norm_key(topic)))
    if alias:
        return alias
    return norm_domain, norm_topic


def _safe_symbolic_hint(raw_hint: Any) -> Dict[str, Any]:
    hint = raw_hint if isinstance(raw_hint, dict) else {}
    primitive = norm_text(hint.get("primitive") or "none") or "none"
    canonical = norm_text(hint.get("canonical") or "")
    required_symbols = ordered_unique(str(item) for item in (hint.get("required_symbols") or []))
    return {
        "primitive": primitive,
        "canonical": canonical,
        "required_symbols": required_symbols,
    }


def _build_match_features(title: str, trigger: str, check_logic: str, symbolic_hint: Dict[str, Any]) -> Dict[str, Any]:
    raw_required_symbols = ordered_unique(str(item) for item in symbolic_hint.get("required_symbols", []))
    primitive = norm_text(symbolic_hint.get("primitive") or "none") or "none"

    def _is_scene_token(token: str) -> bool:
        lowered = norm_text(token).casefold()
        if not lowered:
            return False
        if lowered in {hint.casefold() for hint in PHYSICAL_CONTEXT_HINTS}:
            return True
        return any(lowered in hint or hint in lowered for hint in PHYSICAL_CONTEXT_HINTS if " " not in hint)

    def _keep_keyword(token: str, *, allow_physical_short: bool = False) -> bool:
        item = norm_text(token)
        if not item:
            return False
        lowered = item.casefold()
        if lowered in {"or", "and", "with", "when", "where", "very", "short", "opening", "time"}:
            return False
        if re.fullmatch(r"[A-Za-z]{1,3}", item):
            return allow_physical_short and _is_scene_token(item)
        if item.isalpha() and len(item) < 4:
            return allow_physical_short and _is_scene_token(item)
        return True

    trigger_keywords = [
        item
        for item in extract_keywords([title, trigger], max_keywords=12)
        if _keep_keyword(item, allow_physical_short=True)
    ][:8]
    object_keywords = [
        item
        for item in extract_keywords([check_logic], max_keywords=12)
        if _keep_keyword(item, allow_physical_short=False)
    ][:8]

    scene_trigger_terms = build_scene_keywords(
        topic_name=title,
        tagged_titles=[trigger],
        tagged_aliases=[],
        rule_texts=[check_logic],
    )[:8]
    scene_trigger_terms = [
        item
        for item in scene_trigger_terms
        if len(norm_text(item)) >= 4 or _is_scene_token(item)
    ]

    formula_trigger_terms = []
    for item in object_keywords + trigger_keywords:
        value = norm_text(item)
        if not value:
            continue
        if re.search(r"[A-Z]_[A-Z]|[A-Za-z]+/[A-Za-z]+|\d", value):
            formula_trigger_terms.append(value)
            continue
        if len(value) >= 4 and value.casefold() not in {"sqrt", "frac"}:
            formula_trigger_terms.append(value)
    formula_trigger_terms = ordered_unique(formula_trigger_terms)[:8]

    required_symbols = [item for item in raw_required_symbols if is_strong_symbol(item)]
    weak_symbol_terms = [item for item in raw_required_symbols if item not in required_symbols]

    return {
        "trigger_keywords": ordered_unique(trigger_keywords),
        "object_keywords": ordered_unique(object_keywords),
        "scene_trigger_terms": ordered_unique(scene_trigger_terms),
        "formula_trigger_terms": ordered_unique(formula_trigger_terms),
        "required_symbols": required_symbols,
        "weak_symbol_terms": weak_symbol_terms,
        "primitive": primitive,
    }


def _build_topic_skeleton(knowledge_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    domains_out: List[Dict[str, Any]] = []
    states: Dict[str, Dict[str, Any]] = {}

    for domain in knowledge_data.get("domains", []) or []:
        domain_name = norm_text(domain.get("name") or "Unknown")
        semantic_profile = DOMAIN_SEMANTIC_PROFILES.get(
            domain_name,
            {
                "description": f"Physics domain covering {domain_name.lower()} topics.",
                "includes": [],
                "excludes": [],
            },
        )
        domain_out = {
            "name": domain_name,
            "description": semantic_profile["description"],
            "includes": ordered_unique(semantic_profile.get("includes") or []),
            "excludes": ordered_unique(semantic_profile.get("excludes") or []),
            "topics": [],
        }

        for topic in domain.get("topics", []) or []:
            topic_name = norm_text(topic.get("name") or "Unknown")
            knowledge_rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            knowledge_rule_ids = ordered_unique(str(item.get("id") or "") for item in knowledge_rules)
            knowledge_texts: List[str] = [topic_name]
            for item in knowledge_rules:
                knowledge_texts.extend(
                    [
                        str(item.get("title") or ""),
                        str(item.get("description") or ""),
                        str(item.get("check_logic") or ""),
                    ]
                )

            entry = {
                "name": topic_name,
                "description": "",
                "includes": [],
                "excludes": [],
                "related_topics": [],
                "rules": [],
                "knowledge_reference": {
                    "rule_ids": knowledge_rule_ids,
                    "keywords": extract_keywords(knowledge_texts, max_keywords=16),
                },
                "tagged_reference": {
                    "source_ids": [],
                    "titles": [],
                    "aliases": [],
                    "keywords": [],
                },
                "retrieval_hints": {
                    "topic_keywords": [],
                    "required_symbols": [],
                },
                "scenario_clusters": [],
            }
            domain_out["topics"].append(entry)

            key = _topic_key(domain_name, topic_name)
            states[key] = {
                "domain": domain_name,
                "topic": topic_name,
                "entry": entry,
                "tagged_keyword_texts": [],
            }

        domains_out.append(domain_out)

    return domains_out, states


def _attach_distilled_rules(states: Dict[str, Dict[str, Any]], distilled_data: Dict[str, Any]) -> None:
    metadata = distilled_data.get("metadata") if isinstance(distilled_data, dict) else {}
    if isinstance(metadata, dict) and metadata.get("generator") == "experience_candidate_generalizer_v1":
        if metadata.get("scope_mode") != "full":
            raise ValueError("Filtered candidate generalization output cannot be used to build a formal catalog.")
        if metadata.get("complete") is not True:
            raise ValueError("Incomplete candidate generalization output cannot be used to build a formal catalog.")
        if metadata.get("formal_support_validated") is not True:
            raise ValueError("Generalized rules must pass formal multi-source support validation before catalog build.")

    raw_rules = distilled_data.get("rules") if isinstance(distilled_data, dict) else []
    if not isinstance(raw_rules, list):
        raise ValueError("Distilled experience data must contain a top-level 'rules' list.")

    unmatched: List[str] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue

        domain, topic = _resolve_distilled_topic(
            str(raw_rule.get("domain") or "Unknown"),
            str(raw_rule.get("topic") or "Unknown"),
        )
        key = _topic_key(domain, topic)
        state = states.get(key)
        if state is None:
            unmatched.append(f"{domain} / {topic} :: {raw_rule.get('rule_id')}")
            continue

        symbolic_hint = _safe_symbolic_hint(raw_rule.get("symbolic_hint"))
        title = norm_text(raw_rule.get("title") or "")
        trigger = norm_text(raw_rule.get("trigger") or "")
        check_logic = norm_text(raw_rule.get("check_logic") or "")
        rule_leaf = {
            "rule_id": norm_text(raw_rule.get("rule_id") or ""),
            "title": title,
            "summary": norm_text(raw_rule.get("summary") or ""),
            "trigger": trigger,
            "check_logic": check_logic,
            "error_type": norm_text(raw_rule.get("error_type") or "logic") or "logic",
            "scope": classify_rule_scope(
                title=title,
                trigger=trigger,
                check_logic=check_logic,
            ),
            "symbolic_hint": symbolic_hint,
                "support": {
                    "count": int(raw_rule.get("count") or 0),
                    "sample_ids": ordered_unique(str(item) for item in (raw_rule.get("sample_ids") or [])),
                },
            "match_features": _build_match_features(title, trigger, check_logic, symbolic_hint),
            "retrieval_flags": {},
        }
        rule_leaf = normalize_rule_for_retrieval(rule_leaf)
        state["entry"]["rules"].append(rule_leaf)

    if unmatched:
        uniq = ordered_unique(unmatched)
        preview = "\n".join(f"- {item}" for item in uniq[:10])
        raise ValueError(
            "Distilled rules contain topics that do not map to the knowledge skeleton.\n"
            f"Unmatched unique topics/rules: {len(uniq)}\n{preview}"
        )


def _attach_tagged_reference(states: Dict[str, Dict[str, Any]], tagged_data: Any) -> int:
    if not isinstance(tagged_data, list):
        raise ValueError("Tagged experience data must be a JSON list.")

    mapped_rules = 0
    for raw_rule in tagged_data:
        if not isinstance(raw_rule, dict):
            continue
        tags = raw_rule.get("tags") if isinstance(raw_rule.get("tags"), dict) else {}
        domain = norm_text(tags.get("domain") or "Unknown")
        topic = _normalize_topic(domain, str(tags.get("topic") or "Unknown"))
        key = _topic_key(domain, topic)
        state = states.get(key)
        if state is None:
            continue

        title = norm_text(raw_rule.get("title") or "")
        description = norm_text(raw_rule.get("description") or "")
        source_id = norm_text(raw_rule.get("id") or "")

        ref = state["entry"]["tagged_reference"]
        ref["source_ids"].append(source_id)
        ref["titles"].append(title)
        ref["aliases"].append(title)
        state["tagged_keyword_texts"].extend([title, description])
        mapped_rules += 1

    for state in states.values():
        ref = state["entry"]["tagged_reference"]
        ref["source_ids"] = ordered_unique(ref["source_ids"])
        ref["titles"] = ordered_unique(ref["titles"])
        ref["aliases"] = ordered_unique(ref["aliases"])
        ref["keywords"] = extract_keywords(state["tagged_keyword_texts"], max_keywords=16)

    return mapped_rules


def _rule_text_for_cluster(rule: Dict[str, Any]) -> str:
    return " ".join(
        [
            norm_text(rule.get("title") or ""),
            norm_text(rule.get("trigger") or ""),
            norm_text(rule.get("check_logic") or ""),
        ]
    ).casefold()


def _build_scenario_clusters(rules: List[Dict[str, Any]], cluster_blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rule_texts = {str(rule.get("rule_id") or ""): _rule_text_for_cluster(rule) for rule in rules}
    assigned_rule_ids: set[str] = set()
    clusters: List[Dict[str, Any]] = []

    for blueprint in cluster_blueprints:
        group_defs = blueprint.get("rule_groups") if isinstance(blueprint.get("rule_groups"), list) else []
        built_groups: List[Dict[str, Any]] = []
        cluster_rule_ids: List[str] = []
        for group_def in group_defs:
            if not isinstance(group_def, dict):
                continue
            explicit_rule_ids = [
                norm_text(item) for item in (group_def.get("rule_ids") or []) if norm_text(item)
            ]
            matched_ids: List[str] = []
            if explicit_rule_ids:
                matched_ids = [
                    rule_id
                    for rule_id in explicit_rule_ids
                    if rule_id in rule_texts and rule_id not in assigned_rule_ids
                ]
            else:
                match_any = [norm_text(item).casefold() for item in (group_def.get("match_any") or []) if norm_text(item)]
                for rule in rules:
                    rule_id = str(rule.get("rule_id") or "")
                    if not rule_id or rule_id in assigned_rule_ids:
                        continue
                    haystack = rule_texts.get(rule_id, "")
                    if any(token in haystack for token in match_any):
                        matched_ids.append(rule_id)
            if not matched_ids:
                continue
            built_groups.append(
                {
                    "group_id": norm_text(group_def.get("group_id") or ""),
                    "name": norm_text(group_def.get("name") or ""),
                    "summary": norm_text(group_def.get("summary") or ""),
                    "activation_condition": norm_text(group_def.get("activation_condition") or ""),
                    "rule_ids": matched_ids,
                }
            )
            cluster_rule_ids.extend(matched_ids)
            assigned_rule_ids.update(matched_ids)
        if not built_groups and rules:
            continue
        clusters.append(
            {
                "cluster_id": norm_text(blueprint.get("cluster_id") or ""),
                "name": norm_text(blueprint.get("name") or ""),
                "description": norm_text(blueprint.get("description") or ""),
                "includes": ordered_unique(blueprint.get("includes") or []),
                "excludes": ordered_unique(blueprint.get("excludes") or []),
                "entry_cues": ordered_unique(blueprint.get("entry_cues") or []),
                "related_clusters": ordered_unique(blueprint.get("related_clusters") or []),
                "rule_groups": built_groups,
                "rule_ids": ordered_unique(cluster_rule_ids),
            }
        )

    remaining_rule_ids = [str(rule.get("rule_id") or "") for rule in rules if str(rule.get("rule_id") or "") not in assigned_rule_ids]
    if remaining_rule_ids:
        fallback_group = {
            "group_id": "general_reasoning_checks",
            "name": "General Topic Reasoning Checks",
            "summary": "Residual topic-specific checks kept outside the first-pass scenario clusters.",
            "activation_condition": "Use only if the problem is clearly in this topic but not in a more specific scenario cluster.",
            "rule_ids": remaining_rule_ids,
        }
        existing_general = next((cluster for cluster in clusters if cluster.get("cluster_id") == "general_reasoning"), None)
        if existing_general is not None:
            existing_general.setdefault("rule_groups", []).append(fallback_group)
            existing_general["rule_ids"] = ordered_unique(list(existing_general.get("rule_ids") or []) + remaining_rule_ids)
        else:
            clusters.append(
                {
                    "cluster_id": "general_reasoning",
                    "name": "General Topic Reasoning",
                    "description": "Fallback cluster for rules that belong to the topic but do not fit the first-pass scenario-specific buckets.",
                    "includes": ["topic-specific residual checks"],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [fallback_group],
                    "rule_ids": remaining_rule_ids,
                }
            )
    return clusters


def _summarize_rule(rule: Dict[str, Any]) -> str:
    summary = norm_text(rule.get("summary") or "")
    if summary:
        return summary
    title = norm_text(rule.get("title") or "")
    trigger = norm_text(rule.get("trigger") or "")
    if title and trigger:
        return f"{title}: {trigger}"
    return title or trigger or norm_text(rule.get("check_logic") or "")


def _project_navigation_rule(rule: Dict[str, Any], *, legacy_symbolic: bool = True) -> Dict[str, Any]:
    trigger = norm_text(rule.get("trigger") or "")
    leaf = {
        "rule_id": norm_text(rule.get("rule_id") or ""),
        "title": norm_text(rule.get("title") or ""),
        "summary": _summarize_rule(rule),
        "trigger": trigger,
        "check_logic": norm_text(rule.get("check_logic") or ""),
        "error_type": norm_text(rule.get("error_type") or "logic") or "logic",
    }
    if legacy_symbolic:
        hint = rule.get("symbolic_hint")
        leaf["symbolic_hint"] = hint if isinstance(hint, dict) else {}
    return leaf


def _project_navigation_group(group: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": norm_text(group.get("group_id") or ""),
        "name": norm_text(group.get("name") or ""),
        "summary": norm_text(group.get("summary") or ""),
        "activation_condition": norm_text(group.get("activation_condition") or ""),
        "rule_ids": ordered_unique(group.get("rule_ids") or []),
    }


def _project_navigation_cluster(cluster: Dict[str, Any]) -> Dict[str, Any]:
    groups = [group for group in (cluster.get("rule_groups") or []) if isinstance(group, dict)]
    group_summary = norm_text(groups[0].get("summary") or "") if groups else ""
    summary = group_summary or norm_text(cluster.get("summary") or cluster.get("description") or cluster.get("name") or "")
    projected: Dict[str, Any] = {
        "id": norm_text(cluster.get("cluster_id") or ""),
        "name": norm_text(cluster.get("name") or ""),
        "summary": summary,
        "rule_ids": ordered_unique(cluster.get("rule_ids") or []),
    }
    rule_groups = [
        _project_navigation_group(group)
        for group in groups
    ]
    if rule_groups:
        projected["rule_groups"] = rule_groups
    return projected


def _project_navigation_topic(domain_id: str, domain_name: str, topic: Dict[str, Any]) -> Dict[str, Any]:
    topic_name = norm_text(topic.get("name") or "Unknown")
    projected: Dict[str, Any] = {
        "id": f"{domain_id}.{_slug(topic_name)}",
        "name": topic_name,
        "summary": norm_text(topic.get("description") or f"Topic for {topic_name} under {domain_name}."),
        "scenario_clusters": [
            _project_navigation_cluster(cluster)
            for cluster in (topic.get("scenario_clusters") or [])
            if isinstance(cluster, dict)
        ],
        "rules": [
            _project_navigation_rule(rule)
            for rule in (topic.get("rules") or [])
            if isinstance(rule, dict)
        ],
    }
    return projected


def _project_navigation_domains(domains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    projected_domains: List[Dict[str, Any]] = []
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        domain_name = norm_text(domain.get("name") or "Unknown")
        domain_id = _slug(domain_name)
        projected_domains.append(
            {
                "id": domain_id,
                "name": domain_name,
                "summary": norm_text(domain.get("description") or f"Physics domain covering {domain_name}."),
                "topics": [
                    _project_navigation_topic(domain_id, domain_name, topic)
                    for topic in (domain.get("topics") or [])
                    if isinstance(topic, dict)
                ],
            }
        )
    return projected_domains


def _finalize_topics(
    states: Dict[str, Dict[str, Any]],
    scenario_cluster_blueprints: Dict[str, List[Dict[str, Any]]],
) -> None:
    for state in states.values():
        entry = state["entry"]
        rules = entry["rules"]
        rules.sort(
            key=lambda item: (
                -int((item.get("support") or {}).get("count", 0)),
                str(item.get("rule_id") or ""),
            )
        )

        domain_rule_texts: List[str] = []
        for rule in rules:
            if norm_text(rule.get("scope") or "domain") != "domain":
                continue
            domain_rule_texts.extend(
                [
                    norm_text(rule.get("title") or ""),
                    norm_text(rule.get("trigger") or ""),
                ]
            )

        topic_keywords = ordered_unique(
            list(extract_keywords([state["topic"]], max_keywords=8))
            + list(entry["knowledge_reference"].get("keywords") or [])
            + list(entry["tagged_reference"].get("keywords") or [])
        )[:20]
        scene_keywords = build_scene_keywords(
            topic_name=state["topic"],
            tagged_titles=entry["tagged_reference"].get("titles") or [],
            tagged_aliases=entry["tagged_reference"].get("aliases") or [],
            rule_texts=domain_rule_texts,
        )
        scene_keywords, topic_keywords = refine_topic_hints(
            scene_keywords=scene_keywords,
            topic_keywords=topic_keywords,
            rule_texts=domain_rule_texts,
        )

        entry["retrieval_hints"] = {
            "scene_keywords": scene_keywords,
            "topic_keywords": topic_keywords,
            "required_symbols": build_topic_required_symbols(rules),
        }
        topic_key = _topic_key(state["domain"], state["topic"])
        structure_override = FOCUS_TOPIC_STRUCTURE.get(topic_key, {})
        blueprint_clusters = scenario_cluster_blueprints.get(topic_key, [])
        if structure_override:
            entry["description"] = norm_text(structure_override.get("description") or "")
            entry["includes"] = ordered_unique(structure_override.get("includes") or [])
            entry["excludes"] = ordered_unique(structure_override.get("excludes") or [])
            entry["related_topics"] = ordered_unique(structure_override.get("related_topics") or [])
            entry["scenario_clusters"] = _build_scenario_clusters(
                rules,
                blueprint_clusters,
            )
        elif blueprint_clusters:
            entry["description"] = f"Topic for {state['topic']} under {state['domain']}."
            entry["includes"] = ordered_unique((entry["retrieval_hints"].get("scene_keywords") or [])[:4] + topic_keywords[:4])
            entry["excludes"] = []
            entry["related_topics"] = []
            entry["scenario_clusters"] = _build_scenario_clusters(rules, blueprint_clusters)
        else:
            entry["description"] = f"Topic for {state['topic']} under {state['domain']}."
            entry["includes"] = ordered_unique((entry["retrieval_hints"].get("scene_keywords") or [])[:4] + topic_keywords[:4])
            entry["excludes"] = []
            entry["related_topics"] = []
            entry["scenario_clusters"] = _build_scenario_clusters(rules, [])


def build_unified_catalog_from_data(
    knowledge_data: Dict[str, Any],
    distilled_data: Dict[str, Any],
    tagged_data: Any,
    scenario_cluster_blueprints: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    scenario_cluster_blueprints = scenario_cluster_blueprints or {}
    domains_out, states = _build_topic_skeleton(knowledge_data)
    _attach_distilled_rules(states, distilled_data)
    mapped_tagged_rules = _attach_tagged_reference(states, tagged_data)
    _finalize_topics(states, scenario_cluster_blueprints)

    total_topics = sum(len(domain["topics"]) for domain in domains_out)
    total_rules = sum(len(topic["rules"]) for domain in domains_out for topic in domain["topics"])
    topics_with_rules = sum(1 for domain in domains_out for topic in domain["topics"] if topic["rules"])
    total_scenario_clusters = sum(len(topic["scenario_clusters"]) for domain in domains_out for topic in domain["topics"])
    scenario_clustered_topics = sum(1 for domain in domains_out for topic in domain["topics"] if topic["scenario_clusters"])
    knowledge_rule_references = sum(
        len(topic["knowledge_reference"]["rule_ids"]) for domain in domains_out for topic in domain["topics"]
    )
    navigation_domains = _project_navigation_domains(domains_out)

    return {
        "metadata": {
            "version": "2.0",
            "catalog_type": "unified_rules_v2",
            "schema_profile": "semantic_navigation_tree_minimal",
            "generated_at": _dt.datetime.now().isoformat(),
            "total_domains": len(domains_out),
            "total_topics": total_topics,
            "topics_with_rules": topics_with_rules,
            "total_executable_rules": total_rules,
            "knowledge_rule_references": knowledge_rule_references,
            "mapped_tagged_reference_rules": mapped_tagged_rules,
            "scenario_clustered_topics": scenario_clustered_topics,
            "total_scenario_clusters": total_scenario_clusters,
        },
        "domains": navigation_domains,
    }


def build_unified_catalog(
    knowledge_path: Path,
    distilled_path: Path,
    tagged_path: Path,
    scenario_cluster_blueprints_path: Path = DEFAULT_SCENARIO_CLUSTER_BLUEPRINTS_PATH,
    scenario_cluster_blueprints_paths: Sequence[Path] | None = None,
) -> Dict[str, Any]:
    knowledge_data = _load_json(knowledge_path)
    distilled_data = _load_json(distilled_path)
    tagged_data = _load_json(tagged_path)
    blueprint_paths = list(scenario_cluster_blueprints_paths or [scenario_cluster_blueprints_path])
    scenario_cluster_blueprints = merge_scenario_cluster_blueprints(
        *[_load_scenario_cluster_blueprints(path) for path in blueprint_paths]
    )
    return build_unified_catalog_from_data(knowledge_data, distilled_data, tagged_data, scenario_cluster_blueprints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified_rules v2 from knowledge skeleton and distilled rules.")
    parser.add_argument("--knowledge", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--experience-tagged", type=str, default="catalogs/rules_300_tagged.json")
    parser.add_argument("--experience-distilled", type=str, default="catalogs/semantic_experience_distilled_300.json")
    parser.add_argument(
        "--scenario-cluster-blueprints",
        action="append",
        default=[],
        help="Repeat to merge multiple blueprint sources. First source is typically the seed blueprint; later sources can be generated blueprints.",
    )
    parser.add_argument("--output", "-o", type=str, default="catalogs/rules_unified.json")
    args = parser.parse_args()

    catalog = build_unified_catalog(
        knowledge_path=Path(args.knowledge),
        distilled_path=Path(args.experience_distilled),
        tagged_path=Path(args.experience_tagged),
        scenario_cluster_blueprints_paths=[
            Path(item) for item in (args.scenario_cluster_blueprints or ["catalogs/scenario_cluster_blueprints.json"])
        ],
    )
    output_path = Path(args.output)
    _write_json(output_path, catalog)

    meta = catalog["metadata"]
    print(f"Done. Unified v2 catalog written to: {output_path}")
    print(f"  Domains:             {meta['total_domains']}")
    print(f"  Topics:              {meta['total_topics']}")
    print(f"  Topics with rules:   {meta['topics_with_rules']}")
    print(f"  Executable rules:    {meta['total_executable_rules']}")
    print(f"  Tagged refs mapped:  {meta['mapped_tagged_reference_rules']}")
    print(f"  Scenario topics:     {meta['scenario_clustered_topics']}")
    print(f"  Scenario clusters:   {meta['total_scenario_clusters']}")


if __name__ == "__main__":
    main()
