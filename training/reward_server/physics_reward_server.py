#!/usr/bin/env python3
"""PhysicsVerifier reward server for OpenRLHF and legacy HTTP clients.

Endpoints:
  POST /            - score single sample (slime)
  POST /batch       - score batch of samples (slime)
  POST /get_reward  - OpenRLHF remote RM ({query, prompts, labels} -> rewards)

Payload (single /):
  {prompt, response, label, question, ...}

Response (/):
  {score, acc, n_errors, diagnostics_summary, ...}

OpenRLHF (/get_reward):
  Request:  {query: [prompt+response...], prompts: [...], labels: [...]}
  Response: {rewards: [...], scores: [...], extra_logs: {...}}
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Project root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_project_env() -> None:
    """Load project-local configuration before reward constants are frozen."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


_load_project_env()

from core.physics_rule_verifier import PhysicsRuleVerifier
from training.compat.math_grading import extract_answer, grade_answer_verl

app = FastAPI(title="PhysicsVerifier Reward Server")

_verifier_local = threading.local()
_semaphore: Optional[asyncio.Semaphore] = None

DEFAULT_LAMBDA = float(os.environ.get("PHYSICS_REWARD_LAMBDA", "0.3"))
DEFAULT_CAP = int(os.environ.get("PHYSICS_REWARD_ERROR_CAP", "3"))
DEFAULT_CONCURRENCY = int(os.environ.get("PHYSICS_REWARD_CONCURRENCY", "4"))
REWARD_MODE = os.environ.get("PHYSICS_REWARD_MODE", "answer_low_verifier").strip().lower()
W_ANSWER = float(os.environ.get("PHYSICS_REWARD_W_ANSWER", "1.0"))
W_FORMAT = float(os.environ.get("PHYSICS_REWARD_W_FORMAT", "0.05"))
W_VERIFIER = float(os.environ.get("PHYSICS_REWARD_W_VERIFIER", "0.1"))
W_LENGTH = float(os.environ.get("PHYSICS_REWARD_W_LENGTH", "0.0"))
VERIFIER_SAMPLE_RATE = float(os.environ.get("PHYSICS_VERIFIER_SAMPLE_RATE", "1.0"))
MAX_RESPONSE_CHARS = int(os.environ.get("PHYSICS_REWARD_MAX_RESPONSE_CHARS", "12000"))
METRICS_LOG = os.environ.get("PHYSICS_REWARD_METRICS_LOG", str(ROOT / "logs/physics_reward_metrics.jsonl"))
UNIFIED_RULES_PATH = os.environ.get("PHYSICSVERIFIER_UNIFIED_RULES", "").strip()
CHECKER_GATE_MODE = os.environ.get("PHYSICSVERIFIER_CHECKER_GATE_MODE", "legacy").strip().lower()
CHECKER_JSON_ATTEMPTS = int(os.environ.get("PHYSICSVERIFIER_CHECKER_JSON_ATTEMPTS", "3"))
SEMANTIC_JSON_ATTEMPTS = int(os.environ.get("PHYSICSVERIFIER_SEMANTIC_JSON_ATTEMPTS", "3"))
UNIFIED_RULE_TOP_N = int(os.environ.get("PHYSICSVERIFIER_UNIFIED_RULE_TOP_N", "6"))
PRECISION_MODE = os.environ.get("PHYSICSVERIFIER_PRECISION_MODE", "strict").strip().lower()
MAX_DIAGNOSTICS_PER_SAMPLE = int(
    os.environ.get("PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_SAMPLE", "12")
)
MAX_DIAGNOSTICS_PER_PARAGRAPH = int(
    os.environ.get("PHYSICSVERIFIER_MAX_DIAGNOSTICS_PER_PARAGRAPH", "2")
)
ENABLE_LLM_CACHE = os.environ.get("PHYSICSVERIFIER_ENABLE_LLM_CACHE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REQUIRE_PROVIDER_IDENTITY = os.environ.get(
    "PHYSICSVERIFIER_REQUIRE_PROVIDER_IDENTITY", "1"
).strip().lower() in {"1", "true", "yes", "on"}
SYMBOLIC_ENABLED = os.environ.get("PHYSICSVERIFIER_SYMBOLIC_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VERIFIER_FAILURE_POLICY = os.environ.get(
    "PHYSICS_REWARD_VERIFIER_FAILURE_POLICY", "raise"
).strip().lower()

VALID_CHECKER_STATUSES = {
    "complete_no_rules",
    "valid_empty",
    "valid_with_diagnostics",
}
VALID_REWARD_MODES = {
    "answer_only",
    "answer_low_verifier",
    "answer_full_verifier",
}


def _validate_runtime_config() -> None:
    if REWARD_MODE not in VALID_REWARD_MODES:
        raise ValueError(
            "PHYSICS_REWARD_MODE must be one of "
            + ", ".join(sorted(VALID_REWARD_MODES))
        )
    if VERIFIER_FAILURE_POLICY not in {"raise", "zero_reward"}:
        raise ValueError(
            "PHYSICS_REWARD_VERIFIER_FAILURE_POLICY must be 'raise' or 'zero_reward'"
        )
    if DEFAULT_CAP < 1:
        raise ValueError("PHYSICS_REWARD_ERROR_CAP must be at least 1")
    if DEFAULT_CONCURRENCY < 1:
        raise ValueError("PHYSICS_REWARD_CONCURRENCY must be at least 1")
    if CHECKER_GATE_MODE not in {
        "legacy",
        "dual_evidence",
        "dual_evidence_consistency",
    }:
        raise ValueError(
            "PHYSICSVERIFIER_CHECKER_GATE_MODE must be legacy, dual_evidence, "
            "or dual_evidence_consistency"
        )
    if not 1 <= CHECKER_JSON_ATTEMPTS <= 5:
        raise ValueError("PHYSICSVERIFIER_CHECKER_JSON_ATTEMPTS must be between 1 and 5")
    if SEMANTIC_JSON_ATTEMPTS < 1:
        raise ValueError("PHYSICSVERIFIER_SEMANTIC_JSON_ATTEMPTS must be at least 1")
    if UNIFIED_RULE_TOP_N < 1:
        raise ValueError("PHYSICSVERIFIER_UNIFIED_RULE_TOP_N must be at least 1")
    if PRECISION_MODE not in {"strict", "balanced", "score_only"}:
        raise ValueError(
            "PHYSICSVERIFIER_PRECISION_MODE must be strict, balanced, or score_only"
        )
    if REQUIRE_PROVIDER_IDENTITY and ENABLE_LLM_CACHE:
        raise ValueError(
            "PHYSICSVERIFIER_ENABLE_LLM_CACHE must be disabled when provider identity is required"
        )
    if not math.isfinite(VERIFIER_SAMPLE_RATE) or not 0.0 <= VERIFIER_SAMPLE_RATE <= 1.0:
        raise ValueError("PHYSICS_VERIFIER_SAMPLE_RATE must be between 0.0 and 1.0")
    for name, value in {
        "PHYSICS_REWARD_LAMBDA": DEFAULT_LAMBDA,
        "PHYSICS_REWARD_W_ANSWER": W_ANSWER,
        "PHYSICS_REWARD_W_FORMAT": W_FORMAT,
        "PHYSICS_REWARD_W_VERIFIER": W_VERIFIER,
        "PHYSICS_REWARD_W_LENGTH": W_LENGTH,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")


_validate_runtime_config()


class VerifierExecutionError(RuntimeError):
    """Raised when a requested Verifier check did not complete reliably."""


class ScoreRequest(BaseModel):
    prompt: str = ""
    response: str = ""
    label: Optional[List[str] | str] = None
    question: Optional[str] = None
    points: Optional[List[float]] = None
    marking: Any = None
    marking_mode: str = "total_score"
    use_xverify: bool = False


class BatchRequest(BaseModel):
    requests: List[ScoreRequest]


def _normalize_label(label: Optional[List[str] | str]) -> List[str]:
    if label is None:
        return []
    if isinstance(label, list):
        return [str(x) for x in label if x is not None]
    text = str(label).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [text]


def _extract_question(req: ScoreRequest) -> str:
    if req.question and req.question.strip():
        return req.question.strip()
    prompt = req.prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        return "\n".join(parts).strip()
    return str(prompt).strip()


def _check_answer(response: str, labels: List[str]) -> tuple[bool, str, str]:
    if not labels:
        return False, "", ""
    extracted_pred = extract_answer(response) or ""
    for gt in labels:
        if grade_answer_verl(response, gt):
            extracted_gt = extract_answer(gt) or str(gt)
            return True, extracted_pred, extracted_gt
    gt0 = labels[0]
    extracted_gt = extract_answer(gt0) or str(gt0)
    return False, extracted_pred, extracted_gt


def _count_error_diagnostics(result: Dict[str, Any]) -> tuple[int, List[Dict[str, str]]]:
    diagnostics = result.get("diagnostics") or []
    errors = [d for d in diagnostics if str(d.get("severity", "")).lower() == "error"]
    summary = []
    for d in errors[:5]:
        summary.append(
            {
                "rule": str(d.get("rule", "")),
                "message": str(d.get("message", ""))[:200],
            }
        )
    return len(errors), summary


def _validate_verifier_result(result: Any) -> Dict[str, Any]:
    """Reject incomplete checks instead of treating them as clean solutions."""
    if not isinstance(result, dict):
        raise VerifierExecutionError("Verifier returned a non-object result")

    semantic_error = str(result.get("semantic_selection_error") or "").strip()
    if semantic_error:
        raise VerifierExecutionError(f"semantic retrieval failed: {semantic_error}")
    if str(result.get("selection_strategy") or "").strip() == "semantic_error":
        raise VerifierExecutionError("semantic retrieval ended with semantic_error")

    checker_status = str(result.get("checker_status") or "").strip().lower()
    if checker_status not in VALID_CHECKER_STATUSES:
        raise VerifierExecutionError(
            f"Verifier checker did not complete (checker_status={checker_status or '<empty>'})"
        )
    checker_failures = result.get("checker_failures") or []
    if checker_failures:
        raise VerifierExecutionError(
            f"Verifier reported {len(checker_failures)} checker failure(s)"
        )

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise VerifierExecutionError("Verifier result is missing a diagnostics list")
    if any(not isinstance(item, dict) for item in diagnostics):
        raise VerifierExecutionError("Verifier diagnostics must be objects")
    return result


def _compute_score(acc: bool, n_errors: int, *, lam: float, cap: int) -> float:
    penalty = min(n_errors, cap) / max(cap, 1)
    return (1.0 if acc else 0.0) - lam * penalty


def _reward_weights() -> Dict[str, float]:
    mode = REWARD_MODE
    if mode == "answer_only":
        return {"answer": 1.0, "format": 0.0, "verifier": 0.0, "length": 0.0}
    if mode == "answer_full_verifier":
        return {
            "answer": max(W_ANSWER, 0.0),
            "format": max(W_FORMAT, 0.0),
            "verifier": max(W_VERIFIER, DEFAULT_LAMBDA),
            "length": max(W_LENGTH, 0.0),
        }
    return {
        "answer": max(W_ANSWER, 0.0),
        "format": max(W_FORMAT, 0.0),
        "verifier": max(W_VERIFIER, 0.1),
        "length": max(W_LENGTH, 0.0),
    }


def _format_component(response: str) -> float:
    extracted = extract_answer(response) or ""
    return 1.0 if extracted else 0.0


def _length_penalty(response: str) -> float:
    if MAX_RESPONSE_CHARS <= 0:
        return 0.0
    over = max(len(response) - MAX_RESPONSE_CHARS, 0)
    return min(over / max(MAX_RESPONSE_CHARS, 1), 1.0)


def _should_run_verifier(acc: bool, sample_idx: int, sample_key: str = "") -> bool:
    if not acc:
        return False
    weights = _reward_weights()
    if weights["verifier"] <= 0.0:
        return False
    if VERIFIER_SAMPLE_RATE >= 1.0:
        return True
    if VERIFIER_SAMPLE_RATE <= 0.0:
        return False
    # OpenRLHF splits rewards into repeated micro-batches, so sampling by the
    # local batch index systematically favors the same rollout positions.
    # A content hash is deterministic across retries while remaining unbiased
    # with respect to batch layout.
    material = f"{sample_idx}\0{sample_key}".encode("utf-8", errors="surrogatepass")
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / float(1 << 64)
    return draw < VERIFIER_SAMPLE_RATE


def _append_metrics(record: Dict[str, Any]) -> None:
    try:
        path = Path(METRICS_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_verifier() -> PhysicsRuleVerifier:
    """Return one mutable verifier per executor thread.

    ``PhysicsRuleVerifier.verify`` updates per-request rule selections and trace
    state. Sharing one cached instance across concurrent reward calls can mix
    those states, so each worker thread owns its own instance.
    """
    verifier = getattr(_verifier_local, "instance", None)
    if verifier is None:
        if not UNIFIED_RULES_PATH:
            raise ValueError(
                "PHYSICSVERIFIER_UNIFIED_RULES must explicitly select a catalog "
                "when Verifier reward is enabled"
            )
        verifier = PhysicsRuleVerifier(
            llm_model=os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "qwen3-30b-a3b"),
            enable_llm_cache=ENABLE_LLM_CACHE,
            unified_rules_path=UNIFIED_RULES_PATH,
            experience_code_manifest_path=os.environ.get(
                "PHYSICSVERIFIER_SYMBOLIC_MANIFEST",
                str(ROOT / "results/experience_symbolic_program_manifest_v2_unified.json"),
            ),
            enable_symbolic_check=SYMBOLIC_ENABLED,
            precision_mode=PRECISION_MODE,
            max_diagnostics_per_sample=MAX_DIAGNOSTICS_PER_SAMPLE,
            max_diagnostics_per_paragraph=MAX_DIAGNOSTICS_PER_PARAGRAPH,
            unified_rule_top_n=UNIFIED_RULE_TOP_N,
            unified_retrieval_mode=os.environ.get(
                "PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE",
                "semantic",
            ),
            semantic_json_attempts=SEMANTIC_JSON_ATTEMPTS,
            semantic_output_adapter=os.environ.get("PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER") or None,
            checker_gate_mode=CHECKER_GATE_MODE,
            checker_json_attempts=CHECKER_JSON_ATTEMPTS,
            require_provider_identity=REQUIRE_PROVIDER_IDENTITY,
            expected_provider_model=os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "qwen3-30b-a3b"),
        )
        _verifier_local.instance = verifier
    return verifier


async def score_one(req: ScoreRequest, sample_idx: int = 0) -> Dict[str, Any]:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)

    started = time.time()
    question = _extract_question(req)
    labels = _normalize_label(req.label)
    acc, extracted_pred, extracted_gt = _check_answer(req.response, labels)

    n_errors = 0
    diagnostics_summary: List[Dict[str, str]] = []
    verifier_mode = "skipped"
    verifier_failed = False
    verifier_error = ""

    sample_key = json.dumps(
        {"question": question, "response": req.response},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if _should_run_verifier(acc, sample_idx, sample_key):
        if not question.strip():
            verifier_failed = True
            verifier_mode = "failed"
            verifier_error = "VerifierExecutionError: verifier question is empty"
            diagnostics_summary = [
                {"rule": "verifier_error", "message": verifier_error}
            ]
        else:
            async with _semaphore:
                loop = asyncio.get_running_loop()
                verifier_mode = "full"
                try:
                    result = await loop.run_in_executor(
                        None, lambda: _get_verifier().verify({"question": question, "prediction": req.response})
                    )
                    _validate_verifier_result(result)
                    n_errors, diagnostics_summary = _count_error_diagnostics(result)
                except Exception as exc:
                    verifier_failed = True
                    verifier_mode = "failed"
                    verifier_error = f"{type(exc).__name__}: {exc}"
                    diagnostics_summary = [{"rule": "verifier_error", "message": verifier_error[:200]}]

    if verifier_failed and VERIFIER_FAILURE_POLICY == "raise":
        latency_ms = (time.time() - started) * 1000.0
        _append_metrics(
            {
                "ts": time.time(),
                "acc": acc,
                "score": None,
                "legacy_score": None,
                "verifier_mode": verifier_mode,
                "verifier_error": verifier_error[:500],
                "n_errors": 0,
                "latency_ms": latency_ms,
                "reward_mode": REWARD_MODE,
            }
        )
        raise VerifierExecutionError(verifier_error)

    weights = _reward_weights()
    r_answer = 1.0 if acc else 0.0
    r_format = _format_component(req.response)
    penalty = min(n_errors, DEFAULT_CAP) / max(DEFAULT_CAP, 1)
    r_verifier = -penalty if verifier_mode == "full" else 0.0
    r_length = -_length_penalty(req.response)
    score = (
        weights["answer"] * r_answer
        + weights["format"] * r_format
        + weights["verifier"] * r_verifier
        + weights["length"] * r_length
    )
    legacy_score = _compute_score(acc, n_errors, lam=DEFAULT_LAMBDA, cap=DEFAULT_CAP)
    if verifier_failed and VERIFIER_FAILURE_POLICY == "zero_reward":
        score = 0.0
        legacy_score = 0.0
    latency_ms = (time.time() - started) * 1000.0
    out = {
        "score": score,
        "point": score,
        "acc": acc,
        "n_errors": n_errors,
        "extracted_pred": extracted_pred,
        "extracted_gt": extracted_gt,
        "diagnostics_summary": diagnostics_summary,
        "scored_by": "physics_verifier",
        "verifier_mode": verifier_mode,
        "score_noxverify": legacy_score,
        "point_noxverify": legacy_score,
        "reward_components": {
            "answer": r_answer,
            "format": r_format,
            "verifier_penalty": penalty,
            "verifier": r_verifier,
            "length": r_length,
            "weights": weights,
            "verifier_failed": verifier_failed,
            "verifier_error": verifier_error,
            "verifier_failure_policy": VERIFIER_FAILURE_POLICY,
        },
        "latency_ms": latency_ms,
    }
    _append_metrics(
        {
            "ts": time.time(),
            "acc": acc,
            "score": score,
            "legacy_score": legacy_score,
            "verifier_mode": verifier_mode,
            "n_errors": n_errors,
            "latency_ms": latency_ms,
            "reward_mode": REWARD_MODE,
        }
    )
    return out


def _http_unavailable(exc: VerifierExecutionError) -> HTTPException:
    return HTTPException(status_code=503, detail=f"PhysicsVerifier unavailable: {exc}")


def _runtime_identity() -> Dict[str, Any]:
    """Configuration fields that determine reward semantics or worker layout."""
    return {
        "reward_mode": REWARD_MODE,
        "lambda": DEFAULT_LAMBDA,
        "error_cap": DEFAULT_CAP,
        "concurrency": DEFAULT_CONCURRENCY,
        "w_answer": W_ANSWER,
        "w_format": W_FORMAT,
        "w_verifier": W_VERIFIER,
        "w_length": W_LENGTH,
        "verifier_sample_rate": VERIFIER_SAMPLE_RATE,
        "verifier_failure_policy": VERIFIER_FAILURE_POLICY,
        "unified_rules": UNIFIED_RULES_PATH,
        "retrieval_mode": os.environ.get(
            "PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE", "semantic"
        ),
        "semantic_output_adapter": os.environ.get(
            "PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER", ""
        ),
        "llm_model": os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "qwen3-30b-a3b"),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "symbolic_enabled": SYMBOLIC_ENABLED,
        "checker_gate_mode": CHECKER_GATE_MODE,
        "checker_json_attempts": CHECKER_JSON_ATTEMPTS,
        "semantic_json_attempts": SEMANTIC_JSON_ATTEMPTS,
        "unified_rule_top_n": UNIFIED_RULE_TOP_N,
        "precision_mode": PRECISION_MODE,
        "max_diagnostics_per_sample": MAX_DIAGNOSTICS_PER_SAMPLE,
        "max_diagnostics_per_paragraph": MAX_DIAGNOSTICS_PER_PARAGRAPH,
        "llm_cache_enabled": ENABLE_LLM_CACHE,
        "require_provider_identity": REQUIRE_PROVIDER_IDENTITY,
        "max_response_chars": MAX_RESPONSE_CHARS,
        "metrics_log": METRICS_LOG,
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "config": _runtime_identity(),
    }


@app.post("/")
async def score_endpoint(req: ScoreRequest) -> Dict[str, Any]:
    try:
        return await score_one(req)
    except VerifierExecutionError as exc:
        raise _http_unavailable(exc) from exc


@app.post("/batch")
async def score_batch(req: BatchRequest) -> Dict[str, List[Dict[str, Any]]]:
    try:
        results = await asyncio.gather(
            *[score_one(item, index) for index, item in enumerate(req.requests)]
        )
    except VerifierExecutionError as exc:
        raise _http_unavailable(exc) from exc
    return {"results": list(results)}


class OpenRLHFRewardRequest(BaseModel):
    """OpenRLHF remote RM payload (see openrlhf.utils.remote_rm_utils)."""

    query: List[str] = Field(default_factory=list)
    prompts: List[str] = Field(default_factory=list)
    labels: List[Any] = Field(default_factory=list)


def _response_from_query(query: str, prompt: str) -> str:
    if not prompt:
        return query
    if isinstance(prompt, str) and query.startswith(prompt):
        return query[len(prompt) :]
    raise ValueError("query does not begin with its paired prompt")


@app.post("/get_reward")
async def openrlhf_get_reward(req: OpenRLHFRewardRequest) -> Dict[str, Any]:
    """OpenRLHF-compatible remote reward endpoint.

    Expects JSON: {query: [...], prompts: [...], labels: [...]}
    Returns: {rewards: [...], scores: [...], extra_logs: {...}}
    """
    n = len(req.query)
    if n == 0:
        raise HTTPException(status_code=422, detail="query must contain at least one sample")
    if len(req.prompts) != n or len(req.labels) != n:
        raise HTTPException(
            status_code=422,
            detail=(
                "query, prompts, and labels must have identical lengths "
                f"(query={n}, prompts={len(req.prompts)}, labels={len(req.labels)})"
            ),
        )
    prompts = list(req.prompts)
    labels = list(req.labels)

    score_reqs = []
    for index, (query, prompt, label) in enumerate(zip(req.query, prompts, labels)):
        if not _normalize_label(label):
            raise HTTPException(
                status_code=422,
                detail=f"labels[{index}] must contain at least one ground-truth answer",
            )
        try:
            response = _response_from_query(str(query), str(prompt or ""))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"query[{index}]: {exc}") from exc
        score_reqs.append(
            ScoreRequest(
                prompt=str(prompt or ""),
                response=response,
                label=label,
                question=str(prompt or ""),
            )
        )

    try:
        results = await asyncio.gather(
            *[score_one(item, index) for index, item in enumerate(score_reqs)]
        )
    except VerifierExecutionError as exc:
        raise _http_unavailable(exc) from exc
    rewards = [float(r.get("score", 0.0)) for r in results]
    accs = [1.0 if r.get("acc") else 0.0 for r in results]
    n_errors = [float(r.get("n_errors", 0) or 0) for r in results]
    verifier_hits = [1.0 if r.get("verifier_mode") in {"full", "failed"} else 0.0 for r in results]
    verifier_successes = [1.0 if r.get("verifier_mode") == "full" else 0.0 for r in results]
    verifier_failed = [1.0 if (r.get("reward_components") or {}).get("verifier_failed") else 0.0 for r in results]
    latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in results]
    return {
        "rewards": rewards,
        "scores": rewards,
        "extra_logs": {
            "physics_acc": sum(accs) / max(len(accs), 1),
            "physics_n_errors_mean": sum(n_errors) / max(len(n_errors), 1),
            "physics_verifier_trigger_rate": sum(verifier_hits) / max(len(verifier_hits), 1),
            "physics_verifier_success_rate": sum(verifier_successes) / max(len(verifier_successes), 1),
            "physics_verifier_fail_rate": sum(verifier_failed) / max(len(verifier_failed), 1),
            "physics_reward_latency_ms_mean": sum(latencies) / max(len(latencies), 1),
            "physics_reward_mode": REWARD_MODE,
            "physics_answer_acc": sum(accs) / max(len(accs), 1),
            "physics_format_weight": float(_reward_weights().get("format", 0.0)),
        },
    }


def main() -> None:
    global DEFAULT_LAMBDA, DEFAULT_CAP, DEFAULT_CONCURRENCY, REWARD_MODE
    global W_ANSWER, W_FORMAT, W_VERIFIER, W_LENGTH, VERIFIER_SAMPLE_RATE

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--lambda-penalty", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--error-cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--reward-mode", default=REWARD_MODE)
    args = parser.parse_args()

    DEFAULT_LAMBDA = args.lambda_penalty
    DEFAULT_CAP = args.error_cap
    DEFAULT_CONCURRENCY = args.concurrency
    REWARD_MODE = str(args.reward_mode).strip().lower()
    _validate_runtime_config()

    if REWARD_MODE != "answer_only":
        _get_verifier()
    print(
        json.dumps(
            {
                "event": "physics_reward_server_start",
                "host": args.host,
                "port": args.port,
                **_runtime_identity(),
                "weights": _reward_weights(),
                "metrics_log": METRICS_LOG,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
