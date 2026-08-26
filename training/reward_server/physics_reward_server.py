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
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

# Project root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier
from training.compat.math_grading import extract_answer, grade_answer_verl

app = FastAPI(title="PhysicsVerifier Reward Server")

_verifier: Optional[PhysicsRuleVerifier] = None
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
DEFAULT_UNIFIED_RULES = ROOT / "catalogs/rules_unified_3000_runtime_backfilled.json"
SYMBOLIC_ENABLED = os.environ.get("PHYSICSVERIFIER_SYMBOLIC_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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


def _should_run_verifier(acc: bool, sample_idx: int) -> bool:
    if not acc:
        return False
    weights = _reward_weights()
    if weights["verifier"] <= 0.0:
        return False
    if VERIFIER_SAMPLE_RATE >= 1.0:
        return True
    if VERIFIER_SAMPLE_RATE <= 0.0:
        return False
    return (sample_idx % max(int(round(1.0 / VERIFIER_SAMPLE_RATE)), 1)) == 0


def _append_metrics(record: Dict[str, Any]) -> None:
    try:
        path = Path(METRICS_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


@lru_cache(maxsize=1)
def _get_verifier() -> PhysicsRuleVerifier:
    return PhysicsRuleVerifier(
        llm_model=os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "qwen3-30b-a3b"),
        unified_rules_path=os.environ.get(
            "PHYSICSVERIFIER_UNIFIED_RULES",
            str(DEFAULT_UNIFIED_RULES),
        ),
        experience_code_manifest_path=os.environ.get(
            "PHYSICSVERIFIER_SYMBOLIC_MANIFEST",
            str(ROOT / "results/experience_symbolic_program_manifest_v2_unified.json"),
        ),
        enable_symbolic_check=SYMBOLIC_ENABLED,
        unified_retrieval_mode=os.environ.get(
            "PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE",
            "semantic",
        ),
        semantic_output_adapter=os.environ.get("PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER") or None,
    )


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

    if _should_run_verifier(acc, sample_idx):
        async with _semaphore:
            loop = asyncio.get_event_loop()
            verifier_mode = "full"
            try:
                result = await loop.run_in_executor(
                    None, lambda: _get_verifier().verify({"question": question, "prediction": req.response})
                )
                n_errors, diagnostics_summary = _count_error_diagnostics(result)
            except Exception as exc:
                verifier_failed = True
                verifier_mode = "failed"
                diagnostics_summary = [{"rule": "verifier_error", "message": str(exc)[:200]}]

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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/")
async def score_endpoint(req: ScoreRequest) -> Dict[str, Any]:
    return await score_one(req)


@app.post("/batch")
async def score_batch(req: BatchRequest) -> Dict[str, List[Dict[str, Any]]]:
    results = await asyncio.gather(*[score_one(r) for r in req.requests])
    return {"results": list(results)}


class OpenRLHFRewardRequest(BaseModel):
    """OpenRLHF remote RM payload (see openrlhf.utils.remote_rm_utils)."""

    query: List[str] = Field(default_factory=list)
    prompts: List[str] = Field(default_factory=list)
    labels: List[Any] = Field(default_factory=list)


def _response_from_query(query: str, prompt: str) -> str:
    if prompt and isinstance(prompt, str) and prompt in query:
        return query[len(prompt) :]
    return query


@app.post("/get_reward")
async def openrlhf_get_reward(req: OpenRLHFRewardRequest) -> Dict[str, Any]:
    """OpenRLHF-compatible remote reward endpoint.

    Expects JSON: {query: [...], prompts: [...], labels: [...]}
    Returns: {rewards: [...], scores: [...], extra_logs: {...}}
    """
    n = len(req.query)
    prompts = list(req.prompts) if req.prompts else [""] * n
    labels = list(req.labels) if req.labels else [None] * n
    if len(prompts) < n:
        prompts = prompts + [""] * (n - len(prompts))
    if len(labels) < n:
        labels = labels + [None] * (n - len(labels))

    score_reqs = []
    for query, prompt, label in zip(req.query, prompts, labels):
        response = _response_from_query(str(query), str(prompt or ""))
        score_reqs.append(
            ScoreRequest(
                prompt=str(prompt or ""),
                response=response,
                label=label,
                question=str(prompt or ""),
            )
        )

    results = await asyncio.gather(*[score_one(r, idx) for idx, r in enumerate(score_reqs)])
    rewards = [float(r.get("score", 0.0)) for r in results]
    accs = [1.0 if r.get("acc") else 0.0 for r in results]
    n_errors = [float(r.get("n_errors", 0) or 0) for r in results]
    verifier_hits = [1.0 if r.get("verifier_mode") == "full" else 0.0 for r in results]
    verifier_failed = [1.0 if (r.get("reward_components") or {}).get("verifier_failed") else 0.0 for r in results]
    latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in results]
    return {
        "rewards": rewards,
        "scores": rewards,
        "extra_logs": {
            "physics_acc": sum(accs) / max(len(accs), 1),
            "physics_n_errors_mean": sum(n_errors) / max(len(n_errors), 1),
            "physics_verifier_trigger_rate": sum(verifier_hits) / max(len(verifier_hits), 1),
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
    REWARD_MODE = args.reward_mode

    if REWARD_MODE != "answer_only":
        _get_verifier()
    print(
        json.dumps(
            {
                "event": "physics_reward_server_start",
                "host": args.host,
                "port": args.port,
                "lambda": DEFAULT_LAMBDA,
                "cap": DEFAULT_CAP,
                "concurrency": DEFAULT_CONCURRENCY,
                "reward_mode": REWARD_MODE,
                "weights": _reward_weights(),
                "verifier_sample_rate": VERIFIER_SAMPLE_RATE,
                "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
                "metrics_log": METRICS_LOG,
                "unified_rules": os.environ.get(
                    "PHYSICSVERIFIER_UNIFIED_RULES",
                    str(DEFAULT_UNIFIED_RULES),
                ),
                "symbolic_enabled": SYMBOLIC_ENABLED,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
