#!/usr/bin/env python3
"""OpenRLHF custom reward_func that calls PhysicsVerifier reward server.

Compatible with OpenRLHF 0.8.x:
  --remote_rm_url /path/to/physics_reward_func.py

Environment:
  PHYSICS_REWARD_URL  default http://127.0.0.1:8770/get_reward
"""
from __future__ import annotations

import math
import os
from typing import Any, List

import requests
import torch

REWARD_URL = os.environ.get("PHYSICS_REWARD_URL", "http://127.0.0.1:8770/get_reward")
TIMEOUT = float(os.environ.get("PHYSICS_REWARD_TIMEOUT", "1800"))
if not math.isfinite(TIMEOUT) or TIMEOUT <= 0.0:
    raise ValueError("PHYSICS_REWARD_TIMEOUT must be a finite positive number")


def reward_func(queries: List[str], prompts: List[str], labels: List[Any], **kwargs) -> torch.Tensor:
    """Return a 1-D float tensor of rewards (OpenRLHF 0.8.x API)."""
    if not queries:
        raise ValueError("queries must contain at least one sample")
    if len(prompts) != len(queries) or len(labels) != len(queries):
        raise ValueError(
            "queries, prompts, and labels must have identical lengths "
            f"(queries={len(queries)}, prompts={len(prompts)}, labels={len(labels)})"
        )
    payload = {"query": list(queries), "prompts": list(prompts), "labels": list(labels)}
    resp = requests.post(REWARD_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("PhysicsVerifier reward response must be a JSON object")
    rewards = data.get("rewards")
    if not isinstance(rewards, list):
        raise RuntimeError(f"PhysicsVerifier reward missing 'rewards': {data}")
    if len(rewards) != len(queries):
        raise RuntimeError(
            "PhysicsVerifier reward count mismatch: "
            f"expected {len(queries)}, received {len(rewards)}"
        )
    try:
        normalized = [float(value) for value in rewards]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PhysicsVerifier rewards must all be numeric") from exc
    if any(not math.isfinite(value) for value in normalized):
        raise RuntimeError("PhysicsVerifier rewards must all be finite")
    return torch.tensor(normalized, dtype=torch.float32)
