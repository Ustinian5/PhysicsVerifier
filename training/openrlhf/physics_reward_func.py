#!/usr/bin/env python3
"""OpenRLHF custom reward_func that calls PhysicsVerifier reward server.

Compatible with OpenRLHF 0.8.x:
  --remote_rm_url /path/to/physics_reward_func.py

Environment:
  PHYSICS_REWARD_URL  default http://127.0.0.1:8770/get_reward
"""
from __future__ import annotations

import os
from typing import Any, List

import requests
import torch

REWARD_URL = os.environ.get("PHYSICS_REWARD_URL", "http://127.0.0.1:8770/get_reward")
TIMEOUT = float(os.environ.get("PHYSICS_REWARD_TIMEOUT", "600"))


def reward_func(queries: List[str], prompts: List[str], labels: List[Any], **kwargs) -> torch.Tensor:
    """Return a 1-D float tensor of rewards (OpenRLHF 0.8.x API)."""
    payload = {"query": list(queries), "prompts": list(prompts), "labels": list(labels)}
    resp = requests.post(REWARD_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    rewards = data.get("rewards")
    if rewards is None:
        raise RuntimeError(f"PhysicsVerifier reward missing 'rewards': {data}")
    return torch.tensor(rewards, dtype=torch.float32)
