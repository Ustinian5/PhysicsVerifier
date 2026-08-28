#!/usr/bin/env bash
# 10-step GRPO pilot to check in-group reward std after SFT + self-judge + longer completions.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
export MAX_STEPS="${MAX_STEPS:-10}"
export JUDGE_REFRESH="${JUDGE_REFRESH:-0}"
export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-strict}"
export MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1536}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
bash "${ROOT}/training/swift/launch_swift_grpo_8gpu.sh"
echo "[pilot] launched; inspect logging.jsonl for reward_std >= 0.15 and frac_reward_zero_std=0"
