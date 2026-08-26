#!/usr/bin/env bash
# CPU-only smoke: answer_only reward + variance filter helper. No Ray, no GPU grab.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_only}"
export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0}"

bash "${ROOT}/training/reward_server/start_reward_server.sh"
curl -sf http://127.0.0.1:8770/health >/dev/null

python3 - <<'PY'
import json, urllib.request
from pathlib import Path
import importlib.util, sys

req = {
    "query": [
        "prompt\nassistant\n\\boxed{1}",
        "prompt\nassistant\nno box",
        "prompt\nassistant\n\\boxed{1}",
        "prompt\nassistant\n\\boxed{2}",
    ],
    "prompts": ["prompt", "prompt", "prompt", "prompt"],
    "labels": ["\\boxed{1}", "\\boxed{1}", "\\boxed{1}", "\\boxed{1}"],
}
data = json.dumps(req).encode()
with urllib.request.urlopen(
    urllib.request.Request("http://127.0.0.1:8770/get_reward", data=data, headers={"Content-Type": "application/json"})
) as resp:
    out = json.loads(resp.read().decode())
rewards = [float(x) for x in out["rewards"]]
print("rewards", rewards, "extra", out.get("extra_logs"))
assert "rewards" in out
assert float(out["extra_logs"].get("physics_format_weight", 0)) == 0.0
assert out["extra_logs"].get("physics_reward_mode") == "answer_only"

path = Path("/slow_share/jinjianhan/workspace/openrlhf_rl/OpenRLHF/openrlhf/trainer/ppo_utils/dynamic_filter.py")
spec = importlib.util.spec_from_file_location("openrlhf_dynamic_filter", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["openrlhf_dynamic_filter"] = mod
spec.loader.exec_module(mod)
cfg = mod.FilterConfig(mode=mod.MODE_REWARD_VARIANCE, n_samples_per_prompt=4, rollout_batch_size=1)
decision = mod.decide_group(rewards, cfg)
print("filter", decision)
assert decision.variance_keep, "mixed answer_only rewards must produce nonzero advantage"
print("[ok] reward+filter smoke")
PY
