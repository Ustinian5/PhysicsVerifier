# Physics RL Training

> **2026-07-09**: Training framework switched from **slime** to **OpenRLHF 0.8.2**
> because the host NVIDIA driver (550.54.15 / CUDA 12.4) cannot be upgraded and
> slime's SGLang/`sgl-kernel` stack is incompatible with torch 2.6.
>
> **Canonical Chinese guide**: [`docs/物理RL训练_OpenRLHF.md`](./物理RL训练_OpenRLHF.md)
>
> Training goals unchanged: Qwen3-30B-A3B + GRPO + PhysicsVerifier remote reward +
> 6-GPU train / GPU6 fixed judge.

## Quick start (OpenRLHF)

```bash
bash training/openrlhf/setup_openrlhf_env.sh
source /slow_share/jinjianhan/workspace/openrlhf_rl/env.sh
bash training/reward_server/start_reward_server.sh
bash training/openrlhf/prepare_openrlhf_data.sh
bash training/openrlhf/launch_training.sh
```

## Status

| Component | Status |
|-----------|--------|
| Framework | **OpenRLHF 0.8.2** (vLLM rollout) |
| Reward server | `:8770` with `/` + `/batch` + **`/get_reward`** |
| vLLM judge (GPU6) | `:8766` |
| Train data | `data/rl/openrlhf_prompts.jsonl` (2336) |
| Held-out | `data/rl/openrlhf_heldout.jsonl` (150) |
| slime path | Deprecated on this host (driver blocker) |

## GPU layout

| GPU | Role |
|-----|------|
| 0-5 | OpenRLHF Hybrid Engine (Actor/Ref + vLLM rollout) |
| 6 | Fixed vLLM judge for PhysicsVerifier |
| 7 | Optional spare / second judge |

## Reward

```
score = acc - 0.3 * min(n_errors, 3) / 3
```

Full verifier runs only when `acc=True`. OpenRLHF calls
`POST /get_reward` via `training/openrlhf/physics_reward_func.py`.

## Key scripts

- `training/openrlhf/setup_openrlhf_env.sh`
- `training/openrlhf/run-qwen3-30b-physics-6gpu-openrlhf.sh`
- `training/openrlhf/physics_reward_func.py`
- `training/reward_server/physics_reward_server.py`

See the Chinese doc for full parameter mapping, troubleshooting, and version pins.
