# 物理 RL 训练指南（OpenRLHF）

> **状态更新（2026-07-09）**：因本机 NVIDIA 驱动 **550.54.15（CUDA 12.4）无法升级**，原 slime 方案依赖的 `sglang/sgl-kernel` 与 PyTorch 2.6 ABI 不兼容，已**切换训练框架为 OpenRLHF 0.8.2**。  
> **训练目标与设置保持不变**：Qwen3-30B-A3B + GRPO + PhysicsVerifier 远程奖励 + 6 卡训练 / GPU6 固定 judge。

---

## 1. 为什么从 slime 切到 OpenRLHF？

| 项目 | slime | OpenRLHF 0.8.2 |
|------|-------|----------------|
| Rollout 引擎 | **SGLang**（需 sgl-kernel） | **vLLM 0.8.5**（本机已有） |
| 驱动 550 / torch 2.6 | **不可用** | **可用** |
| GRPO | 支持 | `--advantage_estimator group_norm` |
| 远程奖励 | `remote_rm` HTTP | `--remote_rm_url`（HTTP 或 Python 函数） |
| MoE | Megatron EP | HF + DeepSpeed ZeRO-3（`--aux_loss_coef`） |
| Checkpoint | 需 HF→torch_dist 转换 | **直接读 HF 权重** |

结论：在无法升级驱动的前提下，OpenRLHF + 现有 vLLM 是可行路径；slime 脚本保留作参考，但**不再作为默认训练入口**。

---

## 2. 训练目标与设置（与原 plan 对齐）

| 项 | 取值 |
|----|------|
| 模型 | Qwen3-30B-A3B-Instruct-2507 |
| 算法 | **GRPO**（`group_norm`） |
| 每题采样数 | **8** |
| 学习率 | **1e-6** |
| Adam | CPU offload |
| 精度 | bf16 + ZeRO-3 |
| 最大生成长度 | **8192** |
| Reward | PhysicsVerifier：`score = acc - 0.3 * min(n_errors,3)/3` |
| GPU | **0–5 训练**；**GPU6 固定 vLLM judge**；GPU7 备用 |
| 题池 | `data/rl/rl_prompts.jsonl`（2336）+ held-out（150） |

---

## 3. 架构

```
GPU 0-5                          GPU 6
┌─────────────────────────┐      ┌──────────────────┐
│ OpenRLHF Hybrid Engine  │      │ vLLM Judge       │
│  · Actor + Ref (ZeRO-3) │      │ Qwen3-30B :8766  │
│  · vLLM rollout ×3 TP2  │      └────────┬─────────┘
└───────────┬─────────────┘               │
            │ remote reward               │ semantic check
            ▼                             ▼
      physics_reward_func.py ──► :8770 /get_reward
                                 PhysicsRuleVerifier
```

Reward 调用链：

1. OpenRLHF 生成 `(prompt, response)`  
2. `physics_reward_func.py` → `POST http://127.0.0.1:8770/get_reward`  
3. Reward server：答案判分 →（若正确）完整 PhysicsVerifier → 返回 `rewards` / `scores`

---

## 4. 目录与关键文件

| 路径 | 说明 |
|------|------|
| `training/openrlhf/setup_openrlhf_env.sh` | 创建独立训练 venv，安装 OpenRLHF 0.8.2 |
| `training/openrlhf/prepare_openrlhf_data.sh` | jsonl 格式转换（label 转字符串） |
| `training/openrlhf/physics_reward_func.py` | OpenRLHF 自定义 reward 函数 |
| `training/openrlhf/run-qwen3-30b-physics-6gpu-openrlhf.sh` | 6 卡 GRPO 启动脚本 |
| `training/openrlhf/check_prerequisites.sh` | 前置检查 |
| `training/openrlhf/launch_training.sh` | 检查 + 启动 |
| `training/reward_server/physics_reward_server.py` | 已新增 `/get_reward`（OpenRLHF 协议） |
| `/slow_share/.../openrlhf_rl/OpenRLHF` | OpenRLHF 源码（v0.8.2） |
| `/slow_share/.../openrlhf_rl/env.sh` | 训练环境变量（setup 后生成） |
| `/data1/jinjianhan/venv/openrlhf_train` | **训练专用 venv**（勿与 judge venv 混用） |

数据：

- 训练：`data/rl/openrlhf_prompts.jsonl`（由 `rl_prompts.jsonl` 转换）  
- 评测：`data/rl/openrlhf_heldout.jsonl`  
- Checkpoint：`/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf/`

---

## 5. 一键流程

```bash
# 0) 安装训练环境（独立 venv，不碰 GPU6 judge）
bash training/openrlhf/setup_openrlhf_env.sh
source /slow_share/jinjianhan/workspace/openrlhf_rl/env.sh

# 1) 确认 GPU6 judge + reward server
curl -sf http://127.0.0.1:8766/v1/models
bash training/reward_server/start_reward_server.sh   # 需含 /get_reward

# 2) 准备 OpenRLHF 数据格式
bash training/openrlhf/prepare_openrlhf_data.sh
bash training/openrlhf/prepare_openrlhf_data.sh \
  data/rl/heldout_eval.jsonl data/rl/openrlhf_heldout.jsonl

# 3) 前置检查
bash training/openrlhf/check_prerequisites.sh

# 4) 启动 GRPO（GPU 0-5）
bash training/openrlhf/launch_training.sh

# 5) 评测闭环（训练出 HF ckpt 后）
bash evaluation/benchmarks/hipho/run_eval_loop.sh
```

---

## 6. 与 slime 参数对照

| 原 slime 参数 | OpenRLHF 0.8.2 对应 |
|---------------|---------------------|
| `--advantage-estimator grpo` | `--advantage_estimator group_norm` |
| `--n-samples-per-prompt 8` | `--n_samples_per_prompt 8` |
| `--rm-type remote_rm --rm-url ...` | `--remote_rm_url training/openrlhf/physics_reward_func.py` |
| `--lr 1e-6` | `--actor_learning_rate 1e-6` |
| `--optimizer-cpu-offload` | `--adam_offload` |
| `--rollout-max-response-len 8192` | `--generate_max_len 8192` |
| `--global-batch-size 256` | `--train_batch_size 256` |
| `--rollout-batch-size 32` | `--rollout_batch_size 32` |
| `--eps-clip 0.2` | `--eps_clip 0.2` |
| `--use-kl-loss --kl-loss-coef 0` | `--use_kl_loss --init_kl_coef 0.0 --kl_estimator k3` |
| dynamic sampling filter | `--dynamic_filtering --dynamic_filtering_reward_range 0.01 0.99` |
| HF→torch_dist 转换 | **不需要**（直接 `--pretrain` HF 路径） |
| Megatron TP2/EP2 | vLLM TP=2 × 3 engines；训练侧 ZeRO-3 |

---

## 7. GPU 布局

| GPU | 角色 |
|-----|------|
| 0–5 | OpenRLHF Hybrid Engine（Actor/Ref + vLLM rollout，colocate） |
| 6 | PhysicsVerifier 语义检查用 vLLM judge（`:8766`） |
| 7 | 可选第二 judge / 空闲 |

**不要**把训练 venv 的 torch 升级到 cu130，否则会破坏 GPU6 上已运行的 vLLM 0.8.5 judge（若误用同一 venv）。训练必须使用 `/data1/jinjianhan/venv/openrlhf_train`。

---

## 8. 依赖与版本钉扎

| 组件 | 版本 | 原因 |
|------|------|------|
| NVIDIA Driver | 550.54.15 | 当前不可升级 |
| PyTorch | **2.6.0+cu124** | 驱动上限 |
| vLLM | **0.8.5** | 与 OpenRLHF 0.8.2 匹配；judge 同版本 |
| OpenRLHF | **0.8.2** | `Requires-Dist: vllm==0.8.5.post1`（0.8.5 可用） |
| DeepSpeed | 0.16.9 | OpenRLHF 0.8.2 钉扎 |
| transformers | 4.52.3 | OpenRLHF 0.8.2 钉扎 |
| Ray | 2.43.0 | OpenRLHF 0.8.2 钉扎 |

> 不要安装 OpenRLHF ≥0.9：其默认依赖更新的 vLLM/torch，会再次触发驱动问题。

---

## 9. Reward 公式与接口

```
score = acc - λ * min(n_errors, cap) / cap
```

默认 `λ=0.3`，`cap=3`。仅当 `acc=True` 时跑完整 PhysicsVerifier。

OpenRLHF HTTP 协议（已实现）：

```http
POST /get_reward
{"query": ["prompt+response", ...], "prompts": [...], "labels": [...]}

→ {"rewards": [...], "scores": [...], "extra_logs": {...}}
```

也可用本地函数：`--remote_rm_url training/openrlhf/physics_reward_func.py`（内部仍调上述 HTTP）。

---

## 10. 已知差异与风险

1. **并行实现不同**：slime 用 Megatron EP；OpenRLHF 用 HF MoE + ZeRO-3。吞吐/显存曲线会不同，但算法目标一致。  
2. **flash-attn**：主机无 nvcc，无法编译 CUDA flash-attn。`setup_openrlhf_env.sh` 会安装纯 Python shim（`bert_padding` 等），训练脚本**默认不加** `--flash_attn/--packing_samples`。若日后装上真正的 flash-attn wheel，可手动加回以加速。  
3. **显存**：30B MoE + 6 卡 colocate 较紧；若 OOM，可下调 `--vllm_gpu_memory_utilization`（如 0.45）或 `--rollout_batch_size`。  
4. **Judge 与训练隔离**：务必使用独立 `openrlhf_train` venv；重启 reward server 后确认 `/get_reward` 可用。  
5. **DeepSpeed / CUDA_HOME**：主机无系统 CUDA toolkit；`env.sh` 指向 `cuda_stub`（假 nvcc）仅用于 DeepSpeed 版本探测，不进行 JIT 编译。  
6. **triton**：需与 torch 2.6 匹配（钉扎 `triton==3.2.0`）；过高版本会导致 `AttrsDescriptor` 导入失败。  
7. **slime 遗留**：完整 slime 与旧训练脚本已移出活动项目并保存在迁移备份；当前仅保留答案判分兼容模块。

---

## 11. 故障排查

| 现象 | 处理 |
|------|------|
| `sgl_kernel` / driver too old | 确认未误装 slime/sglang；使用 OpenRLHF 路径 |
| `/get_reward` 404 | 重启 `start_reward_server.sh`（需含新端点） |
| Ray GPU index 错误 | `export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`（env.sh 已设） |
| `MissingCUDAException: CUDA_HOME` | `source .../openrlhf_rl/env.sh`（已指向 `cuda_stub`；主机无系统 CUDA toolkit） |
| OOM | 降 `vllm_gpu_memory_utilization` / `rollout_batch_size` / `micro_train_batch_size` |
| reward 全 0 | 检查 label 格式、`\boxed{}`、GPU6 judge 是否存活 |
| import openrlhf 失败 | 重新跑 `setup_openrlhf_env.sh` |

---

## 12. 后续评测

训练保存 HF checkpoint 后：

```bash
# 用新 ckpt 起 vLLM（可用 GPU7，避免打断 GPU6 judge）
# 然后：
bash evaluation/benchmarks/hipho/run_eval_loop.sh
```

双轨：HiPhO 外部 bench + PhysicsVerifier 内部指标。
