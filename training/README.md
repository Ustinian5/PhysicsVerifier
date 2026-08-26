# Training

训练层以 OpenRLHF 0.8.2 为唯一正式入口，通过 HTTP reward server 调用远程仓库的 `PhysicsRuleVerifier`。

## 快速入口

```bash
bash training/openrlhf/setup_openrlhf_env.sh
source /slow_share/jinjianhan/workspace/openrlhf_rl/env.sh
bash training/reward_server/start_reward_server.sh
bash training/openrlhf/prepare_openrlhf_data.sh
bash training/openrlhf/launch_training.sh
```

训练前检查：

```bash
bash training/openrlhf/check_prerequisites.sh
bash training/reward_server/verify_external_api.sh
```

## Verifier profile

Verifier reward 不再隐式选择规则库；启动时必须通过环境显式绑定 catalog。当前 development 基线是 `catalogs/rules_unified_3000.json`（1123 条），4875 条 runtime-backfilled 库只保留为已否决的历史消融。检索模式默认为 `semantic`，符号核查默认关闭。

可通过以下环境变量显式覆盖：

- `PHYSICSVERIFIER_UNIFIED_RULES`
- `PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE`
- `PHYSICSVERIFIER_CHECKER_GATE_MODE`
- `PHYSICSVERIFIER_CHECKER_JSON_ATTEMPTS`
- `PHYSICSVERIFIER_SEMANTIC_JSON_ATTEMPTS`
- `PHYSICSVERIFIER_SYMBOLIC_ENABLED`
- `PHYSICSVERIFIER_SYMBOLIC_MANIFEST`
- `PHYSICSVERIFIER_LLM_MODEL`
- `PHYSICSVERIFIER_REQUIRE_PROVIDER_IDENTITY`

根目录 `.env.example` 给出共享配置模板。

## OpenRLHF engine contract

四卡入口使用项目定制的 variance-filter 参数；官方 OpenRLHF 0.8.2 本身不包含这组扩展。训练会在分配 Ray/GPU 前运行 `openrlhf_contract.py`，校验所需参数与 `dynamic_filter.py` 接口，并把 OpenRLHF commit、dirty diff hash、规则库、训练数据和 Reward Server 身份写入运行目录。缺少外部补丁时流程会明确失败，不再静默使用不等价实现。

## 边界

- `openrlhf/`：训练、pilot、watchdog 和曲线工具。
- `reward_server/`：`/health`、`/batch`、`/get_reward`。
- `rl_data/`：prompt 构建、切分与离线过滤。
- `compat/`：从旧 slime 路径保留的最小答案判分工具。
- `docs/`：运行指南和历史训练报告。

完整 slime 源码和旧 `scripts/rl_train` 已保存在外部迁移备份，不再是项目运行依赖。
