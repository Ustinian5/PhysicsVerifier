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

默认规则库为 `catalogs/rules_unified_3000_runtime_backfilled.json`，检索模式为 `semantic`。由于该库的 `norm_*` Rule ID 没有匹配的历史 `exp_*` manifest，默认关闭符号核查。

可通过以下环境变量显式覆盖：

- `PHYSICSVERIFIER_UNIFIED_RULES`
- `PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE`
- `PHYSICSVERIFIER_SYMBOLIC_ENABLED`
- `PHYSICSVERIFIER_SYMBOLIC_MANIFEST`
- `PHYSICSVERIFIER_LLM_MODEL`

根目录 `.env.example` 给出共享配置模板。

## 边界

- `openrlhf/`：训练、pilot、watchdog 和曲线工具。
- `reward_server/`：`/health`、`/batch`、`/get_reward`。
- `rl_data/`：prompt 构建、切分与离线过滤。
- `compat/`：从旧 slime 路径保留的最小答案判分工具。
- `docs/`：运行指南和历史训练报告。

完整 slime 源码和旧 `scripts/rl_train` 已保存在外部迁移备份，不再是项目运行依赖。
