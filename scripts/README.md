# 脚本入口

所有 Python 入口通过 conda `physicsverifier` 执行。完整参数以各脚本 `--help` 为准；方法和指标见 [效果测评](../docs/效果测评.md)。

## 运行与评测

| 脚本 | 用途 |
|---|---|
| `run_verifier.py` | 完整 verifier；`--retrieval-only` 只运行检索 |
| `run_physics_eval_pipeline.py` | 构造数据、运行和评测，自动保存实验清单 |
| `run_llm_checker_baseline.py` | 无规则的直接检查基线 |
| `extract_holdout_eval_samples.py` | 去重并排除历史样本 |
| `audit_correct_eval_candidates.py` | 复核正确解答候选 |
| `build_physics_eval_sets.py` / `audit_eval_set_quality.py` | 构造数据及检查 GT 质量 |
| `evaluate_physics_eval_sets.py` | 错误位置匹配指标 |
| `evaluate_question_level_sets.py` | 题目级指标 |
| `prepare_retrieval_comparison.py` | 从历史池冻结 50 条，隔离参考答案与待复核标注 |
| `run_retrieval_comparison.py` | 四组检索；`development` 预跑、`retrieval_batch` 运行已冻结的 50 条；保存 API 请求和工具记录，不含 checker 与物理计分 |
| `run_retrieval_checker.py` | 将上述冻结候选接回原 Qwen checker、定位和发布；`--prepare-only` 无 API 调用，完整请求一致时共享原始响应 |

`run_physics_eval_pipeline.py` 要求显式传入规则库，使用当前 conda 解释器；`validation/final` 要求 clean worktree。运行配置与输出身份按 [实验清单](../experiments/README.md)保存。

API 先行时使用 `run_retrieval_comparison.py --phase retrieval_batch --input … --selection-manifest … --output … --arms 2 3 4 --api-model gpt-5.6-terra --reasoning-effort medium --workers 1 --overload-backoff`。当前默认 Terra medium；历史 Astra 需显式传入 `--api-model gpt-6-astra --reasoning-effort max`。沿用 `.env` 的组内 `ASTRA_BASE_URL/ASTRA_API_KEY` 或 `OPENAI_BASE_URL/OPENAI_API_KEY`，变量名不决定实际模型。脚本校验冻结输入的 SHA、50 个 ID 和模型输入字段，并核验响应中的模型和推理档位；配置变化须独立运行，禁止混入旧模型结果。并发共用请求/token 预算，额度查询最多每分钟一次；查询接口故障只记警告、5 分钟后重试。确认低额度、鉴权失败或不明原因的 429 会停止新请求。GT 复核和 Qwen 不阻塞检索采集，但仍是最终计分的前提。

`--overload-backoff` 仅用于单并发 API 批量。对明确的上游容量错误、408/502/503/504，以及传输超时和断连，按 5、10、15 分钟重试完全相同的请求；**每次逻辑请求最多重试三次，成功后下一请求重新计数**。尊重 `Retry-After`，单次请求计划等待最多 30 分钟；持续故障则停止整批，状态为 `stopped_upstream`。每任务仍最多 32 次实际请求加已重放响应，全批请求/token 与余额保护继续生效；未知用量保留全批预留值，不能无限重试。全部失败保留，等待单列于 `overload_retry.json`，不占原 30 分钟执行预算；`waiting_for_capacity` 表示正常等待，不能当成进程中断。未核实的模型名/推理档位不自动接受或重试，单列技术失败。不启用此选项时保留原错误停止行为。

接续预算/网络中断时使用新的结果子目录，追加 `--retry-transient-from 原运行目录 --continue-pending --replay-prefix`。原始失败保留；在全部关联记录中跳过已成功项，未启动任务继续执行。已收到的成功响应只在完整请求逐项相同时重放以恢复运行状态，不重复付费；实际新请求、重放数量和历史用量分开记录，全批预算累计计算。没有明确完成状态的在途请求仍须先核对，不能覆盖重跑。

## Checker 回放

`run_checker_replay.py` 从冻结候选运行 checker 与发布流程，不重新检索。候选可来自真实检索或显式指定规则，二者不能混称。

1. 用 `--prepare-manifest-only` 绑定 dataset、候选 trace 和 catalog。
2. 为每个条件指定 `--mode`、`--model`、独立的 `--output`、`--report`、`--llm-trace`。
3. 三种模式比较可使用 `build_checker_common_valid.py` 冻结共同有效集，并同时保留各组全量失败统计。

| 模式 | 发布条件 |
|---|---|
| `legacy` | 原 checker 与发布路径 |
| `dual_evidence` | 题目侧适用性与解答侧违规证据均通过 source/span 校验和置信门 |
| `dual_evidence_consistency` | 再要求结论为确认违规；自我修正、等价解法和不确定结论不发布 |

回放记录模型身份、运行参数、原始响应与失败；续跑校验输入和已有输出。三模式的提示与候选诊断也不同，不能描述为仅替换发布过滤条件。

四组检索对照使用 `run_retrieval_checker.py --input … --retrieval-run … --output … --arms 1 2 3 4`，只读取 `QWEN_BASE_URL` 指向的指定 Qwen 服务。先加 `--prepare-only` 核对输入、候选和规则门槛；去掉后才运行 checker。多个 `--retrieval-run` 路径按原始批次、哈希关联的技术补跑依次填写，不选择重复运行中的最好结果。该实验入口保留自主检索的真实来源，并在适配层共用原有 0–1 语义分数门槛；不改主系统阈值，尚不开放正式计分。

历史机制实验工具仍可用：`build_checker_mechanism_dataset.py` 生成指定规则的正反例，`evaluate_checker_mechanism_gate.py` 评估规则级对照，`audit_checker_mechanism_semantic_retrieval.py` 审计检索 trace。它们不构成当前已完成的批量实验，也不代替实际运行凭据。

## 规则构建

| 脚本 | 用途 |
|---|---|
| `unified_rules_pipeline.py` | 统一规则库构建 |
| `prepare_incremental_update.py` | 建立增量工作区并冻结范围、配置和预算 |
| `finalize_incremental_update.py` | 校验来源、变化范围及产物依赖，生成待审目录 |
| `validate_unified_catalog_structure.py` | 校验 ID、引用和可达性 |

流程见 [增量规则说明](../docs/增量规则流程.md)。`legacy/` 仅用于复现旧实验，不能直接沿用其中的路径、环境或模型配置。
