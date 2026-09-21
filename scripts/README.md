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

API 先行时使用 `run_retrieval_comparison.py --phase retrieval_batch --input … --selection-manifest … --output … --arms 2 3 4 --workers 3`。脚本校验冻结输入的 SHA、50 个 ID 和模型输入字段，保存工具配置及源码哈希；各题独立会话，成功记录续跑时跳过。并发共用请求/token 预算，检查组内平台当前令牌额度；额度不足或遇到账户/限流错误时停止新请求、保留断点。GT 复核和 Qwen 不阻塞检索采集，但仍是最终计分的前提。旧中断记录必须先核对，不能覆盖重跑。

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
