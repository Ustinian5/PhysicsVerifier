# 脚本入口

`scripts/` 只在根层保留仍参与当前流程的 Python 入口与其辅助模块。历史实验包装器和专用分析程序统一放在 `scripts/legacy/`。

## 运行与评测

| 入口 | 用途 |
|---|---|
| `run_verifier.py` | verifier 主入口 |
| `run_llm_checker_baseline.py` | 同模型无规则基线 |
| `run_physics_eval_pipeline.py` | 构造并运行完整效果测评 |
| `extract_holdout_eval_samples.py` | 抽取去重、排除历史样本的候选集 |
| `audit_correct_eval_candidates.py` | 审计 precision 正确样本 |
| `build_physics_eval_sets.py` | 构造错误级和题目级冻结数据集 |
| `audit_eval_set_quality.py` | 检查 GT 数量与可定位性 |
| `evaluate_physics_eval_sets.py` | 错误定位级指标 |
| `evaluate_question_level_sets.py` | 题目级指标 |

`run_physics_eval_pipeline.py` 是正式实验入口：必须显式传入 `unified_rules_v2` catalog，默认复用当前 conda Python，自动关闭跨运行 LLM cache、逐题 checkpoint，并生成 `schema_version=1` 的实验 manifest。development 可记录 dirty 状态；validation/final 强制 clean worktree。冻结清单使用 `--manifest-output experiments/manifests/<run-id>.json`。

## unified_rules 生命周期

主要入口是 `unified_rules_pipeline.py`。其余 `prepare_*`、`generalize_*`、`run_rule_embedding_clustering.py`、`build_unified_catalog.py`、`finalize_incremental_update.py` 和 `validate_*` 脚本分别承担候选准备、概括、聚类、构建、增量合并和验证。

审计类脚本使用 `audit_*` 命名；分析和对比工具使用 `analyze_*`、`compare_*`、`evaluate_*` 命名。它们默认不应覆盖正式 catalog。

## 历史实验

`legacy/` 中的脚本固定了旧数据目录、旧 catalog、旧虚拟环境或特定实验表格，仅用于追溯历史实验。新的测评不应复制这些脚本，而应从 `run_physics_eval_pipeline.py` 和[效果测评](../docs/效果测评.md)中的冻结命令开始。
