# 脚本入口

`scripts/` 只在根层保留仍参与当前流程的 Python 入口与其辅助模块。历史实验包装器和专用分析程序统一放在 `scripts/legacy/`。

## 运行与评测

| 入口 | 用途 |
|---|---|
| `run_verifier.py` | verifier 主入口 |
| `run_llm_checker_baseline.py` | 同模型无规则基线 |
| `run_physics_eval_pipeline.py` | 构造并运行完整效果测评 |
| `build_checker_mechanism_dataset.py` | 从 1123 catalog 固定抽样并用 Gemini 3 Flash 构造 P3 五机制 GT |
| `run_checker_replay.py` | 基于冻结 semantic trace 或显式 target binding 运行单模式 Checker + release gate 回放 |
| `build_checker_common_valid.py` | 校验三臂身份并冻结共同有效配对数据集及 manifest |
| `evaluate_checker_mechanism_gate.py` | 审计 P3 三臂三重复并判定条件 Checker 门禁 |
| `audit_checker_mechanism_semantic_retrieval.py` | 独立审计 schema-compatible semantic trace 的目标规则命中与失败 |
| `extract_holdout_eval_samples.py` | 抽取去重、排除历史样本的候选集 |
| `audit_correct_eval_candidates.py` | 审计 precision 正确样本 |
| `build_physics_eval_sets.py` | 构造错误级和题目级冻结数据集 |
| `audit_eval_set_quality.py` | 检查 GT 数量与可定位性 |
| `evaluate_physics_eval_sets.py` | 错误定位级指标 |
| `evaluate_question_level_sets.py` | 题目级指标 |

`run_physics_eval_pipeline.py` 是正式实验入口：必须显式传入 `unified_rules_v2` catalog，可显式传入 `--checker-gate-mode` 和 `--checker-json-attempts`，默认复用当前 conda Python，自动关闭跨运行 LLM cache、逐题 checkpoint，并生成 `schema_version=1` 的实验 manifest。development 可记录 dirty 状态；validation/final 强制 clean worktree。冻结清单使用 `--manifest-output experiments/manifests/<run-id>.json`。

本机脚本统一使用 conda `physicsverifier`，不直接使用系统 Python。个人计算资源的环境与命令不进入共享文档。

## P2 Checker 与受控回放

Checker 现支持三种模式：

- `legacy`：历史 Checker/发布路径；
- `dual_evidence`：题目侧 `question/context` 适用性证据与解答侧 `prediction` 违规证据都必须通过严格 source/span 校验；
- `dual_evidence_consistency`：在双证据上再要求 `confirmed_violation`，拒绝自我修正、等价/替代解法和不确定结论。

适用性、违规与 consistency 的最低可发布置信度固定为 `0.8`。它在 P2 效果测试前已固定，对齐历史 Prompt 的 80% 保守要求，不使用历史或未来测试样本调参。历史 5+5、20+20 及消融样本不参与 P2 开发、模式选择或阈值调整。

`run_checker_replay.py` 不重跑 retrieval，也不执行 bottom-up 诊断。首先冻结 manifest：

```bash
conda run -n physicsverifier python scripts/run_checker_replay.py \
  --dataset data/p2_development.json \
  --frozen-retrieval results/p2/frozen_retrieval.json \
  --catalog catalogs/rules_unified_3000.json \
  --frozen-manifest experiments/manifests/p2-checker-replay.json \
  --prepare-manifest-only
```

然后每次只运行一个模式：

```bash
conda run -n physicsverifier python scripts/run_checker_replay.py \
  --dataset data/p2_development.json \
  --frozen-retrieval results/p2/frozen_retrieval.json \
  --catalog catalogs/rules_unified_3000.json \
  --frozen-manifest experiments/manifests/p2-checker-replay.json \
  --mode dual_evidence_consistency \
  --model qwen3-30b-a3b-instruct-2507 \
  --checker-json-attempts 3 \
  --run-kind development \
  --output results/p2/dual_evidence_consistency.json \
  --report results/p2/dual_evidence_consistency.report.json \
  --llm-trace results/p2/dual_evidence_consistency.llm_trace.jsonl
```

三种模式必须使用独立 output（同时作为原子 checkpoint）/report/LLM trace。它们只共享冻结 retrieval；由于 Checker prompt/schema、Checker candidates 和 release gate 会随模式变化，三组是独立的 **Checker + release gate system arms**，不是 shared-candidate gate-only 消融。

回放配置会绑定 frozen manifest、Git/source tree、conda Python/package set、API endpoint 摘要和传输参数。每次非 transport LLM 响应还必须记录供应商返回的 `actual_model` 和非空、格内唯一的 `response_id`；实际模型必须精确等于配置模型。`--run-kind validation`/`final` 强制 clean worktree 且禁止 Checker cache，运行结束再次核验 source tree；raw-response trace 不保存 prompt。`--resume` 会校验 sidecar 与 trace 前缀，并显式审计崩溃后已写入但尚未 checkpoint 的 orphan trace。

三臂结束后冻结共同有效集：

```bash
conda run -n physicsverifier python scripts/build_checker_common_valid.py \
  --dataset data/p2_development.json \
  --arm legacy results/p2/legacy.json results/p2/legacy.report.json \
  --arm dual_evidence results/p2/dual_evidence.json results/p2/dual_evidence.report.json \
  --arm dual_evidence_consistency results/p2/dual_evidence_consistency.json results/p2/dual_evidence_consistency.report.json \
  --output-dataset results/p2/common_valid.json \
  --manifest experiments/manifests/p2-common-valid.json
```

该工具要求三臂禁用 Checker cache；除 `system_arm` 和臂本地产物路径外，完整配置必须一致。它严格校验 typed ID、输入哈希、source/runtime/API 身份、report/result 绑定及逐题终态，并实际读取三份不同的 LLM trace，核验 SHA/size/record/raw-response/parse-status，拒绝缺失、篡改、共享路径和 prompt 字段。输出 dataset 与 manifest 均不可覆盖。三个系统臂分别用现有 evaluator 在 `common_valid.json` 上做配对比较，同时保留原始三臂 report 的全量 coverage/failure_counts，以及 common-valid manifest 中各臂的 failure_by_stage。

Checker 传输/解析/schema 失败会保存状态并进入 evaluator coverage，不计为 TN/FN。非 `legacy` 下，bottom-up experience-code fail 仅保留为 audit，主诊断和 evaluator 都不将其当作预测。

当前状态：**P2 工程已收口；P3 构造、target-binding 回放、门禁评估和 semantic trace schema 审计入口已经落地。正式 300 例生成与 Qwen30B 九格回放尚未运行，因而没有效果结论**。合并 P3 与响应身份门禁后全量回归为 368/368 通过。

## P3 规则级机制门禁

`build_checker_mechanism_dataset.py` 从 1123 条 catalog 规则中按固定 seed 和精确分层抽取 60 条，用 `gemini-3-flash-preview` 为每条构造 5 类机制样本，共 300 例。它严格校验 GT 证据区间，并用 generation manifest 绑定模型、源码、运行时、API 身份和全部产物。失败规则不得替换。

`run_checker_replay.py` 在每例唯一 `frozen_target_binding` 上运行 `legacy`、`dual_evidence`、`dual_evidence_consistency` 三臂，每臂独立重复 3 次，共 2700 个 Checker 单元。前两臂是消融基线，只有 `dual_evidence_consistency` 参与判门。每格关闭 cache，并使用独立 result、report 和 LLM trace。

`evaluate_checker_mechanism_gate.py` 核验三臂配置、产物、供应商实际模型和 response ID，分别构造 `C[r]`、`T[a]`、`G` 共同有效集。API、传输、解析、schema 或绑定失败只降低 coverage，不计为 TN/FN。

`audit_checker_mechanism_semantic_retrieval.py` 是独立的次级 trace schema 审计，不参与主门禁。在 execution sidecar 完成前，其结果不能称为 production provenance 或外部效度证据。

唯一正式运行命令和判门阈值见[效果测评](../docs/效果测评.md#p3-规则级条件-checker-门禁)。脚本细节以 `--help` 为准，不在本文重复保存命令副本。

## unified_rules 生命周期

主要入口是 `unified_rules_pipeline.py`。增量流程由 `prepare_incremental_update.py` 生成 manifest v2、冻结六步运行配置和预注册预算，再由 `finalize_incremental_update.py` 重算候选差分，校验逐阶段父产物 SHA 链、确定性 formal/precluster 重放、API 完整性、蓝图 exact-once 覆盖、来源、generalized 与 catalog diff；纯逻辑比较集中在 `rule_framework/incremental_validation.py`。additive 不改变旧拓扑，`scoped_recluster` 仅能完整替换预声明 Topic。其余 `prepare_*`、`generalize_*`、`run_rule_embedding_clustering.py`、`build_unified_catalog.py` 和 `validate_*` 脚本分别承担候选准备、概括、聚类、构建和验证。

审计类脚本使用 `audit_*` 命名；分析和对比工具使用 `analyze_*`、`compare_*`、`evaluate_*` 命名。它们默认不应覆盖正式 catalog。

## 历史实验

`legacy/` 中的脚本固定了旧数据目录、旧 catalog、旧虚拟环境或特定实验表格，仅用于追溯历史实验。新的测评不应复制这些脚本，而应从 `run_physics_eval_pipeline.py` 和[效果测评](../docs/效果测评.md)中的冻结命令开始。
