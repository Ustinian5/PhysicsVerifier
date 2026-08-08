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

所有 Python 脚本必须在 conda `physicsverifier` 环境运行；独立命令统一写为 `conda run -n physicsverifier python scripts/<entry>.py ...`，不直接使用本机 Python。

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

回放配置会绑定 frozen manifest、Git/source tree、conda Python/package set、API endpoint 摘要和传输参数。`--run-kind validation`/`final` 强制 clean worktree 且禁止 Checker cache，运行结束再次核验 source tree；raw-response trace 不保存 prompt。`--resume` 会校验 sidecar 与 trace 前缀，并显式审计崩溃后已写入但尚未 checkpoint 的 orphan trace。

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

当前状态：**P2 工程已收口；P3 构造、target-binding 回放、门禁评估和 semantic trace schema 审计入口已经落地。正式 300 例生成与 Qwen30B 九格回放尚未运行，因而没有效果结论**。合并 P3 后全量回归为 365/365 通过。

## P3 规则级机制门禁

### 冻结抽样与 GT

`build_checker_mechanism_dataset.py` 只读取 `catalogs/rules_unified_3000.json`，不会读取历史测评数据或结果。目录实际含 1123 条唯一规则；固定 seed 为 `P3_RULE_SAMPLE_V1_20260808`，catalog SHA-256 为 `838506bfa01b67cc0038ee05d5832dcaecbc55a260f736b9107bfbfd52612005`。抽样严格得到 60 条规则：6 个 Domain 配额分别为 11、11、10、10、10、8，`gen`/`exp` 各 30，`broad_proxy`/`narrow_proxy` 各 30，无 symbolic primitive 14 条、有 primitive 46 条。任何分层不足均失败关闭，不用其他规则替补。

`trigger_scope_proxy` 只是可复现的词法长度分层：trigger 经 NFC 规范化和空白折叠后，在每个 `Domain × origin` 单元内按 Unicode 字符数、规范化 trigger、Rule ID 排序，前 `ceil(n/2)` 标记为 `broad_proxy`。该标签**不表示语义上的宽/窄**，不能据此声称规则覆盖范围更广或更窄。

每条规则由且只能由 `gemini-3-flash-preview` 构造以下 5 类机制，共 300 例：

1. `true_violation`；
2. `applicable_correct`；
3. `symbol_overlap_inapplicable`；
4. `equivalent_alternative`；
5. `insufficient_or_self_corrected`，其中 30 条规则为 `self_corrected`、30 条为 `insufficient_information`。

生成器要求 exact JSON key、固定 mechanism/expected 映射、题目侧 applicability span、答案侧 violation/correction span、唯一 quote 与精确字符区间；失败最多按冻结策略重试，不允许因难生成而更换规则。generation manifest 绑定 catalog、固定计划、模型/Prompt、源码与 conda 身份、API transport、dataset、target trace、原始响应 trace 和 checkpoint 的 SHA-256。target trace 为显式 `frozen_target_binding`：每例恰好绑定目标规则，`fixed_control_0_1=1.0` 只是控制实验常量，不是 semantic 相似度或置信度。

只冻结 60 条计划，不调用 API：

```bash
conda run -n physicsverifier python scripts/build_checker_mechanism_dataset.py \
  --catalog catalogs/rules_unified_3000.json \
  --plan results/p3_checker_mechanism/rule_plan.json \
  --prepare-plan-only
```

本机小批只用于链路 smoke。最终协议单规则产物已通过 `audit_generation_artifacts`：1/1 规则、5/5 样本、无 orphan，请求与实际响应模型均为 `gemini-3-flash-preview`；这仍不构成效果证据。复现命令为：

```bash
conda run -n physicsverifier python scripts/build_checker_mechanism_dataset.py \
  --catalog catalogs/rules_unified_3000.json \
  --plan results/p3_checker_mechanism/rule_plan.json \
  --dataset results/p3_checker_mechanism/smoke/dataset.json \
  --candidate-trace results/p3_checker_mechanism/smoke/target_binding.json \
  --raw-trace results/p3_checker_mechanism/smoke/gemini_raw.jsonl \
  --checkpoint results/p3_checker_mechanism/smoke/checkpoint.json \
  --manifest results/p3_checker_mechanism/smoke/generation_manifest.json \
  --model gemini-3-flash-preview \
  --max-rules 1 \
  --run-kind development
```

完整 60 规则/300 例必须在 clean worktree 上使用 `--run-kind validation` 或 `final`；中断后以同一配置增加 `--resume`，不得覆盖不可变 manifest。

Gemini development 产物通过公共审计后，可在同一源码状态下跑三臂各一次的 Qwen 联调。先生成一次共享 replay manifest，再为每个 arm 使用独立 output/report/LLM trace；不要传 `--enable-cache`：

```bash
conda run -n physicsverifier python scripts/run_checker_replay.py \
  --dataset results/p3_checker_mechanism/smoke/dataset.json \
  --frozen-retrieval results/p3_checker_mechanism/smoke/target_binding.json \
  --catalog catalogs/rules_unified_3000.json \
  --frozen-manifest results/p3_checker_mechanism/smoke/replay_manifest.json \
  --prepare-manifest-only

for arm in legacy dual_evidence dual_evidence_consistency; do
  conda run -n physicsverifier python scripts/run_checker_replay.py \
    --dataset results/p3_checker_mechanism/smoke/dataset.json \
    --frozen-retrieval results/p3_checker_mechanism/smoke/target_binding.json \
    --catalog catalogs/rules_unified_3000.json \
    --frozen-manifest results/p3_checker_mechanism/smoke/replay_manifest.json \
    --mode "$arm" \
    --model qwen3-30b-a3b-instruct-2507 \
    --checker-json-attempts 3 \
    --llm-temperature 0.1 \
    --llm-max-output-tokens 2048 \
    --precision-mode strict \
    --run-kind development \
    --output "results/p3_checker_mechanism/smoke/${arm}.json" \
    --report "results/p3_checker_mechanism/smoke/${arm}.report.json" \
    --llm-trace "results/p3_checker_mechanism/smoke/${arm}.llm_trace.jsonl"
done

conda run -n physicsverifier python scripts/evaluate_checker_mechanism_gate.py \
  --dataset results/p3_checker_mechanism/smoke/dataset.json \
  --generation-manifest results/p3_checker_mechanism/smoke/generation_manifest.json \
  --run legacy 1 results/p3_checker_mechanism/smoke/legacy.json results/p3_checker_mechanism/smoke/legacy.report.json \
  --run dual_evidence 1 results/p3_checker_mechanism/smoke/dual_evidence.json results/p3_checker_mechanism/smoke/dual_evidence.report.json \
  --run dual_evidence_consistency 1 results/p3_checker_mechanism/smoke/dual_evidence_consistency.json results/p3_checker_mechanism/smoke/dual_evidence_consistency.report.json \
  --development-smoke \
  --allow-incomplete-matrix \
  --output results/p3_checker_mechanism/smoke/mechanism_gate.development.json
```

当前三臂各一次的真实 Qwen30B smoke 均 5/5 完成、无 Checker failure，development evaluator 的共同有效集为 5/5；`formal_gate` 仍为未判定。因缺少另外两次重复，`candidate_acceptance.overall_pass=false` 也不是效果结论。

### 主实验：条件 Checker + release gate

主实验的 estimand 是：**预注册目标规则已提供时，Checker + release gate 能否正确发布或抑制诊断**。它不包含 retrieval，不能被表述为端到端 verifier 效果。三臂均独立调用 Qwen30B：`legacy` 和 `dual_evidence` 是消融基线，只有 `dual_evidence_consistency` 是候选判门臂；三者共享同一冻结 target trace，但不共享 Checker 响应或候选诊断。

正式矩阵为 60 规则 × 5 机制 × 3 臂 × 3 次独立重复，即 2700 个运行单元。每个格子关闭 cache，使用独立 output/report/raw-response trace，并固定 `--checker-json-attempts 3 --llm-temperature 0.1 --llm-max-output-tokens 2048 --precision-mode strict`。先从生成器输出冻结 replay manifest：

```bash
conda run -n physicsverifier python scripts/run_checker_replay.py \
  --dataset results/p3_checker_mechanism/formal/dataset.json \
  --frozen-retrieval results/p3_checker_mechanism/formal/target_binding.json \
  --catalog catalogs/rules_unified_3000.json \
  --frozen-manifest results/p3_checker_mechanism/formal/replay_manifest.json \
  --prepare-manifest-only
```

单格模板如下；`ARM` 与 `REP` 分别替换为三个模式和 `1/2/3`，九格必须使用不同路径：

```bash
conda run -n physicsverifier python scripts/run_checker_replay.py \
  --dataset results/p3_checker_mechanism/formal/dataset.json \
  --frozen-retrieval results/p3_checker_mechanism/formal/target_binding.json \
  --catalog catalogs/rules_unified_3000.json \
  --frozen-manifest results/p3_checker_mechanism/formal/replay_manifest.json \
  --mode ARM \
  --model qwen3-30b-a3b-instruct-2507 \
  --checker-json-attempts 3 \
  --llm-temperature 0.1 \
  --llm-max-output-tokens 2048 \
  --precision-mode strict \
  --run-kind validation \
  --output results/p3_checker_mechanism/formal/ARM_rREP.json \
  --report results/p3_checker_mechanism/formal/ARM_rREP.report.json \
  --llm-trace results/p3_checker_mechanism/formal/ARM_rREP.llm_trace.jsonl
```

`evaluate_checker_mechanism_gate.py` 严格构造三个交集：`C[r]` 是同一重复内三臂共同有效集，`T[a]` 是同一臂三次重复共同有效集，`G` 是九格全局共同有效集。API、传输、解析、schema、缺失结果或 target-binding 审计失败只降低 coverage，不得当作 TN/FN 或负类正确预测。

候选臂必须在每次重复分别满足：结构化输出率 ≥ 99%、真实违规 Recall ≥ 90%、四类负机制的 FPR **各自** ≤ 5%、self-corrected 诊断数为 0、可评估的 protocol contradiction 数为 0；三次发布决策一致率还必须 ≥ 95%。不允许跨重复平均后过线，也不允许把四类负例合并以稀释误报。两个 baseline 仅用于消融比较，不参与候选 acceptance。

### 次实验：semantic retrieval trace schema 审计

`run_verifier.py --retrieval-only` 的输出可由 `audit_checker_mechanism_semantic_retrieval.py` 报告 `target_hit_executable`、`target_present_suppressed`、`wrong_only`、`empty`、`retrieval_failure`、目标排名和候选数。当前入口只验证 trace schema 与分类；尚未用 sidecar 绑定运行源码、conda、模型、API transport、完整参数和 output SHA，因此**不能作为 production provenance 或外部效度证据**。它不得与 target-binding 主门禁合并，也不得用于反向修改 GT 或替换困难样本。

## unified_rules 生命周期

主要入口是 `unified_rules_pipeline.py`。增量流程由 `prepare_incremental_update.py` 生成 manifest v2、冻结六步运行配置和预注册预算，再由 `finalize_incremental_update.py` 重算候选差分，校验逐阶段父产物 SHA 链、确定性 formal/precluster 重放、API 完整性、蓝图 exact-once 覆盖、来源、generalized 与 catalog diff；纯逻辑比较集中在 `rule_framework/incremental_validation.py`。additive 不改变旧拓扑，`scoped_recluster` 仅能完整替换预声明 Topic。其余 `prepare_*`、`generalize_*`、`run_rule_embedding_clustering.py`、`build_unified_catalog.py` 和 `validate_*` 脚本分别承担候选准备、概括、聚类、构建和验证。

审计类脚本使用 `audit_*` 命名；分析和对比工具使用 `analyze_*`、`compare_*`、`evaluate_*` 命名。它们默认不应覆盖正式 catalog。

## 历史实验

`legacy/` 中的脚本固定了旧数据目录、旧 catalog、旧虚拟环境或特定实验表格，仅用于追溯历史实验。新的测评不应复制这些脚本，而应从 `run_physics_eval_pipeline.py` 和[效果测评](../docs/效果测评.md)中的冻结命令开始。
