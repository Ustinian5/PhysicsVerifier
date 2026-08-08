# PhysicsVerifier

PhysicsVerifier 用于检查物理竞赛题的模型解答。系统根据题目背景检索适用规则，再对待检查答案进行语义和可选符号核查。

## 当前主流程

```text
题目与上下文
  → 背景分析与 Domain
  → Topic
  → Scenario Cluster
  → Rule
  → 语义检查
  → 可选符号核查
  → 最终诊断
```

正式检索使用 API 语义导航。题目背景负责规则适用性，`prediction` 负责答案证据，参考答案不进入 verifier。

## 快速入口

- 运行检查：`scripts/run_verifier.py`
- 效果测评主流程：`scripts/run_physics_eval_pipeline.py`
- P3 规则级机制数据：`scripts/build_checker_mechanism_dataset.py`
- P3 条件 Checker 门禁：`scripts/evaluate_checker_mechanism_gate.py`
- P3 semantic retrieval 审计：`scripts/audit_checker_mechanism_semantic_retrieval.py`
- 语义导航：`core/unified_semantic_matcher.py`
- 检查主流程：`core/physics_rule_verifier.py`
- 当前 development 基线：`catalogs/rules_unified_3000.json`（1123 条）
- 增量规则治理：`scripts/prepare_incremental_update.py` / `scripts/finalize_incremental_update.py`（manifest v2 + 配置/来源/拓扑预注册门禁）
- 历史运行时参考库：`catalogs/rules_unified_3000_runtime_backfilled.json`（4875 条，当前效果消融已否决）
- 共享文档：[文档索引](docs/文档索引.md)
- 目录职责与清理边界：[项目结构](docs/项目结构.md)

两个 catalog 不是纯 metadata 变体。当前保留 1123 条库作为高召回开发基线；4875 条库仅作历史参考。`run_verifier.py` 不会自动选择 unified catalog，所有正式实验必须显式传入路径并记录 SHA-256，禁止静默回退到旧 catalog。

正式测评统一使用 `run_physics_eval_pipeline.py`。它强制 conda 解释器和 `unified_rules_v2` catalog，自动生成包含 Git、源码、数据、catalog、Prompt、输出及失败阶段指纹的 manifest；validation/final 启动时要求 clean worktree。受版本控制的冻结清单放在 [`experiments/manifests/`](experiments/README.md)。

所有 Python 命令统一通过 conda `physicsverifier` 环境执行，不直接使用本机 Python。

```bash
conda env create -f environment.yml
conda run -n physicsverifier python scripts/run_verifier.py --help
```

只运行语义检索：

```bash
conda run -n physicsverifier python scripts/run_verifier.py \
  --retrieval-only \
  --continue-on-semantic-error \
  --unified-retrieval-mode semantic \
  --input data/input.json \
  --output results/background_retrieval/semantic_tree_results.json \
  --unified-catalog catalogs/rules_unified_3000.json \
  --model qwen3-30b-a3b-instruct-2507 \
  --semantic-output-adapter forced_tool_call \
  --semantic-json-attempts 3 \
  --unified-rule-top-n 6 \
  --checkpoint-every 1
```

## 测试

```bash
conda run -n physicsverifier python -m unittest discover -s tests -p 'test_*.py'
```

当前 P3 主实验使用 honest target binding，估计目标规则已提供时的 Checker + release gate 性能。最终协议的单规则 Gemini 与三臂各一次 Qwen30B development smoke 已闭环，但正式 300 例与九格回放尚未运行，没有新效果结论。semantic retrieval 当前只做 secondary trace schema audit，尚不提供 production provenance。项目状态和下一步见[项目进展](docs/项目进展.md)，数据隔离、完整命令和当前结果见[效果测评](docs/效果测评.md)。
