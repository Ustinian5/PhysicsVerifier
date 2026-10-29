# PhysicsVerifier

检查物理竞赛题的模型解答，输出对应规则、错误说明和原文位置，并提供训练奖励接口。

当前科研主线：GPT-OSS-120B在连续上下文中完成全文检查、规则检索和诊断，使用冻结C4提示比较旧1,123叶与新2,065叶。入口是[run_catalog_comparison.py](scripts/run_catalog_comparison.py)，实际状态与对照边界见[项目进展](docs/项目进展.md)。

现有默认verifier及训练接口保留分段流程；下方示例属于这一入口：

```text
题目与上下文 → 背景分析 → Domain → Topic → Scenario Cluster → Rule
                                                    ↓
候选解答 ──────────────────────────────────────────→ Checker → 定位与发布
```

参考答案不进入 verifier。符号核查可选；规则可达、流程完成和诊断正确是不同的验收项目。

## 运行

Python 统一使用 conda `physicsverifier` 环境：

```bash
conda env create -f environment.yml
conda run -n physicsverifier python scripts/run_verifier.py --help
```

按 [.env.example](.env.example) 配置 API。运行前将 `PHYSICSVERIFIER_LLM_MODEL` 设为实际可用的模型 ID，并按服务商支持情况选择结构化输出协议。以下为参数模板：

```bash
conda run -n physicsverifier python scripts/run_verifier.py \
  --input data/input.json \
  --output results/run/results.json \
  --model "$PHYSICSVERIFIER_LLM_MODEL" \
  --unified-catalog catalogs/rules_unified_3000.json \
  --unified-retrieval-mode semantic \
  --no-symbolic-check \
  --continue-on-semantic-error \
  --checkpoint-every 1
```

增加 `--retrieval-only` 可只运行检索。规则库必须显式指定；文件名中的 3000 不代表规则数，当前该文件实际包含 1,123 条规则。

## 文档与入口

- [项目状态](docs/项目进展.md)：已完成工作、证据边界和当前方向。
- [共享文档索引](docs/文档索引.md)：结构、规则维护和评测说明。
- [脚本入口](scripts/README.md)：检索、检查、评测和规则构建。
- [训练接入](training/README.md)：OpenRLHF、Swift 与奖励服务。
- [实验清单](experiments/README.md)：可复现运行需要保存的配置。

`data/`、`results/`、`private_docs/` 和 `.env` 不进入 Git；复现时需要另外提供相应数据和运行产物。

## 测试

```bash
conda run -n physicsverifier python -m unittest discover -s tests -p 'test_*.py'
```
