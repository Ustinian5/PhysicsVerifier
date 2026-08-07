# 实验清单

本目录只保存需要进入 Git 的冻结实验 manifest；大型数据、模型输出和 trace 继续保存在被忽略的 `data/` 与 `results/`。

正式运行使用 `scripts/run_physics_eval_pipeline.py --manifest-output experiments/manifests/<run-id>.json`。脚本会自动记录：

- Git HEAD、分支、dirty 状态、tracked diff SHA；
- 全部运行源码的逐文件 SHA 和聚合 SHA；
- conda 解释器、Python 版本和包集合 SHA；
- 输入、冻结数据集、catalog、结果、trace 和指标 SHA；
- 模型、Prompt 版本、有效参数、阶段状态和失败原因。

规则：

- development 允许 dirty，但 manifest 必须如实记录；
- validation/final 启动时强制 clean worktree；
- validation/final 必须显式指定 `unified_rules_v2` catalog；
- 受控重复默认禁用跨运行 LLM cache，并逐题原子 checkpoint；
- 完成后检查 manifest 的 `status=completed`，再将 manifest 与结论文档一起提交；
- 不提交 `.env`、原始数据、结果或 trace；服务器同步这些产物时按 manifest 校验 SHA-256。
