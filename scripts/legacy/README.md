# 历史实验包装器

本目录保存早期 combined-language、dual-chain、scale、precision ablation 和服务器批处理的包装器及专用分析程序。归档的目的，是保留实验线索但不让它们与当前正式入口混在一起。

注意：

- 默认路径多为旧 `.venv`、旧数据切分或 `catalogs/legacy/`；
- 部分模型名和服务器路径依赖当时环境；
- 运行前必须逐项检查输入、catalog、模型和输出目录；
- 新实验统一使用 conda `physicsverifier`，并以 `scripts/run_physics_eval_pipeline.py` 为主入口。

旧版提交的 `run_scale_checkpoints.sh` 写死 `.venv` 并依赖已删除入口，已作为不可运行冗余移除。若需要追溯 scale 实验，可用 `generate_scale_runbook.py` 按当前 legacy 路径重新生成脚本；生成结果仍只用于历史复现，不得作为新实验入口。
