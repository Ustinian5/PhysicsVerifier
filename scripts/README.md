# 脚本入口

所有 Python 入口通过 conda `physicsverifier` 执行。完整参数以各脚本 `--help` 为准；方法和指标见 [效果测评](../docs/效果测评.md)。

换标签离线重评使用 `rescore_frozen_bench.py`。主线以 `prepare --scope all39 --data-root <原材料目录> --plan <新清单路径>` 冻结E01–E39、原50题、修订参考及输出哈希，复现旧成绩并运行已知案例合成探针；不传scope时仍为近期连续上下文4组。执行工作区使用同一代码与清单依次 `verify --data-root <原材料目录> --plan <清单路径>`、`run --data-root <原材料目录> --plan <清单路径> --output <新结果目录>`。verify返回2表示原件缺失，返回1表示哈希/计分等异常，须回主线核查，不用指标摘要代替原件。

只有run计算保存终稿的新标签成绩，输出report.json、summary.csv及带哈希的completion.json。全量报告包括78行旧/新完整P/R/F1、28项原定对照与配对区间、跨轮纯模型波动，以及按固定哈希抽取的最多8条变分样例；样例附原文和两版标签，默认未裁定。错误理由但相同位置的负控用于揭示评分器局限，不算被测模型输出。原失败与协议不合规保留，不能为读取失败外层样本ID而伪造检查完成。源文件只读、目标目录不覆盖；不调用模型、不需要凭据/GPU。新旧指标均为位置代理，不能仅凭涨分确认bench物理准确性。

2026-10-18全量已完成，入口为结果目录 `bench_rescore_20261018/all39/plan.json` 和 `all39/scored/`，无需重新运行。首批4组保留在同级父目录 `plan.json`、`scored/`；全量清单关联首批报告哈希，run逐题验证对应4组成绩一致。首批冻结代码保存在Git引用 `refs/experiments/bench-rescore-20261018-recent4`（提交对象 `50330659b12bba590478b99457cb1fb6537e0ff0`），不把扩展后的代码伪装成首批原代码，不改旧清单哈希。

服务器实验必须将模型服务、请求客户端、全批调度、计分和保存一并放在服务器上的 tmux、nohup 或已核验的进程托管方式中执行。客户端直连服务器回环服务；SSH 只负责启动、观察和取回结果，不使用本机客户端＋SSH 隧道作为正式执行链。显式指定隔离 conda 环境、工作目录与配置，标准输入脱离终端，日志及逐题原始结果保存在服务器个人持久化目录，使用互斥锁、断点和最终退出状态。正式批量前必须以预跑验证：退出启动 SSH 会话后重新连接，进程仍存活且日志/进度继续落盘。重连先查进程和锁，禁止重复启动；成功项不重跑。下述 macOS 本机运行措施仅适用于独立本机任务，不能替代服务器断连验收。

## 运行与评测

| 脚本 | 用途 |
|---|---|
| `run_verifier.py` | 完整 verifier；`--retrieval-only` 只运行检索 |
| `run_physics_eval_pipeline.py` | 构造数据、运行和评测，自动保存实验清单 |
| `rescore_frozen_bench.py` | 全39组或近期4组×50题离线重评；prepare/verify复现旧分数，run换参考并导出逐题分数、配对区间、变分样例；不调用模型、不覆盖历史输出 |
| `run_llm_checker_baseline.py` | 直接检查基线；Responses 模式可用 `--rule-catalog` 运行 Terra 规则树自主检查（无独立 checker） |
| `extract_holdout_eval_samples.py` | 去重并排除历史样本 |
| `audit_correct_eval_candidates.py` | 复核正确解答候选 |
| `build_physics_eval_sets.py` / `audit_eval_set_quality.py` | 构造数据及检查 GT 质量 |
| `evaluate_physics_eval_sets.py` | 错误位置匹配指标 |
| `evaluate_question_level_sets.py` | 题目级指标 |
| `prepare_retrieval_comparison.py` | 从历史池冻结 50 条，隔离参考答案与待复核标注 |
| `run_retrieval_comparison.py` | 四组检索；`development` 预跑、`retrieval_batch` 运行已冻结的 50 条；保存 API 请求和工具记录，不含 checker 与物理计分 |
| `run_retrieval_checker.py` | 将上述冻结候选接回原 Qwen checker、定位和发布；`--prepare-only` 无 API 调用，完整请求一致时共享原始响应 |
| `run_continuous_context_v3.py` | C0p/C4 各50题：同会话先独立检查，再实际检索并核对规则；复用v2预算、工具、引文修正及计分。使用 `python -m scripts.run_continuous_context_v3` 的prepare/run/confirm-detached/score模式 |
| `run_continuous_context_v2.py` | 固定 GPT-OSS 连续会话对照：C0p 纯模型与 C3 问题驱动规则辅助，各 50 题；共用引文协议，自动保存及计分 |
| `run_bench_reannotation.py` | 完整 99 题的四轮机器重标；盲复标不看初标，旧 GT 与 verifier 输出不进入提示；`audit` 离线核验；当前修订发布见结果目录 `assistant_review/release_manifest.json`，原四轮已完成，不要重启生成 |

`run_physics_eval_pipeline.py` 要求显式传入规则库，使用当前 conda 解释器；`validation/final` 要求 clean worktree。运行配置与输出身份按 [实验清单](../experiments/README.md)保存。

API 先行时使用 `run_retrieval_comparison.py --phase retrieval_batch --input … --selection-manifest … --output … --arms 2 3 4 --api-model gpt-5.6-terra --reasoning-effort medium --workers 1 --overload-backoff`。当前默认 Terra medium；历史 Astra 需显式传入 `--api-model gpt-6-astra --reasoning-effort max`。沿用 `.env` 的组内 `ASTRA_BASE_URL/ASTRA_API_KEY` 或 `OPENAI_BASE_URL/OPENAI_API_KEY`，变量名不决定实际模型。脚本校验冻结输入的 SHA、50 个 ID 和模型输入字段，并核验响应中的模型和推理档位；配置变化须独立运行，禁止混入旧模型结果。并发共用请求/token 预算，额度查询最多每分钟一次；查询接口故障只记警告、5 分钟后重试。确认低额度、鉴权失败或不明原因的 429 会停止新请求。GT 复核和 Qwen 不阻塞检索采集，但仍是最终计分的前提。

`--overload-backoff` 仅用于单并发 API 批量。对明确的上游容量错误、408/502/503/504，以及传输超时和断连，按 5、10、15 分钟重试完全相同的请求；**每次逻辑请求最多重试三次，成功后下一请求重新计数**。尊重 `Retry-After`，单次请求计划等待最多 30 分钟；持续故障则停止整批，状态为 `stopped_upstream`。每任务仍最多 32 次实际请求加已重放响应，全批请求/token 与余额保护继续生效；未知用量保留全批预留值，不能无限重试。全部失败保留，等待单列于 `overload_retry.json`，不占原 30 分钟执行预算；`waiting_for_capacity` 表示正常等待，不能当成进程中断。未核实的模型名/推理档位不自动接受或重试，单列技术失败。不启用此选项时保留原错误停止行为。

需要并行补齐时，使用 `--workers 6 --drain-transient`，不要同时加 `--overload-backoff`。临时失败先让出位置，跑完本轮后自动创建哈希关联的补跑轮次，精确重放成功前缀；已完成任务不重跑，真实空选也不重跑。模型身份被拒绝的响应只留档、不采纳，下一轮重发同一请求，仍要求精确匹配；解析、规则选择等非网络失败不自动抽样重做。根目录 `progress.json` 指向当前轮，`runner.pid`、`runner.log` 和 `STOP` 始终位于根目录。完全无有效响应时逐轮冷却至最多 5 分钟，并尊重 `Retry-After`；额度不足、鉴权或不明限流仍停止。可用 `--deadline-unix` 指定停止新请求的时间，为保存和汇总留余量。只允许显式增加全批请求/token 计数上限以容纳技术失败，变更写入 manifest，既有用量继续累计；每题模型预算、输入、提示和规则须与父批次一致。

跨机器续跑时，先确认旧端已停止，复制完整关联记录，再增加 `--relocate-root /旧项目绝对路径 /新项目绝对路径`；旧 manifest 和响应保持原字节，仍逐项验证哈希，新 manifest 记录路径映射，后续续跑自动继承。`--lock-file …/active.lock` 为本机各次尝试指定同一把锁，防止重复调用；锁不能代替跨机器停机确认。脚本记录运行中的 `runner.pid`，正常退出后移除。API 批量可在当前输出目录创建空文件 `STOP`，待正在发送的请求结束并落盘后停止新请求；不要强杀。macOS 可加 `--notify-on-exit` 发送结束通知（是否显示由系统通知权限决定），并使用 `caffeinate -i` 防止空闲休眠；合盖、断网仍会影响运行。

接续预算/网络中断时使用新的结果子目录，追加 `--retry-transient-from 原运行目录 --continue-pending --replay-prefix`。原始失败保留；在全部关联记录中跳过已成功项，未启动任务继续执行。已收到的成功响应只在完整请求逐项相同时重放以恢复运行状态，不重复付费；实际新请求、重放数量和历史用量分开记录，全批预算累计计算。没有明确完成状态的在途请求仍须先核对，不能覆盖重跑。

## 连续会话对照：C0p / C3

完整提示在 [continuous_context_v2.py](continuous_context_v2.py) 的 `BASE`、`RULES` 和 `repair_message`；实验调度和冻结参数在 [run_continuous_context_v2.py](run_continuous_context_v2.py)。被测模型固定 GPT-OSS-120B medium，两组共用全文检查提示，C3 增加按具体疑点检索、查询无进展时换路线、核对规则前提及返回全文查漏。检索和诊断在同一会话完成，不另接 checker。

每题累计输出 16,384 token（含推理），最多 12 响应，规则组最多 6 个不同完整叶。程序核对原文引文，必要时在原会话最多进行一次只改引文的修正；计入原预算，诊断数量、顺序、理由和规则引用不允许改动，修正失败保留原诊断。两组都执行此协议，并分别计分修正前、修正后输出。

在服务器原仓库、个人 conda 环境中执行以下命令；`RUN_DIR` 为个人持久化结果目录，`PYTHON_BIN` 为该环境的 Python 绝对路径，所需只读运行库按已有服务器配置设置。输入固定为 `results/retrieval_comparison_50_20260920/frozen50/inputs.json`，旧标签文件只供离线计分；`prepare` 会验证样本、规则、标签和源码哈希。

```bash
"$PYTHON_BIN" -m scripts.run_continuous_context_v2 prepare --output "$RUN_DIR"
nohup "$PYTHON_BIN" -u -m scripts.run_continuous_context_v2 run --output "$RUN_DIR" >> "$RUN_DIR/runner.log" 2>&1 < /dev/null &
```

退出启动 SSH，再新建 SSH 会话，在同一环境和目录中执行：

```bash
"$PYTHON_BIN" -m scripts.run_continuous_context_v2 confirm-detached --output "$RUN_DIR"
```

启动后先等待断连确认（最长 10 分钟），再执行 1 次原生工具 IO 探针和 5 个集外样例×2 组；工程验收不要求物理判定全正确。通过后按题目轮换两组，自动完成 100 项正式任务及旧位置 P/R/F1 计分。不修改提示、模型或预算参数，不重跑空诊断/零检索结果；技术故障保存原始证据并停止，不自动重试或换模型。创建 `STOP` 可在当前题结束后停止新任务；重启先核实进程和锁。

查看 `execution_status.json`、`formal/progress.json` 和 `runner.log`。最终 `metrics_automatic_proxy_unreviewed.json` 包含完整三指标、修正前后分数、未定位数、用量和位置命中增减。未完成组不给伪完整分数；旧位置分数不是物理正确率。只读重计分用同入口 `score` 模式。配置发生变化时拒绝混入同批次，不能删除原文件绕过校验。

## Bench 重标

重标入口使用 `conda run -n physicsverifier python scripts/run_bench_reannotation.py --source <99题历史输入> --output <原结果目录> --env .env --mode pilot|formal|audit --recover-technical`，`.env` 使用 `OPENAI_BASE_URL/OPENAI_API_KEY`，实际模型固定 `gpt-6.1-sol`、high。先做固定 5 题四轮试跑；`supervise_bench_reannotation.py` 可在工程验收后接续正式 396 项，同协议试跑结果复用。正式续跑先核实进程和锁，已有终稿不重抽。整批技术退避为 5/10/20/30 分钟并跨重启保留；新未知 429、鉴权、余额或内容协议异常停止，成功响应与原文哈希不符也停止。历史 HTTP 发出前的连接失败可在一次性凭据下补回请求计数，原始记录及未知 token 预留不清零；后续重启不自动加额度。查看 `progress.json`、`budget.json`、`scheduler.json` 和 `quality_audit.json`；创建 `STOP` 或发送 SIGTERM 后当前请求先落盘，再停止。

本批99题四轮已完成。2026-10-18先冻结原机器参考v1，随后按用户要求由主线Codex助手处理未决、引文、遗漏与来源标记，0新增API实验。当前入口为结果目录 `assistant_review/release_manifest.json`，对应 `reference.json` 和含逐项理由、原值及可重放补丁的 `review_decisions.json`；原四轮记录、`machine_reference.json` 和根目录v1清单保持不变。修订参考含402条confirmed、7条unresolved及1条合并记录；404条唯一引文、5条遗漏证据，未决不作正负金标，遗漏不伪造位置。各方案重评必须使用同一版本与口径，旧分数保留。助手复核不等于人工金标；原采集pending字段和质量审核是历史状态，不应据此重启生成。全量 `audit` 会重写原质量审核；检查冻结原记录应只读调用 `validate_saved`，不要对冻结目录重新运行会写文件的模式。

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
| `unified_rules_pipeline.py` | 旧规则库构建入口；120B逐题提炼核心；正式调度见run_experience_first_120b.py |
| `build_120b_rules.py` | 已完成83条库的原构建器，保留两题支持协议与恢复证据，不用于新逐题方案 |
| `rebuild_qwen30b_corpus.py` | 冻结训练题与bench隔离，原版30B每题生成一次，120B medium独立筛错；保存原始响应与可恢复断点 |
| `run_built_catalog.py` | 新构建库的C0p/C4连续会话对照，各50题；原120B预算、无独立checker、新参考位置计分 |
| `run_rule_skill_comparison.py` | 新规则树 vs 120B从同批1,726份30B错答提炼的skills；两组各50题连续检查、配对P/R/F1与双向案例；不修改默认库 |
| `prepare_incremental_update.py` | 建立增量工作区并冻结范围、配置和预算 |
| `finalize_incremental_update.py` | 校验来源、变化范围及产物依赖，生成待审目录 |
| `validate_unified_catalog_structure.py` | 校验 ID、引用和可达性 |

流程见 [增量规则说明](../docs/增量规则流程.md)。`legacy/` 仅用于复现旧实验，不能直接沿用其中的路径、环境或模型配置。

## 120B逐题建库

10.24起在旧流程上改造：`generate_experience_rules.experience_120b_prompt`与`normalize_120b_experience`完成完整输入的审查＋提炼提示和行号取证；`unified_rules_pipeline.build_120b_experience_catalog`串联原候选去重、来源合并、120B归组及原导航叶/结构验证。`experience_topic_skeleton`只提取旧6领域/123主题的名称与层级，旧规则和摘要不传入。单来源候选保留，跨题概括不作为产出前置门槛。

当前v5只收集未验收机器候选：沿用v3逐题初稿提示，不采用试验中未解决物理误判的固定复查。结构异常最多一次完整上下文修复；仍不合规者进入invalid_outputs，不能算正确、空选或入库规则，也不阻塞其余题。技术故障、模型身份、隔离、树成员守恒仍严格停止。输出发布清单固定usable_for_runtime=false、physical_quality_passed=false，不能绕过物理验收接入默认库。试跑收集许可记录engineering_passed、collection_only与已发现的质量问题，不冒充物理验收通过。`prepare --parent-preparation DIR`复用相同生成/隔离证据；父v3已完成目录/初稿仅在请求体完全一致时重放，调用计数保留。

正式入口为`python -m scripts.run_experience_first_120b`，必须在服务器个人conda环境中执行。依次使用`prepare --output DIR --corpus-release RELEASE`、后台`run --output DIR --phase pilot`、断开SSH重新连接后`confirm-detached --output DIR`。固定12题试跑输出保存在DIR/pilot，由主线审读并以`acceptance.json`绑定counts及semantic_experience的SHA；然后后台`run --output DIR --phase full`并再次断连确认。不能跳过验收，不能让执行助手自行改提示或内容门槛。

prepare核对已发布的全部完整30B回答及实际生成证据，不导入旧筛错标签，重新检查所有已登记bench。Domain/Topic按领域分批补摘要，保持原名称/层级；逐题审查＋提炼和场景归组都用本地120B medium。成功调用精确重放，只有明确技术故障在同一请求最多3次尝试内恢复；模型身份、语义/结构异常停止，保存原件由主线修复。正式输出不覆盖旧83条或默认库。`execution_status.json`、`budget.json`、`runner.log`与原始arm目录分别记录状态、用量、过程和证据；release_manifest仅表示候选库构建完成，仍不是物理效果验收。

离线验证：`conda run -n physicsverifier python -m pytest -q tests/test_experience_first_runner.py tests/test_experience_first_120b.py tests/test_released_corpus_build.py`。以下命令仅属于历史83条协议。

## 已完成83条120B构建与历史恢复

10.21起使用`oss120b_all_stages_v2`。最新决定先由原版Qwen3-30B-A3B-Instruct-2507重新答题，随后筛错、目录、机制归组、通用规则提炼和ScenarioCluster组织全部由GPT-OSS-120B medium完成；不调用外部教师、embedding或导入旧叶。实现与提示在[oss120b_construction.py](../rule_framework/oss120b_construction.py)。空结果和拒绝不重抽，每条正式叶仍须至少两个不同题干模板支持。

来源必须先核实再显式指定，不能凭verifier模型推断答题模型。旧300/3000份均已追溯为Qwen3-4B；另找到的30B训练首轮材料仅33道筛后独立题，均不混入新采集。新采集入口如下，准备目录内须已有`question_pool.json`（id、question）及`source_manifest.json`；题目只用原题字段，旧回答、参考、评分标准和历史消息中的提示均不传给模型。旧领域标签也不可靠（曾发现反帕斯卡三角纯数学题混入physics）。同一120B先按固定提示每批16题筛查physics/non_physics/uncertain，只有physics继续，歧义与格式失败隔离、不补抽。正式模型调用完全在服务器后台进行，先退出启动SSH再确认断连，工程控制通过后才能生成正式题：

```bash
python -m scripts.rebuild_qwen30b_corpus prepare --output <采集目录> --size 3000
python -m scripts.rebuild_qwen30b_corpus verify --output <采集目录>
# 服务器个人conda，后台运行；30B服务18084，既有120B服务18083
python -u -m scripts.rebuild_qwen30b_corpus run --output <采集目录>
python -m scripts.rebuild_qwen30b_corpus confirm-detached --output <采集目录>
```

已有v2批次遇到技术故障时，先等旧客户端停止、锁释放，保留所有原始请求，再用恢复入口，不重新prepare原数据：

```bash
python -m scripts.resume_qwen30b_corpus prepare --output <采集目录>
# 服务器后台启动后退出SSH、重连确认
python -u -m scripts.resume_qwen30b_corpus run --output <采集目录>
python -m scripts.resume_qwen30b_corpus confirm-detached --output <采集目录>
```

恢复清单绑定原配置、全部既有请求和扩展代码。仅恢复完整覆盖输入且相同ID标签一致的领域重复项；缺ID、陌生ID和矛盾不推断。原收据不改，派生结果单独关联。生成与领域筛查重叠，之后审查与剩余生成重叠，并发仍为生成4/审查1；所有实际请求保持原指纹，成功和截断不重抽。最终发布清单包含恢复依据。vLLM 0.11.2启动服务时CUDA_VISIBLE_DEVICES需用数字序号，并另核对GPU UUID，不能直接填GPU UUID。

已执行旧恢复代码时，升级须先等待客户端退出、两把锁释放及在途请求落盘，再以Git中的旧源码核验并绑定已有结果；不能直接替换后启动。本轮旧调度源码Git对象为`58c2671977b03b1ec0f9e0980ade6c37b72b80fd`：

```bash
/home/visitor/huhengyuan/envs/physicsverifier/bin/python -m scripts.resume_qwen30b_corpus upgrade \
  --output /slow_share2/huhengyuan/physicsverifier_results/qwen30b_corpus \
  --previous-source-git-blob 58c2671977b03b1ec0f9e0980ade6c37b72b80fd
```

升级只修复实时进度、异常及时停止新请求及审查续跑误等已释放30B服务的问题，原请求与三次技术尝试上限不变。实时查看`continuation_progress.json`；原核心阶段字段在等待期间可能滞后。CUDA graph工程探测使用`probe_qwen30b_runtime.py --role qwen|auditor`，仅在独立空闲卡运行合成控制，不能用其结果冒充正式采集或效果测评。

已冻结旧升级清单的批次若需修复最终发布路径，先等客户端退出、两把锁释放，使用`patch-publication --output <原目录> --previous-source-git-blob 7451a9eaeebab9d1ac58bcf9d00a5000f8b6545a`。它保留原`continuation_upgrade.json`，以`continuation_publication_patch.json`关联旧源码与全部既有证据；不重写原请求、提示或采样。此补丁只防止校验运行证据时覆盖最终清单的目标路径，最终发布同时绑定两份技术恢复清单。已完成该步骤的批次直接用run续跑，不重复升级或打补丁。

两项服务分别由`start_qwen30b_corpus_service.sh`、`start_oss120b_corpus_service.sh`在服务器后台启动。显式设置`CORPUS_GRAPH_MODE=1`可使用已验证的level=0、FULL_DECODE_ONLY模式，不做Torch编译；缓存位于个人`/dev/shm/huhengyuan/`，原始记录仍持久化。默认未设置时保留eager模式。切换须先排空客户端，核验并停止自己的原服务；不能覆盖运行中的启动脚本或抢占GPU。实际服务的切换前后验证用`probe_qwen30b_runtime --validate-existing before|after`，后台启动、退出SSH，再重连以同参数加`--confirm`放行。每阶段只复用三项既有合成控制，收到的响应不重跑；须审读生成控制并保存验收，最终发布绑定`service/runtime_transition.json`关联的证据。

重启恢复时，可显式设置`CORPUS_RUNTIME_SITE`复用已核验的个人持久化运行库；30B可同时设置`CORPUS_GPU_INDEX`与`CORPUS_GPU_UUID`，启动器仍要求该GPU空闲、型号容量合格且UUID一致，不直接沿用重启前编号。120B仍检查原GPU1及其UUID。重启后的合成控制使用`--validate-existing reboot`，与before/after记录分开；模型、提示、种子、预算及既有结果均保持不变。环境包解压不得修改他人的运行库，正式客户端仍显式使用个人conda。

固定种子、每题1份回答；30B为BF16原始权重、temperature=0.7/top_p=0.8/top_k=20、最多16,384生成token，120B medium独立判断，不接旧标签。筛后不足3,000道时用全部合格题，不能重复题目补数。采集记录正常/截断、明确错误/未检出错误/不确定/格式失败；只有有连续原文证据的错误候选进入`wrong_answers.json`，不称人工金标。只恢复技术失败（最多3次），不重抽成功或内容异常；成功项与原始响应绑定哈希。生成结束后仅释放本任务专用30B服务；120B继续筛错。输出`release_manifest.json`后才可用于后续建库，不能把正在写入的中间结果当完整来源。

10.23采集批已完成：2,627份生成（2,626完整、1截断），2,626份审查中468份进入`wrong_answers.json`；1,202份未检出错误、169份不确定、787份格式/引文无效分别保留。发布清单为结果目录内`release_manifest.json`，离线工程/协议及候选文本隔离验收为`completion_verification.json`，均不将机器判断升级为人工金标。已完成采集目录禁止重跑；后续建库使用独立结果目录和以下入口。

后续构建按实际独立题量冻结，不直接重启旧pilot：

```bash
# 在项目conda环境内；prepare/verify不调用模型
python -m scripts.build_120b_rules prepare \
  --source <已发布的错题文件> --source-model Qwen3-30B-A3B-Instruct-2507 \
  --corpus-release <采集目录>/release_manifest.json \
  --output <新的构建结果目录> --size <不同题干数量> --distill-max-requests <本批提炼请求上限>
python -m scripts.build_120b_rules verify --output <新的构建结果目录>
```

新30B来源必须加`--corpus-release`：核验完整发布清单、逐题生成和审查原始请求/响应，复用已接纳错误，不重做筛错、不修复已拒绝引文。120B依次建立通用目录、分配错误Topic、按机制归组、结合完整题目/回答核实并提炼规则、组织ScenarioCluster；拒绝项逐条保留，每条叶仍需两个不同题干模板支持。plan绑定源码、提示、模型角色、来源及隔离证据和预算；历史无发布清单模式仅保留来源声明，不能混称同一协议。必查bench由`BENCHMARK_FILES`登记，包括新增的服务器历史评测题干表；同步至执行环境时该表不可遗漏，`--exclude`只能追加。检查同题、数字变体、近重复和各可见字段嵌入的题干；verify/run及在线接入均重放筛查与抽样。文件缺失/改变、实际在线题目重叠即停止。文本筛查不证明全部语义题族独立。

服务器客户端只连本机18083的既有个人120B服务；不要求OPENAI_API_KEY或外部额度，不自动部署或抢占GPU。先核对进程/锁、服务身份、模型配置/索引哈希及磁盘，再用个人conda后台运行`python -u -m scripts.build_120b_rules run --output <构建目录>`。新来源首次先给`run`加`--control-only`，仅跑两个合成错误的完整提炼/建树控制。退出启动SSH后重新连接，执行`python -m scripts.build_120b_rules confirm-detached --output <构建目录>`，放行后才发请求。主线审读控制规则后写入`controls/acceptance.json`（actor为Codex assistant、passed为true、summary_sha256绑定controls/summary.json），再以不带`--control-only`的`run`启动正式批并重新断连确认；控制成功响应直接复用。新输出优先放在个人slow_share2目录，服务和客户端均须在服务器。

成功请求及完整落盘响应直接复用。每任务最多3次尝试，只有明确技术故障且冷却至少300秒/满足Retry-After后才可用`--retry-transient`；身份、内容、未知限流错误不重抽。STOP阻止新请求，在途响应先落盘；查看`execution_status.json`、`budget.json`和`arm_*`。总预算包含失败的保守预留，提炼次数上限也含失败，不等于美元额度。历史回答中的Harmony标记仅作JSON无损转义，不删原文。引用/类别/schema异常继续保留证据回主线处理。

10.24主线增加Topic分类的无损结构恢复：仅对同一错误编号、同一Topic完全重复的输出去重，缺项、未知编号、冲突分类仍停止。停机且无在途请求时，执行`python -m scripts.build_120b_rules recover-topic-duplicates --output <构建目录>`离线登记；`topic_assignment_recovery.json`绑定原plan、完整成功前缀和修改后源码，每次实际去重另存`normalization.json`。原请求、响应和缓存值不改写、不重抽；随后按上述后台启动和新SSH确认流程续跑。发布清单包含恢复及逐项归一化记录。该入口不修复其他物理或格式内容。

同日增加提炼阶段重复拒绝ID的结构恢复：`recover-rejection-duplicates --output <构建目录>`只合并同一ID的拒绝记录、保留全部不同理由，不改任何规则正文或支持关系。`distill_rejection_recovery.json`串联前次恢复清单、原始前缀和源码；每个实际变换仍存逐项`normalization.json`。同一错误既被支持又被拒绝、来源缺失/伪造、多条规则重复使用同一支持、两道独立题支持不足等仍停止；不得以此入口修复物理判断或重新抽取答案。

机制归组已有编号被缩写时，可用`recover-group-prefixes --output <构建目录>`登记恢复：仅解析至少7位十六进制、唯一对应输入已有组的前缀，完整编号和新组key优先精确匹配；不按名称或物理含义猜测。歧义、未知编号、重复/缺失成员仍停止。`group_id_recovery.json`串联此前恢复及原件，逐项`normalization.json`记录编号展开，原响应不改写。

归组分配完全重复时使用`recover-group-duplicates --output <构建目录>`；只删除完全相同的分配对象，保留首次出现顺序。`group_member_recovery.json`绑定此前恢复和原件，变换写入`group_member_normalization.json`，不覆盖旧编号展开记录；不同组的冲突分配、缺失或未知成员仍停止，不重抽内容。

完整组ID被错误拼写且无法无损还原时，本轮已授权`prepare-short-ids --output <构建目录>`登记协议修订。它保留全部旧请求/响应和成功前缀，仅对登记的无效批及尚未请求的group_errors/cluster_rules使用请求内I1/G1短编号，由程序精确映射回持久ID；新提示、受影响批次和源码记录在`short_id_protocol.json`。新请求在`arm_*_shortids/`，结果和`id_mapping.json`绑定原件，不覆盖旧批，不按效果择优。原模型、温度、token/总调用上限不变，同一批新旧请求共用原尝试上限；新协议仍有无效输出时停止。该产物属于有明确协议修订的接续批，不声称与原提示完全一致。

提炼来源编号错误时，本轮接续使用`prepare-short-distill --output <构建目录>`登记`short_distill_protocol.json`。新distill请求将来源ID映射为E1/E2，并明确全体来源在各规则support与rejected之间互斥分配；仍要求每条规则至少两道题支持，不允许跨规则重复支持。旧有效提炼复用，只补清单登记的无效批；新旧请求合计计入原提炼/尝试/全局预算，原失败不覆盖，短编号结果和映射在`arm_distill_shortids/`。

短编号提炼仅在rejected中多出输入外编号、真实来源覆盖和支持全部合格时，可登记`recover-unknown-rejections`。不存在的拒绝项完整保存在原响应与`unknown_rejections.json`，不参与计数；任何未知支持、真实来源缺失/重复、两题支持不足仍停止。`unknown_rejection_recovery.json`绑定改动及历史原件，不再调用模型、不更改规则正文或真实归属。

单值error_type收到明确的合法多标签（如logic|calculation）时，`recover-error-types`登记固定取首项的元数据投影；原列表及选择保存在逐批error_type_projection.json，规则正文和支持关系不改，未知标签不猜测。该项属于有记录的字段兼容转换，不声称保留了全部分类标签，不增加模型调用。

目录响应的JSON字符串包含未转义LaTeX反斜杠时，`recover-json-escapes`登记序列化恢复：只将字符串内JSON不支持的反斜杠转义编码为字面反斜杠；合法转义不改，截断、缺逗号等其他JSON错误不修补。原响应不覆盖，逐请求json_escape_view_request_*.json记录字符位置和哈希，复用原响应、0新增模型调用。

本入口启动和候选库接入均强制校验无embedding、无外部教师、无旧规则导入。机制归组、ScenarioCluster和在线语义检查使用120B；在线BM25/TF-IDF工具仅作本地词频计算，不调用embedding模型。旧`unified_rules_pipeline.py`的embedding步骤只保留用于历史复现，不属于本批链路。

原Sol试跑及`resume_120b_construction.py`是历史协议证据，新入口拒绝旧plan；不能用换模型方式续跑或覆盖旧结果。此前引用、修正内容与归组问题尚不能因模型改动视为解决。

完成后输出候选库、逐规则来源/拒绝、统计和`release_manifest.json`；零规则结果保留但不能进入在线评测。有可用新库后用以下入口，不改变默认库：

```bash
python -m scripts.run_built_catalog prepare --build <构建目录> --output <新对照目录>
# 在服务器后台运行；退出SSH后重连，执行confirm-detached
python -u -m scripts.run_built_catalog run --build <构建目录> --output <新对照目录>
python -m scripts.run_built_catalog confirm-detached --build <构建目录> --output <新对照目录>
```

C0p纯120B与C4规则辅助各50题，同一120B配置、每题累计12次响应/16384生成token、最多读6叶；C4保留连续上下文，不拆checker。沿用10个工程控制与工具IO探针，再做正式对照。冻结新参考输出完整P/R/F1、敏感性、失败和配对区间，同时报告实际调用/token与树/叶使用。相同预算上限不等于实际等成本，位置命中不等于物理正确，开发结果不作独立论文结论。

### 累计5,000题扩容（独立入口，不修改运行中的逐题建库）

`prepare_qwen30b_expansion.py`从固定版本[MegaScience/TextbookReasoning](https://huggingface.co/datasets/MegaScience/TextbookReasoning)的train/physics只导出题干，校验源SHA，做bench、旧题及新题间去重。许可CC-BY-NC-SA-4.0；上游题目曾经其他模型整理，本项目不导入其答案或解答。仅此离线导出需要pyarrow，推理端不需要。所有Python命令在个人conda环境执行。

```bash
python -m scripts.prepare_qwen30b_expansion export-source --parquet <固定源parquet> --output <题干目录>
python -m scripts.prepare_qwen30b_expansion prepare --source <题干目录> --base-corpus <原30B发布目录> --parent <v5逐题目录> --output <扩容目录> --target 5000 --reserve 4000
python -m scripts.run_qwen30b_expansion prepare --output <扩容目录>
python -m scripts.run_qwen30b_expansion verify --output <扩容目录>
```

已有2,626份完整回答直接复用；120B仅按领域从固定备选顺序选2,374新题，随后冻结一次性30B作答。生成前不按预测错题率选题，生成后不补抽；截断、技术失败和bench隔离排除另计，因此完整入库回答可能不足5,000份。新来源与规模同时变化，不是严格同分布规模消融。

`start_qwen30b_expansion_service.sh <扩容目录绝对路径> <空闲GPU编号> <对应UUID>`只启动本人的30B，复用个人权重SHA边界，检查GPU/端口/空间。120B沿用原服务。模型服务与下面的客户端都须在服务器nohup/tmux后台运行，客户端直连回环地址。客户端要求v5已完成发布及锁释放，条件不满足或源文件不符即停止；不改运行中的父批。退出启动SSH并重连后再confirm，首次状态为waiting_disconnect_acceptance，未确认不推理。

```bash
python -u -m scripts.run_qwen30b_expansion run --output <扩容目录绝对路径> --phase all
# 在新的SSH会话核对存活进程后执行；不能再启动第二个run。
python -m scripts.run_qwen30b_expansion confirm-detached --output <扩容目录绝对路径>
```

生成阶段写generation_release.json后关闭本批自己启动的30B；只对新回答提炼，父批初稿/规定结构修复按完整请求体重放，合并候选后重新组织ScenarioCluster。复用Domain/Topic名称、单来源规则保留，不调用embedding或外部教师；新库只作机器候选，build/release_manifest.json明确physical_quality_passed=false、usable_for_runtime=false，不自动接默认库或在线评测。支持collect/build分阶段及幂等恢复，所有模式均保留失败和来源证据；成功项不重跑。验证见`tests/test_qwen30b_expansion.py`。

#### 大目录建树与快速恢复

已完成提炼的扩容批次使用`reorganize_qwen30b_expansion.py`：逐规则重新分配Topic，再以每批4规则、每页16已有簇完成场景归组；全部语义步骤仍为120B。旧规则正文/来源不变，旧归组原件保留，新产物在同批`tree_v2/`。这是导航协议调整，不能当作旧归组无损重放或仅规模变化的消融。

```bash
python -m scripts.reorganize_qwen30b_expansion prepare --root <已有扩容目录>
python -m scripts.reorganize_qwen30b_expansion verify --root <已有扩容目录>
# run须在服务器后台运行；退出启动SSH后，在新会话confirm-detached。
python -u -m scripts.reorganize_qwen30b_expansion run --root <已有扩容目录>
python -m scripts.reorganize_qwen30b_expansion confirm-detached --root <已有扩容目录>
python -m scripts.reorganize_qwen30b_expansion status --root <已有扩容目录>
```

迁入前先让旧客户端保存当前请求后停止并释放锁；保留根STOP，新的停止文件为`tree_v2/STOP`。prepare一次性导入并核验全部历史；恢复使用原子检查点及内容绑定的证据缓存，不重新解析全批响应或重算相同输入/bench的隔离。首次导入、最终发布全量SHA；平时核查清单、当前bench/代码/配置和变更文件，可用`verify --full`强制全量核验。每页均被访问后才允许新建簇，超出64页或判断不清保留未决；每次输入≤48,000字节，短编号精确映射，响应预算不增加。分页限制单次负载，最坏总比较量仍可能二次增长。

仅漏编号可一次补齐，成功分配不能推翻；内容失败隔离，连续三次内容失败停止。历史用量与崩溃预留不清零。发布同时保留`unplaced_rules.json`，所有输入候选必须守恒；`physical_quality_passed=false`、`usable_for_runtime=false`，不自动替换默认库。工程覆盖见`tests/test_bounded_tree.py`，工程通过不代表物理质量验收。

### 同一批30B错答生成规则树与skills的对照

`run_rule_skill_comparison.py`比较同一错题材料的知识生成方式：A为已有30B错答经120B审查提炼的新规则树；B为120B直接读取同一批1,726份完整30B错答，在连续上下文中核查并凝练skills，不另让120B重新解题。1,726份由原4,995份完整回答的机器筛错结果确定，覆盖入树前全部候选来源，不只取2,065条已入树叶的来源；准备时核验来源守恒。B只见原题、30B原答和题设上下文，不提供原判词、规则或参考答案/测试标签。每题最多两项skill，仅精确去重；无法证实错误、信息不足、有效空提炼、技术失败和未入树分别记账。skill的目标、条件、步骤、陷阱、自检方法无损映射到旧叶接口，正文不改写为判错规则。两组共享6领域/123主题目录和同一120B分页归组机制；不使用embedding或外部教师。

测试继续检查原50份固定解答。两组采用相同120B medium、连续上下文、工具、提示和每题预算（最多12响应、总16,384生成token含推理及引文修正、最多6张完整知识卡）。可独立发现未被库覆盖的错误；不使用规则不重抽。两组共100项，按题交替先后顺序。原库质量标记不翻转，明确只允许原样含噪声库用于本次开发对照。比较的是同一30B错答凝练规则或skills的完整方法，不能单独证明树形结构的因果效果；共有机器筛错与各自建库计算量分别记账，不声称计算量相等。

以下按顺序执行，Python必须来自个人conda。`build-pilot`、`build`、`run-eval`均须在服务器后台执行，退出启动SSH后重新连接并调用对应`confirm-detached`；不能在本机经隧道运行。先核验个人服务、占用和锁，不部署第二个模型。

```bash
python -m scripts.run_rule_skill_comparison prepare --rule-tree <5000批次/tree_v2> --output <对照目录>
python -u -m scripts.run_rule_skill_comparison build-pilot --output <对照目录>
# 新SSH会话：
python -m scripts.run_rule_skill_comparison confirm-detached --phase skills --output <对照目录>
# 固定12题完成后，只验工程结构；失败留在主线处理，不修改prompt后混用旧响应。
python -m scripts.run_rule_skill_comparison accept-pilot --output <对照目录>
python -u -m scripts.run_rule_skill_comparison build --output <对照目录>
# 再次新SSH确认，phase仍为skills；成功试跑原样复用。
python -m scripts.run_rule_skill_comparison prepare-eval --output <对照目录>
python -u -m scripts.run_rule_skill_comparison run-eval --output <对照目录>
# 新SSH会话：
python -m scripts.run_rule_skill_comparison confirm-detached --phase evaluation --output <对照目录>
python -m scripts.run_rule_skill_comparison score --output <对照目录>
```

每次启动前确认没有同目录客户端，完整环境和后台启动方式见项目运行约定。`sources.json`保存原题与30B原答，`skills/`保留逐题skill、来源复核判断、导航、未入树项及请求；`evaluation/metrics.json`给出P/R/F1、全50题状态、实际调用/token、知识读取和配对差值区间。未完成组不输出完整分数，技术失败不记为成功空结果。沿用新bench位置匹配代理，不称为语义正确率或人工金标。

`evaluation/case_review.json`按预先固定的哈希顺序，分别抽取规则组独有命中、skill组独有命中、共同漏检、命中相同但诊断数量不同的题，最多4个不同题；不存在的类别不补造。保存原题/原答、两组诊断、实际工具记录、已确认及未决参考，由主线Codex assistant补充物理解释和知识实际贡献，再整理为人类可读案例。自动抽样完成不等于案例复核完成。两组各50题均是开发对照，论文独立验证另留隔离测试集。
