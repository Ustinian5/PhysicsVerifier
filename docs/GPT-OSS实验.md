# GPT-OSS-120B 三组对照

## 实验与结论

固定同一冻结50题与1,123叶规则，GPT-OSS-120B（vLLM 0.11.2）分别直接检查、只提供叶规则检索自检、提供规则树和叶规则检索自检；每组50题，全程不接Qwen。正式批次`formal_vllm_008`已150/150完成、0失败。经模型、输入、提示、工具、预算及哈希核验，续跑复用126条已保存响应且未重发；续跑新增24次模型请求。最终保留150条请求/响应记录，合计625,749总token、214,123 completion token（含推理）。

| 条件 | P | R | F1 |
|---|---:|---:|---:|
| 纯模型 | 36.73% | 19.25% | 25.26% |
| 叶规则自检 | 40.00% | 16.04% | 22.90% |
| 树＋叶规则自检 | 37.18% | 15.51% | 21.89% |
| 纯Terra参考 | 39.57% | 39.57% | 39.57% |

叶、树自检相对GPT-OSS纯模型的F1差分别为−2.36和−3.38个百分点，配对95%区间分别为[−8.30,+3.30]和[−9.63,+2.38]；树减叶为−1.11pp，区间[−8.20,+5.41]。区间均跨零，未确认任一自检方案带来收益或树结构增益。纯模型相对纯Terra低14.31pp，95%区间[−20.02,−8.60]。

**9.29主线复核补充：规则工具实际使用为零。**150项每题均只有1次请求；叶组50题均没有检索或读取叶规则，树组50题均只有程序预载的树根，没有主动遍历或读叶节点。例：`cl_257_181380`叶组直接输出空诊断，树组直接给出诊断，均无后续工具调用。因此本轮结论仅适用于“可选工具说明/树根＋直接检查”，不能归因于规则内容，也不能据此否定规则库增益。定向接口验收通过不代表自主调用发生。证据：`formal_vllm_008/review_summary.json`；450个请求/响应/SSE哈希已核对。

位置计分另有局限：187条旧主张只有130条可定位；P的预测分母为可定位诊断98/75/78，而全部诊断为148/125/116，R仍以187为分母。必须保留这一口径，不能把这些数称为人工物理准确率。

分数按50题、187条未独立复核旧标签的位置重合计算，只是代理P/R/F1，不代表物理正确率；尚无独立人工物理裁定。旧`formal_adapter_001`全零批次存在输入/协议故障，不纳入比较。本次三组原始输出和迁移链均保留；受限JSON边界修复不改变诊断文字。计分器的纯Terra参考字段路径错误已修复，离线复算通过，未增加模型调用。

## 复核入口

服务器结果：`results/gpt_oss_120b_baseline_20260928/formal_vllm_008/`。其中`manifest.json`冻结配置；各组`results.json`及`response_evidence/`保存逐题结果和请求响应；`response_migration.json`记录126条续用来源；`metrics_automatic_proxy_unreviewed.json`保存分数、配对bootstrap及数据哈希。正式运行后已停止专用服务，GPU7/18083已释放。

复现实验入口为`scripts/run_gpt_oss_selfcheck.py`，所有Python命令使用操作者的隔离conda环境；具体模型身份、冻结哈希、每题预算及服务参数以正式manifest为准。计分可通过`--score-only`离线执行，不会调用模型。

## 运行与交接

这是固定50题协议的复现入口，不是任意模型/数据的通用评测器。`--model-path`仍要求历史权重路径 `/slow_share/jinjianhan/models/gpt-oss-120b`；输入、规则和模型身份必须通过冻结校验。该路径不可读时须先做显式移植和新manifest，不能换模型冒充同一运行。原始证据含绝对路径，异机计分须保留可读路径及哈希；仅复制指标摘要不够。数据通过授权交接获得，不在Git提交原始题目、GT或响应。

vLLM启动脚本要求显式指定个人目录、空闲GPU和个人conda解释器：`GPTOSS_PERSONAL_ROOT`、`GPTOSS_GPU`、`GPTOSS_PYTHON`；可选 `GPTOSS_SHARED_SITE_PACKAGES` 只读复用兼容运行库，`GPTOSS_PORT`默认18083。启动前检查磁盘和GPU；脚本拒绝明显占用的GPU，不停止任何已有进程。个人缓存 `harmony/o200k_base.tiktoken` 必须准备好。实际使用vLLM 0.11.2、Transformers 4.57.1、Torch 2.9.1，单卡、TRITON_ATTN、131072上下文，具体以manifest为准。

在隔离conda环境中按顺序执行（尖括号为需替换参数）：

```text
bash scripts/start_gpt_oss120b_vllm.sh
python scripts/check_gpt_oss_selfcheck_health.py --backend vllm --catalog catalogs/rules_unified_3000.json --api-base http://127.0.0.1:18083/v1 --server-adapter-sha256 <服务实现SHA256> --output <新验收目录>
python scripts/run_gpt_oss_selfcheck.py --backend vllm --input <冻结目录>/inputs.json --catalog catalogs/rules_unified_3000.json --api-base http://127.0.0.1:18083/v1 --server-adapter-sha256 <服务实现SHA256> --adapter-launcher-sha256 <启动脚本SHA256> --health-check <验收目录> --output <新实验目录> --prepare-only
```

服务在单独终端运行；验收通过且准备清单确认后，移除`--prepare-only`才采集。正式调用失败默认暂停；`continue_gpt_oss_selfcheck_after_reviewed_failure.py`用于核实失败后继续未处理样本，保留已成功和失败结果，不补采失败挑结果。已完成批次仅使用同一runner参数加`--score-only`离线计分；它仍需要冻结50题的参考根目录和纯Terra对照材料，不启动模型。

`start_gpt_oss120b_adapter.py`与`gpt_oss120b_warmup_and_exec.py`仅保留为旧Transformers故障/修复实验的代码依据，不用于当前vLLM批次。JSON受限格式修复保留原响应和修复记录；不改写诊断文字，越界或截断不能算成功。共享脚本纠正了bootstrap说明中的seed为20260928，与实际计算一致；未改写已冻结的历史指标文件。

离线验证（0次模型调用）：

```bash
python -m unittest tests.test_gpt_oss_input_boundaries tests.test_gpt_oss_selfcheck_protocol
```

工程正反例验收不检查模型是否主动读规则；下一轮必须另加实际规则使用门槛，不能用本轮验收代替。
