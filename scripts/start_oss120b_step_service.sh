#!/usr/bin/env bash
set -euo pipefail
RUN=/slow_share2/huhengyuan/physicsverifier_results/qwen30b_step_verifier/oss_service
ENV=/home/visitor/huhengyuan/envs/physicsverifier
export CONDA_PREFIX="$ENV" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/visitor/huhengyuan/envs/vllm_gptoss120b/lib/python3.10/site-packages
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PV_HARMONY_DATE=2026-11-07
export TIKTOKEN_ENCODINGS_BASE=/home/visitor/huhengyuan/.cache/gptoss120b/harmony
export XDG_CACHE_HOME=/slow_share2/huhengyuan/cache/oss120b_steps
export TMPDIR="$XDG_CACHE_HOME/tmp" VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export NO_PROXY=127.0.0.1,localhost
mkdir -p "$RUN" "$TMPDIR" "$VLLM_CACHE_ROOT"
exec 9>"$RUN/active.lock"
flock -n 9 || exit 1
echo "$$" > "$RUN/launcher.pid"
trap 'echo "$?" > "$RUN/exit_code"; rm -f "$RUN/launcher.pid"' EXIT
cd /home/visitor/huhengyuan/PhysicsVerifier
"$ENV/bin/python" - <<'PY'
import json,os,socket,subprocess,time
from pathlib import Path
from scripts import run_continuous_context_v2 as runtime
from scripts import run_gpt_oss_selfcheck as r
root=Path('/slow_share/jinjianhan/models/gpt-oss-120b')
gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert gpu[0].strip()=='GPU-1f91ae4b-c681-68b6-385e-250e0acc1941' and int(gpu[1])<100
with socket.socket() as s:
 s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 s.bind(('127.0.0.1',18083))
assert {n:r.sha256_file(root/n) for n in runtime.MODEL_FILES}==runtime.MODEL_FILES
probe=Path('/slow_share2/huhengyuan/physicsverifier_results/qwen30b_corpus/service/runtime_probe_auditor/manifest.json')
prior=json.loads(probe.read_text())['auditor_weight_stat_snapshot']
for name,entry in prior.items():
 st=(root/name).stat();assert {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}==entry
Path('/slow_share2/huhengyuan/physicsverifier_results/qwen30b_step_verifier/oss_service/startup.json').write_text(json.dumps({
 'fixed_harmony_date':os.environ['PV_HARMONY_DATE'],'model_files':runtime.MODEL_FILES,'unix':time.time()}))
PY
"$ENV/bin/python" -u -m scripts.serve_fixed_date_vllm \
 --model /slow_share/jinjianhan/models/gpt-oss-120b --served-model-name /slow_share/jinjianhan/models/gpt-oss-120b \
 --host 127.0.0.1 --port 18083 --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
 --max-model-len 131072 --max-num-seqs 1 \
 --compilation-config '{"level":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
 --safetensors-load-strategy eager --seed 42 --generation-config vllm --disable-log-requests
