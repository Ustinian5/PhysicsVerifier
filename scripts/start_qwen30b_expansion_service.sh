#!/usr/bin/env bash
# New service logs/lock; reuse verified personal weights without editing old runs.
set -euo pipefail
RUN=${1:?Pass the prepared expansion directory}
GPU=${2:?Pass an available GPU index}
UUID=${3:?Pass its matching GPU UUID}
ENV=/home/visitor/huhengyuan/envs/physicsverifier
MODEL=/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507
SITE=/home/visitor/huhengyuan/runtime/vllm_gptoss120b/lib/python3.10/site-packages
export CONDA_PREFIX="$ENV" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SITE" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
export NO_PROXY=127.0.0.1,localhost
export TMPDIR=/home/visitor/huhengyuan/tmp/q30_expansion
export XDG_CACHE_HOME=/slow_share2/huhengyuan/cache/qwen30b_expansion
export HF_HOME="$XDG_CACHE_HOME/hf" VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
case "$RUN" in /slow_share2/huhengyuan/*|/slow_share/huhengyuan/*|/home/visitor/huhengyuan/*) ;; *) echo 'Use a personal output directory'; exit 1;; esac
test -f "$RUN/plan.json"
mkdir -p "$RUN/service" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
exec 9>"$RUN/service/active.lock"
flock -n 9 || { echo 'This expansion service already has a launcher'; exit 1; }
echo "$$" > "$RUN/service/launcher.pid"
trap 'code=$?; echo "$code" > "$RUN/service/exit_code"; rm -f "$RUN/service/launcher.pid"' EXIT
"$ENV/bin/python" - "$RUN" "$UUID" "$MODEL" <<'PY'
import hashlib,json,os,pathlib,shutil,socket,subprocess,sys
root=pathlib.Path(sys.argv[1]);expected_uuid=sys.argv[2];model=pathlib.Path(sys.argv[3])
config=json.loads((root/'selection.json').read_text())
prior=pathlib.Path(config['base_corpus'])/'service/checkpoint_manifest.json'
data=json.loads(prior.read_text())
assert data['complete'] and data['destination']==str(model)
assert shutil.disk_usage(root).free >= 2*1024**3,'Less than 2 GiB free for logs/results'
for name,entry in data['files'].items():
 p=model/name;st=p.stat()
 assert st.st_size==entry['bytes'] and st.st_mtime<=data['completed_unix'],'Verified weight changed'
 if not name.endswith('.safetensors'):
  assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],'Checkpoint metadata changed'
gpu=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
 '--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert gpu[0].strip()==expected_uuid and int(gpu[1])<=100 and int(gpu[2])>=80000,'GPU occupied/changed; no preemption'
with socket.socket() as sock:sock.bind(('127.0.0.1',18084))
out=root/'service/checkpoint_manifest.json'
if out.exists():assert json.loads(out.read_text())==data,'Checkpoint receipt drift'
else:out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('Verified prior full-SHA checkpoint boundary and current metadata; free GPU admitted.',flush=True)
PY
"$ENV/bin/python" -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --host 127.0.0.1 --port 18084 --dtype bfloat16 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 8 \
  --safetensors-load-strategy eager --seed 20261021 --generation-config vllm --disable-log-requests
