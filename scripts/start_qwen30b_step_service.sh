#!/usr/bin/env bash
# Separate logs/lock; reuse verified personal weights without copying old runs.
set -euo pipefail
RUN=${STEP_RUN_ROOT:-/slow_share2/huhengyuan/physicsverifier_results/qwen30b_step_verifier}
export STEP_RUN_ROOT="$RUN"
ENV=/home/visitor/huhengyuan/envs/physicsverifier
export CONDA_PREFIX="$ENV" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/visitor/huhengyuan/envs/vllm_gptoss120b/lib/python3.10/site-packages
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${STEP_GPU_INDEX:?set free GPU index}"
export STEP_GPU_UUID="${STEP_GPU_UUID:?set expected GPU UUID}"
export XDG_CACHE_HOME="${STEP_CACHE_ROOT:-/slow_share2/huhengyuan/cache/qwen30b_steps}"
export TMPDIR="$XDG_CACHE_HOME/tmp" HF_HOME="$XDG_CACHE_HOME/huggingface"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export NO_PROXY=127.0.0.1,localhost
mkdir -p "$RUN/service" "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
exec 9>"$RUN/service/active.lock"
flock -n 9 || exit 1
echo "$$" > "$RUN/service/launcher.pid"
trap 'echo "$?" > "$RUN/service/exit_code"; rm -f "$RUN/service/launcher.pid"' EXIT
"$ENV/bin/python" - <<'PY'
import hashlib,json,os,pathlib,socket,subprocess,time
root=pathlib.Path(os.environ['STEP_RUN_ROOT'])/'service'
manifest=pathlib.Path('/slow_share2/huhengyuan/physicsverifier_results/qwen30b_corpus/service/checkpoint_manifest.json')
prior=json.loads(manifest.read_text())
assert prior['complete']
dest=pathlib.Path(prior['destination'])
assert str(dest)=='/slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507'
for name,entry in prior['files'].items():
    path=dest/name; st=path.stat()
    assert st.st_size==entry['bytes'] and st.st_mtime <= prior['completed_unix'], name
    if not name.endswith('.safetensors'):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],name
gpu=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
    '--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert gpu[0].strip()==os.environ['STEP_GPU_UUID'] and int(gpu[1])<100 and int(gpu[2])>=80000
with socket.socket() as s:s.bind(('127.0.0.1',18084))
(root/'admission.json').write_text(json.dumps({'prior_full_weight_sha_manifest':str(manifest),
 'prior_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'verified':'prior full SHA, unchanged size/mtime, fresh metadata SHA', 'gpu':gpu,'unix':time.time()},indent=2))
PY
"$ENV/bin/python" -u -m vllm.entrypoints.openai.api_server \
 --model /slow_share2/huhengyuan/models/Qwen3-30B-A3B-Instruct-2507 \
 --served-model-name Qwen3-30B-A3B-Instruct-2507 --host 127.0.0.1 --port 18084 \
 --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
 --max-model-len 32768 --max-num-seqs 1 --enforce-eager --safetensors-load-strategy eager \
 --seed 20261107 --generation-config vllm --disable-log-requests
