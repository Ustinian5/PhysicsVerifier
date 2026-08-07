#!/usr/bin/env bash
# Four experiments on the dual-chain test pool (error 100 + question 100):
#   B1 同模型基线: qwen3-30b-a3b-instruct-2507
#   B2 大规模 MoE 基线: BASELINE_MODEL_NEXT80（默认 qwen3-next-80b-a3b-instruct）
#   B3 更大规模 MoE 基线: BASELINE_MODEL_MO235（默认 qwen3-235b-a22b-instruct-2507）
#   B4 消融: 关闭符号核查（NO_SYMBOLIC_CHECK=1）
#
# 默认 **并行** 跑四个任务（PARALLEL=1）；串行请设置 PARALLEL=0。
#
# Usage:
#   bash scripts/legacy/run_four_qwen_baselines_and_no_symbolic_ablation.sh
#   PARALLEL=0 bash scripts/legacy/run_four_qwen_baselines_and_no_symbolic_ablation.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] PYTHON not executable: $PYTHON" >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain}"
ERROR_DS="${ERROR_DS:-$DATASET_DIR/error_eval_dataset_100.json}"
QUESTION_DS="${QUESTION_DS:-$DATASET_DIR/question_eval_dataset_50_50.json}"

for f in "$ERROR_DS" "$QUESTION_DS"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing dataset: $f" >&2
    exit 3
  fi
done

EMPTY_AUDIT="${EMPTY_AUDIT:-$ROOT_DIR/results/_empty_symbolic_audit.json}"
echo '[]' > "$EMPTY_AUDIT"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
PARALLEL="${PARALLEL:-1}"
MASTER_LOG="$ROOT_DIR/results/four_exp_qwen3_${STAMP}.log"

run_baseline_pair() {
  local tag="$1"
  local model="$2"
  local out="$ROOT_DIR/results/baseline_qwen3_${tag}_${STAMP}"
  mkdir -p "$out"
  echo "$model" > "$out/model.txt"
  echo "$tag" > "$out/tag.txt"
  {
    echo "=== Baseline [$tag] model=$model start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    "$PYTHON" scripts/run_llm_checker_baseline.py \
      --input "$ERROR_DS" --model "$model" \
      --out_json "$out/error_verifier_results.json"
    "$PYTHON" scripts/run_llm_checker_baseline.py \
      --input "$QUESTION_DS" --model "$model" \
      --out_json "$out/question_verifier_results.json"
    "$PYTHON" scripts/evaluate_physics_eval_sets.py \
      --dataset "$ERROR_DS" \
      --results "$out/error_verifier_results.json" \
      --audit "$EMPTY_AUDIT" \
      --output "$out/error_metrics.json" \
      --match-mode location
    "$PYTHON" scripts/evaluate_question_level_sets.py \
      --dataset "$QUESTION_DS" \
      --results "$out/question_verifier_results.json" \
      --audit "$EMPTY_AUDIT" \
      --output "$out/question_metrics.json"
    echo "=== Baseline [$tag] done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  } >> "$out/run.log" 2>&1
}

{
  echo "STAMP=$STAMP"
  echo "PARALLEL=$PARALLEL"
  echo "MASTER_LOG=$MASTER_LOG"
  echo "Datasets: ERROR=$ERROR_DS QUESTION=$QUESTION_DS"
} | tee "$MASTER_LOG"

run_ablation_fixed() {
  local AB_TAG="e2e_no_symbolic_ablation_${STAMP}"
  mkdir -p "$ROOT_DIR/results/$AB_TAG"
  {
    echo "=== Ablation NO_SYMBOLIC_CHECK=1 -> $AB_TAG start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    cd "$ROOT_DIR"
    NO_SYMBOLIC_CHECK=1 RUN_TAG="$AB_TAG" SKIP_BUILD=1 bash scripts/legacy/run_e2e_with_experience_symbolic.sh
    echo "=== Ablation done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  } >> "$ROOT_DIR/results/$AB_TAG/run.log" 2>&1
}

BASELINE_MODEL_NEXT80="${BASELINE_MODEL_NEXT80:-qwen3-next-80b-a3b-instruct}"
BASELINE_MODEL_MO235="${BASELINE_MODEL_MO235:-qwen3-235b-a22b-instruct-2507}"

if [[ "$PARALLEL" == "1" ]]; then
  echo "[parallel] launching 4 jobs..." | tee -a "$MASTER_LOG"
  run_baseline_pair "same" "qwen3-30b-a3b-instruct-2507" &
  p1=$!
  run_baseline_pair "next80" "$BASELINE_MODEL_NEXT80" &
  p2=$!
  run_baseline_pair "mo235" "$BASELINE_MODEL_MO235" &
  p3=$!
  run_ablation_fixed &
  p4=$!
  echo "PIDs baseline_same=$p1 next80=$p2 mo235=$p3 ablation=$p4" | tee -a "$MASTER_LOG"
  ec=0
  wait $p1 || ec=1
  wait $p2 || ec=1
  wait $p3 || ec=1
  wait $p4 || ec=1
  echo "[parallel] all finished ec=$ec" | tee -a "$MASTER_LOG"
  "$PYTHON" scripts/legacy/collect_four_exp_metrics.py --stamp "$STAMP" --write-md "$ROOT_DIR/results/four_exp_table_${STAMP}.md" || true
  exit "$ec"
else
  echo "[sequential] running..." | tee -a "$MASTER_LOG"
  run_baseline_pair "same" "qwen3-30b-a3b-instruct-2507"
  run_baseline_pair "next80" "$BASELINE_MODEL_NEXT80"
  run_baseline_pair "mo235" "$BASELINE_MODEL_MO235"
  run_ablation_fixed
  "$PYTHON" scripts/legacy/collect_four_exp_metrics.py --stamp "$STAMP" --write-md "$ROOT_DIR/results/four_exp_table_${STAMP}.md" || true
  echo "All four experiments finished." | tee -a "$MASTER_LOG"
fi
