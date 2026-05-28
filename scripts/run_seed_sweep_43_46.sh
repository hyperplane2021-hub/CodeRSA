#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
cd "${REPO_ROOT}"

seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
  seeds=(43 44 45 46)
fi

export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-20}"
export GORSA_EVAL_TIMEOUT_SECONDS="${GORSA_EVAL_TIMEOUT_SECONDS:-2}"
export GORSA_EVAL_TASK_TIMEOUT_SECONDS="${GORSA_EVAL_TASK_TIMEOUT_SECONDS:-30}"
export GORSA_EVAL_WORKERS="${GORSA_EVAL_WORKERS:-4}"
export VLLM_TASK_BATCH_SIZE="${VLLM_TASK_BATCH_SIZE:-8}"
export VLLM_L0_PROMPT_BATCH_SIZE="${VLLM_L0_PROMPT_BATCH_SIZE:-16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

roots=()
for seed in "${seeds[@]}"; do
  export GORSA_SEED="${seed}"
  export GORSA_ROOT_DIR="${WORKSPACE}/runs/codersa_mbpp_llama3_limit378_seed${seed}"
  roots+=("${GORSA_ROOT_DIR}")
  echo "========== SEED ${seed} =========="
  echo "root=${GORSA_ROOT_DIR}"
  bash scripts/run_full_vllm_mbpp.sh
done

python scripts/summarize_seed_results.py \
  "${roots[@]}" \
  --out "${WORKSPACE}/runs/codersa_mbpp_llama3_seed_sweep_summary.csv"
