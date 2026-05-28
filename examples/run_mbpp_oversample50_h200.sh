#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:-${WORKSPACE}/codersa_mbpp_llama3_limit974_seed42_oversample50_2gpu_repro}"
export GORSA_MODEL_ID="${GORSA_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
export GORSA_SEED="${GORSA_SEED:-42}"
export GORSA_LIMIT="${GORSA_LIMIT:-974}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-50}"
export GORSA_EVAL_TIMEOUT_SECONDS="${GORSA_EVAL_TIMEOUT_SECONDS:-2}"
export GORSA_EVAL_WORKERS="${GORSA_EVAL_WORKERS:-4}"
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-64}"
export GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE="${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE:-0.0}"
export GORSA_ADDITIONAL_INSTRUCTION_TOP_P="${GORSA_ADDITIONAL_INSTRUCTION_TOP_P:-1.0}"
export VLLM_TASK_BATCH_SIZE="${VLLM_TASK_BATCH_SIZE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "${GORSA_ROOT_DIR}/logs"

echo "root: ${GORSA_ROOT_DIR}"
echo "seed: ${GORSA_SEED}"
echo "limit: ${GORSA_LIMIT}"
echo "candidate oversample: ${GORSA_CANDIDATE_OVERSAMPLE}"
echo "candidate/instruction backend: vLLM"
echo "L0 backend: transformers sharded data parallel"
echo "pairwise writeback: staged"

for stage in \
  scripts/01_init_tasks_staged.py \
  scripts/02_generate_candidates_vllm.py \
  scripts/03_evaluate_candidates_parallel.py \
  scripts/04_score_baselines.py \
  scripts/05_generate_instructions_vllm.py
do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done

echo "===== RUNNING Stage 6 sharded transformers L0 ====="
examples/run_l0_transformers_2gpu_sharded.sh
echo "===== DONE Stage 6 sharded transformers L0 ====="

echo "===== RUNNING Stage 7 staged pairwise writeback ====="
scripts/run_stage_logged.sh scripts/07_pairwise_writeback_staged.py
echo "===== DONE Stage 7 staged pairwise writeback ====="

echo "===== RUNNING Stage 8 report ====="
scripts/run_stage_logged.sh scripts/08_report.py
echo "===== DONE Stage 8 report ====="
