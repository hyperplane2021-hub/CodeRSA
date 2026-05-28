#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:-${WORKSPACE}/codersa_mbpp_llama3_limit378_seed42}"
export GORSA_LIMIT="${GORSA_LIMIT:-378}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-50}"
export GORSA_EVAL_TIMEOUT_SECONDS="${GORSA_EVAL_TIMEOUT_SECONDS:-2}"
export GORSA_EVAL_WORKERS="${GORSA_EVAL_WORKERS:-4}"

export VLLM_TASK_BATCH_SIZE="${VLLM_TASK_BATCH_SIZE:-8}"
export VLLM_L0_PROMPT_BATCH_SIZE="${VLLM_L0_PROMPT_BATCH_SIZE:-16}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

export HF_HOME="${HF_HOME:-${WORKSPACE}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${WORKSPACE}/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE}/hf_home/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKSPACE}/.cache}"
export TMPDIR="${TMPDIR:-${WORKSPACE}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE}/.cache/triton}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${GORSA_ROOT_DIR}/logs" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

echo "root: ${GORSA_ROOT_DIR}"
echo "limit: ${GORSA_LIMIT}"
echo "candidate oversample: ${GORSA_CANDIDATE_OVERSAMPLE}"
echo "eval workers: ${GORSA_EVAL_WORKERS}"
echo "vLLM task batch size: ${VLLM_TASK_BATCH_SIZE}"
echo "L0 backend: transformers"
echo "score batch size: ${GORSA_SCORE_BATCH_SIZE:-8}"

for stage in \
  scripts/01_init_tasks.py \
  scripts/02_generate_candidates_vllm.py \
  scripts/03_evaluate_candidates_parallel.py \
  scripts/04_score_baselines.py \
  scripts/05_generate_instructions_vllm.py \
  scripts/06_compute_l0.py \
  scripts/07_pairwise_results_mbpp.py \
  scripts/08_report.py
do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done
