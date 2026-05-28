#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:-${WORKSPACE}/runs/codersa_mbpp_llama3_limit378_seed42}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKSPACE}/.cache}"
export TMPDIR="${TMPDIR:-${WORKSPACE}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE}/.cache/triton}"

mkdir -p "${GORSA_ROOT_DIR}/logs" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

echo "MBPP notebook-style repro root: ${GORSA_ROOT_DIR}"
echo "Candidate/instruction backend: vLLM"
echo "L0 backend: transformers"

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
