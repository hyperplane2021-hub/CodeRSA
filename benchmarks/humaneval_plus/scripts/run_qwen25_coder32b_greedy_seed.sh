#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: scripts/run_qwen25_coder32b_greedy_seed.sh SEED}"
root="${2:-/workspace/HumanEval+Qwen2p5-Coder-32B_vllm_os20_seed${seed}}"

cd /workspace

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export GORSA_MODEL_ID="${GORSA_MODEL_ID:-Qwen/Qwen2.5-Coder-32B-Instruct}"
export GORSA_ROOT_DIR="${root}"
export GORSA_SEED="${seed}"
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-8}"
export GORSA_EVAL_WORKERS="${GORSA_EVAL_WORKERS:-12}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-20}"
export GORSA_CANDIDATE_TEMPERATURE="${GORSA_CANDIDATE_TEMPERATURE:-1.0}"
export GORSA_CANDIDATE_TOP_P="${GORSA_CANDIDATE_TOP_P:-1.0}"
export GORSA_VLLM_PER_TASK_SEED="${GORSA_VLLM_PER_TASK_SEED:-1}"
export GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE="${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE:-0.0}"
export GORSA_ADDITIONAL_INSTRUCTION_TOP_P="${GORSA_ADDITIONAL_INSTRUCTION_TOP_P:-1.0}"
export GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS="${GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS:-48}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

mkdir -p "${root}/logs" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

echo "seed=${seed}"
echo "root=${root}"
echo "model=${GORSA_MODEL_ID}"
echo "candidate oversample=${GORSA_CANDIDATE_OVERSAMPLE}"
echo "candidate temperature=${GORSA_CANDIDATE_TEMPERATURE}"
echo "instruction temperature=${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE}"

python scripts/01_init_tasks.py

echo "===== RUNNING Stage 1 candidates vLLM TP=2 ====="
CUDA_VISIBLE_DEVICES=0,1 /workspace/.venvs/vllm/bin/python scripts/02_generate_candidates_vllm.py \
  --tensor-parallel-size 2 \
  --task-batch-size "${VLLM_TASK_BATCH_SIZE:-16}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}"

python - <<'PY'
from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import pad_candidate_pool
pad_candidate_pool(prepare_config())
PY

echo "===== RUNNING Stage 2 candidate eval ====="
python scripts/03_evaluate_candidates_parallel.py

echo "===== Candidate eval stats ====="
python - <<'PY'
from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import print_candidate_eval_stats
print_candidate_eval_stats(prepare_config())
PY

echo "===== RUNNING Stage 3 baseline scoring ====="
python scripts/04_score_baselines.py

echo "===== RUNNING Stage 4 instructions vLLM TP=2 ====="
CUDA_VISIBLE_DEVICES=0,1 /workspace/.venvs/vllm/bin/python scripts/05_generate_instructions_vllm.py \
  --tensor-parallel-size 2 \
  --task-batch-size "${VLLM_TASK_BATCH_SIZE:-16}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}"

echo "===== RUNNING Stage 6 L0 scoring ====="
python scripts/06_compute_l0.py

echo "===== RUNNING Stage 7/8 reports ====="
python scripts/07_pairwise_results.py
python scripts/08_report.py
