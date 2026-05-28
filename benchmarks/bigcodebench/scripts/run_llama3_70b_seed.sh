#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: scripts/run_llama3_70b_seed.sh SEED [ROOT_DIR]}"
root="${2:-/workspace/HumanEval+Llama3-70B-Instruct_vllm_os20_temp1_seed${seed}}"

cd /workspace

export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/workspace/.venvs/vllm/bin/python}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export GORSA_MODEL_ID="${GORSA_MODEL_ID:-meta-llama/Meta-Llama-3-70B-Instruct}"
export GORSA_ROOT_DIR="${root}"
export GORSA_SEED="${seed}"
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-4}"
export GORSA_EVAL_WORKERS="${GORSA_EVAL_WORKERS:-12}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-20}"
export GORSA_CANDIDATE_TEMPERATURE="${GORSA_CANDIDATE_TEMPERATURE:-1.2}"
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

export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_TASK_BATCH_SIZE="${VLLM_TASK_BATCH_SIZE:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_L0_PROMPT_BATCH_SIZE="${VLLM_L0_PROMPT_BATCH_SIZE:-4}"
export VLLM_L0_GPU_MEMORY_UTILIZATION="${VLLM_L0_GPU_MEMORY_UTILIZATION:-0.82}"

if [[ -z "${HF_TOKEN:-}" && -f /workspace/.env ]]; then
  hf_token_line="$(grep -E '^[[:space:]]*HF_TOKEN=' /workspace/.env | tail -n 1 || true)"
  if [[ -n "${hf_token_line}" ]]; then
    hf_token="${hf_token_line#*=}"
    hf_token="${hf_token#"${hf_token%%[![:space:]]*}"}"
    hf_token="${hf_token%"${hf_token##*[![:space:]]}"}"
    hf_token="${hf_token%\"}"
    hf_token="${hf_token#\"}"
    hf_token="${hf_token%\'}"
    hf_token="${hf_token#\'}"
    export HF_TOKEN="${hf_token}"
  fi
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

mkdir -p "${root}/logs" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

echo "seed=${seed}"
echo "root=${root}"
echo "model=${GORSA_MODEL_ID}"
echo "vllm tensor parallel=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "candidate oversample=${GORSA_CANDIDATE_OVERSAMPLE}"
echo "candidate temperature=${GORSA_CANDIDATE_TEMPERATURE}"
echo "instruction temperature=${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE}"

scripts/run_stage_logged.sh scripts/01_init_tasks.py

echo "===== RUNNING Stage 1 candidates vLLM TP=${VLLM_TENSOR_PARALLEL_SIZE} ====="
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${GORSA_ROOT_DIR}/logs/${stamp}_02_generate_candidates_vllm_tp${VLLM_TENSOR_PARALLEL_SIZE}.log"
echo "logging to: ${log}"
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/.venvs/vllm/bin/python scripts/02_generate_candidates_vllm.py \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --task-batch-size "${VLLM_TASK_BATCH_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" 2>&1 | tee "${log}"

"${PYTHON_BIN}" - <<'PY'
from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import pad_candidate_pool
pad_candidate_pool(prepare_config())
PY

for stage in scripts/03_evaluate_candidates_parallel.py scripts/04_score_baselines.py; do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done

echo "===== Candidate eval stats ====="
"${PYTHON_BIN}" - <<'PY'
from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import print_candidate_eval_stats
print_candidate_eval_stats(prepare_config())
PY

echo "===== RUNNING Stage 4 instructions vLLM TP=${VLLM_TENSOR_PARALLEL_SIZE} ====="
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${GORSA_ROOT_DIR}/logs/${stamp}_05_generate_instructions_vllm_tp${VLLM_TENSOR_PARALLEL_SIZE}.log"
echo "logging to: ${log}"
CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/.venvs/vllm/bin/python scripts/05_generate_instructions_vllm.py \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --task-batch-size "${VLLM_TASK_BATCH_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" 2>&1 | tee "${log}"

echo "===== RUNNING Stage 6 L0 vLLM TP=${VLLM_TENSOR_PARALLEL_SIZE} ====="
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${GORSA_ROOT_DIR}/logs/${stamp}_06_compute_l0_vllm_tp${VLLM_TENSOR_PARALLEL_SIZE}_eager_bs${VLLM_L0_PROMPT_BATCH_SIZE}.log"
echo "logging to: ${log}"
VLLM_ENFORCE_EAGER=1 CUDA_VISIBLE_DEVICES=0,1,2,3 /workspace/.venvs/vllm/bin/python scripts/06_compute_l0_vllm.py \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --prompt-batch-size "${VLLM_L0_PROMPT_BATCH_SIZE}" \
  --gpu-memory-utilization "${VLLM_L0_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" 2>&1 | tee "${log}"

for stage in scripts/07_pairwise_results.py scripts/08_report.py; do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done
