#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: run_bcb_full1140_seed.sh SEED}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
BIGCODEBENCH_REPO="${BIGCODEBENCH_REPO:-${WORKSPACE}/src/bigcodebench}"
BCB_EVAL_PYTHON="${BCB_EVAL_PYTHON:-${WORKSPACE}/.venvs/bcb_eval_py310/bin/python}"

set -a
if [[ -f "${WORKSPACE}/.env" ]]; then
  source "${WORKSPACE}/.env"
fi
set +a

export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKSPACE}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE}/.cache/pip}"
export TMPDIR="${TMPDIR:-${WORKSPACE}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE}/.cache/triton}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"

export PYTHONPATH="${REPO_ROOT}/benchmarks/bigcodebench:${BIGCODEBENCH_REPO}:${PYTHONPATH:-}"
export BIGCODEBENCH_REPO

export BCB_SEED="$seed"
export GORSA_SEED="$seed"
export GORSA_LIMIT="${GORSA_LIMIT:-1140}"
model_id="${BCB_MODEL_ID:-${GORSA_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}}"
model_revision="${BCB_MODEL_REVISION:-main}"
model_file_slug="${model_id//\//--}--${model_revision}"
run_slug="${BCB_RUN_SLUG:-llama3_8b}"
export GORSA_MODEL_ID="$model_id"
export GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE="${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE:-0.0}"
export GORSA_ADDITIONAL_INSTRUCTION_TOP_P="${GORSA_ADDITIONAL_INSTRUCTION_TOP_P:-1.0}"
export GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS="${GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS:-80}"
export BCB_ASYNC_SANITIZE="${BCB_ASYNC_SANITIZE:-1}"
export BCB_SUPPRESS_SANITIZE_WARNINGS="${BCB_SUPPRESS_SANITIZE_WARNINGS:-1}"
export BCB_TOP_P="${BCB_TOP_P:-1.0}"

repo="${REPO_ROOT}/benchmarks/bigcodebench"
run_root="${WORKSPACE}/runs/bcb_full_instruct_${run_slug}_temp12_full1140_n10_tok768_seed${seed}"
results_dir="$run_root/bcb_results"
gorsa_root="$run_root/gorsa"
log_dir="$run_root/logs"
mkdir -p "$results_dir" "$gorsa_root" "$log_dir" "$TMPDIR"
export GORSA_ROOT_DIR="$gorsa_root"
start_stage="${START_STAGE:-1}"

sample_jsonl="$results_dir/${model_file_slug}--bigcodebench-instruct--vllm-1.2-10-sanitized_calibrated.jsonl"
eval_json="$results_dir/${model_file_slug}--bigcodebench-instruct--vllm-1.2-10-sanitized_calibrated_eval_results.json"

cd "$repo"

run_stage() {
  local name="$1"
  shift
  local stage_no="${name%%_*}"
  stage_no=$((10#$stage_no))
  if (( stage_no < start_stage )); then
    echo "[$(date -Is)] seed=${seed} skip  ${name}" | tee -a "$log_dir/00_controller.log"
    return
  fi
  echo "[$(date -Is)] seed=${seed} start ${name}" | tee -a "$log_dir/00_controller.log"
  "$@" >"$log_dir/${name}.log" 2>&1
  echo "[$(date -Is)] seed=${seed} done  ${name}" | tee -a "$log_dir/00_controller.log"
}

run_stage 01_generate \
  python -m bigcodebench.generate "$model_id" instruct full \
    --backend vllm \
    --root "$results_dir" \
    --n_samples 10 \
    --temperature 1.2 \
    --max_new_tokens 768 \
    --max_model_len 4096 \
    --tp 1 \
    --bs 24

run_stage 02_eval \
  "$BCB_EVAL_PYTHON" scripts/eval_bcb_local_chunk_pool.py \
    --samples "$sample_jsonl" \
    --output "$eval_json" \
    --work-dir "$results_dir/local_eval_chunks_j16_timeout20" \
    --subset full \
    --split instruct \
    --chunk-tasks 1 \
    --jobs 16 \
    --keep-n 10 \
    --timeout 20 \
    --chunk-timeout 120 \
    --resume

run_stage 03_bcb_to_gorsa \
  python scripts/bcb_official_to_gorsa_tasks.py \
    --root "$gorsa_root" \
    --samples "$sample_jsonl" \
    --eval-results "$eval_json" \
    --subset full \
    --split instruct \
    --limit "$GORSA_LIMIT"

run_stage 04_score_baselines_vllm \
  python scripts/04_score_baselines_vllm.py \
    --gpu-memory-utilization "${VLLM_SCORE_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}" \
    --prompt-batch-size 8 \
    --max-model-len "$VLLM_MAX_MODEL_LEN"

run_stage 05_generate_instructions_vllm \
  python scripts/05_generate_instructions_vllm.py \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --task-batch-size 24 \
    --max-model-len "$VLLM_MAX_MODEL_LEN"

run_stage 06_compute_l0_vllm_bs32 \
  python scripts/06_compute_l0_vllm.py \
    --gpu-memory-utilization "${VLLM_L0_GPU_MEMORY_UTILIZATION:-$VLLM_GPU_MEMORY_UTILIZATION}" \
    --prompt-batch-size "${VLLM_L0_PROMPT_BATCH_SIZE:-32}" \
    --max-model-len "$VLLM_MAX_MODEL_LEN"

run_stage 07_pairwise_results \
  python scripts/07_pairwise_results.py

run_stage 08_report \
  python scripts/08_report.py

echo "[$(date -Is)] seed=${seed} all done" | tee -a "$log_dir/00_controller.log"
