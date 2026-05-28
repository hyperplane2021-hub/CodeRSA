#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: run_bcb_temp10_seed.sh SEED}"
subset="${BCB_SUBSET:-hard}"
split="${BCB_SPLIT:-instruct}"
model="${GORSA_MODEL_ID:-meta-llama/Meta-Llama-3-70B-Instruct}"
temp="${GORSA_CANDIDATE_TEMPERATURE:-1.0}"
temp_tag="${GORSA_TEMP_TAG:-temp10}"
root="${GORSA_ROOT_DIR:-/workspace/BigCodeBench-${subset}-${split}-Llama3-70B_os20_${temp_tag}_seed${seed}}"
results_dir="${root}/bcb_results"

mkdir -p "${root}/logs" "${results_dir}" /workspace/logs /workspace/tmp /workspace/.cache/triton

export GORSA_ROOT_DIR="${root}"
export GORSA_DATASET=bigcodebench
export GORSA_SEED="${seed}"
export GORSA_MODEL_ID="${model}"
export GORSA_CANDIDATE_TEMPERATURE="${temp}"
export GORSA_CANDIDATE_TOP_P="${GORSA_CANDIDATE_TOP_P:-1.0}"
export BCB_TOP_P="${BCB_TOP_P:-${GORSA_CANDIDATE_TOP_P}}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-20}"
export BCB_KEEP_N="${BCB_KEEP_N:-10}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export PYTHONPATH="/workspace/bigcodebench:/workspace:${PYTHONPATH:-}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_N_GPUS="${VLLM_N_GPUS:-${VLLM_TENSOR_PARALLEL_SIZE}}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_SCORE_PROMPT_BATCH_SIZE="${VLLM_SCORE_PROMPT_BATCH_SIZE:-32}"
export VLLM_TASK_BATCH_SIZE="${VLLM_TASK_BATCH_SIZE:-16}"
export VLLM_L0_PROMPT_BATCH_SIZE="${VLLM_L0_PROMPT_BATCH_SIZE:-32}"
export GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE="${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE:-0.0}"
export GORSA_ADDITIONAL_INSTRUCTION_TOP_P="${GORSA_ADDITIONAL_INSTRUCTION_TOP_P:-1.0}"
export GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS="${GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS:-128}"

python_bin="${PYTHON_BIN:-/workspace/.venvs/vllm/bin/python}"
raw_samples="${results_dir}/raw_n${GORSA_CANDIDATE_OVERSAMPLE}.jsonl"
filtered_samples="${results_dir}/filtered_n${BCB_KEEP_N}.jsonl"
filter_report="${results_dir}/filter_report.jsonl"
eval_results="${results_dir}/filtered_n${BCB_KEEP_N}_eval_results.json"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed${seed} temp${temp} full BigCodeBench+GoRSA start"
echo "root=${root}"
echo "temperature=${GORSA_CANDIDATE_TEMPERATURE} oversample=${GORSA_CANDIDATE_OVERSAMPLE} keep=${BCB_KEEP_N}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 1 official generate"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "${python_bin}" -m bigcodebench.generate \
  --model "${model}" \
  --backend vllm \
  --split "${split}" \
  --subset "${subset}" \
  --root "${results_dir}" \
  --n_samples "${GORSA_CANDIDATE_OVERSAMPLE}" \
  --temperature "${GORSA_CANDIDATE_TEMPERATURE}" \
  --max_new_tokens "${BCB_MAX_NEW_TOKENS:-1280}" \
  --max_model_len "${BCB_MAX_MODEL_LEN:-8192}" \
  --tp "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --bs "${BCB_BS:-4}"

generated="$(find "${results_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name 'filtered_n*.jsonl' ! -name 'raw_n*.jsonl' | sort -t/ -k2 | tail -1)"
cp "${generated}" "${raw_samples}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 2 exact de-dup/filter"
"${python_bin}" scripts/filter_bcb_exact_duplicates.py \
  --input "${raw_samples}" \
  --output "${filtered_samples}" \
  --report "${filter_report}" \
  --keep-n "${BCB_KEEP_N}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 2 local eval chunk pool"
/workspace/.venvs/bcb_eval_py310/bin/python scripts/eval_bcb_local_chunk_pool.py \
  --samples "${filtered_samples}" \
  --output "${eval_results}" \
  --work-dir "${results_dir}/local_eval_chunks_j4_timeout20" \
  --subset "${subset}" \
  --split "${split}" \
  --chunk-tasks 1 \
  --jobs "${BCB_LOCAL_EVAL_JOBS:-4}" \
  --keep-n "${BCB_KEEP_N}" \
  --timeout "${BCB_LOCAL_EVAL_TIMEOUT:-20}" \
  --resume

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 2 adapt to GoRSA tasks"
"${python_bin}" scripts/bcb_official_to_gorsa_tasks.py \
  --root "${root}" \
  --samples "${filtered_samples}" \
  --eval-results "${eval_results}" \
  --subset "${subset}" \
  --split "${split}"

export VLLM_MAX_MODEL_LEN="${VLLM_GORSA_MAX_MODEL_LEN:-2048}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 3 score baselines"
"${python_bin}" scripts/04_score_baselines_vllm.py

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 4 generate detailed instructions"
"${python_bin}" scripts/05_generate_instructions_vllm.py

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 6 compute L0"
"${python_bin}" scripts/06_compute_l0_vllm.py

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 7 pairwise results"
"${python_bin}" scripts/07_pairwise_results.py

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage 8 report"
"${python_bin}" scripts/08_report.py

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed${seed} temp${temp} full complete"
