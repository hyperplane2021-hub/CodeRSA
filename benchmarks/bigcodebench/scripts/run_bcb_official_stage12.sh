#!/usr/bin/env bash
set -euo pipefail

seed="${1:-42}"
subset="${BCB_SUBSET:-hard}"
split="${BCB_SPLIT:-instruct}"
model="${GORSA_MODEL_ID:-meta-llama/Meta-Llama-3-70B-Instruct}"
root="${GORSA_ROOT_DIR:-/workspace/BigCodeBench-${subset}-${split}-Llama3-70B_os20_temp12_seed${seed}}"
results_dir="${BCB_RESULTS_DIR:-${root}/bcb_results}"

export GORSA_ROOT_DIR="${root}"
export GORSA_DATASET="bigcodebench"
export GORSA_SEED="${seed}"
export GORSA_MODEL_ID="${model}"
export GORSA_CANDIDATE_TEMPERATURE="${GORSA_CANDIDATE_TEMPERATURE:-1.2}"
export GORSA_CANDIDATE_TOP_P="${GORSA_CANDIDATE_TOP_P:-1.0}"
export BCB_TOP_P="${BCB_TOP_P:-${GORSA_CANDIDATE_TOP_P}}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-20}"
export BCB_KEEP_N="${BCB_KEEP_N:-10}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"
export PYTHONPATH="/workspace/bigcodebench:/workspace:${PYTHONPATH:-}"

mkdir -p "${root}/logs" "${results_dir}" /workspace/tmp /workspace/.cache/triton
if [ -f /workspace/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/.env
  set +a
fi

python_bin="${PYTHON_BIN:-/workspace/.venvs/vllm/bin/python}"
tp="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_N_GPUS="${VLLM_N_GPUS:-${tp}}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
bs="${BCB_BS:-4}"
max_model_len="${VLLM_MAX_MODEL_LEN:-8192}"
max_new_tokens="${BCB_MAX_NEW_TOKENS:-1280}"
execution="${BCB_EXECUTION:-gradio}"
parallel="${BCB_EVAL_PARALLEL:-8}"

echo "root=${root}"
echo "model=${model}"
echo "subset=${subset} split=${split}"
echo "generate_n=${GORSA_CANDIDATE_OVERSAMPLE} keep_n=${BCB_KEEP_N}"
echo "temperature=${GORSA_CANDIDATE_TEMPERATURE} top_p=${BCB_TOP_P}"

raw_samples="${results_dir}/raw_n${GORSA_CANDIDATE_OVERSAMPLE}.jsonl"
filtered_samples="${results_dir}/filtered_n${BCB_KEEP_N}.jsonl"
filter_report="${results_dir}/filter_report.jsonl"
eval_results="${filtered_samples%.jsonl}_eval_results.json"

echo "===== BigCodeBench official generate ====="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" "${python_bin}" -m bigcodebench.generate \
  --model "${model}" \
  --backend vllm \
  --split "${split}" \
  --subset "${subset}" \
  --root "${results_dir}" \
  --n_samples "${GORSA_CANDIDATE_OVERSAMPLE}" \
  --temperature "${GORSA_CANDIDATE_TEMPERATURE}" \
  --max_new_tokens "${max_new_tokens}" \
  --max_model_len "${max_model_len}" \
  --tp "${tp}" \
  --bs "${bs}"

generated="$(find "${results_dir}" -maxdepth 1 -type f -name '*.jsonl' ! -name 'filtered_n*.jsonl' ! -name 'raw_n*.jsonl' | sort -t/ -k2 | tail -1)"
cp "${generated}" "${raw_samples}"

echo "===== exact function-body de-dup ${GORSA_CANDIDATE_OVERSAMPLE} -> ${BCB_KEEP_N} ====="
"${python_bin}" scripts/filter_bcb_exact_duplicates.py \
  --input "${raw_samples}" \
  --output "${filtered_samples}" \
  --report "${filter_report}" \
  --keep-n "${BCB_KEEP_N}"

echo "===== BigCodeBench official evaluate ====="
"${python_bin}" -m bigcodebench.evaluate \
  --split "${split}" \
  --subset "${subset}" \
  --samples "${filtered_samples}" \
  --execution "${execution}" \
  --parallel "${parallel}"

echo "===== adapt official artifacts to GoRSA tasks ====="
"${python_bin}" scripts/bcb_official_to_gorsa_tasks.py \
  --root "${root}" \
  --samples "${filtered_samples}" \
  --eval-results "${eval_results}" \
  --subset "${subset}" \
  --split "${split}"

echo "===== DONE BigCodeBench official stage1/2 ====="
echo "root=${root}"
echo "samples=${filtered_samples}"
echo "eval_results=${eval_results}"
