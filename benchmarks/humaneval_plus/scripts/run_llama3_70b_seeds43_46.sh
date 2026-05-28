#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

mkdir -p /workspace/logs "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

seeds=("$@")
if [[ ${#seeds[@]} -eq 0 ]]; then
  seeds=(43 44 45 46)
fi

for seed in "${seeds[@]}"; do
  root="/workspace/HumanEval+Llama3-70B-Instruct_vllm_os20_temp12_greedyinst_seed${seed}"
  echo "===== START seed ${seed} root=${root} ====="
  bash scripts/run_llama3_70b_seed.sh "${seed}" "${root}"
  echo "===== DONE seed ${seed} ====="
done

roots=(/workspace/HumanEval+Llama3-70B-Instruct_vllm_os20_temp12_greedyinst_seed{42..46})
existing_roots=()
for root in "${roots[@]}"; do
  if [[ -f "${root}/baseline_pairwise_avg.csv" ]]; then
    existing_roots+=("${root}")
  fi
done

if [[ ${#existing_roots[@]} -gt 0 ]]; then
  /workspace/.venvs/vllm/bin/python scripts/summarize_seed_results.py \
    "${existing_roots[@]}" \
    --out /workspace/HumanEval+Llama3-70B-Instruct_vllm_os20_temp12_greedyinst_seed_summary.csv
fi
