#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export GORSA_SWEEP_ROOT_TEMPLATE="${GORSA_SWEEP_ROOT_TEMPLATE:-/workspace/HumanEval+Llama3-70B-Instruct_vllm_os20_temp12_greedyinst_seed{seed}}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

if [[ $# -eq 0 ]]; then
  seeds=(42 43 44 45 46)
else
  seeds=("$@")
fi

python scripts/monitor_seed_sweep.py --seeds "${seeds[@]}" --watch --interval "${GORSA_MONITOR_INTERVAL:-10}"
