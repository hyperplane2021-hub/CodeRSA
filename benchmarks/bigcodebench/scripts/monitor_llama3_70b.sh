#!/usr/bin/env bash
set -euo pipefail

seed="${1:-42}"
root="${2:-/workspace/HumanEval+Llama3-70B-Instruct_vllm_os20_temp1_seed${seed}}"

cd /workspace

export GORSA_ROOT_DIR="${root}"
export PYTHON_BIN="${PYTHON_BIN:-/workspace/.venvs/vllm/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

"${PYTHON_BIN}" scripts/monitor_progress.py --watch --interval "${GORSA_MONITOR_INTERVAL:-10}"
