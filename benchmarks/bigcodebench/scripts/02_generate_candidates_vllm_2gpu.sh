#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

root_dir="${GORSA_ROOT_DIR:-/workspace/HumanEval+Llama3}"
mkdir -p "${root_dir}/logs" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log0="${root_dir}/logs/${stamp}_02_generate_candidates_vllm_gpu0.log"
log1="${root_dir}/logs/${stamp}_02_generate_candidates_vllm_gpu1.log"

echo "GPU 0 vLLM shard log: ${log0}"
echo "GPU 1 vLLM shard log: ${log1}"

CUDA_VISIBLE_DEVICES=0 /workspace/.venvs/vllm/bin/python scripts/02_generate_candidates_vllm.py --shard-index 0 --shard-count 2 >"${log0}" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 /workspace/.venvs/vllm/bin/python scripts/02_generate_candidates_vllm.py --shard-index 1 --shard-count 2 >"${log1}" 2>&1 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

python - <<'PY'
from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import pad_candidate_pool

pad_candidate_pool(prepare_config())
PY
