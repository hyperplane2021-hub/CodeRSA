#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: scripts/run_vllm_os20_temp1_seed.sh SEED}"
root="${2:-/workspace/HumanEval+Llama3_vllm_os20_temp1_seed${seed}}"

cd /workspace

export GORSA_SEED="${seed}"
export GORSA_ROOT_DIR="${root}"
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-32}"
export GORSA_EVAL_WORKERS="${GORSA_EVAL_WORKERS:-12}"
export GORSA_CANDIDATE_OVERSAMPLE="${GORSA_CANDIDATE_OVERSAMPLE:-20}"
export GORSA_CANDIDATE_TEMPERATURE="${GORSA_CANDIDATE_TEMPERATURE:-1.0}"
export GORSA_CANDIDATE_TOP_P="${GORSA_CANDIDATE_TOP_P:-1.0}"
export GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE="${GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE:-0.7}"
export GORSA_ADDITIONAL_INSTRUCTION_TOP_P="${GORSA_ADDITIONAL_INSTRUCTION_TOP_P:-0.95}"
export GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS="${GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS:-48}"
export VLLM_TASK_BATCH_SIZE="${VLLM_TASK_BATCH_SIZE:-64}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf_cache/hub}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

mkdir -p "${root}/logs"

echo "seed=${seed}"
echo "root=${root}"

python scripts/01_init_tasks.py

bash scripts/02_generate_candidates_vllm_2gpu.sh

python - <<'PY'
import json
from pathlib import Path

root = Path(__import__("os").environ["GORSA_ROOT_DIR"])
rows = sum(len(json.loads(p.read_text()).get("candidates") or []) for p in (root / "tasks").glob("*.json"))
print("candidate_rows_after_pad", rows)
PY

for stage in scripts/03_evaluate_candidates_parallel.py scripts/04_score_baselines.py; do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done

echo "===== RUNNING scripts/05_generate_instructions_vllm.py ====="
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="${GORSA_ROOT_DIR}/logs/${stamp}_05_generate_instructions_vllm.log"
echo "logging to: ${log}"
CUDA_VISIBLE_DEVICES=0 /workspace/.venvs/vllm/bin/python scripts/05_generate_instructions_vllm.py \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 2>&1 | tee "${log}"
echo "===== DONE scripts/05_generate_instructions_vllm.py ====="

for stage in scripts/06_compute_l0.py scripts/07_pairwise_results.py scripts/08_report.py; do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done
