#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:-/workspace/HumanEval+Llama3_notebook_repro}"
export GORSA_INSTRUCTION_MODE=serial

mkdir -p "${GORSA_ROOT_DIR}/logs"

echo "Notebook-style repro root: ${GORSA_ROOT_DIR}"
echo "Instruction generation mode: ${GORSA_INSTRUCTION_MODE}"

for stage in \
  scripts/01_init_tasks.py \
  scripts/02_generate_candidates.py \
  scripts/03_evaluate_candidates.py \
  scripts/04_score_baselines.py \
  scripts/05_generate_instructions.py \
  scripts/06_compute_l0.py \
  scripts/07_pairwise_results.py \
  scripts/08_report.py
do
  echo "===== RUNNING ${stage} ====="
  scripts/run_stage_logged.sh "${stage}"
  echo "===== DONE ${stage} ====="
done
