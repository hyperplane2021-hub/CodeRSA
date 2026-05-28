#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:-/workspace/HumanEval+Llama3_repro_fast}"
export GORSA_INSTRUCTION_MODE=batched
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-32}"
export GORSA_GENERATION_BATCH_SIZE="${GORSA_GENERATION_BATCH_SIZE:-16}"

mkdir -p "${GORSA_ROOT_DIR}/logs"

echo "Fast repro root: ${GORSA_ROOT_DIR}"
echo "Instruction generation mode: ${GORSA_INSTRUCTION_MODE}"
echo "Score batch size: ${GORSA_SCORE_BATCH_SIZE}"
echo "Generation batch size: ${GORSA_GENERATION_BATCH_SIZE}"

echo "===== RUNNING scripts/01_init_tasks.py ====="
scripts/run_stage_logged.sh scripts/01_init_tasks.py
echo "===== DONE scripts/01_init_tasks.py ====="

echo "===== RUNNING scripts/02_generate_candidates_2gpu.sh ====="
scripts/02_generate_candidates_2gpu.sh
echo "===== DONE scripts/02_generate_candidates_2gpu.sh ====="

for stage in \
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
