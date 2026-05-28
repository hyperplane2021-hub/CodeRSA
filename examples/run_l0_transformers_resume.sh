#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:?set GORSA_ROOT_DIR before running L0}"
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-64}"

mkdir -p "${GORSA_ROOT_DIR}/logs" /tmp/gorsa_stage_logs
log="/tmp/gorsa_stage_logs/06_l0_transformers_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "root: ${GORSA_ROOT_DIR}"
echo "L0 backend: transformers"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "GORSA_SCORE_BATCH_SIZE=${GORSA_SCORE_BATCH_SIZE}"
echo "log: ${log}"

python -u scripts/06_compute_l0.py 2>&1 | tee "${log}"
cp "${log}" "${GORSA_ROOT_DIR}/logs/$(basename "${log}")"
