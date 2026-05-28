#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export GORSA_ROOT_DIR="${GORSA_ROOT_DIR:?set GORSA_ROOT_DIR before running L0 shards}"
export GORSA_TASK_SHARD_COUNT="${GORSA_TASK_SHARD_COUNT:-2}"
export GORSA_SCORE_BATCH_SIZE="${GORSA_SCORE_BATCH_SIZE:-64}"

mkdir -p "${GORSA_ROOT_DIR}/logs" /tmp/gorsa_stage_logs

echo "root: ${GORSA_ROOT_DIR}"
echo "L0 backend: transformers sharded data parallel"
echo "GORSA_TASK_SHARD_COUNT=${GORSA_TASK_SHARD_COUNT}"
echo "GORSA_SCORE_BATCH_SIZE=${GORSA_SCORE_BATCH_SIZE}"

pids=()
for shard in $(seq 0 $((GORSA_TASK_SHARD_COUNT - 1))); do
  log="/tmp/gorsa_stage_logs/06_l0_transformers_gpu${shard}_shard${shard}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "starting shard ${shard}/${GORSA_TASK_SHARD_COUNT} on GPU ${shard}: ${log}"
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    export GORSA_TASK_SHARD_INDEX="${shard}"
    python -u -X faulthandler scripts/06_compute_l0.py 2>&1 | tee "${log}"
    cp "${log}" "${GORSA_ROOT_DIR}/logs/$(basename "${log}")"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
