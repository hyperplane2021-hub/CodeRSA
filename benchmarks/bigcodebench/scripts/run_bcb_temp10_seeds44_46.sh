#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/logs
queue_ts="$(date -u +%Y%m%dT%H%M%SZ)"
queue_log="/workspace/logs/${queue_ts}_bcb_hard_instruct_temp10_seeds44_46_queue.log"

echo "queue_log=${queue_log}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] queue start seeds 44 45 46" | tee -a "${queue_log}"

for seed in 44 45 46; do
  log="/workspace/logs/${queue_ts}_bcb_hard_instruct_seed${seed}_temp10_full.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start seed ${seed}, log=${log}" | tee -a "${queue_log}"
  GORSA_CANDIDATE_TEMPERATURE=1.0 GORSA_TEMP_TAG=temp10 \
    /workspace/scripts/run_bcb_temp10_seed.sh "${seed}" > "${log}" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done seed ${seed}" | tee -a "${queue_log}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] queue complete" | tee -a "${queue_log}"
