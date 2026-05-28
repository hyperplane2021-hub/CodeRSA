#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_stage_logged.sh scripts/02_generate_candidates_vllm.py [args...]"
  exit 2
fi

WORKSPACE="${WORKSPACE:-/root/workspace}"
cd "${WORKSPACE}"
root_dir="${GORSA_ROOT_DIR:-${WORKSPACE}/HumanEval+Llama3}"
mkdir -p "${root_dir}/logs"

stage_name="$(basename "$1" .py)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${root_dir}/logs/${stamp}_${stage_name}.log"

echo "logging to: ${log_path}"
python -u "$@" 2>&1 | stdbuf -oL awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' | tee "${log_path}"
