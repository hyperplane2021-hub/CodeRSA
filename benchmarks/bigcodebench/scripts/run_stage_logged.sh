#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_stage_logged.sh scripts/02_generate_candidates.py [args...]"
  exit 2
fi

cd /workspace
root_dir="${GORSA_ROOT_DIR:-/workspace/HumanEval+Llama3}"
mkdir -p "${root_dir}/logs"

stage_name="$(basename "$1" .py)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${root_dir}/logs/${stamp}_${stage_name}.log"

echo "logging to: ${log_path}"
"${PYTHON_BIN:-python}" "$@" 2>&1 | stdbuf -oL awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' | tee "${log_path}"
