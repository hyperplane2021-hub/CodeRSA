#!/usr/bin/env bash
set -euo pipefail
root="${1:-/workspace/BigCodeBench-hard-instruct-Llama3-70B_os20_temp12_seed42}"
python scripts/monitor_bcb_progress.py --root "${root}"
