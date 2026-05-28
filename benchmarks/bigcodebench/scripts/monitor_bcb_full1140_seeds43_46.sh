#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "${SCRIPT_DIR}")"
python scripts/monitor_seed_sweep.py "$@"
