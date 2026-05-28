#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 43 44 45 46; do
  "${SCRIPT_DIR}/run_bcb_full1140_seed.sh" "$seed"
done
