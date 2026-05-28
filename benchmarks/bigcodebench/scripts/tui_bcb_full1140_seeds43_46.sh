#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "${SCRIPT_DIR}")"
python scripts/tui_bcb_full1140_seeds43_46.py "$@"
