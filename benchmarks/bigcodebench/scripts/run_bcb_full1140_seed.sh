#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${REPO_ROOT}/examples/run_bcb_full1140_seed.sh" "$@"
