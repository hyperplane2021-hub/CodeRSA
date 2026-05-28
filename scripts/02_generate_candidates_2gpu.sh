#!/usr/bin/env bash
set -euo pipefail

cd /workspace
root_dir="${GORSA_ROOT_DIR:-/workspace/HumanEval+Llama3}"
mkdir -p "${root_dir}/logs"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log0="${root_dir}/logs/${stamp}_02_generate_candidates_gpu0.log"
log1="${root_dir}/logs/${stamp}_02_generate_candidates_gpu1.log"

echo "GPU 0 shard log: ${log0}"
echo "GPU 1 shard log: ${log1}"

CUDA_VISIBLE_DEVICES=0 python scripts/02_generate_candidates.py --shard-index 0 --shard-count 2 --skip-pad >"${log0}" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/02_generate_candidates.py --shard-index 1 --shard-count 2 --skip-pad >"${log1}" 2>&1 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

python - <<'PY'
import _bootstrap  # noqa: F401
from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import pad_candidate_pool

pad_candidate_pool(prepare_config())
PY
