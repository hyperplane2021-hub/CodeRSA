#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SAMPLE_JSONL [subset] [split] [parallel]" >&2
  exit 2
fi

SAMPLES="$1"
SUBSET="${2:-hard}"
SPLIT="${3:-instruct}"
PARALLEL="${4:-8}"

EVAL_VENV="${BCB_EVAL_VENV:-/workspace/.venvs/bcb_eval_py310}"
ROOT="$(cd "$(dirname "$SAMPLES")/.." && pwd)"
OUT="${SAMPLES%.jsonl}_eval_results.json"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
NLTK_DATA="${NLTK_DATA:-/workspace/.cache/nltk_data}"

mkdir -p "$LOG_DIR" "$NLTK_DATA" /workspace/tmp /workspace/.cache
LOG="$LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)_bcb_local_eval.log"

IDS="$("$EVAL_VENV/bin/python" - "$SAMPLES" <<'PY'
import json
import sys
from collections import OrderedDict

seen = OrderedDict()
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        tid = json.loads(line)["task_id"]
        if not str(tid).startswith("BigCodeBench/"):
            tid = f"BigCodeBench/{tid}"
        seen[tid] = 1
print(",".join(seen))
PY
)"

export PYTHONPATH="/workspace/bigcodebench:/workspace"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export NLTK_DATA
export PYTHONUNBUFFERED=1

rm -f "$OUT" "${SAMPLES%.jsonl}_pass_at_k.json"

setsid bash -c "
  '$EVAL_VENV/bin/python' -m bigcodebench.evaluate \
    --execution local \
    --split '$SPLIT' \
    --subset '$SUBSET' \
    --samples '$SAMPLES' \
    --parallel '$PARALLEL' \
    --selective_evaluate '$IDS'
  /workspace/.venvs/vllm/bin/python /workspace/scripts/bcb_official_to_gorsa_tasks.py \
    --root '$ROOT' \
    --samples '$SAMPLES' \
    --eval-results '$OUT' \
    --subset '$SUBSET' \
    --split '$SPLIT'
" > "$LOG" 2>&1 < /dev/null &

echo "pid=$!"
echo "log=$LOG"
echo "out=$OUT"
