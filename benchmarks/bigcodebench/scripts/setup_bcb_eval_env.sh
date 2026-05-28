#!/usr/bin/env bash
set -euo pipefail

EVAL_VENV="${BCB_EVAL_VENV:-/workspace/.venvs/bcb_eval_py310}"
PYTHON_BIN="${BCB_EVAL_PYTHON:-python3.10}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
TMPDIR="${TMPDIR:-/workspace/tmp}"
NLTK_DATA="${NLTK_DATA:-/workspace/.cache/nltk_data}"

export PIP_CACHE_DIR XDG_CACHE_HOME TMPDIR NLTK_DATA

mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$NLTK_DATA"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sysconfig
from pathlib import Path

include = Path(sysconfig.get_paths()["include"]) / "Python.h"
raise SystemExit(0 if include.exists() else 1)
PY
then
  echo "missing Python.h for $PYTHON_BIN; install python3.10-dev or equivalent" >&2
  exit 1
fi

if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$EVAL_VENV"
fi

"$EVAL_VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# Official docs recommend installing evaluation requirements in an isolated
# environment. Use the local checkout so the evaluated package version matches
# the scripts we are using.
"$EVAL_VENV/bin/python" -m pip install -e /workspace/bigcodebench --no-deps
"$EVAL_VENV/bin/python" -m pip install -I --timeout 2000 -r /workspace/bigcodebench/Requirements/requirements-eval.txt

# The official Docker image pins these after requirements-eval to keep dataset
# loading stable.
"$EVAL_VENV/bin/python" -m pip install \
  appdirs fire multipledispatch pqdm tempdir termcolor tqdm \
  tree_sitter tree-sitter-python wget datasets==2.17.0 pyarrow==14.0.1 \
  gradio-client transformers

# Keep the evaluation stack compatible with the older libraries in
# requirements-eval. In particular, librosa pulls numba==0.55.0, which needs
# numpy<1.22, and numba still imports pkg_resources from setuptools.
"$EVAL_VENV/bin/python" -m pip install numpy==1.21.2 setuptools==65.5.1 protobuf==3.19.6

"$EVAL_VENV/bin/python" - <<'PY'
import os
import nltk

target = os.environ.get("NLTK_DATA", "/workspace/.cache/nltk_data")
os.makedirs(target, exist_ok=True)
for package in ("punkt", "punkt_tab"):
    nltk.download(package, download_dir=target, quiet=True)
print("bcb eval env ready")
print("python:", os.sys.executable)
print("nltk_data:", target)
PY
