# CodeRSA Reproduction

This repository contains the reproduction code for **CodeRSA**, a pragmatic
reranking method for natural-language-to-code generation.

CodeRSA samples multiple candidate programs, reverse-generates a behavior
description for each candidate, scores candidates under the original and
candidate-induced instructions, and reranks with local pairwise pragmatic
contests plus a global support term.

## Repository Layout

```text
src/gorsa_pipeline/              MBPP+ pipeline used as the main runnable path
scripts/                         MBPP+ stage scripts and monitors
benchmarks/humaneval_plus/       HumanEval+ pipeline snapshot
benchmarks/bigcodebench/         BigCodeBench pipeline snapshot and adapters
examples/                        End-to-end shell entry points
docs/                            Method and artifact notes
```

The package name remains `gorsa_pipeline` because the original experimental
code used that name. In the paper and documentation, this method is referred to
as CodeRSA.

## Installation

Use Python 3.10 or 3.11. For GPU runs, install a CUDA-compatible PyTorch build
first if your environment does not already provide one.

```bash
git clone <your-repo-url> CodeRSA-repro
cd CodeRSA-repro
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For vLLM generation/scoring stages:

```bash
pip install -r requirements-vllm.txt
```

Set `HF_TOKEN` if the selected model requires gated Hugging Face access.

## Quick MBPP+ Run

The MBPP+ path is the cleanest runnable entry point in this repo.

```bash
export WORKSPACE=/workspace
export HF_TOKEN=...
export GORSA_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
export GORSA_ROOT_DIR=$WORKSPACE/runs/codersa_mbpp_seed42
export GORSA_SEED=42
export GORSA_LIMIT=974

bash scripts/run_full_vllm_mbpp.sh
```

This executes:

1. task initialization
2. candidate generation
3. candidate evaluation
4. Coder and CoderReviewer scoring
5. candidate-induced instruction generation
6. L0 matrix scoring
7. CodeRSA pairwise + average reranking
8. report generation

Main outputs are written under `$GORSA_ROOT_DIR`:

```text
tasks/*.json
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
pairwise_avg_curve.csv
humanevalplus_pairwise_avg_sweep.png
```

The plot filename is inherited from the original scripts; for MBPP+ it still
uses the historical `humanevalplus_...` name.

## H200 / Two-GPU MBPP+ Flow

For the oversample-50 flow used in the artifact package:

```bash
export WORKSPACE=/workspace
export HF_TOKEN=...
bash examples/run_mbpp_oversample50_h200.sh
```

The L0 stage is sharded across two independent transformer processes. Adjust
`GORSA_TASK_SHARD_COUNT`, `CUDA_VISIBLE_DEVICES`, and `GORSA_SCORE_BATCH_SIZE`
for your machine.

## HumanEval+ and BigCodeBench

Benchmark-specific snapshots are kept under `benchmarks/`:

```bash
cd benchmarks/humaneval_plus
python scripts/run_all.py
```

For BigCodeBench, install the official BigCodeBench package/repository and set:

```bash
export BIGCODEBENCH_REPO=/workspace/src/bigcodebench
export BCB_EVAL_PYTHON=/workspace/.venvs/bcb_eval_py310/bin/python
bash examples/run_bcb_full1140_seed.sh 42
```

## Method

For task record fields and scoring details, see:

- [docs/method.md](docs/method.md)
- [docs/task_record_schema.md](docs/task_record_schema.md)
- [docs/artifacts.md](docs/artifacts.md)

## Notes

- Large run outputs, model caches, virtual environments, and compressed artifact
  archives are intentionally ignored by git.
- The reproduction scripts do not require Consensus-WUCS; the main comparisons
  included here are Random, Coder, CoderReviewer, Avg-all L0, Pairwise only, and
  Pairwise + Avg.
- Candidate code is executed during evaluation. Run in a sandboxed environment.
