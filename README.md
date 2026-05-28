# CodeRSA Reproduction

This anonymous repository provides the reproduction code for **CodeRSA**, a
pragmatic reranking method for natural-language-to-code generation.

CodeRSA samples multiple candidate programs, derives candidate-induced
instructions from those programs, scores candidates in the induced local
instruction neighborhood, and reranks them using local pairwise pragmatic
contests together with a global support term.

## Benchmarks

The repository supports the three benchmarks used in the paper:

```text
HumanEval+
MBPP+
BigCodeBench
```

Each benchmark follows the same high-level pipeline:

1. initialize benchmark tasks
2. generate candidate programs
3. evaluate candidates with benchmark tests
4. compute Coder and CoderReviewer scores
5. generate candidate-induced instructions
6. compute the L0 instruction-candidate score matrix
7. run CodeRSA pairwise + global-support reranking
8. write reports and summary tables

## Repository Layout

```text
src/                         shared pipeline implementation
scripts/                     benchmark stage scripts and utilities
benchmarks/humaneval_plus/   HumanEval+ reproduction scripts
benchmarks/bigcodebench/     BigCodeBench reproduction scripts and adapters
examples/                    end-to-end launcher scripts
docs/                        method and artifact documentation
```

## Installation

Use Python 3.10 or 3.11. For GPU runs, install a CUDA-compatible PyTorch build
appropriate for your machine.

```bash
git clone <anonymous-repo-url> CodeRSA-repro
cd CodeRSA-repro
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For vLLM-backed generation and scoring:

```bash
pip install -r requirements-vllm.txt
```

Set `HF_TOKEN` if the selected model requires Hugging Face access.

## Running MBPP+

```bash
export WORKSPACE=/workspace
export HF_TOKEN=...
export GORSA_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
export GORSA_ROOT_DIR=$WORKSPACE/runs/codersa_mbpp_seed42
export GORSA_SEED=42
export GORSA_LIMIT=378

bash scripts/run_full_vllm_mbpp.sh
```

For the two-GPU oversample-50 configuration:

```bash
export WORKSPACE=/workspace
export HF_TOKEN=...
bash examples/run_mbpp_oversample50_h200.sh
```

## Running HumanEval+

```bash
cd benchmarks/humaneval_plus
export WORKSPACE=/workspace
export HF_TOKEN=...
export GORSA_ROOT_DIR=$WORKSPACE/runs/codersa_humaneval_seed42
export GORSA_SEED=42

python scripts/run_all.py
```

## Running BigCodeBench

BigCodeBench requires the official BigCodeBench codebase and its evaluation
environment. Set their locations before launching the run:

```bash
export WORKSPACE=/workspace
export HF_TOKEN=...
export BIGCODEBENCH_REPO=/workspace/src/bigcodebench
export BCB_EVAL_PYTHON=/workspace/.venvs/bcb_eval_py310/bin/python

bash examples/run_bcb_full1140_seed.sh 42
```

## Outputs

Each run writes artifacts under `$GORSA_ROOT_DIR`, including:

```text
tasks/*.json
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
pairwise_avg_curve.csv
*.png
logs/*.log
```

The main table is `baseline_pairwise_avg.csv`. The lambda sweep for the final
CodeRSA score is stored in `pairwise_avg_curve.csv`.

## Documentation

- [docs/method.md](docs/method.md): scoring and reranking details
- [docs/task_record_schema.md](docs/task_record_schema.md): per-task JSON format
- [docs/artifacts.md](docs/artifacts.md): output and storage notes

## Notes

- Candidate programs are executed only for benchmark evaluation. Run evaluation
  stages in a sandboxed environment.
- Large generated outputs, model caches, and virtual environments are excluded
  from git.
- Consensus-WUCS is not required for this reproduction package.
