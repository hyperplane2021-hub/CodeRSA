# CodeRSA Reproduction

This anonymous repository provides the reproduction code for **CodeRSA**, a
pragmatic reranking method for natural-language-to-code generation.

CodeRSA samples a pool of candidate programs, derives candidate-induced
instructions, computes local pairwise pragmatic contests, and combines those
pairwise scores with global L0 support to select the final program.

## Benchmarks

The repository supports the three benchmarks used in the paper:

```text
HumanEval+      164 tasks
MBPP+           378 tasks
BigCodeBench    1140 tasks
```

All three benchmarks follow the same high-level pipeline:

1. initialize benchmark tasks
2. generate 10 candidate programs per task
3. evaluate candidates with benchmark tests
4. compute Coder and CoderReviewer scores
5. generate candidate-induced instructions
6. compute the L0 instruction-candidate score matrix
7. run CodeRSA reranking
8. write the summary table

## Paper Settings

The default settings match the paper:

- candidate pool size: `n=10`
- raw candidate samples per task for MBPP+: `50`, from which 10 valid
  candidates are kept
- candidate sampling: temperature `1.2`, top-p `1.0`
- reported seed sweep: `42, 43, 44, 45, 46`
- induced-instruction generation: greedy decoding
- final CodeRSA score: `z(pairwise) + z(avg-all L0)`

HumanEval+ and MBPP+ use the concise one-sentence behavior-description prompt
for induced instructions. BigCodeBench uses the detailed behavior-description
prompt.

To reproduce a different paper model, change `GORSA_MODEL_ID`; the benchmark
and reranking settings stay fixed.

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

## Default Run: MBPP+

The default launcher runs the MBPP+ reproduction.

```bash
export WORKSPACE=/workspace
export HF_TOKEN=...
export GORSA_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
export GORSA_ROOT_DIR=$WORKSPACE/runs/codersa_mbpp_seed42
export GORSA_SEED=42
export GORSA_LIMIT=378
export GORSA_CANDIDATE_OVERSAMPLE=50

bash scripts/run_full_vllm_mbpp.sh
```

For the same MBPP+ setting with staged task writes and sharded two-GPU L0
scoring:

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
export GORSA_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
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
export GORSA_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
export BIGCODEBENCH_REPO=/workspace/src/bigcodebench
export BCB_EVAL_PYTHON=/workspace/.venvs/bcb_eval_py310/bin/python

bash examples/run_bcb_full1140_seed.sh 42
```

Repeat each command with seeds `42` through `46` for the reported seed sweep,
using a distinct `$GORSA_ROOT_DIR` for each seed.

## Outputs

Each run writes artifacts under `$GORSA_ROOT_DIR`:

```text
tasks/*.json
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
logs/*.log
```

The main result table is `baseline_pairwise_avg.csv`; the `CodeRSA` row is the
paper's fixed equal-weight reranker.

## Documentation

- [docs/method.md](docs/method.md): scoring and reranking details
- [docs/task_record_schema.md](docs/task_record_schema.md): per-task JSON format
- [docs/artifacts.md](docs/artifacts.md): output and storage notes

## Notes

- Candidate programs are executed only for benchmark evaluation. Run evaluation
  stages in a sandboxed environment.
- Large generated outputs, model caches, and virtual environments are excluded
  from git.
