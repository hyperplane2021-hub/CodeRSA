# Artifact Notes

This repository is intended to contain source code and lightweight
configuration files. Large generated artifacts should be stored outside git.

## Ignored Outputs

The `.gitignore` excludes:

- model and dataset caches
- virtual environments
- compressed archives
- run directories
- per-task generated JSON records
- evaluation JSONL files
- plots and summary CSV/JSON files

## Run Outputs

For each benchmark run, the primary output directory is `$GORSA_ROOT_DIR`.
Typical contents are:

```text
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
pairwise_avg_curve.csv
tasks/*.json
logs/*.log
```

The per-task JSON files contain candidate programs, candidate-induced
instructions, L0 score matrices, and final reranking decisions. See
[task_record_schema.md](task_record_schema.md) for field-level details.

## External Artifact Release

For reviewer-facing artifact bundles, include only reproducibility-relevant
outputs:

```text
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
pairwise_avg_curve.csv
tasks/*.json
logs/*.log
```

Do not include model checkpoints, Hugging Face caches, local virtual
environments, or private credentials.
