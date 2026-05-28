# Artifact Notes

This repository contains source code and lightweight configuration files. Large
generated artifacts should be stored outside git.

## Ignored Outputs

The `.gitignore` excludes:

- model and dataset caches
- virtual environments
- compressed archives
- run directories
- per-task generated JSON records
- evaluation JSONL files
- summary CSV/JSON files

## Run Outputs

For each benchmark run, the primary output directory is `$GORSA_ROOT_DIR`.
Typical contents are:

```text
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
tasks/*.json
logs/*.log
```

The per-task JSON files contain the intermediate records needed to audit and
recompute the reported results.

## Reviewer Artifact Bundle

For reviewer-facing artifact bundles, include only reproducibility-relevant
outputs:

```text
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
tasks/*.json
logs/*.log
```

Do not include model checkpoints, Hugging Face caches, local virtual
environments, or private credentials.
