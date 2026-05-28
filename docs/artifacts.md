# Artifact Notes

This repository is meant to be pushed to GitHub as source code. Large generated
artifacts should be kept outside git.

## Ignored Outputs

The `.gitignore` excludes:

- model and dataset caches
- virtual environments
- compressed archives
- run directories
- per-task generated JSON records
- evaluation JSONL files
- plots and summary CSV/JSON files

If you want to publish finished experiment outputs, use GitHub Releases,
Hugging Face Datasets, Zenodo, or an external storage bucket.

## Original Local Artifact Packages

The local workspace used to create this repo contained:

```text
GoRSA_workspace_scripts_and_human_artifacts_20260514.zip
GoRSA_BigCodeBench_20260514_artifacts.zip
gorsa_mbpp_oversample50_h200_flow_20260515.tar.gz
```

Those archives are not copied into this repository. Their reusable source code
has been extracted into `src/`, `scripts/`, `benchmarks/`, and `examples/`.

## Suggested Release Layout

For an external artifact release, include:

```text
run_config.json
summary_pairwise_avg.json
baseline_pairwise_avg.csv
pairwise_avg_curve.csv
tasks/*.json
logs/*.log
```

Do not include model checkpoints, Hugging Face caches, or virtual environments.
