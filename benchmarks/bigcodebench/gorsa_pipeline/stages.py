"""Stage implementations for BigCodeBench reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .core import (
    compute_task_results_pairwise_avg,
    read_json,
    summarize_run_pairwise_avg,
    task_path,
    write_json,
)


PAIRWISE_TIE_MARGIN = 0.0


def _task_id_from_raw(raw) -> str:
    return str(raw["task_id"])


def compute_pairwise_results(config, dataset) -> None:
    for raw in tqdm(dataset, desc="Stage 7: compute BigCodeBench CodeRSA results"):
        task_id = _task_id_from_raw(raw)
        path = task_path(config.root_dir, task_id)
        if not Path(path).exists():
            print(f"[WARNING] task file missing, skip task_id={task_id}")
            continue

        record = read_json(path)
        required_fields = [
            "candidates",
            "candidate_eval",
            "coder_logprobs",
            "reviewer_logprobs",
            "additional_instructions",
            "l0_logprobs",
        ]
        missing = [key for key in required_fields if record.get(key) is None]
        if missing:
            print(f"[WARNING] skip task {task_id} missing fields: {missing}")
            continue

        try:
            record["results_pairwise_avg"] = compute_task_results_pairwise_avg(
                record=record,
                tie_margin=PAIRWISE_TIE_MARGIN,
            )
        except Exception as e:
            print(f"[WARNING] CodeRSA failed task={task_id} error={e}")
            continue

        write_json(record, path)

    summary_pairwise_avg = summarize_run_pairwise_avg(config.root_dir)
    summary_path = Path(config.root_dir) / "summary_pairwise_avg.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_pairwise_avg, f, indent=2, ensure_ascii=False)

    print("Saved:", summary_path)


def write_report(config) -> None:
    summary_path = Path(config.root_dir) / "summary_pairwise_avg.json"
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    baseline_df = pd.DataFrame(
        [
            {"method": "Random", "accuracy": summary["random_acc"]},
            {"method": "Coder", "accuracy": summary["coder_acc"]},
            {"method": "CoderReviewer", "accuracy": summary["coderreviewer_acc"]},
            {"method": "Oracle@10", "accuracy": summary["oracle10"]},
            {"method": "Avg-all L0", "accuracy": summary["avg_all_l0_acc"]},
            {"method": "Pairwise only", "accuracy": summary["pairwise_only_acc"]},
            {"method": "CodeRSA", "accuracy": summary["codersa_acc"]},
        ]
    )

    baseline_path = Path(config.root_dir) / "baseline_pairwise_avg.csv"
    baseline_df.to_csv(baseline_path, index=False)

    print(baseline_df.to_string(index=False))
    print("Baseline CSV saved to:", baseline_path)
    print("CodeRSA:", summary["codersa"])
