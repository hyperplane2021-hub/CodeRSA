import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from gorsa_pipeline.core import (
    compute_task_results_pairwise_avg,
    normalize_mbpp_doc,
    read_json,
    task_path,
)
from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.stages import PAIRWISE_LAMBDA_VALUES, PAIRWISE_TIE_MARGIN


def summarize_records(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        raise RuntimeError("No completed pairwise+avg records.")

    summary = {
        "num_tasks": total,
        "oracle10": sum(bool(r["results_pairwise_avg"]["oracle_any"]) for r in records) / total,
        "random_acc": (
            sum(bool(ev["passed"]) for r in records for ev in r.get("candidate_eval", []))
            / max(1, sum(len(r.get("candidate_eval", [])) for r in records))
        ),
        "coder_acc": sum(bool(r["results_pairwise_avg"]["coder"]["selected_passed"]) for r in records) / total,
        "coderreviewer_acc": sum(
            bool(r["results_pairwise_avg"]["coderreviewer"]["selected_passed"]) for r in records
        )
        / total,
        "orig_only_l0_acc": sum(
            bool(r["results_pairwise_avg"]["orig_only_l0"]["selected_passed"]) for r in records
        )
        / total,
        "avg_all_l0_acc": sum(bool(r["results_pairwise_avg"]["avg_all_l0"]["selected_passed"]) for r in records)
        / total,
        "pairwise_only_acc": sum(
            bool(r["results_pairwise_avg"]["pairwise_only"]["selected_passed"]) for r in records
        )
        / total,
        "pairwise_avg_curve": [],
    }

    for lam in PAIRWISE_LAMBDA_VALUES:
        lam = float(lam)
        acc = (
            sum(
                next(
                    x
                    for x in r["results_pairwise_avg"]["pairwise_avg_curve"]
                    if float(x["lambda"]) == lam
                )["selected_passed"]
                for r in records
            )
            / total
        )
        summary["pairwise_avg_curve"].append({"lambda": lam, "accuracy": float(acc)})

    summary["pairwise_avg_best"] = max(summary["pairwise_avg_curve"], key=lambda x: x["accuracy"])
    summary["pairwise_avg_lambda1"] = next(
        x for x in summary["pairwise_avg_curve"] if float(x["lambda"]) == 1.0
    )
    return summary


def write_report_files(root: Path, summary: dict) -> None:
    summary_path = root / "summary_pairwise_avg.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    curve_df = pd.DataFrame(summary["pairwise_avg_curve"])
    baseline_df = pd.DataFrame(
        [
            {"method": "Random", "accuracy": summary["random_acc"]},
            {"method": "Coder", "accuracy": summary["coder_acc"]},
            {"method": "CoderReviewer", "accuracy": summary["coderreviewer_acc"]},
            {"method": "Oracle@10", "accuracy": summary["oracle10"]},
            {"method": "Avg-all L0", "accuracy": summary["avg_all_l0_acc"]},
            {"method": "Pairwise only", "accuracy": summary["pairwise_only_acc"]},
            {"method": "Pairwise + Avg(lambda=1)", "accuracy": summary["pairwise_avg_lambda1"]["accuracy"]},
            {"method": "Pairwise + Avg(best lambda)", "accuracy": summary["pairwise_avg_best"]["accuracy"]},
        ]
    )

    baseline_path = root / "baseline_pairwise_avg.csv"
    curve_path = root / "pairwise_avg_curve.csv"
    baseline_df.to_csv(baseline_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(curve_df["lambda"], curve_df["accuracy"], label="Pairwise + Avg-all")
    plt.axhline(summary["coder_acc"], linestyle="--", label="Coder")
    plt.axhline(summary["coderreviewer_acc"], linestyle="--", label="CoderReviewer")
    plt.axhline(summary["avg_all_l0_acc"], linestyle="--", label="Avg-all L0")
    plt.axhline(summary["pairwise_only_acc"], linestyle="--", label="Pairwise only")
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("MBPP+ pairwise+avg accuracy vs lambda")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = root / "mbpp_pairwise_avg_sweep.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(baseline_df.to_string(index=False))
    print("Summary saved to:", summary_path)
    print("Baseline CSV saved to:", baseline_path)
    print("Curve CSV saved to:", curve_path)
    print("Plot saved to:", plot_path)
    print("Pairwise+avg lambda=1:", summary["pairwise_avg_lambda1"])
    print("Best pairwise+avg:", summary["pairwise_avg_best"])


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    root = Path(config.root_dir)

    records = []
    for raw in tqdm(dataset, desc="Stage 7: pairwise summary in memory"):
        doc = normalize_mbpp_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        if not Path(path).exists():
            print(f"[WARNING] task file missing, skip task_id={doc['task_id']}", file=sys.stderr)
            continue
        record = read_json(path)
        required = [
            "candidates",
            "candidate_eval",
            "coder_logprobs",
            "reviewer_logprobs",
            "additional_instructions",
            "l0_logprobs",
        ]
        missing = [key for key in required if record.get(key) is None]
        if missing:
            print(f"[WARNING] skip task {doc['task_id']} missing fields: {missing}", file=sys.stderr)
            continue
        record["results_pairwise_avg"] = compute_task_results_pairwise_avg(
            record=record,
            lambda_values=PAIRWISE_LAMBDA_VALUES,
            tie_margin=PAIRWISE_TIE_MARGIN,
        )
        records.append(record)

    summary = summarize_records(records)
    write_report_files(root, summary)


if __name__ == "__main__":
    main()
