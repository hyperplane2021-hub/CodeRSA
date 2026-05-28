import json
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from tqdm.auto import tqdm

from gorsa_pipeline.core import (
    compute_task_results_pairwise_avg,
    normalize_mbpp_doc,
    read_json,
    summarize_run_pairwise_avg,
    task_path,
)
from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.stages import PAIRWISE_TIE_MARGIN


def write_json_compact(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{Path(str(path)).name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    root = Path(config.root_dir)
    workspace_tasks = root / "tasks"
    temp_tasks = Path("/tmp/codersa_pairwise") / root.name / "tasks"

    if temp_tasks.exists():
        shutil.rmtree(temp_tasks)
    temp_tasks.mkdir(parents=True, exist_ok=True)

    completed = []
    for raw in tqdm(dataset, desc="Stage 7: compute CodeRSA"):
        doc = normalize_mbpp_doc(raw)
        src = Path(task_path(config.root_dir, doc["task_id"]))
        if not src.exists():
            print(f"[WARNING] task file missing, skip task_id={doc['task_id']}", file=sys.stderr)
            continue

        record = read_json(src)
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
            tie_margin=PAIRWISE_TIE_MARGIN,
        )
        dst = temp_tasks / src.name
        write_json_compact(record, dst)
        completed.append((dst, workspace_tasks / src.name))

    if not completed:
        raise RuntimeError("No completed task JSON files for CodeRSA.")

    print(f"Completed {len(completed)} task JSON files.", flush=True)
    for src, dst in tqdm(completed, desc="Stage 7: write CodeRSA results"):
        tmp = dst.with_name(f".{dst.name}.pairwise.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)

    summary = summarize_run_pairwise_avg(config.root_dir)
    summary_path = root / "summary_pairwise_avg.json"
    write_json_compact(summary, summary_path)
    print("Saved:", summary_path, flush=True)
    print(f"Wrote CodeRSA results into {len(completed)} task JSON files.", flush=True)


if __name__ == "__main__":
    main()
