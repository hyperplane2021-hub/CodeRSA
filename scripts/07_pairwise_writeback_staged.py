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
    staged_tasks = Path("/tmp/gorsa_pairwise_writeback") / root.name / "tasks"

    if staged_tasks.exists():
        shutil.rmtree(staged_tasks)
    staged_tasks.mkdir(parents=True, exist_ok=True)

    staged = []
    for raw in tqdm(dataset, desc="Stage 7: compute pairwise to /tmp"):
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
        dst = staged_tasks / src.name
        write_json_compact(record, dst)
        staged.append((dst, workspace_tasks / src.name))

    if not staged:
        raise RuntimeError("No task JSON files staged for pairwise writeback.")

    print(f"Staged {len(staged)} task JSON files under {staged_tasks}", flush=True)
    for src, dst in tqdm(staged, desc="Stage 7: copy pairwise JSON to workspace"):
        tmp = dst.with_name(f".{dst.name}.pairwise.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)

    print(f"Wrote pairwise results into {len(staged)} task JSON files.", flush=True)


if __name__ == "__main__":
    main()
