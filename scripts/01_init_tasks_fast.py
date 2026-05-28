import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from gorsa_pipeline.core import (
    build_mbpp_generation_prompt,
    normalize_mbpp_doc,
    task_path,
)
from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config


def write_json_compact(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)

    root = Path(config.root_dir)
    temp_root = Path("/tmp/codersa_init") / root.name
    temp_tasks = temp_root / "tasks"

    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_tasks.mkdir(parents=True, exist_ok=True)

    write_json_compact({"config": config.to_dict()}, temp_root / "run_config.json")

    count = 0
    for raw in dataset:
        doc = normalize_mbpp_doc(raw)
        prompt = build_mbpp_generation_prompt(doc)
        record = {
            "task_id": doc["task_id"],
            "text": doc["text"],
            "test_list": doc["test_list"],
            "test_setup_code": doc["test_setup_code"],
            "challenge_test_list": doc["challenge_test_list"],
            "reference_code": doc["reference_code"],
            "generation_prompt": prompt,
            "candidates": None,
            "candidate_eval": None,
            "coder_logprobs": None,
            "reviewer_logprobs": None,
            "prior_logprobs": None,
            "additional_instructions": None,
            "equiv_yes_edges": None,
            "clusters": None,
            "l0_logprobs": None,
            "results": None,
            "results_pairwise_avg": None,
        }
        dst = Path(str(task_path(temp_root, doc["task_id"])))
        write_json_compact(record, dst)
        count += 1

    print(f"task files: {count}", flush=True)

    root.mkdir(parents=True, exist_ok=True)
    workspace_tasks = root / "tasks"
    if workspace_tasks.exists():
        shutil.rmtree(workspace_tasks)
    shutil.copytree(temp_tasks, workspace_tasks)
    shutil.copyfile(temp_root / "run_config.json", root / "run_config.json")
    print(f"wrote init to: {root}", flush=True)


if __name__ == "__main__":
    main()
