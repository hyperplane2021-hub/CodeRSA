import _bootstrap  # noqa: F401

import concurrent.futures
import os
from pathlib import Path

from tqdm.auto import tqdm

from gorsa_pipeline.core import (
    candidate_passes_humanevalplus_relaxed,
    read_json,
    task_path,
    write_json,
)
from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.core import normalize_humanevalplus_doc


def evaluate_one(config_dict: dict, raw_doc: dict) -> tuple[str, int, bool, str | None]:
    class Config:
        pass

    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)

    doc = normalize_humanevalplus_doc(raw_doc)
    path = task_path(config.root_dir, doc["task_id"])
    if not Path(path).exists():
        return doc["task_id"], 0, False, f"missing task file: {path}"

    record = read_json(path)
    if record.get("candidate_eval") is not None and not config.force_rescore:
        return doc["task_id"], len(record.get("candidate_eval") or []), False, None

    candidates = record.get("candidates")
    if candidates is None:
        return doc["task_id"], 0, False, "missing candidates"

    evals = []
    for cand_idx, cand in enumerate(candidates):
        passed, stderr = candidate_passes_humanevalplus_relaxed(
            code=cand["exec_code"],
            context_code=record.get("context_code", ""),
            test_code=record["test"],
            entry_point=record["entry_point"],
            timeout_seconds=config.eval_timeout_seconds,
        )
        evals.append({"candidate_id": cand_idx, "passed": bool(passed), "stderr": stderr})

    record["candidate_eval"] = evals
    write_json(record, path)
    return doc["task_id"], len(evals), True, None


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    workers = int(os.environ.get("GORSA_EVAL_WORKERS", "8"))
    config_dict = config.to_dict()

    completed = 0
    skipped = 0
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_one, config_dict, raw) for raw in dataset]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 2: evaluate candidates parallel"):
            task_id, _num, wrote, error = fut.result()
            if error:
                errors.append((task_id, error))
            elif wrote:
                completed += 1
            else:
                skipped += 1

    print("parallel eval workers =", workers)
    print("tasks evaluated =", completed)
    print("tasks skipped =", skipped)
    if errors:
        print("errors =", len(errors))
        for task_id, error in errors[:20]:
            print(f"[ERROR] {task_id}: {error}")
        raise SystemExit(1)
    print("Stage 2 parallel complete.")


if __name__ == "__main__":
    main()
