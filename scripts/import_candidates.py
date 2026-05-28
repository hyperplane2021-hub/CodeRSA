import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any

from gorsa_pipeline.core import (
    clean_code_generation,
    get_first_top_level_function_name,
    mask_first_function_name_ast,
    read_json,
    try_extract_first_function_from_text,
    write_json,
)
from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.stages import init_tasks, pad_candidate_pool


def normalize_task_id(value: Any) -> str:
    text = str(value)
    if text.startswith("HumanEval/"):
        return text
    if text.startswith("HumanEval_"):
        return "HumanEval/" + text.split("_", 1)[1]
    if text.isdigit():
        return "HumanEval/" + text
    return text


def task_filename(task_id: str) -> str:
    return task_id.replace("/", "_") + ".json"


def load_payload(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            return data["tasks"]
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("task_id", key)
            else:
                row = {"task_id": key, "candidates": value}
            rows.append(row)
        return rows
    raise ValueError(f"Unsupported payload type: {type(data)}")


def candidate_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("raw_code", "completion", "code", "text", "solution", "exec_code"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def build_candidate(raw_text: str, candidate_id: int) -> dict | None:
    raw_code = clean_code_generation(raw_text)
    exec_code = try_extract_first_function_from_text(raw_code)
    if not exec_code.strip():
        return None

    original_function_name = get_first_top_level_function_name(exec_code)
    try:
        scoring_code = mask_first_function_name_ast(exec_code, new_name="f")
    except Exception:
        scoring_code = exec_code

    return {
        "candidate_id": candidate_id,
        "raw_code": raw_text,
        "exec_code": exec_code,
        "scoring_code": scoring_code,
        "original_function_name": original_function_name,
        "generator": "external",
    }


def extract_candidates(row: dict) -> list[Any]:
    for key in ("candidates", "completions", "generations", "solutions", "outputs"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    if any(key in row for key in ("exec_code", "code", "completion", "raw_code", "text", "solution")):
        return [row]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Import externally generated candidates into a GoRSA run root.")
    parser.add_argument("--input", required=True, help="JSON/JSONL file exported from Colab.")
    parser.add_argument("--no-pad", action="store_true", help="Do not pad short candidate pools to n_candidates.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing candidates.")
    args = parser.parse_args()

    config = prepare_config()
    dataset = load_dataset_for_config(config)
    init_tasks(config, dataset)

    rows = load_payload(Path(args.input))
    imported_tasks = 0
    imported_candidates = 0
    skipped_existing = 0
    empty_tasks = []

    for row in rows:
        task_id = normalize_task_id(row.get("task_id", row.get("id", row.get("name", ""))))
        if not task_id:
            raise ValueError(f"Missing task_id in row: {row.keys()}")

        path = Path(config.root_dir) / "tasks" / task_filename(task_id)
        if not path.exists():
            print(f"[WARNING] task file missing, skip {task_id}")
            continue

        record = read_json(path)
        if record.get("candidates") is not None and not args.force:
            skipped_existing += 1
            continue

        candidates = []
        seen = set()
        for item in extract_candidates(row):
            text = candidate_text(item)
            cand = build_candidate(text, len(candidates))
            if cand is None:
                continue
            norm = cand["exec_code"].strip()
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append(cand)
            if len(candidates) >= config.n_candidates:
                break

        record["candidates"] = candidates
        record["candidate_eval"] = None
        record["coder_logprobs"] = None
        record["reviewer_logprobs"] = None
        record["prior_logprobs"] = None
        record["additional_instructions"] = None
        record["l0_logprobs"] = None
        record["results_pairwise_avg"] = None
        write_json(record, path)

        imported_tasks += 1
        imported_candidates += len(candidates)
        if not candidates:
            empty_tasks.append(task_id)

    if not args.no_pad:
        pad_candidate_pool(config)

    print("imported_tasks:", imported_tasks)
    print("imported_candidates_before_pad:", imported_candidates)
    print("skipped_existing:", skipped_existing)
    print("empty_tasks:", empty_tasks)
    print("root:", config.root_dir)


if __name__ == "__main__":
    main()
