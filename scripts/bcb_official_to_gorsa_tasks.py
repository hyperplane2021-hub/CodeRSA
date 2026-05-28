#!/usr/bin/env python3
"""Convert official BigCodeBench samples/eval results into GoRSA task files."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

bigcodebench_repo = os.environ.get("BIGCODEBENCH_REPO")
if bigcodebench_repo:
    sys.path.insert(0, bigcodebench_repo)

from bigcodebench.data import get_bigcodebench  # noqa: E402

from gorsa_pipeline.core import mask_first_function_name_ast, task_path, write_json  # noqa: E402
from gorsa_pipeline.runtime import prepare_config  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def scoring_code_for(solution: str) -> str:
    try:
        return mask_first_function_name_ast(solution, new_name="f")
    except Exception:
        return solution


def status_passed(status: str) -> bool:
    return str(status).strip().lower() == "pass"


def fill_empty_candidates(candidates: list[dict], candidate_eval: list[dict], seed: int, task_id: str) -> int:
    non_empty = [idx for idx, cand in enumerate(candidates) if str(cand.get("scoring_code") or "").strip()]
    empty = [idx for idx, cand in enumerate(candidates) if not str(cand.get("scoring_code") or "").strip()]
    if not non_empty:
        return 0
    rng = random.Random(f"{seed}:{task_id}")
    for idx in empty:
        donor_idx = rng.choice(non_empty)
        donor = dict(candidates[donor_idx])
        donor["candidate_id"] = idx
        candidates[idx] = donor
        if idx < len(candidate_eval) and donor_idx < len(candidate_eval):
            donor_eval = dict(candidate_eval[donor_idx])
            donor_eval["candidate_id"] = idx
            candidate_eval[idx] = donor_eval
    return len(empty)


def extract_behavior_instruction(prompt: str) -> str:
    """Extract the natural-language behavior spec used as GoRSA I_0."""
    text = str(prompt or "").strip()
    markers = [
        "\nThe function should output with:",
        "\nYou should write self-contained code starting with:",
        "\nYou should write code starting with:",
    ]
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    behavior = text[:cut].strip()
    return behavior or text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GoRSA tasks from official BigCodeBench artifacts.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--eval-results", required=True, type=Path)
    parser.add_argument("--subset", choices=["full", "hard"], default="hard")
    parser.add_argument("--split", choices=["instruct", "complete"], default="instruct")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["GORSA_ROOT_DIR"] = str(args.root)
    config = prepare_config()

    problems = get_bigcodebench(subset=args.subset)
    samples = read_jsonl(args.samples)
    with open(args.eval_results, "r", encoding="utf-8") as f:
        eval_payload = json.load(f)

    samples_by_task = defaultdict(list)
    for row in samples:
        samples_by_task[row["task_id"]].append(row)

    eval_by_task = eval_payload.get("eval", {})
    task_ids = [task_id for task_id in problems if task_id in samples_by_task and task_id in eval_by_task]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    tasks_dir = args.root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for task_id in task_ids:
        problem = problems[task_id]
        task_samples = samples_by_task[task_id]
        task_evals = eval_by_task[task_id]
        n = min(config.n_candidates, len(task_samples), len(task_evals))

        candidates = []
        candidate_eval = []
        for idx in range(n):
            sample = task_samples[idx]
            solution = sample.get("solution") or sample.get("completion") or ""
            raw_solution = sample.get("raw_solution") or solution
            candidates.append(
                {
                    "candidate_id": idx,
                    "raw_code": raw_solution,
                    "exec_code": solution,
                    "scoring_code": scoring_code_for(solution),
                    "original_function_name": problem.get("entry_point"),
                    "generator": "bigcodebench_official",
                }
            )
            ev = task_evals[idx]
            candidate_eval.append(
                {
                    "candidate_id": idx,
                    "passed": status_passed(ev.get("status")),
                    "stderr": "" if status_passed(ev.get("status")) else str(ev.get("details", "")),
                    "status": ev.get("status"),
                }
            )
        filled_empty = fill_empty_candidates(candidates, candidate_eval, config.seed, task_id)
        if filled_empty:
            print(f"[WARNING] filled {filled_empty} empty candidates for {task_id}", flush=True)

        prompt_key = f"{args.split}_prompt"
        raw_prompt = problem[prompt_key]
        behavior_instruction = extract_behavior_instruction(raw_prompt)
        record = {
            "task_id": task_id,
            "dataset": "bigcodebench",
            "subset": args.subset,
            "split": args.split,
            "text": behavior_instruction,
            "raw_prompt": raw_prompt,
            "full_instruct_prompt": raw_prompt,
            "context_code": "",
            "function_prompt": problem.get("code_prompt", ""),
            "entry_point": problem.get("entry_point"),
            "test": problem.get("test", ""),
            "test_list": [],
            "test_setup_code": "",
            "challenge_test_list": [problem.get("test", "")],
            "reference_code": problem.get("complete_prompt", "") + "\n" + problem.get("canonical_solution", ""),
            "generation_prompt": raw_prompt,
            "candidates": candidates,
            "candidate_eval": candidate_eval,
            "coder_logprobs": None,
            "reviewer_logprobs": None,
            "prior_logprobs": None,
            "additional_instructions": None,
            "l0_logprobs": None,
            "results_pairwise_avg": None,
        }
        write_json(record, task_path(args.root, task_id))
        written += 1

    print("root:", args.root)
    print("tasks written:", written)
    print("tasks dir:", tasks_dir)


if __name__ == "__main__":
    main()
