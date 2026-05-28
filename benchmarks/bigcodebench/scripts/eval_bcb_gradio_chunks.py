#!/usr/bin/env python3
"""Evaluate BigCodeBench samples in task chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/workspace/bigcodebench")

from bigcodebench.evaluate import evaluate  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_order(rows: list[dict]) -> list[str]:
    order = OrderedDict()
    for row in rows:
        order[row["task_id"]] = None
    return list(order)


def merge_eval_results(chunk_results: list[Path], output: Path) -> None:
    merged = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval": {}}
    for path in chunk_results:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for task_id, results in payload.get("eval", {}).items():
            merged["eval"][task_id] = results
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BigCodeBench samples in smaller chunks.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--subset", choices=["full", "hard"], default="hard")
    parser.add_argument("--split", choices=["instruct", "complete"], default="instruct")
    parser.add_argument("--execution", choices=["gradio", "local", "e2b"], default="gradio")
    parser.add_argument("--chunk-tasks", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.samples)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)

    tasks = task_order(rows)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    chunk_result_paths = []

    print("samples:", len(rows))
    print("tasks:", len(tasks))
    print("chunk_tasks:", args.chunk_tasks)

    for chunk_idx, start in enumerate(range(0, len(tasks), args.chunk_tasks)):
        chunk_tasks = tasks[start : start + args.chunk_tasks]
        chunk_samples = args.work_dir / f"chunk_{chunk_idx:03d}.jsonl"
        chunk_eval = args.work_dir / f"chunk_{chunk_idx:03d}_eval_results.json"
        chunk_rows = [row for task_id in chunk_tasks for row in grouped[task_id]]
        write_jsonl(chunk_samples, chunk_rows)
        chunk_result_paths.append(chunk_eval)

        if args.resume and chunk_eval.exists():
            print(f"[{chunk_idx}] skip existing {chunk_eval}")
            continue

        selective = ",".join(chunk_tasks)
        print(f"[{chunk_idx}] evaluating tasks={chunk_tasks} samples={len(chunk_rows)}")
        evaluate(
            split=args.split,
            subset=args.subset,
            samples=str(chunk_samples),
            execution=args.execution,
            parallel=args.parallel,
            selective_evaluate=selective,
        )
        produced = chunk_samples.with_name(chunk_samples.stem + "_eval_results.json")
        if produced != chunk_eval:
            produced.replace(chunk_eval)
        print(f"[{chunk_idx}] done -> {chunk_eval}")

    existing = [p for p in chunk_result_paths if p.exists()]
    print("chunks complete:", len(existing), "/", len(chunk_result_paths))
    if len(existing) == len(chunk_result_paths):
        merge_eval_results(existing, args.output)
        print("merged:", args.output)
    else:
        raise SystemExit("not all chunks completed")


if __name__ == "__main__":
    main()
