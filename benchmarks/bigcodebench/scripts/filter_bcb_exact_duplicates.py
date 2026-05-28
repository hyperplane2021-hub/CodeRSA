#!/usr/bin/env python3
"""Keep BigCodeBench samples after exact sanitized-function de-duplication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


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


def function_body_key(sample: dict) -> str:
    code = sample.get("solution") or sample.get("completion") or sample.get("raw_solution") or ""
    code = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in code.splitlines()).strip()


def filter_task_samples(rows: list[dict], keep_n: int) -> tuple[list[dict], dict]:
    unique = []
    duplicates = []
    seen = set()

    for row in rows:
        key = function_body_key(row)
        if key and key not in seen:
            seen.add(key)
            unique.append(row)
        else:
            duplicates.append(row)

    kept = unique[:keep_n]
    if len(kept) < keep_n:
        kept.extend(duplicates[: keep_n - len(kept)])

    report = {
        "task_id": rows[0]["task_id"] if rows else None,
        "input_n": len(rows),
        "unique_n": len(unique),
        "duplicate_n": len(duplicates),
        "kept_n": len(kept),
        "filled_from_duplicates_n": max(0, len(kept) - min(len(unique), keep_n)),
    }
    return kept, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter BigCodeBench JSONL samples by exact sanitized function duplicates.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--keep-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)

    kept_rows = []
    reports = []
    for task_id in sorted(grouped):
        kept, report = filter_task_samples(grouped[task_id], args.keep_n)
        kept_rows.extend(kept)
        reports.append(report)

    write_jsonl(args.output, kept_rows)
    write_jsonl(args.report, reports)
    print("input samples:", len(rows))
    print("output samples:", len(kept_rows))
    print("tasks:", len(reports))
    print("total duplicates removed before refill:", sum(r["duplicate_n"] for r in reports))
    print("saved:", args.output)
    print("report:", args.report)


if __name__ == "__main__":
    main()
