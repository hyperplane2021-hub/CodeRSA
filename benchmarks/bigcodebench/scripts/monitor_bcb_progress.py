#!/usr/bin/env python3
"""Monitor BigCodeBench official stage1/2 plus GoRSA task adaptation."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from time import strftime, gmtime


def read_jsonl_count(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    tasks = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                tasks.add(json.loads(line)["task_id"])
            except Exception:
                pass
    return total, len(tasks)


def latest_log(root: Path) -> Path | None:
    logs = sorted(Path("/workspace/logs").glob("*run_bcb*stage12.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    root_s = str(root)
    for path in logs:
        try:
            if root_s in path.read_text(encoding="utf-8", errors="ignore")[:1000]:
                return path
        except Exception:
            continue
    return logs[0] if logs else None


def count_tasks(root: Path) -> tuple[int, int, int, int]:
    task_paths = sorted((root / "tasks").glob("*.json"))
    n_tasks = len(task_paths)
    n_eval = 0
    n_instr = 0
    n_l0 = 0
    for path in task_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_eval += int(data.get("candidate_eval") is not None)
        n_instr += int(data.get("additional_instructions") is not None)
        n_l0 += int(data.get("l0_logprobs") is not None)
    return n_tasks, n_eval, n_instr, n_l0


def gpu_lines() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def process_lines() -> list[str]:
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid,pcpu,pmem,etime,cmd"], text=True)
    except Exception:
        return []
    needles = ("run_bcb", "bigcodebench", "VllmWorkerProcess", "Meta-Llama-3-70B")
    return [line for line in ps.splitlines() if any(n in line for n in needles) and "monitor_bcb_progress" not in line]


def detect_files(root: Path) -> dict[str, Path | None]:
    results = root / "bcb_results"
    raw = results / "raw_n20.jsonl"
    filtered = results / "filtered_n10.jsonl"
    eval_results = results / "filtered_n10_eval_results.json"
    if not raw.exists():
        candidates = sorted(list(results.glob("raw_n20*.jsonl")) + list(results.glob("*-20-sanitized_calibrated.jsonl")))
        raw = candidates[-1] if candidates else raw
    if not filtered.exists():
        candidates = sorted(results.glob("filtered_n10*.jsonl"))
        filtered = candidates[-1] if candidates else filtered
    if not eval_results.exists() and filtered.exists():
        candidate = filtered.with_name(filtered.stem + "_eval_results.json")
        eval_results = candidate
    return {"raw": raw if raw.exists() else None, "filtered": filtered if filtered.exists() else None, "eval": eval_results if eval_results.exists() else None}


def eval_count(path: Path | None) -> tuple[int, int]:
    if not path or not path.exists():
        return 0, 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    evals = payload.get("eval", {})
    return sum(len(v) for v in evals.values()), len(evals)


def chunk_progress(root: Path) -> tuple[int, int, int]:
    results = root / "bcb_results"
    dirs = sorted(results.glob("gradio_chunks*"))
    if not dirs:
        return 0, 0, 0
    d = dirs[-1]
    chunks = sorted(d.glob("chunk_*.jsonl"))
    evals = sorted(d.glob("chunk_*_eval_results.json"))
    samples = 0
    for path in evals:
        n, _ = eval_count(path)
        samples += n
    return len(evals), len(chunks), samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor BigCodeBench official stage1/2 progress.")
    parser.add_argument("--root", default="/workspace/BigCodeBench-hard-instruct-Llama3-70B_os20_temp12_seed42")
    parser.add_argument("--expected-tasks", type=int, default=148)
    parser.add_argument("--generate-n", type=int, default=20)
    parser.add_argument("--keep-n", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    files = detect_files(root)
    raw_samples, raw_tasks = read_jsonl_count(files["raw"]) if files["raw"] else (0, 0)
    filtered_samples, filtered_tasks = read_jsonl_count(files["filtered"]) if files["filtered"] else (0, 0)
    eval_samples, eval_tasks = eval_count(files["eval"])
    chunk_done, chunk_total, chunk_eval_samples = chunk_progress(root)
    gorsa_tasks, gorsa_eval, gorsa_instr, gorsa_l0 = count_tasks(root)

    generated_target = args.expected_tasks * args.generate_n
    filtered_target = args.expected_tasks * args.keep_n
    stage = "generate"
    if filtered_samples:
        stage = "evaluate"
    if gorsa_tasks:
        stage = "adapted/goRSA"

    print(f"BigCodeBench monitor | {strftime('%Y-%m-%d %H:%M:%S UTC', gmtime())}")
    print("root:", root)
    print("stage:", stage)
    print(f"official raw samples:      {raw_samples}/{generated_target} ({raw_tasks}/{args.expected_tasks} tasks)")
    print(f"filtered samples:          {filtered_samples}/{filtered_target} ({filtered_tasks}/{args.expected_tasks} tasks)")
    print(f"official evaluated samples:{eval_samples}/{filtered_target} ({eval_tasks}/{args.expected_tasks} tasks)")
    if chunk_total:
        print(f"gradio chunks:             {chunk_done}/{chunk_total} chunks, {chunk_eval_samples} samples returned")
    print(f"goRSA tasks:               {gorsa_tasks}/{args.expected_tasks} eval={gorsa_eval} instr={gorsa_instr} l0={gorsa_l0}")
    log = latest_log(root)
    if log:
        print("log:", log)
    print("\nGPU:")
    for line in gpu_lines():
        idx, name, used, total, util = [x.strip() for x in line.split(",")]
        print(f"  GPU {idx} {name}: {used}/{total} MiB, util {util}%")
    print("\nRunning processes:")
    lines = process_lines()
    if lines:
        for line in lines[:12]:
            print(" ", line)
    else:
        print("  none detected")


if __name__ == "__main__":
    main()
