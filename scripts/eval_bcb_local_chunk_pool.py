#!/usr/bin/env python3
"""Run BigCodeBench local evaluation as a bounded pool of task chunks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path


THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_failed_eval(chunk_samples: Path, chunk_eval: Path, reason: str) -> None:
    rows = read_jsonl(chunk_samples)
    eval_data = defaultdict(list)
    for row in rows:
        eval_data[row["task_id"]].append(
            {
                "task_id": row["task_id"],
                "solution": row.get("solution") or row.get("completion") or "",
                "status": "fail",
                "details": {"error": reason},
            }
        )
    chunk_eval.parent.mkdir(parents=True, exist_ok=True)
    with chunk_eval.open("w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval": eval_data}, f, indent=2, ensure_ascii=False)
    pass_path = chunk_eval.with_name(chunk_eval.name.replace("eval_results.json", "pass_at_k.json"))
    with pass_path.open("w", encoding="utf-8") as f:
        json.dump({"gt_pass_rate": 0.0, "failed_tasks": sorted(eval_data)}, f, indent=2)


def cleanup_eval_processes(needle: Path | str) -> int:
    """Kill leftover BigCodeBench evaluator descendants for this work area."""
    text = str(needle)
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except Exception:
        return 0
    pids = []
    for line in out.splitlines():
        if "-m bigcodebench.evaluate" not in line or text not in line:
            continue
        parts = line.strip().split(maxsplit=1)
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid != os.getpid():
            pids.append(pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        if sig == signal.SIGTERM and pids:
            try:
                subprocess.run(["sleep", "1"], check=False)
            except Exception:
                pass
    return len(pids)


def load_eval(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("eval") or {}
    except Exception:
        return {}


def is_complete_eval(path: Path, task_ids: list[str], keep_n: int) -> bool:
    data = load_eval(path)
    return all(len(data.get(task_id, [])) >= keep_n for task_id in task_ids)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    out = 1.0
    for value in range(n - c + 1, n + 1):
        out *= 1.0 - k / value
    return 1.0 - out


def merge_eval_results(chunk_results: list[Path], output: Path, pass_k: list[int], subset: str, split: str) -> None:
    merged = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval": {}}
    gt_rates = []
    failed_tasks = []
    for path in chunk_results:
        for task_id, results in load_eval(path).items():
            merged["eval"][task_id] = results
        pass_path = path.with_name(path.name.replace("eval_results.json", "pass_at_k.json"))
        if pass_path.exists():
            with pass_path.open(encoding="utf-8") as f:
                chunk_pass = json.load(f)
            gt_rates.append(float(chunk_pass.get("gt_pass_rate", 0.0)))
            failed_tasks.extend(chunk_pass.get("failed_tasks") or [])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    totals = [len(results) for results in merged["eval"].values()]
    correct = [sum(item.get("status") == "pass" for item in results) for results in merged["eval"].values()]
    pass_at_k = {
        f"pass@{k}": sum(estimate_pass_at_k(n, c, k) for n, c in zip(totals, correct)) / len(totals)
        for k in pass_k
        if totals and min(totals) >= k
    }
    pass_at_k.update(
        {
            "model": output.name.split("--bigcodebench-")[0],
            "split": split,
            "subset": subset,
            "calibrated": True,
            "gt_pass_rate": sum(gt_rates) / len(gt_rates) if gt_rates else 0.0,
            "failed_tasks": sorted(set(failed_tasks)),
        }
    )
    with output.with_name(output.name.replace("eval_results.json", "pass_at_k.json")).open("w", encoding="utf-8") as f:
        json.dump(pass_at_k, f, indent=2)


def run_chunk(
    *,
    python_bin: Path,
    chunk_samples: Path,
    chunk_eval: Path,
    chunk_log: Path,
    task_ids: list[str],
    subset: str,
    split: str,
    timeout: int,
    chunk_timeout: int | None,
    keep_n: int,
    resume: bool,
) -> tuple[Path, bool, str]:
    if resume and is_complete_eval(chunk_eval, task_ids, keep_n):
        return chunk_eval, True, "skip"

    chunk_eval.unlink(missing_ok=True)
    chunk_eval.with_name(chunk_eval.name.replace("eval_results.json", "pass_at_k.json")).unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(THREAD_LIMIT_ENV)
    env.update(
        {
            "BIGCODEBENCH_TIMEOUT_PER_TASK": str(timeout),
            "PYTHONPATH": "/workspace/bigcodebench:/workspace",
            "HF_HOME": env.get("HF_HOME", "/workspace/hf_cache"),
            "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE", "/workspace/hf_cache"),
            "XDG_CACHE_HOME": env.get("XDG_CACHE_HOME", "/workspace/.cache"),
            "TMPDIR": env.get("TMPDIR", "/workspace/tmp"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [
        str(python_bin),
        "-m",
        "bigcodebench.evaluate",
        "--execution",
        "local",
        "--split",
        split,
        "--subset",
        subset,
        "--samples",
        str(chunk_samples),
        "--parallel",
        "1",
        "--selective_evaluate",
        ",".join(task_ids),
    ]
    chunk_log.parent.mkdir(parents=True, exist_ok=True)
    with chunk_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=chunk_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            cleanup_eval_processes(chunk_samples)
            reason = f"chunk timeout>{chunk_timeout}s"
            write_failed_eval(chunk_samples, chunk_eval, reason)
            return chunk_eval, True, reason
    if returncode != 0:
        reason = f"failed rc={returncode}"
        write_failed_eval(chunk_samples, chunk_eval, reason)
        return chunk_eval, True, reason
    if not is_complete_eval(chunk_eval, task_ids, keep_n):
        reason = "missing/incomplete eval output"
        write_failed_eval(chunk_samples, chunk_eval, reason)
        return chunk_eval, True, reason
    return chunk_eval, True, "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--subset", choices=["full", "hard"], default="hard")
    parser.add_argument("--split", choices=["instruct", "complete"], default="instruct")
    parser.add_argument("--chunk-tasks", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--keep-n", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--chunk-timeout", type=int, default=int(os.environ.get("BCB_CHUNK_TIMEOUT", "120")))
    parser.add_argument("--pass-k", default="1,5,10")
    parser.add_argument("--python-bin", type=Path, default=Path("/workspace/.venvs/bcb_eval_py310/bin/python"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.samples)
    grouped: dict[str, list[dict]] = defaultdict(list)
    task_order: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        task_id = row["task_id"]
        grouped[task_id].append(row)
        task_order[task_id] = None

    tasks = list(task_order)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.work_dir / "logs"
    chunks = []
    for chunk_idx, start in enumerate(range(0, len(tasks), args.chunk_tasks)):
        chunk_tasks = tasks[start : start + args.chunk_tasks]
        chunk_samples = args.work_dir / f"chunk_{chunk_idx:03d}.jsonl"
        chunk_eval = args.work_dir / f"chunk_{chunk_idx:03d}_eval_results.json"
        chunk_log = log_dir / f"chunk_{chunk_idx:03d}.log"
        chunk_rows = [row for task_id in chunk_tasks for row in grouped[task_id]]
        write_jsonl(chunk_samples, chunk_rows)
        chunks.append((chunk_idx, chunk_tasks, chunk_samples, chunk_eval, chunk_log))

    print(f"samples={len(rows)} tasks={len(tasks)} chunks={len(chunks)} jobs={args.jobs} timeout={args.timeout}", flush=True)
    completed = sum(is_complete_eval(chunk_eval, chunk_tasks, args.keep_n) for _, chunk_tasks, _, chunk_eval, _ in chunks)
    print(f"initial_complete={completed}/{len(chunks)}", flush=True)

    try:
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_chunk = {
                executor.submit(
                    run_chunk,
                    python_bin=args.python_bin,
                    chunk_samples=chunk_samples,
                    chunk_eval=chunk_eval,
                    chunk_log=chunk_log,
                    task_ids=chunk_tasks,
                    subset=args.subset,
                    split=args.split,
                    timeout=args.timeout,
                    chunk_timeout=args.chunk_timeout,
                    keep_n=args.keep_n,
                    resume=args.resume,
                ): (chunk_idx, chunk_tasks, chunk_eval)
                for chunk_idx, chunk_tasks, chunk_samples, chunk_eval, chunk_log in chunks
            }
            done_count = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx, chunk_tasks, chunk_eval = future_to_chunk[future]
                path, ok, status = future.result()
                done_count += 1
                completed_now = sum(is_complete_eval(p, t, args.keep_n) for _, t, _, p, _ in chunks)
                label = f"[{chunk_idx:03d}] {','.join(chunk_tasks)} {status}"
                print(f"{label} complete={completed_now}/{len(chunks)} checked={done_count}/{len(chunks)}", flush=True)
                if not ok:
                    failures.append(f"{chunk_idx}:{status}:{path}")

        if failures:
            print("failures:", file=sys.stderr)
            for item in failures:
                print(f"  {item}", file=sys.stderr)
            raise SystemExit(1)

        chunk_results = [chunk_eval for _, _, _, chunk_eval, _ in chunks]
        pass_k = [int(k) for k in args.pass_k.split(",") if k.strip()]
        merge_eval_results(chunk_results, args.output, pass_k, args.subset, args.split)
        print(f"merged={args.output}", flush=True)
    finally:
        killed = cleanup_eval_processes(args.work_dir)
        if killed:
            print(f"cleanup_eval_processes killed={killed}", flush=True)


if __name__ == "__main__":
    main()
