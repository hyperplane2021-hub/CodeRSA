#!/usr/bin/env python3
"""Pretty live monitor for the BigCodeBench + GoRSA workspace run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def clear() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count_jsonl(path: Path | None) -> tuple[int, int]:
    if not path or not path.exists():
        return 0, 0
    samples = 0
    tasks = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            samples += 1
            try:
                tasks.add(json.loads(line)["task_id"])
            except Exception:
                pass
    return samples, len(tasks)


def eval_count(path: Path | None) -> tuple[int, int]:
    if not path or not path.exists():
        return 0, 0
    payload = read_json(path)
    data = payload.get("eval") or {}
    return sum(len(v) for v in data.values()), len(data)


def chunk_eval_count(root: Path, n_keep: int) -> tuple[int, int, int, Path | None]:
    results = root / "bcb_results"
    dirs = sorted(
        [p for p in results.glob("*chunks*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for directory in dirs:
        sample_chunks = sorted(directory.glob("chunk_*.jsonl"))
        if not sample_chunks:
            continue
        eval_chunks = sorted(directory.glob("chunk_*_eval_results.json"))
        eval_tasks = 0
        eval_samples = 0
        for path in eval_chunks:
            samples, tasks = eval_count(path)
            eval_samples += samples
            eval_tasks += tasks
        expected_samples = len(sample_chunks) * n_keep
        return eval_samples, eval_tasks, expected_samples, directory
    return 0, 0, 0, None


def latest(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def detect_files(root: Path, n_generate: int, n_keep: int) -> dict[str, Path | None]:
    results = root / "bcb_results"
    official = latest(list(results.glob("*bigcodebench*instruct*vllm-*.jsonl")))
    raw = results / f"raw_n{n_generate}.jsonl"
    filtered = results / f"filtered_n{n_keep}.jsonl"
    eval_path = results / f"filtered_n{n_keep}_eval_results.json"
    return {
        "official": official,
        "raw": raw if raw.exists() else None,
        "filtered": filtered if filtered.exists() else None,
        "eval": eval_path if eval_path.exists() else None,
    }


def task_counts(root: Path) -> dict[str, int]:
    fields = [
        "candidate_eval",
        "coder_logprobs",
        "reviewer_logprobs",
        "prior_logprobs",
        "additional_instructions",
        "l0_logprobs",
        "results_pairwise_avg",
    ]
    counts = {"tasks": 0, **{field: 0 for field in fields}}
    task_dir = root / "tasks"
    if not task_dir.exists() and (root / "gorsa" / "tasks").exists():
        task_dir = root / "gorsa" / "tasks"
    for path in sorted(task_dir.glob("*.json")):
        counts["tasks"] += 1
        data = read_json(path)
        for field in fields:
            counts[field] += int(data.get(field) is not None)
    return counts


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def gpu_rows() -> list[tuple[str, str, str, str]]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            rows.append(tuple(parts))
    return rows


def process_rows() -> list[str]:
    out = run(["ps", "-eo", "pid,ppid,stat,pcpu,pmem,etime,cmd"])
    needles = [
        "bigcodebench.generate",
        "bigcodebench.evaluate",
        "eval_bcb_gradio_chunks",
        "bcb_official_to_gorsa_tasks",
        "04_score_baselines_vllm",
        "05_generate_instructions_vllm",
        "06_compute_l0_vllm",
        "run_bcb_official_stage12",
    ]
    rows = []
    for line in out.splitlines():
        if "watch_bcb_dashboard" in line or " rg " in line:
            continue
        if any(n in line for n in needles):
            rows.append(line)
    return rows[:8]


def latest_log(root: Path) -> Path | None:
    logs = list((root / "logs").glob("*.log")) + list(Path("/workspace/logs").glob("*.log"))
    logs = [p for p in logs if p.exists()]
    if not logs:
        return None
    root_s = str(root)
    candidates = []
    for p in logs:
        name = p.name.lower()
        relevant = "bcb" in name or "rerank" in name
        try:
            head = p.read_text(encoding="utf-8", errors="ignore")[:2000]
            if root_s in head:
                relevant = True
        except Exception:
            pass
        if relevant:
            candidates.append(p)
    if not candidates:
        candidates = logs
    return max(candidates, key=lambda p: p.stat().st_mtime)


def tail(path: Path | None, n: int = 10) -> list[str]:
    if not path or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return lines[-n:]


def read_log_text(path: Path | None, max_chars: int = 200_000) -> str:
    if not path or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return text[-max_chars:]


def parse_live_eval_progress(log_text: str, expected_tasks: int, expected_samples: int) -> tuple[str, int, int] | None:
    """Parse tqdm-style progress fragments from BigCodeBench local eval logs."""
    if not log_text:
        return None

    # tqdm uses carriage returns, so normalize to searchable line-like chunks.
    normalized = log_text.replace("\r", "\n")
    matches = re.findall(r"(\d+)%\|[^\n]*?\|\s*(\d+)/(\d+)\s*\[", normalized)
    if not matches:
        return None

    done, total = map(int, matches[-1][1:])
    if total == expected_tasks and "Reading samples..." not in normalized[normalized.rfind(matches[-1][0] + "%") :]:
        return "groundtruth", done, total
    if total == expected_samples:
        return "candidate tests", done, total
    if total == expected_tasks:
        return "groundtruth", done, total
    return "eval", done, total


def bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, done / total))
    filled = int(round(width * pct))
    body = "█" * filled + "░" * (width - filled)
    code = GREEN if pct >= 1 else CYAN if pct >= 0.5 else YELLOW
    return f"{color(body, code)} {done:>5}/{total:<5} {pct * 100:5.1f}%"


def fmt_age(ts: float | None) -> str:
    if not ts:
        return "-"
    seconds = max(0, int(time.time() - ts))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s ago"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m ago"


def render(args: argparse.Namespace) -> str:
    root = Path(args.root)
    files = detect_files(root, args.generate_n, args.keep_n)
    official_samples, official_tasks = count_jsonl(files["official"])
    raw_samples, raw_tasks = count_jsonl(files["raw"])
    filtered_samples, filtered_tasks = count_jsonl(files["filtered"])
    eval_samples, eval_tasks = eval_count(files["eval"])
    chunk_samples, chunk_tasks, chunk_expected_samples, chunk_dir = chunk_eval_count(root, args.keep_n)
    tasks = task_counts(root)

    expected_raw = args.expected_tasks * args.generate_n
    expected_filtered = args.expected_tasks * args.keep_n
    width = max(80, min(shutil.get_terminal_size((110, 30)).columns, 140))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    log = latest_log(root)
    log_text = read_log_text(log)
    live_eval = parse_live_eval_progress(log_text, args.expected_tasks, expected_filtered)

    lines = []
    lines.append(color(" BigCodeBench hard/instruct dashboard ".center(width, "═"), BOLD + BLUE))
    lines.append(f"{color('time', DIM)} {now}")
    lines.append(f"{color('root', DIM)} {root}")
    lines.append("")
    lines.append(color("Official Stage 1/2", BOLD))
    lines.append(f"  generate target    {bar(official_samples, expected_raw)}  tasks {official_tasks}/{args.expected_tasks}")
    lines.append(f"  raw_n{args.generate_n:<2}          {bar(raw_samples, expected_raw)}  tasks {raw_tasks}/{args.expected_tasks}")
    lines.append(f"  filtered_n{args.keep_n:<2}     {bar(filtered_samples, expected_filtered)}  tasks {filtered_tasks}/{args.expected_tasks}")
    lines.append(f"  evaluated          {bar(eval_samples, expected_filtered)}  tasks {eval_tasks}/{args.expected_tasks}")
    if chunk_expected_samples and eval_samples == 0:
        lines.append(
            f"  chunk checkpoint   {bar(chunk_samples, expected_filtered)}  "
            f"chunks/tasks {chunk_tasks}/{args.expected_tasks}  {color(chunk_dir.name if chunk_dir else '', DIM)}"
        )
    if live_eval and eval_samples == 0:
        phase, done, total = live_eval
        lines.append(f"  live {phase:<15} {bar(done, total)}  {color('(parsed from log)', DIM)}")
    lines.append("")
    lines.append(color("GoRSA Fields", BOLD))
    lines.append(f"  tasks              {bar(tasks['tasks'], args.expected_tasks)}")
    lines.append(f"  eval               {bar(tasks['candidate_eval'], args.expected_tasks)}")
    lines.append(f"  baseline scores    {bar(min(tasks['coder_logprobs'], tasks['reviewer_logprobs'], tasks['prior_logprobs']), args.expected_tasks)}")
    lines.append(f"  instructions       {bar(tasks['additional_instructions'], args.expected_tasks)}")
    lines.append(f"  L0 matrices        {bar(tasks['l0_logprobs'], args.expected_tasks)}")
    lines.append(f"  pairwise results   {bar(tasks['results_pairwise_avg'], args.expected_tasks)}")
    lines.append("")
    if files["official"]:
        lines.append(f"{color('official file', DIM)} {files['official'].name}  updated {fmt_age(files['official'].stat().st_mtime)}")
    if files["filtered"]:
        lines.append(f"{color('filtered file', DIM)} {files['filtered'].name}  updated {fmt_age(files['filtered'].stat().st_mtime)}")

    lines.append("")
    lines.append(color("GPU", BOLD))
    gpu = gpu_rows()
    if not gpu:
        lines.append("  nvidia-smi unavailable")
    for idx, used, total, util in gpu:
        try:
            used_i, total_i, util_i = int(used), int(total), int(util)
        except ValueError:
            used_i = total_i = util_i = 0
        mem_bar = bar(used_i, total_i, width=18).split()[0]
        util_code = GREEN if util_i >= 70 else YELLOW if util_i else DIM
        lines.append(f"  GPU {idx}: {mem_bar} {used_i:>6}/{total_i:<6} MiB  util {color(f'{util_i:>3}%', util_code)}")

    if not args.no_log:
        lines.append("")
        lines.append(color("Latest Log", BOLD))
        lines.append(f"  {log}" if log else "  none")
        for line in tail(log, args.tail):
            if "ERROR" in line or "Traceback" in line:
                line = color(line, RED)
            elif "WARNING" in line or "WARN" in line:
                line = color(line, YELLOW)
            elif "complete" in line.lower() or "done" in line.lower():
                line = color(line, GREEN)
            lines.append("  " + line[-(width - 4) :])

    lines.append("")
    lines.append(color("Ctrl-C to quit. Refresh interval: ", DIM) + f"{args.interval}s")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live BigCodeBench + GoRSA dashboard.")
    parser.add_argument("--root", default="/workspace/BigCodeBench-hard-instruct-Llama3-70B_os20_temp12_seed42")
    parser.add_argument("--expected-tasks", type=int, default=148)
    parser.add_argument("--generate-n", type=int, default=20)
    parser.add_argument("--keep-n", type=int, default=10)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--no-log", action="store_true", help="Hide the Latest Log section.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        clear()
        print(render(args), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
