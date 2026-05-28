import _bootstrap  # noqa: F401

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    ("candidates", "1 candidates"),
    ("candidate_eval", "2 eval"),
    ("coder_logprobs", "3 coder"),
    ("reviewer_logprobs", "3 reviewer"),
    ("prior_logprobs", "3 prior"),
    ("additional_instructions", "4 instructions"),
    ("l0_logprobs", "6 L0"),
    ("results_pairwise_avg", "7 pairwise"),
]


def default_root_dir() -> Path:
    env_root = os.environ.get("GORSA_ROOT_DIR")
    if env_root:
        return Path(env_root)

    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    candidate = workspace / "codersa_mbpp_llama3_limit378_seed42_2gpu_repro"
    if candidate.exists():
        return candidate
    return workspace / "codersa_mbpp_llama3_limit378_seed42"


def load_records(task_dir: Path):
    records = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            records.append((path, json.load(open(path, "r", encoding="utf-8"))))
        except Exception as e:
            records.append((path, {"_error": str(e)}))
    return records


def bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def percent(done: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{done / total * 100:5.1f}%"


def summarize(records):
    total = len(records)
    counts = {key: 0 for key, _ in FIELDS}
    candidates_total = 0
    candidates_tasks = 0
    passed = 0
    evaluated = 0
    oracle = 0
    recent = []

    for path, rec in records:
        if rec.get("_error"):
            continue
        for key, _ in FIELDS:
            if rec.get(key) is not None:
                counts[key] += 1

        candidates = rec.get("candidates") or []
        if candidates:
            candidates_total += len(candidates)
            candidates_tasks += 1

        evals = rec.get("candidate_eval") or []
        if evals:
            evaluated += len(evals)
            passed += sum(bool(x.get("passed")) for x in evals)
            oracle += int(any(bool(x.get("passed")) for x in evals))

        try:
            mtime = path.stat().st_mtime
            recent.append((mtime, path.name, rec.get("task_id", "")))
        except OSError:
            pass

    recent.sort(reverse=True)
    return {
        "total": total,
        "counts": counts,
        "candidates_total": candidates_total,
        "candidates_tasks": candidates_tasks,
        "passed": passed,
        "evaluated": evaluated,
        "oracle": oracle,
        "recent": recent[:8],
    }


def gpu_lines() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return [f"GPU status unavailable: {e}"]

    lines = []
    for raw in out.strip().splitlines():
        idx, name, used, total, util = [x.strip() for x in raw.split(",")]
        lines.append(f"GPU {idx} {name}: {used}/{total} MiB, util {util}%")
    return lines


def process_lines() -> list[str]:
    try:
        out = subprocess.check_output(
            "ps -eo pid,pcpu,pmem,etime,cmd | rg 'scripts/(0[2-8]|run_all)|python scripts' | rg -v 'rg |monitor_progress' || true",
            shell=True,
            text=True,
        ).strip()
    except Exception:
        return []
    return out.splitlines() if out else []


def resolve_monitor_root(root_dir: Path) -> tuple[Path, str]:
    scratch_base = Path(os.environ.get("GORSA_EVAL_SCRATCH_ROOT", "/tmp/gorsa_eval_scratch"))
    scratch_root = scratch_base / root_dir.name
    scratch_tasks = scratch_root / "tasks"
    if scratch_tasks.exists() and any(scratch_tasks.glob("*.json")):
        workspace_records = load_records(root_dir / "tasks")
        scratch_records = load_records(scratch_tasks)
        workspace_counts = summarize(workspace_records)["counts"]
        scratch_counts = summarize(scratch_records)["counts"]
        workspace_done = sum(workspace_counts.values())
        scratch_done = sum(scratch_counts.values())
        if scratch_done > workspace_done:
            return scratch_root, "scratch"
    return root_dir, "workspace"


def render_once(root_dir: Path) -> str:
    monitor_root, source = resolve_monitor_root(root_dir)
    task_dir = monitor_root / "tasks"
    records = load_records(task_dir)
    summary = summarize(records)
    total = summary["total"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"GoRSA MBPP+ pairwise pipeline monitor | {now}",
        f"root: {root_dir}",
        f"source: {source} ({monitor_root})",
        "",
        "Stages:",
    ]
    for key, label in FIELDS:
        done = summary["counts"][key]
        lines.append(f"  {label:16} {bar(done, total)} {done:3d}/{total:<3d} {percent(done, total)}")

    lines.extend(["", "Candidate/eval stats:"])
    lines.append(
        f"  candidate tasks: {summary['candidates_tasks']}/{total}, "
        f"candidate rows: {summary['candidates_total']}"
    )
    if summary["evaluated"]:
        lines.append(
            f"  evaluated candidates: {summary['evaluated']}, "
            f"pass ratio: {summary['passed'] / summary['evaluated']:.4f}, "
            f"oracle tasks: {summary['oracle']}/{total}"
        )
    else:
        lines.append("  evaluated candidates: 0")

    lines.extend(["", "GPU:"])
    lines.extend("  " + line for line in gpu_lines())

    procs = process_lines()
    lines.extend(["", "Running stage processes:"])
    lines.extend(("  " + line for line in procs) if procs else ["  none detected"])

    lines.extend(["", "Recently touched task files:"])
    if summary["recent"]:
        for mtime, name, task_id in summary["recent"]:
            stamp = datetime.fromtimestamp(mtime, timezone.utc).strftime("%H:%M:%S")
            lines.append(f"  {stamp}  {name:22} {task_id}")
    else:
        lines.append("  none")

    summary_path = monitor_root / "summary_pairwise_avg.json"
    if summary_path.exists():
        try:
            data = json.load(open(summary_path, "r", encoding="utf-8"))
            lines.extend(["", "Summary:"])
            lines.append(f"  num_tasks: {data.get('num_tasks')}")
            lines.append(f"  random_acc: {data.get('random_acc')}")
            lines.append(f"  coder_acc: {data.get('coder_acc')}")
            lines.append(f"  coderreviewer_acc: {data.get('coderreviewer_acc')}")
            lines.append(f"  pairwise_only_acc: {data.get('pairwise_only_acc')}")
            lines.append(f"  pairwise_avg_lambda1: {data.get('pairwise_avg_lambda1')}")
            lines.append(f"  pairwise_avg_best: {data.get('pairwise_avg_best')}")
        except Exception as e:
            lines.append(f"summary read failed: {e}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor MBPP+ pairwise pipeline progress.")
    parser.add_argument("--root", type=Path, default=None, help="Run root directory.")
    parser.add_argument("--watch", action="store_true", help="Refresh continuously.")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds.")
    args = parser.parse_args()

    root_dir = args.root or default_root_dir()

    if not args.watch:
        print(render_once(root_dir), flush=True)
        return

    while True:
        print("\033[2J\033[H", end="")
        print(render_once(root_dir), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
