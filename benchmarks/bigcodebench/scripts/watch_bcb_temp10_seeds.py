#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


SEEDS = [44, 45, 46]
ROOT_TEMPLATE = "/workspace/BigCodeBench-hard-instruct-Llama3-70B_os20_temp10_seed{seed}"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count_jsonl(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    n = 0
    tasks = set()
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                tasks.add(json.loads(line)["task_id"])
            except Exception:
                pass
    return n, len(tasks)


def eval_count(path: Path) -> tuple[int, int]:
    data = read_json(path).get("eval") or {}
    return sum(len(v) for v in data.values()), len(data)


def latest_official_samples(results: Path) -> Path | None:
    candidates = [
        p
        for p in results.glob("*bigcodebench-hard-instruct*vllm-1.0-20*.jsonl")
        if not p.name.startswith("raw_n") and not p.name.startswith("filtered_n")
    ]
    if not candidates:
        candidates = [
            p
            for p in results.glob("*bigcodebench-hard-instruct*.jsonl")
            if not p.name.startswith("raw_n") and not p.name.startswith("filtered_n")
        ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def task_counts(root: Path) -> dict[str, int]:
    fields = ["candidate_eval", "coder_logprobs", "additional_instructions", "l0_logprobs", "results_pairwise_avg"]
    counts = {"tasks": 0, **{field: 0 for field in fields}}
    for p in (root / "tasks").glob("*.json"):
        counts["tasks"] += 1
        d = read_json(p)
        for field in fields:
            counts[field] += int(d.get(field) is not None)
    return counts


def bar(done: int, total: int, width: int = 18) -> str:
    if total <= 0:
        return "░" * width
    filled = round(width * min(1.0, max(0.0, done / total)))
    return "█" * filled + "░" * (width - filled)


def stage_progress(stage: str, raw: int, filt: int, ev: int, counts: dict[str, int]) -> tuple[int, int, str]:
    steps = [
        ("generate", raw, 2960),
        ("filter", filt, 1480),
        ("eval", ev, 1480),
        ("adapt", counts["candidate_eval"], 148),
        ("baseline", counts["coder_logprobs"], 148),
        ("instructions", counts["additional_instructions"], 148),
        ("L0", counts["l0_logprobs"], 148),
        ("pairwise", counts["results_pairwise_avg"], 148),
    ]
    done_units = 0
    total_units = len(steps) * 100
    active = stage
    for name, done, total in steps:
        pct = 100 if total and done >= total else int(100 * done / total) if total else 0
        done_units += max(0, min(100, pct))
        if pct < 100 and active in {"pending", "started", "generating", "filtered", "evaluated", "adapted", "baseline", "instructions", "L0/rerank"}:
            active = name
            break
    if stage == "complete":
        done_units = total_units
        active = "complete"
    elif stage == "pending":
        done_units = 0
        active = "pending"
    elif stage == "started":
        active = "starting"
    return done_units, total_units, active


def fmt_pct(x) -> str:
    return "-" if x is None else f"{100 * float(x):5.2f}%"


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def process_summary() -> str:
    out = run(["ps", "-eo", "pid,ppid,stat,pcpu,etime,cmd"])
    lines = []
    needles = ["run_bcb_temp10_seeds44_46", "run_bcb_temp10_seed", "bigcodebench.generate", "04_score", "05_generate", "06_compute"]
    for line in out.splitlines():
        if "watch_bcb_temp10_seeds" in line:
            continue
        if any(n in line for n in needles):
            lines.append(line[:140])
    return "\n".join("  " + line for line in lines[:8]) or "  none"


def gpu_summary() -> str:
    out = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, used, total, util = parts
        rows.append(f"GPU{idx} {int(used):5d}/{int(total):5d}MiB util {int(util):3d}%")
    return " | ".join(rows) if rows else "nvidia-smi unavailable"


def render(args: argparse.Namespace) -> str:
    lines = [
        "BigCodeBench hard/instruct | Llama3-70B | temp=1.0 | seeds 44-46",
        f"time {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    for seed in args.seeds:
        root = Path(ROOT_TEMPLATE.format(seed=seed))
        results = root / "bcb_results"
        raw_path = results / "raw_n20.jsonl"
        official_path = latest_official_samples(results)
        raw_n, _ = count_jsonl(raw_path if raw_path.exists() else official_path or raw_path)
        filt_n, _ = count_jsonl(results / "filtered_n10.jsonl")
        eval_n, _ = eval_count(results / "filtered_n10_eval_results.json")
        counts = task_counts(root)
        p = read_json(results / "filtered_n10_pass_at_k.json")
        s = read_json(root / "summary_pairwise_avg.json")
        if s:
            stage = "complete"
        elif counts["l0_logprobs"]:
            stage = "L0/rerank"
        elif counts["additional_instructions"]:
            stage = "instructions"
        elif counts["coder_logprobs"]:
            stage = "baseline"
        elif counts["tasks"]:
            stage = "adapted"
        elif eval_n:
            stage = "evaluated"
        elif filt_n:
            stage = "filtered"
        elif raw_n:
            stage = "generating"
        elif root.exists():
            stage = "started"
        else:
            stage = "pending"
        overall_done, overall_total, active = stage_progress(stage, raw_n, filt_n, eval_n, counts)
        summary = read_json(root / "summary_pairwise_avg.json")
        lines.append(f"seed {seed}  {active:<12} {bar(overall_done, overall_total, width=34)} {overall_done / overall_total * 100:5.1f}%")
        lines.append(
            f"  data   raw {raw_n:4d}/2960   filtered {filt_n:4d}/1480   eval {eval_n:4d}/1480"
        )
        lines.append(
            f"  gorsa  baseline {counts['coder_logprobs']:3d}/148   inst {counts['additional_instructions']:3d}/148   "
            f"L0 {counts['l0_logprobs']:3d}/148   pair {counts['results_pairwise_avg']:3d}/148"
        )
        lines.append(
            f"  pass   @1 {fmt_pct(p.get('pass@1')):>7}   @5 {fmt_pct(p.get('pass@5')):>7}   @10 {fmt_pct(p.get('pass@10')):>7}"
        )
        if summary:
            lines.append(
                "  final  "
                f"random {fmt_pct(summary.get('random_acc')):>7}   "
                f"CR {fmt_pct(summary.get('coderreviewer_acc')):>7}   "
                f"avg {fmt_pct(summary.get('avg_all_l0_acc')):>7}   "
                f"pair {fmt_pct(summary.get('pairwise_only_acc')):>7}   "
                f"best {fmt_pct((summary.get('pairwise_avg_best') or {}).get('accuracy')):>7}"
            )
        lines.append("")
    lines += ["GPU", "  " + gpu_summary(), "", "Processes", process_summary(), "", "Ctrl-C to quit."]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()
    while True:
        print("\033[2J\033[H" + render(args), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
