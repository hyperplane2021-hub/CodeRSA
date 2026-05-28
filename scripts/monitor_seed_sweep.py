#!/usr/bin/env python
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


def load_records(root: Path):
    task_dir = root / "tasks"
    rows = []
    if not task_dir.exists():
        return rows
    for path in sorted(task_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            rows.append({"_error": True})
    return rows


def stage_counts(records):
    counts = {key: 0 for key, _label in FIELDS}
    candidate_rows = 0
    eval_rows = 0
    passed = 0
    oracle = 0
    for rec in records:
        if rec.get("_error"):
            continue
        for key, _label in FIELDS:
            if rec.get(key) is not None:
                counts[key] += 1
        candidates = rec.get("candidates") or []
        evals = rec.get("candidate_eval") or []
        candidate_rows += len(candidates)
        eval_rows += len(evals)
        passed += sum(bool(x.get("passed")) for x in evals)
        oracle += int(any(bool(x.get("passed")) for x in evals))
    return counts, candidate_rows, eval_rows, passed, oracle


def read_summary(root: Path):
    path = root / "summary_pairwise_avg.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def process_lines():
    try:
        out = subprocess.check_output(
            "ps -eo pid,pcpu,pmem,etime,cmd | rg 'run_full_vllm_mbpp|02_generate|03_evaluate|04_score|05_generate|06_compute|07_pairwise|08_report' | rg -v 'rg |monitor_seed_sweep' || true",
            shell=True,
            text=True,
        ).strip()
    except Exception:
        return []
    return out.splitlines() if out else []


def gpu_lines():
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
        return [f"GPU unavailable: {e}"]
    lines = []
    for raw in out.strip().splitlines():
        idx, name, used, total, util = [x.strip() for x in raw.split(",")]
        lines.append(f"GPU {idx} {name}: {used}/{total} MiB util {util}%")
    return lines


def pct(done, total):
    if not total:
        return "  0.0%"
    return f"{done / total * 100:5.1f}%"


def render(roots):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"GoRSA MBPP+ seed sweep monitor | {now}", ""]
    header = (
        "seed  tasks  cand     eval     score    instr    L0       pair     "
        "pass/oracle      best      lambda"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for root in roots:
        seed = root.name.rsplit("seed", 1)[-1]
        records = load_records(root)
        total = len(records)
        counts, cand_rows, eval_rows, passed, oracle = stage_counts(records)
        summary = read_summary(root)
        best = ""
        lam = ""
        if summary:
            best_obj = summary.get("pairwise_avg_best") or {}
            best = f"{best_obj.get('accuracy', ''):.4f}" if isinstance(best_obj.get("accuracy"), (int, float)) else ""
            lam = str(best_obj.get("lambda", ""))

        score_done = min(counts["coder_logprobs"], counts["reviewer_logprobs"], counts["prior_logprobs"])
        pass_ratio = passed / eval_rows if eval_rows else 0.0
        oracle_ratio = oracle / total if total else 0.0
        lines.append(
            f"{seed:>4}  {total:>5}  "
            f"{counts['candidates']:>3}/{total:<3} {pct(counts['candidates'], total):>7}  "
            f"{counts['candidate_eval']:>3}/{total:<3} {pct(counts['candidate_eval'], total):>7}  "
            f"{score_done:>3}/{total:<3} {pct(score_done, total):>7}  "
            f"{counts['additional_instructions']:>3}/{total:<3} {pct(counts['additional_instructions'], total):>7}  "
            f"{counts['l0_logprobs']:>3}/{total:<3} {pct(counts['l0_logprobs'], total):>7}  "
            f"{counts['results_pairwise_avg']:>3}/{total:<3} {pct(counts['results_pairwise_avg'], total):>7}  "
            f"{pass_ratio:.3f}/{oracle_ratio:.3f}  {best:>8}  {lam:>6}"
        )

    lines.extend(["", "GPU:"])
    lines.extend("  " + line for line in gpu_lines())

    procs = process_lines()
    lines.extend(["", "Running processes:"])
    lines.extend(("  " + line for line in procs) if procs else ["  none"])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor MBPP+ seed sweep progress.")
    parser.add_argument("roots", nargs="*", help="Run root directories.")
    workspace = os.environ.get("WORKSPACE", "/workspace")
    parser.add_argument("--root-template", default=f"{workspace}/runs/codersa_mbpp_llama3_limit378_seed{{seed}}")
    parser.add_argument("--seeds", default="43,44,45,46")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.roots:
        roots = [Path(x) for x in args.roots]
    else:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
        roots = [Path(args.root_template.format(seed=seed)) for seed in seeds]

    if not args.watch:
        print(render(roots), flush=True)
        return

    while True:
        print("\033[2J\033[H", end="")
        print(render(roots), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
