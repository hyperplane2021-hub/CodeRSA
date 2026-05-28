import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from monitor_progress import FIELDS, bar, gpu_lines, load_records, percent, summarize


STAGE_WEIGHTS = {
    "candidates": 1.0,
    "candidate_eval": 1.0,
    "coder_logprobs": 1.0 / 3.0,
    "reviewer_logprobs": 1.0 / 3.0,
    "prior_logprobs": 1.0 / 3.0,
    "additional_instructions": 1.0,
    "l0_logprobs": 1.0,
    "results_pairwise_avg": 1.0,
}


def root_for_seed(seed: int) -> Path:
    template = os.environ.get("GORSA_SWEEP_ROOT_TEMPLATE")
    if template:
        return Path(template.format(seed=seed))
    if seed == 42:
        return Path("/workspace/HumanEval+Llama3_vllm_os20_temp1")
    return Path(f"/workspace/HumanEval+Llama3_vllm_os20_temp1_seed{seed}")


def process_lines() -> list[str]:
    try:
        out = subprocess.check_output(
            "ps -eo pid,pcpu,pmem,etime,cmd | rg 'run_vllm_os20_temp1_seed|02_generate_candidates_vllm|03_evaluate_candidates|04_score_baselines|05_generate_instructions|06_compute_l0|07_pairwise|08_report|run_stage_logged' | rg -v 'rg |monitor_seed_sweep' || true",
            shell=True,
            text=True,
        ).strip()
    except Exception:
        return []
    return out.splitlines() if out else []


def run_summary(root: Path) -> dict | None:
    path = root / "summary_pairwise_avg.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def root_progress(summary: dict) -> float:
    total = summary["total"]
    if total <= 0:
        return 0.0
    done = 0.0
    max_done = 0.0
    for key, _label in FIELDS:
        weight = STAGE_WEIGHTS[key]
        done += summary["counts"][key] * weight
        max_done += total * weight
    return done / max_done if max_done else 0.0


def render_once(seeds: list[int]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = []
    total_progress = 0.0

    for seed in seeds:
        root = root_for_seed(seed)
        records = load_records(root / "tasks")
        summary = summarize(records)
        progress = root_progress(summary)
        total_progress += progress
        result = run_summary(root)
        best = ""
        lam1 = ""
        random_acc = ""
        if result:
            random_acc = f"{result.get('random_acc', 0):.4f}"
            lam = result.get("pairwise_avg_lambda1") or {}
            best_lam = result.get("pairwise_avg_best") or {}
            lam1 = f"{lam.get('accuracy', 0):.4f}"
            best = f"{best_lam.get('accuracy', 0):.4f}@{best_lam.get('lambda')}"
        rows.append((seed, root, summary, progress, random_acc, lam1, best))

    overall = total_progress / len(seeds) if seeds else 0.0
    lines = [
        f"GoRSA seed sweep monitor | {now}",
        f"seeds: {', '.join(str(s) for s in seeds)}",
        f"overall {bar(round(overall * 1000), 1000, width=36)} {overall * 100:5.1f}%",
        "",
        "Per-seed:",
        " seed  progress   cand eval score instr  L0 pair  rows evals pass   rand   lam1   best",
    ]

    for seed, _root, summary, progress, random_acc, lam1, best in rows:
        total = summary["total"]
        counts = summary["counts"]
        score_done = min(counts["coder_logprobs"], counts["reviewer_logprobs"], counts["prior_logprobs"])
        pass_ratio = summary["passed"] / summary["evaluated"] if summary["evaluated"] else 0.0
        lines.append(
            f" {seed:>4}  {progress*100:6.1f}%  "
            f"{counts['candidates']:4d} {counts['candidate_eval']:4d} {score_done:5d} "
            f"{counts['additional_instructions']:5d} {counts['l0_logprobs']:3d} {counts['results_pairwise_avg']:4d} "
            f"{summary['candidates_total']:5d} {summary['evaluated']:5d} {pass_ratio:5.3f} "
            f"{random_acc:>6} {lam1:>6} {best:>9}"
            f"  {percent(counts['results_pairwise_avg'], total)}"
        )

    lines.extend(["", "GPU:"])
    lines.extend("  " + line for line in gpu_lines())

    procs = process_lines()
    lines.extend(["", "Running processes:"])
    lines.extend(("  " + line for line in procs[:20]) if procs else ["  none detected"])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor the 42-46 seed sweep.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    if not args.watch:
        print(render_once(args.seeds), flush=True)
        return

    while True:
        print("\033[2J\033[H", end="")
        print(render_once(args.seeds), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
