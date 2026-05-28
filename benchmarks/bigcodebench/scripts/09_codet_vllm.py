import _bootstrap  # noqa: F401

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from gorsa_pipeline.core import get_first_top_level_function_name, read_json, seed_everything, write_json
from gorsa_pipeline.settings import load_config


def local_model_path(model_id: str) -> str:
    override = os.environ.get("GORSA_MODEL_LOCAL_PATH")
    if override:
        return override
    cache_root = Path(os.environ.get("HF_HOME", "/workspace/hf_cache"))
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if repo_dir.exists():
        snapshots = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])
    return model_id


def build_test_prompt(record: dict, n_tests: int) -> str:
    entry = record["entry_point"]
    return textwrap.dedent(
        f"""\
        You are testing a Python function named {entry}.
        Write {n_tests} diverse assert statements that check the function's behavior.
        Use only Python literals as inputs and expected outputs.
        Output only assert statements, one per line.
        Do not include markdown, explanations, imports, or function definitions.

        Problem:
        {record["text"]}

        Function name: {entry}

        Assert statements:
        """
    )


def extract_asserts(text: str, entry_point: str, max_tests: int) -> list[str]:
    tests = []
    seen = set()
    for raw in text.replace("\r\n", "\n").splitlines():
        line = raw.strip()
        line = re.sub(r"^[-*]\s*", "", line)
        if line.startswith("```") or not line.startswith("assert "):
            continue
        if entry_point + "(" not in line:
            continue
        try:
            ast.parse(line)
        except SyntaxError:
            continue
        if line in seen:
            continue
        seen.add(line)
        tests.append(line)
        if len(tests) >= max_tests:
            break
    return tests


def run_candidate_on_asserts(record: dict, code: str, tests: list[str], timeout: int) -> tuple[int, tuple[str, ...]]:
    preamble = textwrap.dedent(
        """\
        import math
        import random
        import re
        import statistics
        import functools
        import itertools
        import collections
        import heapq
        import bisect
        from typing import *
        """
    )
    actual_name = get_first_top_level_function_name(code)
    alias = ""
    if actual_name and actual_name != record["entry_point"]:
        alias = f"{record['entry_point']} = {actual_name}"

    statuses = []
    for idx, test in enumerate(tests):
        parts = [preamble]
        if record.get("context_code", "").strip():
            parts.append(record["context_code"].strip())
        parts.append(code.rstrip())
        if alias:
            parts.append(alias)
        parts.append(test)
        script = "\n\n".join(parts) + "\n"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "codet_exec.py"
            path.write_text(script, encoding="utf-8")
            try:
                proc = subprocess.run(
                    ["python", str(path)],
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                statuses.append(f"{idx}:timeout")
                continue
        statuses.append(f"{idx}:pass" if proc.returncode == 0 else f"{idx}:fail")
    pass_count = sum(x.endswith(":pass") for x in statuses)
    return pass_count, tuple(statuses)


def choose_codet(record: dict, tests: list[str], timeout: int) -> dict:
    candidates = record.get("candidates") or []
    candidate_eval = record.get("candidate_eval") or []
    if not candidates or not tests:
        return {"selected_idx": None, "selected_passed": False, "reason": "missing candidates or tests"}

    rows = []
    for idx, cand in enumerate(candidates):
        code = cand.get("exec_code") or cand.get("raw_code") or ""
        pass_count, signature = run_candidate_on_asserts(record, code, tests, timeout)
        rows.append({"candidate_id": idx, "generated_tests_passed": pass_count, "signature": list(signature)})

    max_pass = max(row["generated_tests_passed"] for row in rows)
    best = [row for row in rows if row["generated_tests_passed"] == max_pass]
    signature_counts = Counter(tuple(row["signature"]) for row in rows)
    best.sort(key=lambda row: (signature_counts[tuple(row["signature"])], -row["candidate_id"]), reverse=True)
    selected_idx = best[0]["candidate_id"]
    selected_passed = bool(candidate_eval[selected_idx]["passed"]) if selected_idx < len(candidate_eval) else False
    return {
        "selected_idx": selected_idx,
        "selected_passed": selected_passed,
        "num_tests": len(tests),
        "max_generated_tests_passed": max_pass,
        "candidate_scores": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight CodeT baseline on existing candidates.")
    parser.add_argument("--num-tests", type=int, default=int(os.environ.get("CODET_NUM_TESTS", "20")))
    parser.add_argument("--raw-generations", type=int, default=int(os.environ.get("CODET_RAW_GENERATIONS", "4")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("CODET_TEMPERATURE", "0.8")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("CODET_TOP_P", "0.95")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("CODET_MAX_TOKENS", "512")))
    parser.add_argument("--task-batch-size", type=int, default=int(os.environ.get("CODET_TASK_BATCH_SIZE", "32")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("CODET_TIMEOUT", "3")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.88")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.cache/triton")

    args = parse_args()
    config = load_config()
    seed_everything(config.seed)
    root = Path(config.root_dir)
    task_paths = sorted((root / "tasks").glob("HumanEval_*.json"), key=lambda p: int(p.stem.split("_")[1]))

    pending = []
    for path in task_paths:
        rec = read_json(path)
        if args.force or rec.get("codet_tests") is None or rec.get("codet_result") is None:
            pending.append(path)

    print("root:", root)
    print("model:", config.model_id)
    print("pending tasks:", len(pending))
    print("num_tests:", args.num_tests)
    if pending:
        llm = LLM(
            model=local_model_path(config.model_id),
            tokenizer=local_model_path(config.model_id),
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            download_dir=os.environ["HF_HOME"],
            enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        )
        sampling_params = SamplingParams(
            n=args.raw_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=["\n\n\n", "```", "<|eot_id|>", "<|end_of_text|>"],
            seed=config.seed + 202405,
        )

        for start in tqdm(range(0, len(pending), args.task_batch_size), desc="CodeT: generate and score tests"):
            chunk = pending[start : start + args.task_batch_size]
            records = [read_json(path) for path in chunk]
            prompts = [build_test_prompt(rec, args.num_tests) for rec in records]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            for path, record, output in zip(chunk, records, outputs):
                tests = []
                for item in output.outputs:
                    tests.extend(extract_asserts(item.text, record["entry_point"], args.num_tests - len(tests)))
                    if len(tests) >= args.num_tests:
                        break
                result = choose_codet(record, tests, args.timeout)
                record["codet_tests"] = tests
                record["codet_result"] = result
                write_json(record, path)

    rows = []
    for path in task_paths:
        rec = read_json(path)
        result = rec.get("codet_result") or {}
        rows.append(
            {
                "task_id": rec.get("task_id"),
                "num_tests": len(rec.get("codet_tests") or []),
                "selected_idx": result.get("selected_idx"),
                "selected_passed": bool(result.get("selected_passed")),
            }
        )
    df = pd.DataFrame(rows)
    accuracy = float(df["selected_passed"].mean()) if len(df) else 0.0
    summary = {
        "num_tasks": int(len(df)),
        "accuracy": accuracy,
        "avg_num_tests": float(df["num_tests"].mean()) if len(df) else 0.0,
        "num_zero_test_tasks": int((df["num_tests"] == 0).sum()) if len(df) else 0,
    }
    df.to_csv(root / "codet_results.csv", index=False)
    with open(root / "codet_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("CodeT summary:", summary)
    print("saved:", root / "codet_results.csv")
    print("saved:", root / "codet_summary.json")


if __name__ == "__main__":
    main()
