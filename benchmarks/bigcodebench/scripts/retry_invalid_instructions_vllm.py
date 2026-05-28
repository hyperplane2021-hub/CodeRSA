import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from gorsa_pipeline.core import (
    build_additional_instruction_prompt,
    build_coder_prefix,
    compute_task_results_pairwise_avg,
    is_unusable_generated_instruction,
    postprocess_generated_instruction,
    read_json,
    summarize_run_pairwise_avg,
    write_json,
)
from gorsa_pipeline.settings import load_config
from gorsa_pipeline.stages import PAIRWISE_LAMBDA_VALUES


def local_model_path(model_id: str) -> str:
    override = os.environ.get("GORSA_MODEL_LOCAL_PATH")
    if override:
        return override
    cache_root = Path(os.environ.get("HF_HOME", "/workspace/hf_cache"))
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if repo_dir.exists():
        snapshots = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snapshot in snapshots:
            has_weights = any(
                path.exists() and path.stat().st_size > 0
                for pattern in ("*.safetensors", "*.bin", "*.pt")
                for path in snapshot.glob(pattern)
            )
            if has_weights:
                return str(snapshot)
    return model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry unusable generated instructions and recompute affected L0.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.70")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--retry-batch-size", type=int, default=int(os.environ.get("GORSA_RETRY_INSTRUCTION_BATCH_SIZE", "64")))
    parser.add_argument("--l0-prompt-batch-size", type=int, default=int(os.environ.get("VLLM_L0_PROMPT_BATCH_SIZE", "32")))
    parser.add_argument("--retry-max-tokens", type=int, default=int(os.environ.get("GORSA_RETRY_INSTRUCTION_MAX_TOKENS", "96")))
    parser.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("VLLM_TRUST_REMOTE_CODE", "0") == "1")
    return parser.parse_args()


def prompt_ids_and_prefix_len(tokenizer, prefix: str, suffix: str) -> tuple[list[int], int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    full_ids = tokenizer(prefix + suffix, add_special_tokens=True).input_ids
    return full_ids, len(prefix_ids)


def prompt_score(output, full_ids: list[int], prefix_len: int) -> float:
    prompt_ids = list(output.prompt_token_ids)
    if prompt_ids != full_ids:
        full_ids = prompt_ids
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM output missing prompt_logprobs")
    total = 0.0
    for pos in range(prefix_len, len(full_ids)):
        choices = prompt_logprobs[pos]
        if not choices:
            continue
        token_id = full_ids[pos]
        item = choices.get(token_id)
        if item is None:
            raise RuntimeError(f"actual token id {token_id} missing from prompt_logprobs at pos {pos}")
        total += float(item.logprob)
    return total


def invalid_candidate_indices(record: dict) -> list[int]:
    add = record.get("additional_instructions") or {}
    original = str(add.get("original") or record.get("text") or "").strip()
    generated = add.get("generated") or []
    raw_generated = add.get("raw_generated") or []
    if isinstance(generated, str):
        generated = [generated]
    if isinstance(raw_generated, str):
        raw_generated = [raw_generated]

    out = []
    for idx, text in enumerate(generated):
        text_s = str(text or "").strip()
        raw_clean = postprocess_generated_instruction(raw_generated[idx]) if idx < len(raw_generated) else ""
        if text_s == original or is_unusable_generated_instruction(text_s) or is_unusable_generated_instruction(raw_clean):
            out.append(idx)
    return out


def rebuild_all_instructions(record: dict) -> None:
    add = record["additional_instructions"]
    original = add.get("original") or record["text"]
    generated = add.get("generated") or []
    if isinstance(generated, str):
        generated = [generated]
    add["original"] = original
    add["generated"] = generated
    add["all"] = [original] + [x for x in generated if isinstance(x, str) and x.strip()]


def recompute_l0_for_task(llm, tokenizer, sampling_params, path: Path, prompt_batch_size: int) -> None:
    record = read_json(path)
    candidates = record.get("candidates") or []
    add = record.get("additional_instructions") or {}
    all_instructions = add.get("all") or []
    if not candidates or not all_instructions:
        return

    prompts = []
    meta = []
    for cand_idx, cand in enumerate(candidates):
        suffix = cand.get("scoring_code")
        if suffix is None:
            continue
        for inst_idx, inst in enumerate(all_instructions):
            prefix = build_coder_prefix(inst)
            full_ids, prefix_len = prompt_ids_and_prefix_len(tokenizer, prefix, suffix)
            prompts.append(prefix + suffix)
            meta.append((cand_idx, inst_idx, full_ids, prefix_len))

    matrix = [[0.0 for _ in all_instructions] for _ in candidates]
    for start in range(0, len(prompts), prompt_batch_size):
        outputs = llm.generate(prompts[start : start + prompt_batch_size], sampling_params, use_tqdm=False)
        for output, (cand_idx, inst_idx, full_ids, prefix_len) in zip(outputs, meta[start : start + prompt_batch_size]):
            matrix[cand_idx][inst_idx] = prompt_score(output, full_ids, prefix_len)

    record["l0_logprobs"] = matrix
    write_json(record, path)


def write_summary_tables(root: Path, summary: dict) -> None:
    baseline_rows = [
        ("Random", summary["random_acc"]),
        ("Coder", summary["coder_acc"]),
        ("CoderReviewer", summary["coderreviewer_acc"]),
        ("Oracle@10", summary["oracle10"]),
        ("Avg-all L0", summary["avg_all_l0_acc"]),
        ("Pairwise only", summary["pairwise_only_acc"]),
        ("Pairwise + Avg(lambda=1)", summary["pairwise_avg_lambda1"]["accuracy"]),
        ("Pairwise + Avg(best lambda)", summary["pairwise_avg_best"]["accuracy"]),
    ]
    with open(root / "baseline_pairwise_avg_retry_invalid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy"])
        writer.writerows(baseline_rows)

    with open(root / "pairwise_avg_curve_retry_invalid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "accuracy"])
        for row in summary["pairwise_avg_curve"]:
            writer.writerow([row["lambda"], row["accuracy"]])


def main() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf_cache/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.cache/triton")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = parse_args()
    config = load_config()
    root = Path(config.root_dir)
    task_paths = sorted((root / "tasks").glob("*.json"))

    retry_items = []
    for path in task_paths:
        record = read_json(path)
        bad_indices = invalid_candidate_indices(record)
        if not bad_indices:
            continue
        for cand_idx in bad_indices:
            candidates = record.get("candidates") or []
            if cand_idx < len(candidates) and candidates[cand_idx].get("scoring_code"):
                retry_items.append((path, cand_idx, build_additional_instruction_prompt(candidates[cand_idx]["scoring_code"])))

    print("root:", root)
    print("model:", config.model_id)
    print("retry items:", len(retry_items))
    print("affected tasks:", len({p for p, _, _ in retry_items}))
    if not retry_items:
        return

    model_path = local_model_path(config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        download_dir=os.environ["HF_HOME"],
        enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
    )

    gen_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.retry_max_tokens,
        stop=["\n\n", "<|eot_id|>", "<|end_of_text|>"],
        seed=config.seed + 202606,
    )

    retry_by_path: dict[Path, list[tuple[int, str, str, bool]]] = {}
    for start in tqdm(range(0, len(retry_items), args.retry_batch_size), desc="Retry invalid instructions"):
        chunk = retry_items[start : start + args.retry_batch_size]
        outputs = llm.generate([x[2] for x in chunk], gen_params, use_tqdm=False)
        for (path, cand_idx, _prompt), output in zip(chunk, outputs):
            raw = output.outputs[0].text if output.outputs else ""
            clean = postprocess_generated_instruction(raw)
            valid = not is_unusable_generated_instruction(clean)
            retry_by_path.setdefault(path, []).append((cand_idx, raw, clean, valid))

    retry_stats = {"items": len(retry_items), "valid": 0, "invalid": 0, "affected_tasks": len(retry_by_path)}
    for path, items in retry_by_path.items():
        record = read_json(path)
        add = record["additional_instructions"]
        generated = add.get("generated") or []
        if isinstance(generated, str):
            generated = [generated]
        raw_retry = add.get("raw_generated_retry_invalid") or {}
        status = add.get("retry_invalid_status") or {}
        for cand_idx, raw, clean, valid in items:
            while cand_idx >= len(generated):
                generated.append(None)
            raw_retry[str(cand_idx)] = raw
            if valid:
                generated[cand_idx] = clean
                retry_stats["valid"] += 1
                status[str(cand_idx)] = "retry_valid"
            else:
                generated[cand_idx] = None
                retry_stats["invalid"] += 1
                status[str(cand_idx)] = "retry_invalid_skipped"
        add["generated"] = generated
        add["raw_generated_retry_invalid"] = raw_retry
        add["retry_invalid_status"] = status
        rebuild_all_instructions(record)
        record["l0_logprobs"] = None
        record["results_pairwise_avg"] = None
        write_json(record, path)

    print("retry stats:", retry_stats)
    logprob_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )

    for path in tqdm(sorted(retry_by_path), desc="Recompute affected L0"):
        recompute_l0_for_task(llm, tokenizer, logprob_params, path, args.l0_prompt_batch_size)

    for path in tqdm(task_paths, desc="Recompute pairwise results"):
        record = read_json(path)
        if record.get("l0_logprobs") is None:
            continue
        record["results_pairwise_avg"] = compute_task_results_pairwise_avg(record, PAIRWISE_LAMBDA_VALUES)
        write_json(record, path)

    summary = summarize_run_pairwise_avg(root, PAIRWISE_LAMBDA_VALUES)
    with open(root / "summary_pairwise_avg_retry_invalid.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_summary_tables(root, summary)
    with open(root / "retry_invalid_instruction_stats.json", "w", encoding="utf-8") as f:
        json.dump(retry_stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
