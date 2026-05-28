"""Stage implementations for the MBPP+ pairwise pipeline."""

from __future__ import annotations

import ast
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .core import (
    CODE_STOP_WORDS,
    MIN_OVERSAMPLE,
    OVERSAMPLE_FACTOR,
    build_additional_instruction_prompt,
    build_coder_prefix,
    build_reviewer_prefix,
    candidate_passes_mbpp,
    clean_code_generation,
    compute_task_results_pairwise_avg,
    generate_one_each,
    get_first_top_level_function_name,
    initialize_task_files_mbpp,
    mask_first_function_name_ast,
    normalize_mbpp_doc,
    postprocess_generated_instruction,
    read_json,
    sample_n_completions,
    stable_int_from_task_id,
    sum_logprobs_texts,
    summarize_run_pairwise_avg,
    task_path,
    try_extract_first_function_from_text,
    write_json,
)


PAIRWISE_LAMBDA_VALUES = [round(x, 2) for x in np.arange(0.0, 3.05, 0.1)]
PAIRWISE_TIE_MARGIN = 0.0
MAX_EXTRA_INSTRUCTION_TASKS = int(os.environ.get("GORSA_MAX_EXTRA_INSTRUCTION_TASKS", "0"))


def init_tasks(config, dataset) -> None:
    initialize_task_files_mbpp(config, dataset)
    print("num tasks in dataset =", len(dataset))
    print("task files written to:", Path(config.root_dir) / "tasks")


def generate_candidates(config, model, tokenizer, shard_index: int = 0, shard_count: int = 1) -> None:
    task_paths = sorted((Path(config.root_dir) / "tasks").glob("*.json"))
    if shard_count > 1:
        task_paths = [path for idx, path in enumerate(task_paths) if idx % shard_count == shard_index]
        print(f"candidate shard {shard_index}/{shard_count}: {len(task_paths)} task files")
    print("num task files =", len(task_paths))

    for path in tqdm(task_paths, desc="Stage 1: generate candidates"):
        record = read_json(path)
        if record.get("candidates") is not None and not config.force_regenerate_candidates:
            continue

        prompt = record["generation_prompt"]
        seed = config.seed + stable_int_from_task_id(record["task_id"])

        raw_candidates = sample_n_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=max(config.n_candidates * OVERSAMPLE_FACTOR, MIN_OVERSAMPLE),
            temperature=config.candidate_temperature,
            top_p=config.candidate_top_p,
            max_new_tokens=config.candidate_max_new_tokens,
            stop_words=CODE_STOP_WORDS,
            seed=seed,
        )

        candidates = []
        seen_exec_code = set()
        for raw_code in raw_candidates:
            exec_code = try_extract_first_function_from_text(clean_code_generation(raw_code))
            if not exec_code.strip():
                continue

            try:
                ast.parse(exec_code)
            except Exception:
                continue

            norm_code = exec_code.strip()
            if norm_code in seen_exec_code:
                continue
            seen_exec_code.add(norm_code)

            original_function_name = get_first_top_level_function_name(exec_code)
            try:
                scoring_code = mask_first_function_name_ast(exec_code, new_name="f")
            except Exception:
                scoring_code = exec_code

            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "raw_code": raw_code,
                    "exec_code": exec_code,
                    "scoring_code": scoring_code,
                    "original_function_name": original_function_name,
                }
            )
            if len(candidates) >= config.n_candidates:
                break

        if len(candidates) < config.n_candidates:
            print(
                f"[WARNING] task {record['task_id']} only kept {len(candidates)} candidates "
                f"out of {len(raw_candidates)} raw generations"
            )

        record["candidates"] = candidates
        write_json(record, path)

    print("Stage 1 complete.")


def pad_candidate_pool(config) -> None:
    task_paths = sorted((Path(config.root_dir) / "tasks").glob("*.json"))
    num_fixed = 0
    num_total = 0

    for path in task_paths:
        record = read_json(path)
        candidates = record.get("candidates")
        if not candidates:
            continue

        num_total += 1
        target = config.n_candidates
        if len(candidates) >= target:
            continue

        filled = candidates + random.choices(candidates, k=target - len(candidates))
        for i, cand in enumerate(filled):
            cand["candidate_id"] = i

        record["candidates"] = filled
        write_json(record, path)
        num_fixed += 1

    print("tasks processed:", num_total)
    print("tasks padded:", num_fixed)
    print("target size:", config.n_candidates)


def evaluate_candidates(config, dataset) -> None:
    for raw in tqdm(dataset, desc="Stage 2: evaluate MBPP+ candidates"):
        doc = normalize_mbpp_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        record = read_json(path)

        if record.get("candidate_eval") is not None and not config.force_rescore:
            continue

        assert record["candidates"] is not None, f"Missing candidates for task {doc['task_id']}"
        evals = []
        for cand_idx, cand in enumerate(record["candidates"]):
            passed, stderr = candidate_passes_mbpp(
                code=cand["exec_code"],
                test_setup_code=record["test_setup_code"],
                test_list=record["test_list"],
                timeout_seconds=config.eval_timeout_seconds,
            )
            evals.append({"candidate_id": cand_idx, "passed": bool(passed), "stderr": stderr})

        record["candidate_eval"] = evals
        write_json(record, path)

    print("Stage 2 complete.")


def print_candidate_eval_stats(config) -> None:
    task_paths = sorted((Path(config.root_dir) / "tasks").glob("*.json"))
    num_tasks = 0
    num_candidates = 0
    num_passed = 0
    num_tasks_with_any_pass = 0
    num_first_candidate_pass = 0

    for path in task_paths:
        record = read_json(path)
        evals = record.get("candidate_eval", [])
        if not evals:
            continue

        pass_flags = [x["passed"] for x in evals]
        num_tasks += 1
        num_candidates += len(pass_flags)
        num_passed += sum(pass_flags)
        num_tasks_with_any_pass += int(any(pass_flags))
        num_first_candidate_pass += int(pass_flags[0])

    if num_tasks == 0 or num_candidates == 0:
        print("No evaluated candidates found.")
        return

    print("num_tasks =", num_tasks)
    print("num_candidates =", num_candidates)
    print("candidate pass ratio =", round(num_passed / num_candidates, 4))
    print("first_candidate_pass@1 =", round(num_first_candidate_pass / num_tasks, 4))
    print("oracle@10 =", round(num_tasks_with_any_pass / num_tasks, 4))


def score_baselines(config, dataset, model, tokenizer) -> None:
    for raw in tqdm(dataset, desc="Stage 3: score Coder/Reviewer/prior"):
        doc = normalize_mbpp_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        if not Path(path).exists():
            print(f"[WARN] Missing record file for task {doc['task_id']} at {path}, skipping.")
            continue

        record = read_json(path)
        need_coder = record.get("coder_logprobs") is None or config.force_rescore
        need_reviewer = record.get("reviewer_logprobs") is None or config.force_rescore
        need_prior = record.get("prior_logprobs") is None or config.force_rescore
        if not (need_coder or need_reviewer or need_prior):
            continue

        candidates = record["candidates"]
        assert candidates is not None, f"Missing candidates for task {doc['task_id']}"
        for i, cand in enumerate(candidates):
            assert "scoring_code" in cand, f"Task {doc['task_id']} candidate {i} missing scoring_code"

        scoring_codes = [cand["scoring_code"] for cand in candidates]
        if need_coder:
            record["coder_logprobs"] = sum_logprobs_texts(
                model,
                tokenizer,
                [build_coder_prefix(record["text"]) for _ in scoring_codes],
                scoring_codes,
                batch_size=config.score_batch_size,
            )

        if need_reviewer:
            record["reviewer_logprobs"] = sum_logprobs_texts(
                model,
                tokenizer,
                [build_reviewer_prefix(code) for code in scoring_codes],
                [record["text"] for _ in scoring_codes],
                batch_size=config.score_batch_size,
            )

        if need_prior:
            record["prior_logprobs"] = sum_logprobs_texts(
                model,
                tokenizer,
                ["" for _ in scoring_codes],
                scoring_codes,
                batch_size=config.score_batch_size,
            )

        write_json(record, path)

    print("Stage 3 complete.")


def generate_extra_instructions(config, dataset, model, tokenizer) -> None:
    for idx, raw in enumerate(tqdm(dataset, desc="Stage 4: generate extra instructions")):
        if MAX_EXTRA_INSTRUCTION_TASKS and idx >= MAX_EXTRA_INSTRUCTION_TASKS:
            break

        doc = normalize_mbpp_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        if not Path(path).exists():
            print(f"[WARNING] task file missing, skip task_id={doc['task_id']}")
            continue

        record = read_json(path)
        if record.get("additional_instructions") is not None and not config.force_recompute_instructions:
            continue

        candidates = record.get("candidates")
        if candidates is None:
            print(f"[WARNING] candidates missing, skip task_id={doc['task_id']}")
            continue

        print("\n==============================")
        print("TASK:", doc["task_id"])
        print("ORIGINAL INSTRUCTION:")
        print(record["text"])
        print("--- GENERATED INSTRUCTIONS ---")

        prompts = []
        for cand_idx, cand in enumerate(candidates):
            if "scoring_code" not in cand:
                print(f"[WARNING] scoring_code missing task={doc['task_id']} cand={cand_idx}")
                continue
            prompts.append(build_additional_instruction_prompt(cand["scoring_code"]))

        raw_generations = []
        clean_generations = []
        seed_base = config.seed + 100000 + stable_int_from_task_id(doc["task_id"])

        instruction_mode = os.environ.get("GORSA_INSTRUCTION_MODE", "batched").strip().lower()
        if instruction_mode == "serial":
            for cand_idx, prompt in enumerate(prompts):
                raw_g = generate_one_each(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    temperature=config.additional_instruction_temperature,
                    top_p=config.additional_instruction_top_p,
                    max_new_tokens=config.additional_instruction_max_new_tokens,
                    stop_words=["\n\n", "<|eot_id|>", "<|end_of_text|>"],
                    batch_size=1,
                    seed=seed_base + cand_idx,
                    normalize_instruction=False,
                )[0]
                raw_generations.append(raw_g)
                clean_g = postprocess_generated_instruction(raw_g)
                clean_generations.append(clean_g)
                print(f"[{cand_idx}] {repr(clean_g)}")
        else:
            raw_generations = generate_one_each(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                temperature=config.additional_instruction_temperature,
                top_p=config.additional_instruction_top_p,
                max_new_tokens=config.additional_instruction_max_new_tokens,
                stop_words=["\n\n", "<|eot_id|>", "<|end_of_text|>"],
                batch_size=config.generation_batch_size,
                seed=seed_base,
                normalize_instruction=False,
            )

            for cand_idx, raw_g in enumerate(raw_generations):
                clean_g = postprocess_generated_instruction(raw_g)
                clean_generations.append(clean_g)
                print(f"[{cand_idx}] {repr(clean_g)}")

        print("==============================\n")
        record["additional_instructions"] = {
            "original": record["text"],
            "generated": clean_generations,
            "all": [record["text"]] + clean_generations,
            "raw_generated": raw_generations,
        }
        write_json(record, path)

    print("Stage 4 complete.")


def compute_l0_matrices(config, dataset, model, tokenizer) -> None:
    shard_count = int(os.environ.get("GORSA_TASK_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("GORSA_TASK_SHARD_INDEX", "0"))
    if shard_count < 1:
        raise ValueError("GORSA_TASK_SHARD_COUNT must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("GORSA_TASK_SHARD_INDEX must be in [0, GORSA_TASK_SHARD_COUNT)")

    if shard_count > 1:
        print(f"Stage 6 shard: {shard_index}/{shard_count}", flush=True)

    shard_rows = [raw for idx, raw in enumerate(dataset) if idx % shard_count == shard_index]
    for raw in tqdm(shard_rows, desc="Stage 6: compute L0 matrices"):
        doc = normalize_mbpp_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        if not Path(path).exists():
            print(f"[WARNING] task file missing, skip task_id={doc['task_id']}")
            continue

        record = read_json(path)
        if record.get("l0_logprobs") is not None and not config.force_recompute_l0:
            continue

        candidates = record.get("candidates")
        if candidates is None:
            print(f"[WARNING] candidates missing, skip task_id={doc['task_id']}")
            continue

        add_inst = record.get("additional_instructions")
        if add_inst is None:
            print(f"[WARNING] additional_instructions missing, skip task_id={doc['task_id']}")
            continue

        all_instructions = add_inst.get("all")
        if not all_instructions:
            print(f"[WARNING] instruction list empty, skip task_id={doc['task_id']}")
            continue

        matrix = [[0.0 for _ in all_instructions] for _ in candidates]
        prefix_texts = []
        suffix_texts = []
        offsets = []
        for cand_idx, cand in enumerate(candidates):
            if "scoring_code" not in cand:
                print(f"[WARNING] scoring_code missing task={doc['task_id']} cand={cand_idx}")
                continue

            for inst_idx, inst in enumerate(all_instructions):
                prefix_texts.append(build_coder_prefix(inst))
                suffix_texts.append(cand["scoring_code"])
                offsets.append((cand_idx, inst_idx))

        if len(prefix_texts) == 0:
            print(f"[WARNING] empty L0 matrix, skip task_id={doc['task_id']}")
            continue

        scores = sum_logprobs_texts(
            model=model,
            tokenizer=tokenizer,
            prefix_texts=prefix_texts,
            suffix_texts=suffix_texts,
            batch_size=config.score_batch_size,
        )
        for score, (cand_idx, inst_idx) in zip(scores, offsets):
            matrix[cand_idx][inst_idx] = score

        record["l0_logprobs"] = matrix
        write_json(record, path)

    print("Stage 6 complete.")


def compute_pairwise_results(config, dataset) -> None:
    for raw in tqdm(dataset, desc="Stage 7: compute MBPP+ pairwise+avg results"):
        doc = normalize_mbpp_doc(raw)
        path = task_path(config.root_dir, doc["task_id"])
        if not Path(path).exists():
            print(f"[WARNING] task file missing, skip task_id={doc['task_id']}")
            continue

        record = read_json(path)
        required_fields = [
            "candidates",
            "candidate_eval",
            "coder_logprobs",
            "reviewer_logprobs",
            "additional_instructions",
            "l0_logprobs",
        ]
        missing = [key for key in required_fields if record.get(key) is None]
        if missing:
            print(f"[WARNING] skip task {doc['task_id']} missing fields: {missing}")
            continue

        try:
            record["results_pairwise_avg"] = compute_task_results_pairwise_avg(
                record=record,
                lambda_values=PAIRWISE_LAMBDA_VALUES,
                tie_margin=PAIRWISE_TIE_MARGIN,
            )
        except Exception as e:
            print(f"[WARNING] pairwise+avg failed task={doc['task_id']} error={e}")
            continue

        write_json(record, path)

    summary_pairwise_avg = summarize_run_pairwise_avg(config.root_dir, PAIRWISE_LAMBDA_VALUES)
    summary_path = Path(config.root_dir) / "summary_pairwise_avg.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_pairwise_avg, f, indent=2, ensure_ascii=False)

    print("Saved:", summary_path)


def write_report(config) -> None:
    summary_path = Path(config.root_dir) / "summary_pairwise_avg.json"
    summary = json.load(open(summary_path, "r", encoding="utf-8"))

    curve_df = pd.DataFrame(summary["pairwise_avg_curve"])
    baseline_df = pd.DataFrame(
        [
            {"method": "Random", "accuracy": summary["random_acc"]},
            {"method": "Coder", "accuracy": summary["coder_acc"]},
            {"method": "CoderReviewer", "accuracy": summary["coderreviewer_acc"]},
            {"method": "Oracle@10", "accuracy": summary["oracle10"]},
            {"method": "Avg-all L0", "accuracy": summary["avg_all_l0_acc"]},
            {"method": "Pairwise only", "accuracy": summary["pairwise_only_acc"]},
            {"method": "Pairwise + Avg(lambda=1)", "accuracy": summary["pairwise_avg_lambda1"]["accuracy"]},
            {"method": "Pairwise + Avg(best lambda)", "accuracy": summary["pairwise_avg_best"]["accuracy"]},
        ]
    )

    baseline_path = Path(config.root_dir) / "baseline_pairwise_avg.csv"
    curve_path = Path(config.root_dir) / "pairwise_avg_curve.csv"
    baseline_df.to_csv(baseline_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    print(baseline_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.plot(curve_df["lambda"], curve_df["accuracy"], label="Pairwise + Avg-all")
    plt.axhline(summary["coder_acc"], linestyle="--", label="Coder")
    plt.axhline(summary["coderreviewer_acc"], linestyle="--", label="CoderReviewer")
    plt.axhline(summary["avg_all_l0_acc"], linestyle="--", label="Avg-all L0")
    plt.axhline(summary["pairwise_only_acc"], linestyle="--", label="Pairwise only")
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("MBPP+ pairwise+avg accuracy vs lambda")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = Path(config.root_dir) / "mbpp_pairwise_avg_sweep.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print("Baseline CSV saved to:", baseline_path)
    print("Curve CSV saved to:", curve_path)
    print("Plot saved to:", plot_path)
    print("Pairwise+avg lambda=1:", summary["pairwise_avg_lambda1"])
    print("Best pairwise+avg:", summary["pairwise_avg_best"])
