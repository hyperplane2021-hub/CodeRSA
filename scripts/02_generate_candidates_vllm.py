import argparse
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from gorsa_pipeline.core import (
    CODE_STOP_WORDS,
    clean_code_generation,
    get_first_top_level_function_name,
    mask_first_function_name_ast,
    read_json,
    seed_everything,
    stable_int_from_task_id,
    try_extract_first_function_from_text,
    write_json,
)
from gorsa_pipeline.stages import pad_candidate_pool
from gorsa_pipeline.runtime import configure_workspace_cache, get_hf_token
from gorsa_pipeline.settings import load_config


def local_model_path(model_id: str) -> str:
    override = os.environ.get("GORSA_MODEL_LOCAL_PATH")
    if override:
        return override

    cache_roots = [
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/workspace/hf_cache/hub")),
        Path(os.environ.get("HF_HOME", "/workspace/hf_cache")) / "hub",
        Path(os.environ.get("HF_HOME", "/workspace/hf_cache")),
    ]
    for cache_root in cache_roots:
        repo_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if repo_dir.exists():
            snapshots = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for snapshot in snapshots:
                if (snapshot / "config.json").exists() and (
                    (snapshot / "model.safetensors.index.json").exists()
                    or (snapshot / "model.safetensors").exists()
                ):
                    return str(snapshot)
    return model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate code for MBPP+ tasks with vLLM.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=int(os.environ.get("VLLM_TASK_BATCH_SIZE", "16")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.88")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("VLLM_TRUST_REMOTE_CODE", "0") == "1")
    return parser.parse_args()


def candidate_sample_count(config) -> int:
    override = os.environ.get("GORSA_CANDIDATE_OVERSAMPLE")
    if override is None:
        return config.n_candidates
    return max(config.n_candidates, int(override))


def candidate_temperature(config) -> float:
    override = os.environ.get("GORSA_CANDIDATE_TEMPERATURE")
    return float(override) if override is not None else config.candidate_temperature


def candidate_top_p(config) -> float:
    override = os.environ.get("GORSA_CANDIDATE_TOP_P")
    return float(override) if override is not None else config.candidate_top_p


def main() -> None:
    configure_workspace_cache()
    get_hf_token(prompt=False)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = parse_args()
    config = load_config()
    seed_everything(config.seed)

    model_path = local_model_path(config.model_id)
    sample_count = candidate_sample_count(config)
    temperature = candidate_temperature(config)
    top_p = candidate_top_p(config)
    task_paths = sorted((Path(config.root_dir) / "tasks").glob("*.json"))
    if args.shard_count > 1:
        task_paths = [path for idx, path in enumerate(task_paths) if idx % args.shard_count == args.shard_index]

    pending_paths = []
    for path in task_paths:
        record = read_json(path)
        if record.get("candidates") is None or config.force_regenerate_candidates:
            pending_paths.append(path)

    print("root:", config.root_dir)
    print("model:", config.model_id)
    print("model path:", model_path)
    print(f"candidate shard {args.shard_index}/{args.shard_count}: {len(task_paths)} task files")
    print("pending tasks:", len(pending_paths))
    print("n candidates:", config.n_candidates)
    print("raw samples per task:", sample_count)
    print("temperature:", temperature)
    print("top_p:", top_p)
    print("max_new_tokens:", config.candidate_max_new_tokens)
    if not pending_paths:
        print("Stage 1 vLLM complete.")
        return

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

    sampling_params = SamplingParams(
        n=sample_count,
        temperature=temperature,
        top_p=top_p,
        max_tokens=config.candidate_max_new_tokens,
        stop=CODE_STOP_WORDS,
        seed=config.seed + args.shard_index,
    )

    for start in tqdm(range(0, len(pending_paths), args.task_batch_size), desc="Stage 1 vLLM: generate candidates"):
        chunk_paths = pending_paths[start : start + args.task_batch_size]
        records = [read_json(path) for path in chunk_paths]
        prompts = [record["generation_prompt"] for record in records]
        if os.environ.get("GORSA_VLLM_PER_TASK_SEED", "0") == "1":
            per_task_params = [
                SamplingParams(
                    n=sample_count,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=config.candidate_max_new_tokens,
                    stop=CODE_STOP_WORDS,
                    seed=config.seed + stable_int_from_task_id(record["task_id"]),
                )
                for record in records
            ]
            outputs = llm.generate(prompts, per_task_params, use_tqdm=False)
        else:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for path, record, output in zip(chunk_paths, records, outputs):
            candidates = []
            seen_exec_code = set()
            raw_outputs = [item.text for item in output.outputs]

            for raw_text in raw_outputs:
                raw_code = clean_code_generation(raw_text)
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
                        "raw_code": raw_text,
                        "exec_code": exec_code,
                        "scoring_code": scoring_code,
                        "original_function_name": original_function_name,
                        "generator": "vllm",
                        "generation_config": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "raw_samples": sample_count,
                        },
                    }
                )
                if len(candidates) >= config.n_candidates:
                    break

            if len(candidates) < config.n_candidates:
                print(
                    f"[WARNING] task {record['task_id']} only kept {len(candidates)} candidates "
                    f"out of {len(raw_outputs)} raw generations"
                )

            record["candidates"] = candidates
            write_json(record, path)

    if args.shard_count == 1:
        pad_candidate_pool(config)

    print("Stage 1 vLLM complete.")


if __name__ == "__main__":
    main()
