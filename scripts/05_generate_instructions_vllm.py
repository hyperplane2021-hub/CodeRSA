import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from gorsa_pipeline.core import (
    build_additional_instruction_prompt,
    postprocess_generated_instruction,
    read_json,
    seed_everything,
    stable_int_from_task_id,
    write_json,
)
from gorsa_pipeline.runtime import configure_workspace_cache, get_hf_token
from gorsa_pipeline.settings import load_config


MAX_EXTRA_INSTRUCTION_TASKS = int(os.environ.get("GORSA_MAX_EXTRA_INSTRUCTION_TASKS", "0"))


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
    parser = argparse.ArgumentParser(description="Generate Stage 4 instructions with vLLM.")
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.88")))
    parser.add_argument("--task-batch-size", type=int, default=int(os.environ.get("VLLM_TASK_BATCH_SIZE", "32")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("VLLM_TRUST_REMOTE_CODE", "0") == "1")
    return parser.parse_args()


def main() -> None:
    configure_workspace_cache()

    args = parse_args()
    config = load_config()
    seed_everything(config.seed)
    token = get_hf_token(prompt=False)

    task_paths = sorted((Path(config.root_dir) / "tasks").glob("*.json"))
    if MAX_EXTRA_INSTRUCTION_TASKS:
        task_paths = task_paths[:MAX_EXTRA_INSTRUCTION_TASKS]
    pending_paths = []
    for path in task_paths:
        record = read_json(path)
        if record.get("additional_instructions") is None or config.force_recompute_instructions:
            pending_paths.append(path)

    print("root:", config.root_dir)
    print("model:", config.model_id)
    model_path = local_model_path(config.model_id)
    print("model path:", model_path)
    print("pending tasks:", len(pending_paths))
    print("temperature:", config.additional_instruction_temperature)
    print("top_p:", config.additional_instruction_top_p)
    print("max_new_tokens:", config.additional_instruction_max_new_tokens)
    if not pending_paths:
        print("Stage 4 vLLM complete.")
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
        temperature=config.additional_instruction_temperature,
        top_p=config.additional_instruction_top_p,
        max_tokens=config.additional_instruction_max_new_tokens,
        stop=["\n\n", "<|eot_id|>", "<|end_of_text|>"],
        seed=config.seed,
    )

    for start in tqdm(range(0, len(pending_paths), args.task_batch_size), desc="Stage 4 vLLM: generate extra instructions"):
        chunk_paths = pending_paths[start : start + args.task_batch_size]
        prompts = []
        offsets = []
        records = []

        for path in chunk_paths:
            record = read_json(path)
            candidates = record.get("candidates")
            if not candidates:
                print(f"[WARNING] candidates missing, skip task_id={record.get('task_id')}")
                continue
            records.append((path, record))
            seed_base = config.seed + 100000 + stable_int_from_task_id(record["task_id"])
            for cand_idx, cand in enumerate(candidates):
                if "scoring_code" not in cand:
                    print(f"[WARNING] scoring_code missing task={record['task_id']} cand={cand_idx}")
                    continue
                offsets.append((len(records) - 1, cand_idx, seed_base + cand_idx))
                prompts.append(build_additional_instruction_prompt(cand["scoring_code"]))

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        grouped_raw = [[] for _ in records]
        grouped_clean = [[] for _ in records]

        for output, (record_idx, _cand_idx, _seed) in zip(outputs, offsets):
            raw_text = output.outputs[0].text if output.outputs else ""
            clean_text = postprocess_generated_instruction(raw_text)
            grouped_raw[record_idx].append(raw_text)
            grouped_clean[record_idx].append(clean_text)

        for record_idx, (path, record) in enumerate(records):
            record["additional_instructions"] = {
                "original": record["text"],
                "generated": grouped_clean[record_idx],
                "all": [record["text"]] + grouped_clean[record_idx],
                "raw_generated": grouped_raw[record_idx],
                "generator": "vllm",
                "generation_config": {
                    "temperature": config.additional_instruction_temperature,
                    "top_p": config.additional_instruction_top_p,
                    "max_new_tokens": config.additional_instruction_max_new_tokens,
                    "tensor_parallel_size": args.tensor_parallel_size,
                },
            }
            write_json(record, path)

    print("Stage 4 vLLM complete.")


if __name__ == "__main__":
    main()
