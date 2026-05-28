import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

from gorsa_pipeline.core import (
    build_additional_instruction_prompt,
    is_unusable_generated_instruction,
    postprocess_generated_instruction,
    read_json,
    seed_everything,
    stable_int_from_task_id,
    write_json,
)
from gorsa_pipeline.runtime import get_hf_token
from gorsa_pipeline.settings import load_config


MAX_EXTRA_INSTRUCTION_TASKS = int(os.environ.get("GORSA_MAX_EXTRA_INSTRUCTION_TASKS", "1000000"))


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
        if snapshots:
            print(f"[WARNING] cached snapshots for {model_id} have no weight files; using repo id to download weights")
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
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.cache/triton")

    args = parse_args()
    config = load_config()
    seed_everything(config.seed)
    token = get_hf_token(prompt=False)

    task_paths = sorted((Path(config.root_dir) / "tasks").glob("*.json"))[:MAX_EXTRA_INSTRUCTION_TASKS]
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
            clean_generations = [
                record["text"] if is_unusable_generated_instruction(text) else text
                for text in grouped_clean[record_idx]
            ]
            record["additional_instructions"] = {
                "original": record["text"],
                "generated": clean_generations,
                "all": [record["text"]] + clean_generations,
                "raw_generated": grouped_raw[record_idx],
                "generator": "vllm",
            }
            write_json(record, path)

    print("Stage 4 vLLM complete.")


if __name__ == "__main__":
    main()
