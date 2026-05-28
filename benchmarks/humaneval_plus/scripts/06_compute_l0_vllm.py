import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from gorsa_pipeline.core import build_coder_prefix, read_json, seed_everything, write_json
from gorsa_pipeline.runtime import load_dataset_for_config
from gorsa_pipeline.settings import load_config


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
    parser = argparse.ArgumentParser(description="Compute Stage 6 L0 matrices with vLLM prompt logprobs.")
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.88")))
    parser.add_argument("--prompt-batch-size", type=int, default=int(os.environ.get("VLLM_L0_PROMPT_BATCH_SIZE", "64")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")))
    parser.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("VLLM_TRUST_REMOTE_CODE", "0") == "1")
    parser.add_argument("--force", action="store_true", help="Recompute l0_logprobs even when present.")
    return parser.parse_args()


def prefix_len_for_pair(tokenizer, prefix: str, suffix: str) -> tuple[list[int], int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    full_ids = tokenizer(prefix + suffix, add_special_tokens=True).input_ids
    return full_ids, len(prefix_ids)


def prompt_score(output, full_ids: list[int], prefix_len: int) -> float:
    prompt_ids = list(output.prompt_token_ids)
    if prompt_ids != full_ids:
        # vLLM and transformers should agree for local HF tokenizers. If they
        # ever differ, score by vLLM's own token positions and keep the same
        # prefix length convention computed from the tokenizer.
        full_ids = prompt_ids

    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM output missing prompt_logprobs")
    if len(prompt_logprobs) != len(full_ids):
        raise RuntimeError(f"prompt_logprobs length mismatch: {len(prompt_logprobs)} vs {len(full_ids)}")

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


def main() -> None:
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf_cache/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.cache/triton")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = parse_args()
    config = load_config()
    dataset = load_dataset_for_config(config)
    seed_everything(config.seed)

    model_path = local_model_path(config.model_id)
    print("root:", config.root_dir)
    print("model:", config.model_id)
    print("model path:", model_path)
    print("prompt batch size:", args.prompt_batch_size)

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
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )

    root = Path(config.root_dir)
    task_paths = []
    for raw in dataset:
        task_id = raw["task_id"]
        path = root / "tasks" / f"{task_id.replace('/', '_')}.json"
        if path.exists():
            task_paths.append(path)

    for path in tqdm(task_paths, desc="Stage 6 vLLM: compute L0 matrices"):
        record = read_json(path)
        if record.get("l0_logprobs") is not None and not (args.force or config.force_recompute_l0):
            continue

        candidates = record.get("candidates")
        add_inst = record.get("additional_instructions")
        if not candidates or not add_inst:
            print(f"[WARNING] skip {record.get('task_id')} missing candidates/additional_instructions")
            continue
        all_instructions = add_inst.get("all") or []
        if not all_instructions:
            print(f"[WARNING] skip {record.get('task_id')} empty instructions")
            continue

        prompts: list[str] = []
        metadata: list[tuple[int, int, list[int], int]] = []
        for cand_idx, cand in enumerate(candidates):
            suffix = cand.get("scoring_code")
            if suffix is None:
                print(f"[WARNING] skip candidate without scoring_code task={record.get('task_id')} cand={cand_idx}")
                continue
            for inst_idx, inst in enumerate(all_instructions):
                prefix = build_coder_prefix(inst)
                full_ids, prefix_len = prefix_len_for_pair(tokenizer, prefix, suffix)
                prompts.append(prefix + suffix)
                metadata.append((cand_idx, inst_idx, full_ids, prefix_len))

        matrix = [[0.0 for _ in all_instructions] for _ in candidates]
        for start in range(0, len(prompts), args.prompt_batch_size):
            chunk_prompts = prompts[start : start + args.prompt_batch_size]
            chunk_meta = metadata[start : start + args.prompt_batch_size]
            outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)
            for output, (cand_idx, inst_idx, full_ids, prefix_len) in zip(outputs, chunk_meta):
                matrix[cand_idx][inst_idx] = prompt_score(output, full_ids, prefix_len)

        record["l0_logprobs"] = matrix
        write_json(record, path)

    print("Stage 6 vLLM complete.")


if __name__ == "__main__":
    main()
