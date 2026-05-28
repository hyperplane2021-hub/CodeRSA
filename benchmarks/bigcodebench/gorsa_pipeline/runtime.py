"""Runtime helpers shared by pipeline stages."""

from __future__ import annotations

import os
import json
from getpass import getpass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import seed_everything
from .settings import load_config, save_run_config


def load_workspace_env() -> None:
    env_path = Path("/workspace/.env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def get_hf_token(prompt: bool = True) -> str | None:
    load_workspace_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and prompt:
        token = getpass("HF token (press Enter to skip): ").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        return token
    return None


def load_dataset_for_config(config=None):
    config = config or load_config()
    task_dir = Path(config.root_dir) / "tasks"
    records = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append({"task_id": str(data["task_id"])})
        except Exception:
            continue
    if not records:
        raise RuntimeError(f"No BigCodeBench task files found under {task_dir}")
    if config.limit is not None:
        records = records[: config.limit]
    return records


def load_model_and_tokenizer(config=None, prompt_for_token: bool = True):
    config = config or load_config()
    seed_everything(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    token = get_hf_token(prompt=prompt_for_token)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto",
        token=token,
    )
    model.eval()
    return model, tokenizer


def prepare_config():
    config = load_config()
    Path(config.root_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(config)
    return config
