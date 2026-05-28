"""Runtime helpers shared by pipeline stages."""

from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import load_mbpp_test_split, seed_everything
from .settings import load_config, save_run_config


def load_workspace_env() -> None:
    workspace = Path(os.environ.get("WORKSPACE", "/root/workspace"))
    env_path = workspace / ".env"
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


def configure_workspace_cache() -> None:
    workspace = Path(os.environ.get("WORKSPACE", "/root/workspace"))
    tmp_dir = workspace / "tmp"
    hf_home = workspace / "hf_home"
    cache_dir = workspace / ".cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir / "triton"))


def get_hf_token(prompt: bool = True) -> str | None:
    configure_workspace_cache()
    load_workspace_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and prompt:
        token = getpass("HF token (press Enter to skip): ").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        return token
    return None


def load_dataset_for_config(config=None):
    configure_workspace_cache()
    config = config or load_config()
    return load_mbpp_test_split(limit=config.limit)


def load_model_and_tokenizer(config=None, prompt_for_token: bool = True):
    configure_workspace_cache()
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
    configure_workspace_cache()
    config = load_config()
    Path(config.root_dir).mkdir(parents=True, exist_ok=True)
    save_run_config(config)
    return config
