"""Configuration for the HumanEval+ pairwise pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .core import ReproConfig, ensure_dir


BASE_ROOT = Path("/workspace")
DEFAULT_ROOT_DIR = Path(os.environ.get("GORSA_ROOT_DIR", BASE_ROOT / "HumanEval+Llama3"))


def default_config() -> ReproConfig:
    return ReproConfig(
        root_dir=str(DEFAULT_ROOT_DIR),
        model_id=os.environ.get("GORSA_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"),
        seed=int(os.environ.get("GORSA_SEED", "42")),
        limit=164,
        n_candidates=10,
        candidate_temperature=1.2,
        candidate_top_p=1.0,
        candidate_max_new_tokens=256,
        additional_instruction_temperature=float(os.environ.get("GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE", "0.0")),
        additional_instruction_top_p=float(os.environ.get("GORSA_ADDITIONAL_INSTRUCTION_TOP_P", "1.0")),
        additional_instruction_max_new_tokens=int(os.environ.get("GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS", "48")),
        score_batch_size=int(os.environ.get("GORSA_SCORE_BATCH_SIZE", "8")),
        generation_batch_size=int(os.environ.get("GORSA_GENERATION_BATCH_SIZE", "8")),
        eval_timeout_seconds=8,
        force_regenerate_candidates=False,
        force_rescore=False,
        force_recompute_instructions=False,
        force_recompute_l0=False,
    )


def save_run_config(config: ReproConfig) -> Path:
    root = ensure_dir(config.root_dir)
    path = root / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": config.to_dict()}, f, indent=2, ensure_ascii=False)
    return path


def load_config() -> ReproConfig:
    path = DEFAULT_ROOT_DIR / "run_config.json"
    if not path.exists():
        config = default_config()
        save_run_config(config)
        return config

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("config", payload)
    data["root_dir"] = str(DEFAULT_ROOT_DIR)
    if "GORSA_SCORE_BATCH_SIZE" in os.environ:
        data["score_batch_size"] = int(os.environ["GORSA_SCORE_BATCH_SIZE"])
    if "GORSA_GENERATION_BATCH_SIZE" in os.environ:
        data["generation_batch_size"] = int(os.environ["GORSA_GENERATION_BATCH_SIZE"])
    if "GORSA_SEED" in os.environ:
        data["seed"] = int(os.environ["GORSA_SEED"])
    if "GORSA_MODEL_ID" in os.environ:
        data["model_id"] = os.environ["GORSA_MODEL_ID"]
    if "GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE" in os.environ:
        data["additional_instruction_temperature"] = float(os.environ["GORSA_ADDITIONAL_INSTRUCTION_TEMPERATURE"])
    if "GORSA_ADDITIONAL_INSTRUCTION_TOP_P" in os.environ:
        data["additional_instruction_top_p"] = float(os.environ["GORSA_ADDITIONAL_INSTRUCTION_TOP_P"])
    if "GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS" in os.environ:
        data["additional_instruction_max_new_tokens"] = int(os.environ["GORSA_ADDITIONAL_INSTRUCTION_MAX_NEW_TOKENS"])
    return ReproConfig(**data)
