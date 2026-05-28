import _bootstrap  # noqa: F401

import argparse

from gorsa_pipeline.runtime import load_model_and_tokenizer, prepare_config
from gorsa_pipeline.stages import generate_candidates, pad_candidate_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate code for HumanEval+ tasks.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-pad", action="store_true", help="Do not pad candidate pools after generation.")
    args = parser.parse_args()

    config = prepare_config()
    model, tokenizer = load_model_and_tokenizer(config)
    generate_candidates(
        config,
        model,
        tokenizer,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not args.skip_pad and args.shard_count == 1:
        pad_candidate_pool(config)


if __name__ == "__main__":
    main()
