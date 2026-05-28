import _bootstrap  # noqa: F401

from gorsa_pipeline.runtime import load_dataset_for_config, load_model_and_tokenizer, prepare_config
from gorsa_pipeline.stages import score_baselines


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    model, tokenizer = load_model_and_tokenizer(config)
    score_baselines(config, dataset, model, tokenizer)


if __name__ == "__main__":
    main()
