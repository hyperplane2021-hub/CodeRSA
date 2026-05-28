import _bootstrap  # noqa: F401

from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.stages import compute_pairwise_results


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    compute_pairwise_results(config, dataset)


if __name__ == "__main__":
    main()
