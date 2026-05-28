import _bootstrap  # noqa: F401

from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.stages import evaluate_candidates, print_candidate_eval_stats


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    evaluate_candidates(config, dataset)
    print_candidate_eval_stats(config)


if __name__ == "__main__":
    main()
