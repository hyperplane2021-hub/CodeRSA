import _bootstrap  # noqa: F401

from gorsa_pipeline.runtime import load_dataset_for_config, load_model_and_tokenizer, prepare_config
from gorsa_pipeline.stages import (
    compute_l0_matrices,
    compute_pairwise_results,
    evaluate_candidates,
    generate_candidates,
    generate_extra_instructions,
    init_tasks,
    pad_candidate_pool,
    print_candidate_eval_stats,
    score_baselines,
    write_report,
)


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)

    init_tasks(config, dataset)
    model, tokenizer = load_model_and_tokenizer(config)

    generate_candidates(config, model, tokenizer)
    pad_candidate_pool(config)
    evaluate_candidates(config, dataset)
    print_candidate_eval_stats(config)
    score_baselines(config, dataset, model, tokenizer)
    generate_extra_instructions(config, dataset, model, tokenizer)
    compute_l0_matrices(config, dataset, model, tokenizer)
    compute_pairwise_results(config, dataset)
    write_report(config)


if __name__ == "__main__":
    main()
