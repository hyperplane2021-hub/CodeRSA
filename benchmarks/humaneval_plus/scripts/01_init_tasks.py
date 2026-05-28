import _bootstrap  # noqa: F401

from gorsa_pipeline.runtime import load_dataset_for_config, prepare_config
from gorsa_pipeline.stages import init_tasks


def main() -> None:
    config = prepare_config()
    dataset = load_dataset_for_config(config)
    init_tasks(config, dataset)


if __name__ == "__main__":
    main()
