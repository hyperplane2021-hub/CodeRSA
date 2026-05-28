import _bootstrap  # noqa: F401

from gorsa_pipeline.runtime import prepare_config
from gorsa_pipeline.stages import write_report


def main() -> None:
    config = prepare_config()
    write_report(config)


if __name__ == "__main__":
    main()
