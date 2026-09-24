"""ECS task entry point: `python -m steam_etl` runs `dbt build` (staging -> marts + tests)."""

import logging
import sys

from steam_etl.config import EtlSettings, configure_logging
from steam_etl.runner import run_dbt

logger = logging.getLogger("steam_etl")


def main() -> int:
    settings = EtlSettings()
    configure_logging(settings.log_level)
    result = run_dbt(settings)
    if result.success:
        logger.info("dbt %s succeeded", result.command)
        return 0
    for failure in result.failures:
        logger.error("dbt failure: %s", failure)
    return 1


if __name__ == "__main__":
    sys.exit(main())
