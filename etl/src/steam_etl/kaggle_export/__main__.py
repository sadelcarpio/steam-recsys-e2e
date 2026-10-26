"""ECS entry point of the `kaggle-export` task (spec 16), started by the pipeline after
`Transform` without waiting for it. Exit 0 = published, disabled or not due; 1 = failed."""

import logging
import sys

import boto3

from steam_etl.config import configure_logging
from steam_etl.kaggle_export.athena import Athena
from steam_etl.kaggle_export.export import run_export
from steam_etl.kaggle_export.publisher import KagglePublisher
from steam_etl.kaggle_export.settings import KaggleExportSettings

logger = logging.getLogger("steam_etl.kaggle_export")


def main() -> int:
    settings = KaggleExportSettings()
    configure_logging(settings.log_level)
    session = boto3.session.Session(region_name=settings.aws_region)

    def publisher() -> KagglePublisher:
        token = session.client("secretsmanager").get_secret_value(
            SecretId=settings.kaggle_token_secret_id
        )["SecretString"]
        return KagglePublisher(
            settings.kaggle_dataset, token.strip(), config_dir=f"{settings.work_dir}/.kaggle"
        )

    try:
        outcome = run_export(
            settings,
            s3=session.client("s3"),
            athena=Athena(
                session.client("athena"),
                settings.athena_work_group,
                poll_seconds=settings.athena_poll_seconds,
            ),
            publisher_factory=publisher,
        )
    except Exception as err:  # the message never contains the token
        logger.error("Kaggle export failed: %s: %s", type(err).__name__, err)
        return 1
    logger.info("Kaggle export: %s", outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
