"""SageMaker job entry point: `python -m steam_training {train,promote}`."""

from __future__ import annotations

import argparse
import logging

from steam_training.artifacts import ArtifactStore
from steam_training.config import TrainingSettings, configure_logging
from steam_training.data import IcebergSource
from steam_training.pipeline import run_promotion, run_training

log = logging.getLogger("steam_training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="steam_training")
    parser.add_argument("mode", choices=["train", "promote"])
    args = parser.parse_args(argv)

    settings = TrainingSettings()
    configure_logging(settings.log_level)
    source = IcebergSource(settings.glue_database, settings.aws_region)
    store = ArtifactStore(settings.model_artifacts_bucket)
    if args.mode == "train":
        metadata = run_training(settings, source, store)
        log.info("trained %s: %s", metadata.model_id, metadata.validation.model_dump_json())
    else:
        report = run_promotion(settings, source, store)
        log.info("promoted=%s: %s", report.promoted, report.reason)


if __name__ == "__main__":
    main()
