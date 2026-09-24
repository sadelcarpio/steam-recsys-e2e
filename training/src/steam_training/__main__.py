"""SageMaker job entry point: `python -m steam_training {train,promote}`.

`python -m steam_training export --model-id <id>` (local, one-off) writes the numpy user tower
(`user_tower.npz`) of a model saved before it existed; export `champion` too when it is one.
"""

from __future__ import annotations

import argparse
import logging

from steam_training.artifacts import ArtifactStore
from steam_training.config import ExportSettings, TrainingSettings, configure_logging
from steam_training.data import IcebergSource
from steam_training.pipeline import run_promotion, run_training

log = logging.getLogger("steam_training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="steam_training")
    parser.add_argument("mode", choices=["train", "promote", "export"])
    parser.add_argument("--model-id", help="export: the model to export (e.g. champion)")
    args = parser.parse_args(argv)

    if args.mode == "export":
        if not args.model_id:
            parser.error("export needs --model-id")
        export_settings = ExportSettings()
        configure_logging(export_settings.log_level)
        store = ArtifactStore(export_settings.model_artifacts_bucket)
        model, metadata = store.load_model(args.model_id)
        # the arrays carry the model's own id (champion files are copies of models/<id>/)
        key = store.save_user_tower_numpy(model, metadata.model_id)
        if args.model_id != metadata.model_id:
            key = store.copy_user_tower_numpy(metadata.model_id, args.model_id)
        log.info("exported the numpy user tower of %s to %s", metadata.model_id, key)
        return

    settings = TrainingSettings()
    configure_logging(settings.log_level)
    store = ArtifactStore(settings.model_artifacts_bucket)
    source = IcebergSource(settings.glue_database, settings.aws_region)
    if args.mode == "train":
        metadata = run_training(settings, source, store)
        log.info("trained %s: %s", metadata.model_id, metadata.validation.model_dump_json())
    else:
        report = run_promotion(settings, source, store)
        log.info("promoted=%s: %s", report.promoted, report.reason)


if __name__ == "__main__":
    main()
