"""SageMaker job entry point: `python -m steam_training {train,promote}`.

`python -m steam_training export --model-id <id>` (local, one-off) writes the numpy user tower
(`user_tower.npz`) of a model saved before it existed; export `champion` too when it is one.

`python -m steam_training track --model-id <id> [--alias champion|challenger]` (local, spec 18)
replays models/<id>/tracking.json into MLflow when the run isn't there yet, then optionally
points the registered model alias at it.
"""

from __future__ import annotations

import argparse
import logging

from steam_training.artifacts import ArtifactStore
from steam_training.config import (
    ExportSettings,
    TrackSettings,
    TrainingSettings,
    configure_logging,
)
from steam_training.data import IcebergSource
from steam_training.pipeline import run_promotion, run_training
from steam_training.tracking import ALIASES, MlflowSink

log = logging.getLogger("steam_training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="steam_training")
    parser.add_argument("mode", choices=["train", "promote", "export", "track"])
    parser.add_argument("--model-id", help="export / track: the model (e.g. champion)")
    parser.add_argument(
        "--alias", choices=ALIASES, help="track: point this registered model alias at the run"
    )
    parser.add_argument(
        "--replace", action="store_true", help="track: replay even if the run is already there"
    )
    args = parser.parse_args(argv)

    if args.mode == "track":
        if not args.model_id:
            parser.error("track needs --model-id")
        track(TrackSettings(), args.model_id, args.alias, args.replace)
        return

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


def track(settings: TrackSettings, model_id: str, alias: str | None, replace: bool) -> str:
    """Make sure the run of `model_id` is in MLflow (replayed from its tracking record), then
    set `alias`. `champion` resolves to the model it is a copy of. Returns the run id."""
    configure_logging(settings.log_level)
    store = ArtifactStore(settings.model_artifacts_bucket)
    if model_id == "champion":
        metadata = store.load_metadata(model_id)
        if metadata is None:
            raise FileNotFoundError("no champion model")
        model_id = metadata.model_id
    sink = MlflowSink(settings.mlflow_tracking_uri, settings.mlflow_experiment)
    run_id = None if replace else sink.find_run(model_id)
    if run_id is None:
        record = store.load_tracking(model_id)
        if record is None:
            raise FileNotFoundError(
                f"no tracking.json for {model_id} (trained before spec 18?): retrain it, or "
                "run it with MLFLOW_TRACKING_URI set"
            )
        run_id = sink.replay(record)
        log.info("replayed %s into mlflow run %s", model_id, run_id)
    else:
        log.info("%s is already mlflow run %s", model_id, run_id)
    if alias:
        version = sink.set_alias(run_id, model_id, alias, store.model_uri(model_id))
        log.info("%s@%s -> version %s (%s)", "steam-two-tower", alias, version, model_id)
    return run_id


if __name__ == "__main__":
    main()
