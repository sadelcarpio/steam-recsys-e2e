"""SageMaker Processing entry point (last step of the Step Functions pipeline):
`python -m steam_inference`.
Exits 0 without writing anything when the model (the champion by default) does not exist."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from steam_training.artifacts import ArtifactStore
from steam_training.data import IcebergSource

from steam_inference.config import InferenceSettings, configure_logging
from steam_inference.online import BundleStore, LocalBundleStore, S3BundleStore
from steam_inference.pipeline import run_inference
from steam_inference.rerank import BedrockRerankLlm
from steam_inference.writer import DynamoWriter, JsonlWriter, Writer

log = logging.getLogger("steam_inference")


def main() -> None:
    settings = InferenceSettings()
    configure_logging(settings.log_level)
    if settings.num_threads:
        torch.set_num_threads(settings.num_threads)
    writer: Writer
    details_writer: Writer | None
    if settings.output_path:
        writer = JsonlWriter(settings.output_path)
        details_writer = JsonlWriter(
            str(Path(settings.output_path).with_name("game-details.jsonl"))
        )
    else:
        writer = DynamoWriter(
            settings.recommendations_table,
            region=settings.aws_region,
            concurrency=settings.write_concurrency,
        )
        details_writer = DynamoWriter(
            settings.game_details_table,
            region=settings.aws_region,
            concurrency=settings.write_concurrency,
            key="game_id",
        )
    if not settings.sync_game_details:
        details_writer = None
    bundle_store: BundleStore | None = None
    if settings.online_bundle_enabled:
        bundle_store = (
            LocalBundleStore(str(Path(settings.output_path).with_name("online")))
            if settings.output_path
            else S3BundleStore(
                settings.model_artifacts_bucket,
                settings.online_bundle_prefix,
                region=settings.aws_region,
                search_key=settings.search_index_key,
                user_index_prefix=settings.user_index_prefix,
            )
        )
    reranker = None
    if settings.rerank_enabled and settings.rerank_max_users > 0:
        reranker = BedrockRerankLlm(
            settings.bedrock_model_id,
            max_tokens=settings.rerank_max_tokens,
            temperature=settings.rerank_temperature,
            region=settings.aws_region,
            concurrency=settings.rerank_concurrency,
        )
    run_inference(
        settings,
        IcebergSource(settings.glue_database, settings.aws_region),
        ArtifactStore(settings.model_artifacts_bucket),
        writer,
        reranker,
        details_writer=details_writer,
        bundle_store=bundle_store,
    )
    if reranker is not None:
        for stage, usage in reranker.usage.items():
            log.info(
                "bedrock usage (%s): %d input / %d output tokens",
                stage,
                usage.input_tokens,
                usage.output_tokens,
            )


if __name__ == "__main__":
    main()
