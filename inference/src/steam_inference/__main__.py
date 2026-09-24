"""ECS task entry point (last step of the Step Functions pipeline): `python -m steam_inference`.
Exits 0 without writing anything when the model (the champion by default) does not exist."""

from __future__ import annotations

import logging

import torch
from steam_training.artifacts import ArtifactStore
from steam_training.data import IcebergSource

from steam_inference.config import InferenceSettings, configure_logging
from steam_inference.pipeline import run_inference
from steam_inference.rerank import BedrockReranker
from steam_inference.writer import DynamoWriter, JsonlWriter, Writer

log = logging.getLogger("steam_inference")


def main() -> None:
    settings = InferenceSettings()
    configure_logging(settings.log_level)
    if settings.num_threads:
        torch.set_num_threads(settings.num_threads)
    writer: Writer = (
        JsonlWriter(settings.output_path)
        if settings.output_path
        else DynamoWriter(
            settings.recommendations_table,
            region=settings.aws_region,
            concurrency=settings.write_concurrency,
        )
    )
    reranker = None
    if settings.rerank_enabled and settings.rerank_max_users > 0:
        reranker = BedrockReranker(
            settings.bedrock_model_id,
            explain_top_n=settings.explain_top_n,
            max_tokens=settings.rerank_max_tokens,
            temperature=settings.rerank_temperature,
            region=settings.aws_region,
        )
    run_inference(
        settings,
        IcebergSource(settings.glue_database, settings.aws_region),
        ArtifactStore(settings.model_artifacts_bucket),
        writer,
        reranker,
    )
    if reranker is not None:
        log.info(
            "bedrock usage: %d input / %d output tokens",
            reranker.input_tokens,
            reranker.output_tokens,
        )


if __name__ == "__main__":
    main()
