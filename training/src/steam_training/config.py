"""Component configuration.

Precedence: init kwargs > environment variables > SSM Parameter Store (`/training/<ENV_VAR>`).
SSM is only consulted when `USE_SSM=true` (set on the SageMaker jobs and by the CD launcher), so
local runs and tests work from plain env vars. Hyperparameters are plain env vars too: the
launcher forwards overrides (e.g. `EPOCHS=10`) as the job's environment.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import boto3
from pydantic import AliasChoices, Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

SSM_PREFIX = "/training/"
BUCKET_PATTERN = r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$"
# Models are keyed by the git commit sha of the training code (or any short slug locally).
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$"
CHAMPION = "champion"


class SsmSettingsSource(PydanticBaseSettingsSource):
    """Loads every parameter under `/training/`, keyed by the env var name."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._values: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._values is not None:
            return self._values
        self._values = {}
        if os.environ.get("USE_SSM", "false").lower() != "true":
            return self._values
        ssm = boto3.client("ssm")
        for page in ssm.get_paginator("get_parameters_by_path").paginate(
            Path=SSM_PREFIX, WithDecryption=True
        ):
            for param in page["Parameters"]:
                self._values[param["Name"].removeprefix(SSM_PREFIX).lower()] = param["Value"]
        return self._values

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._load().get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        values = self._load()
        return {name: values[name] for name in self.settings_cls.model_fields if name in values}


class _SsmSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, SsmSettingsSource(settings_cls)


class TrainingSettings(_SsmSettings):
    """Settings of the training / promotion jobs (`python -m steam_training <mode>`)."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    model_id: str = Field(pattern=MODEL_ID_PATTERN)
    # Glue database of the ETL marts (interactions, game_features, lkp_*).
    glue_database: str = "steam_marts"
    aws_region: str = "us-east-1"
    # Local copy of the Iceberg data files read (local runs: each file downloads once, later
    # runs on the same snapshot read from disk). Unset in SageMaker jobs.
    data_cache_dir: str | None = None

    # ---- data ----
    # Temporal split by row share: [train | validation | test], test = the newest rows.
    # FIT_ON=train trains on train and evaluates on validation (test is never read: model
    # selection). FIT_ON=train+validation trains through validation and evaluates on test (the
    # final check of a chosen recipe). TEST_FRACTION=0: no test window (validation = the rest).
    validation_fraction: float = Field(0.1, gt=0, lt=0.5)
    test_fraction: float = Field(0.0, ge=0, lt=0.5)
    fit_on: Literal["train", "train+validation"] = "train"
    # Time-based windows (spec 9, "Time windows"), counted back from the newest interaction N:
    # train before N - HOLDOUT_WEEKS, validate on [N - HOLDOUT_WEEKS, N - TEST_WEEKS), and the
    # last TEST_WEEKS are the test window only gates read (rows there are never used by
    # training). Replaces VALIDATION_FRACTION / TEST_FRACTION when set.
    holdout_weeks: int | None = Field(None, ge=1)
    test_weeks: int = Field(2, ge=0)
    # Share of empty-history (first positive review) rows kept for training. Inference never
    # sees an empty history; a few of them teach the model a "no history" fallback query.
    cold_row_fraction: float = Field(0.1, ge=0, le=1)
    # Probability of truncating a warm history (>= 2 games) to a random shorter prefix.
    history_dropout: float = Field(0.3, ge=0, le=1)
    # Probability of replacing a game_idx with OOV so the OOV row learns a generic item and
    # games without training interactions fall back to their content features.
    item_id_dropout: float = Field(0.1, ge=0, le=1)
    # The same id dropout for the hard negatives (explicit, mined). Without it the OOV row only
    # ever learns from positives (in-batch negatives are logQ-corrected, so a trending dropped
    # target is barely pushed away), and a trending unseen game scores high for everyone.
    hard_negative_id_dropout: float = Field(0.0, ge=0, le=1)
    # A6: id dropout for games released at most NEW_ITEM_DAYS before the example (targets and
    # hard negatives): the games that are unseen when serving. None = ITEM_ID_DROPOUT.
    new_item_id_dropout: float | None = Field(None, ge=0, le=1)
    new_item_days: int = Field(90, ge=1)
    # A6: probability of zeroing an item's recency features in training (targets and hard
    # negatives), so "trending this week" can't carry a score alone (popular games flooding)
    recency_dropout: float = Field(0.0, ge=0, lt=1)
    # Exposure check over the current catalog (`pipeline.current_concentration`): top-K lists of
    # CONCENTRATION_ROWS warm validation users (0 = off).
    concentration_rows: int = Field(20_000, ge=0)
    concentration_k: int = Field(10, ge=1)
    # Steam user tags as an item feature (marts `game_tags` + `lkp_tags`, spec 8).
    use_game_tags: bool = True
    # A4 (spec 17): the user tower also reads a tag profile of the history (needs the tags)
    user_tag_profile: bool = False
    # A3 (spec 17): games in the user history. 5 = the mart's `games_reviewed_positive`; any
    # other length is derived from `interactions` (the last N positive reviews strictly before
    # each row). Models with N != 5 or attention can't be promoted yet (inference / serving
    # read the 5-game mart history).
    history_length: int = Field(5, ge=1, le=512)
    # "mean": masked mean of the history games. "attention": + self-attention over the history
    # (ATTENTION_LAYERS pre-LN blocks, ATTENTION_HEADS heads, learned positions, a summary token)
    history_encoder: Literal["mean", "attention"] = "mean"
    attention_heads: int = Field(2, ge=1)
    attention_layers: int = Field(1, ge=1)
    attention_dropout: float = Field(0.1, ge=0, lt=1)
    # A6 (spec 17): item recency features (`ModelConfig.item_recency`) from
    # `game_popularity_weekly` + `game_details` release dates (inference computes them too)
    item_recency: bool = False
    # A7 (spec 17): the language of the user's previous review (`ModelConfig.user_language`),
    # from the `review_languages` mart (inference reads it; online requests: unknown)
    user_language: bool = False

    # ---- model ----
    game_embedding_dim: int = Field(64, ge=4)
    attribute_embedding_dim: int = Field(16, ge=2)
    hidden_dim: int = Field(128, ge=4)
    output_dim: int = Field(64, ge=4)

    # ---- optimisation ----
    epochs: int = Field(5, ge=1)
    batch_size: int = Field(1024, ge=2)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-6, ge=0)
    temperature: float = Field(0.05, gt=0)
    # Subtract log(sampling probability) from in-batch logits (popularity correction).
    logq_correction: bool = True
    # Also re-sync the user tower's frozen game table every N training steps within an epoch
    # (0 = only at the start of each epoch). In epoch 1 the per-epoch copy is the random init.
    user_table_sync_steps: int = Field(0, ge=0)
    seed: int = 42
    num_threads: int = Field(0, ge=0)  # 0 = torch default
    # auto = cuda when available (GPU image, `cu128` extra), else cpu
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Resume from s3://<bucket>/checkpoints/<model_id>/ when its fingerprint matches.
    resume: bool = True

    # ---- hard negatives ----
    # One of the user's negatively reviewed games (is_positive=false, training split).
    explicit_negatives: bool = True
    # One game sampled from the ranks [mine_skip_top, mine_skip_top + mine_pool_size) of the
    # current model, re-mined at the start of each epoch >= mining_start_epoch. The top ranks
    # are skipped because they are the likeliest false negatives.
    mined_negatives: bool = True
    mining_start_epoch: int = Field(1, ge=0)
    mine_skip_top: int = Field(5, ge=0)
    mine_pool_size: int = Field(50, ge=1)
    # Examples re-mined per epoch (a random subset when there are more; the others keep their
    # previous mined negative). Bounds mining time on large datasets.
    mine_max_rows: int = Field(2_000_000, ge=1)

    # ---- evaluation ----
    recall_ks: list[int] = Field(default_factory=lambda: [30, 50, 100], min_length=1)
    primary_k: int = 50
    # Rows of the validation split scored after each epoch (monitoring only); 0 = skip.
    epoch_eval_rows: int = Field(50_000, ge=0)
    # Keep the epoch with the best monitor warm recall@PRIMARY_K (needs EPOCH_EVAL_ROWS > 0;
    # without monitoring, e.g. FIT_ON=train+validation, the last epoch is kept).
    keep_best_epoch: bool = True
    # Stop after this many epochs without a better monitor score (0 = run all EPOCHS).
    early_stopping_patience: int = Field(1, ge=0)
    eval_batch_size: int = Field(4096, ge=1)
    # Positive validation rows kept (uniform sample): ~500k gives recall within ~0.002.
    eval_max_rows: int = Field(500_000, ge=1)
    # Promotion: candidate must beat the champion's warm recall@primary_k by more than this.
    min_improvement: float = Field(0.0, ge=0)
    # Promotion: promote even when the candidate loses (the report keeps the real metrics and
    # why it would have been rejected). For deliberate swaps, e.g. a first full-data model.
    force_promotion: bool = False

    # ---- experiment tracking (spec 18) ----
    # MLflow server to log the run to while it trains (e.g. http://localhost:5000); unset = only
    # models/<model_id>/tracking.json is written (replay it with `python -m steam_training track`)
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "steam-two-tower"
    # lineage tag; unset = `git rev-parse HEAD` of the working directory, when it is a checkout
    git_commit: str | None = None

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> TrainingSettings:
        if self.primary_k not in self.recall_ks:
            raise ValueError(f"primary_k={self.primary_k} must be one of recall_ks")
        if self.model_id == CHAMPION:
            raise ValueError(f"model_id '{CHAMPION}' is reserved")
        if self.new_item_id_dropout is not None and not self.item_recency:
            raise ValueError("NEW_ITEM_ID_DROPOUT needs ITEM_RECENCY (release dates)")
        if self.recency_dropout and not self.item_recency:
            raise ValueError("RECENCY_DROPOUT needs ITEM_RECENCY")
        if self.validation_fraction + self.test_fraction >= 0.5:
            raise ValueError("validation_fraction + test_fraction must be < 0.5")
        if self.game_embedding_dim % self.attention_heads:
            raise ValueError("GAME_EMBEDDING_DIM must be divisible by ATTENTION_HEADS")
        if self.user_tag_profile and not self.use_game_tags:
            raise ValueError("USER_TAG_PROFILE needs USE_GAME_TAGS")
        if self.holdout_weeks is not None:
            if self.test_weeks >= self.holdout_weeks:
                raise ValueError("TEST_WEEKS must be < HOLDOUT_WEEKS")
            if self.test_fraction > 0:
                raise ValueError("HOLDOUT_WEEKS replaces TEST_FRACTION: set only one")
            if self.fit_on == "train+validation" and self.test_weeks == 0:
                raise ValueError("FIT_ON=train+validation needs TEST_WEEKS > 0")
        elif self.fit_on == "train+validation" and self.test_fraction == 0:
            raise ValueError("FIT_ON=train+validation needs TEST_FRACTION > 0 to evaluate on")
        return self


# Settings that do not change what is trained (a checkpoint stays valid when they change).
NOT_FINGERPRINTED = {
    "concentration_rows",
    "concentration_k",
    "model_artifacts_bucket",
    "model_id",
    "glue_database",
    "aws_region",
    "epochs",
    "num_threads",
    "device",
    "resume",
    "recall_ks",
    "primary_k",
    "epoch_eval_rows",
    "keep_best_epoch",
    "early_stopping_patience",
    "eval_batch_size",
    "eval_max_rows",
    "min_improvement",
    "force_promotion",
    "mlflow_tracking_uri",
    "mlflow_experiment",
    "git_commit",
    "data_cache_dir",
    "log_level",
}


class RankerSettings(_SsmSettings):
    """Settings of `python -m steam_training rank` (spec 9): train a ranker on a retriever's
    candidates after that retriever's cutoff. Separate from `TrainingSettings`, so ranker
    settings never change a retriever's checkpoint fingerprint or metadata."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    # the ranker's id (rankers/<ranker_id>/): the commit sha in CD, local-<sha>-<suffix> locally.
    # RANKER_ID, or MODEL_ID (what the SageMaker launcher sets)
    ranker_id: str = Field(
        pattern=MODEL_ID_PATTERN, validation_alias=AliasChoices("ranker_id", "model_id")
    )
    # the retriever whose candidates train the ranker (`champion` = the current champion)
    retriever_id: str = Field(CHAMPION, pattern=MODEL_ID_PATTERN)
    glue_database: str = "steam_marts"
    aws_region: str = "us-east-1"
    data_cache_dir: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    eval_batch_size: int = Field(4096, ge=1)
    seed: int = 42

    # ---- groups ----
    # candidates per group (RETRIEVAL_CANDIDATES at inference) and the ranker's list length
    ranker_candidates: int = Field(100, ge=2)
    ranker_top_k: int = Field(20, ge=1)
    # liked games the user-side features read (MAX_LIKED_GAMES online)
    ranker_history_length: int = Field(20, ge=1, le=512)
    # share of the window's groups (by time) before T2: train; the rest: test. Only for
    # retrievers without a split end: with time-based windows (HOLDOUT_WEEKS), T2 is the
    # retriever's split end (its test window starts there)
    ranker_train_fraction: float = Field(0.7, gt=0, lt=1)
    # last share (by time) of the training groups used for early stopping
    ranker_early_stopping_fraction: float = Field(0.2, gt=0, lt=1)
    ranker_max_groups_per_user: int = Field(20, ge=1)
    # positive rows of the window read (uniform sample) before the per-user cap
    ranker_max_rows: int = Field(2_000_000, ge=1)
    ranker_max_train_groups: int = Field(300_000, ge=1)
    ranker_max_test_groups: int = Field(100_000, ge=1)
    ranker_min_groups: int = Field(5000, ge=1)
    # weekly game popularity: the `game_popularity_weekly` mart, or the same counts computed
    # from `interactions` (marts deployed before the mart existed)
    ranker_popularity_source: Literal["mart", "interactions"] = "mart"

    # ---- per-user lists (the batch path's view; spec 9) ----
    # test-window users whose lists are rebuilt at weekly snapshots (0 = skip the list metrics)
    ranker_list_users: int = Field(20_000, ge=0)
    # relevant games: positives in [T, T + horizon) (the batch cadence)
    ranker_list_horizon_days: int = Field(7, ge=1)
    ranker_list_max_snapshots: int = Field(6, ge=1)

    # ---- model (LightGBM lambdarank) ----
    ranker_num_leaves: int = Field(31, ge=2)
    ranker_learning_rate: float = Field(0.05, gt=0)
    ranker_max_rounds: int = Field(500, ge=1)
    ranker_early_stopping_rounds: int = Field(30, ge=1)
    ranker_min_data_in_leaf: int = Field(100, ge=1)
    ranker_feature_fraction: float = Field(0.8, gt=0, le=1)
    # features left out of the ranker (names from `ranker.features.FEATURES`; the env takes a
    # JSON list, RANKER_EXCLUDE_FEATURES='["log_positives"]'), e.g. the popularity ones,
    # which pile the most popular games into every list. Inference and serving compute every
    # feature; the ranker reads its subset.
    ranker_exclude_features: list[str] = Field(default_factory=list)
    num_threads: int = Field(0, ge=0)  # 0 = LightGBM default

    # ---- report / gate ----
    # retrain without each feature family and report the recall@top_k change
    ranker_ablation: bool = True
    # the ranker must beat retrieval order's recall@top_k by this much (absolute)
    ranker_min_gain: float = Field(0.005, ge=0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> RankerSettings:
        if self.ranker_id == CHAMPION:
            raise ValueError(f"ranker_id '{CHAMPION}' is reserved")
        if self.ranker_top_k > self.ranker_candidates:
            raise ValueError("RANKER_TOP_K must be <= RANKER_CANDIDATES")
        from steam_training.ranker.features import FEATURES

        unknown = sorted(set(self.ranker_exclude_features) - set(FEATURES))
        if unknown:
            raise ValueError(f"RANKER_EXCLUDE_FEATURES: unknown features {unknown}")
        if set(FEATURES) <= set(self.ranker_exclude_features):
            raise ValueError("RANKER_EXCLUDE_FEATURES leaves no feature")
        return self


class PairSettings(_SsmSettings):
    """Settings of `python -m steam_training pair` (spec 9): evaluate challenger (retriever,
    ranker) pairs against the serving pair on a window none of them trained on, and promote the
    best one; or roll back to the previous pair (`ROLLBACK=true`)."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    # challenger retriever (unset = the serving one: a ranker-only challenge)
    retriever_id: str | None = Field(None, pattern=MODEL_ID_PATTERN)
    # challenger ranker trained on the challenger retriever's candidates (unset = none)
    ranker_id: str | None = Field(None, pattern=MODEL_ID_PATTERN)
    rollback: bool = False
    # Evaluate and write the decision (the report's `chosen` = the pair that would be promoted)
    # without promoting: the training Step Function's step before its approval.
    pair_dry_run: bool = False
    # Key of the report (default evaluation/pairs/<chosen retriever>__<ranker>.json); the Step
    # Function sets a per-execution key to read the decision back.
    pair_report_key: str | None = None
    # Promote the pair a dry-run report chose (no re-evaluation), refused when the serving pair
    # changed since that report: the Step Function's step after the approval.
    apply_report: str | None = None
    glue_database: str = "steam_marts"
    aws_region: str = "us-east-1"
    data_cache_dir: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    eval_batch_size: int = Field(4096, ge=1)
    seed: int = 42

    pair_candidates: int = Field(100, ge=2)
    pair_top_k: int = Field(20, ge=1)
    # retrieval recall@K the retriever gate compares (as the retriever's own promotion)
    pair_retrieval_k: int = Field(50, ge=1)
    pair_history_length: int = Field(20, ge=1, le=512)
    pair_max_rows: int = Field(1_000_000, ge=1)
    pair_max_groups_per_user: int = Field(20, ge=1)
    pair_max_groups: int = Field(100_000, ge=1)
    pair_list_users: int = Field(20_000, ge=0)
    pair_list_horizon_days: int = Field(7, ge=1)
    pair_list_max_snapshots: int = Field(4, ge=1)
    # Exposure gate (`evaluation.concentration`): a new retriever whose top-PAIR_CONCENTRATION_K
    # lists over the current catalog send their most common game to more than this share of
    # recent users is rejected (the old champion: ~0.15 at k=10; the flooded 2155771: ~0.6).
    pair_max_game_reach: float = Field(0.3, gt=0, le=1)
    pair_concentration_k: int = Field(10, ge=1)
    pair_concentration_rows: int = Field(20_000, ge=1)
    pair_popularity_source: Literal["mart", "interactions"] = "mart"

    # final list: recall@top_k must beat the serving pair by this much, NDCG@top_k not drop
    min_improvement: float = Field(0.0, ge=0)
    # a new retriever: retrieval recall@PAIR_RETRIEVAL_K must beat the serving retriever's by
    # this much and recall@PAIR_CANDIDATES (the ranker's ceiling) must not drop
    retriever_min_improvement: float = Field(0.0, ge=0)
    # promote even when the gates reject (the report keeps the real reason)
    force_promotion: bool = False
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> PairSettings:
        for value in (self.retriever_id, self.ranker_id):
            if value == CHAMPION:
                raise ValueError("challenger ids can't be 'champion' (it names the serving pair)")
        if self.rollback + (self.apply_report is not None) + self.pair_dry_run > 1:
            raise ValueError("ROLLBACK, APPLY_REPORT and PAIR_DRY_RUN are exclusive")
        if (
            not self.rollback
            and self.apply_report is None
            and self.retriever_id is None
            and self.ranker_id is None
        ):
            raise ValueError("set RETRIEVER_ID and / or RANKER_ID (or ROLLBACK / APPLY_REPORT)")
        if self.pair_top_k > self.pair_candidates:
            raise ValueError("PAIR_TOP_K must be <= PAIR_CANDIDATES")
        return self


class LaunchSettings(_SsmSettings):
    """Settings of the CD launcher that starts SageMaker jobs from GitHub Actions."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    sagemaker_role_arn: str = Field(pattern=r"^arn:aws:iam::\d{12}:role/.+$")
    # ECR repository URL without tag; the launcher appends `:<model_id>`.
    training_image_repository: str
    instance_type: str = "ml.m5.2xlarge"
    volume_size_gb: int = Field(30, ge=1)
    max_runtime_seconds: int = Field(4 * 3600, ge=60)
    aws_region: str = "us-east-1"


class ExportSettings(_SsmSettings):
    """Settings of `python -m steam_training export` (numpy user tower of a saved model)."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    log_level: str = "INFO"


class TrackSettings(_SsmSettings):
    """Settings of `python -m steam_training track` (replay a run into MLflow, set aliases)."""

    model_artifacts_bucket: str = Field(pattern=BUCKET_PATTERN)
    mlflow_tracking_uri: str
    mlflow_experiment: str = "steam-two-tower"
    log_level: str = "INFO"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for noisy in ("botocore", "boto3", "urllib3", "pyiceberg", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
