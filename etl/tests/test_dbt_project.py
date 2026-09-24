"""Static checks of the dbt project (`dbt parse`, no warehouse connection)."""

import pytest
import yaml
from dbt.cli.main import dbtRunner

from steam_etl.config import DEFAULT_DBT_PROJECT_DIR, EtlSettings
from steam_etl.contracts import MART_CONTRACTS, USER_HISTORY_LENGTH
from steam_etl.runner import dbt_env


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for key, value in dbt_env(
        EtlSettings(
            athena_work_group="wg",
            athena_s3_staging_dir="s3://b/r/",
            iceberg_s3_data_dir="s3://b/i/",
        )
    ).items():
        mp.setenv(key, value)
    target = tmp_path_factory.mktemp("target")
    res = dbtRunner().invoke(
        [
            "parse",
            "--project-dir",
            str(DEFAULT_DBT_PROJECT_DIR),
            "--profiles-dir",
            str(DEFAULT_DBT_PROJECT_DIR),
            "--target-path",
            str(target),
        ]
    )
    mp.undo()
    assert res.success, res.exception
    return res.result


def _models(manifest):
    return {n.name: n for n in manifest.nodes.values() if n.resource_type == "model"}


def test_layers(manifest):
    models = _models(manifest)
    assert {m.name for m in models.values() if m.config.schema == "staging"} == {
        "stg_steam__games",
        "stg_steam__reviews",
    }
    assert set(MART_CONTRACTS) == {m.name for m in models.values() if m.config.schema == "marts"}
    for m in models.values():
        if m.config.schema == "staging":
            assert m.config.materialized == "view"
        else:
            assert m.config.materialized == "incremental", m.name
            assert m.config.get("table_type") == "iceberg", m.name


def test_incremental_models_are_idempotent_by_construction(manifest):
    for m in _models(manifest).values():
        if m.config.materialized != "incremental":
            continue
        if m.name.startswith("lkp_"):
            # append-only vocabularies: never rebuilt, ids stay stable
            assert m.config.incremental_strategy == "append"
            assert m.config.full_refresh is False
        else:
            assert m.config.incremental_strategy == "merge", m.name
            assert m.config.unique_key, m.name


def test_interactions_depend_on_feature_tables(manifest):
    deps = _models(manifest)["interactions"].depends_on.nodes
    assert {"model.steam_recsys.user_features", "model.steam_recsys.game_features"} <= set(deps)


def test_history_length_matches_contract():
    project = yaml.safe_load((DEFAULT_DBT_PROJECT_DIR / "dbt_project.yml").read_text())
    assert project["vars"]["user_history_length"] == USER_HISTORY_LENGTH
