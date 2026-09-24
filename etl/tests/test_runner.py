import json
import os
from types import SimpleNamespace

import pytest

from steam_etl import __main__ as entrypoint
from steam_etl import runner
from steam_etl.config import EtlSettings


def test_dbt_args_and_env(etl_env):
    s = EtlSettings(full_refresh=True, iceberg_maintenance=False, reviews_lookback_days=1)
    args = runner.dbt_args(s)
    assert args[0] == "build"
    assert args[args.index("--project-dir") + 1] == str(s.dbt_project_dir)
    assert "--full-refresh" in args
    assert json.loads(args[args.index("--vars") + 1]) == {
        "reviews_lookback_days": 1,
        "games_lookback_days": 3,
        "iceberg_maintenance": False,
    }
    assert "--full-refresh" not in runner.dbt_args(s, "test")
    env = runner.dbt_env(s)
    assert env["DBT_SCHEMA"] == "steam"


class FakeRunner:
    calls: list[list[str]] = []
    result = SimpleNamespace(success=True, exception=None, result=SimpleNamespace(results=[]))

    def invoke(self, args):
        FakeRunner.calls.append(args)
        return FakeRunner.result


@pytest.fixture
def fake_dbt(monkeypatch):
    import dbt.cli.main

    FakeRunner.calls = []
    monkeypatch.setattr(dbt.cli.main, "dbtRunner", FakeRunner)
    return FakeRunner


def test_run_dbt_sets_env_and_reports_failures(etl_env, fake_dbt, monkeypatch):
    monkeypatch.setenv("DBT_SCHEMA", "ci_1")
    node = SimpleNamespace(
        status="fail", node=SimpleNamespace(unique_id="test.x"), message="3 rows"
    )
    ok = SimpleNamespace(status="success", node=SimpleNamespace(unique_id="model.y"), message="")
    fake_dbt.result = SimpleNamespace(
        success=False, exception=None, result=SimpleNamespace(results=[ok, node])
    )
    res = runner.run_dbt(EtlSettings())
    assert os.environ["DBT_SCHEMA"] == "ci_1"
    assert os.environ["ATHENA_WORK_GROUP"] == "wg"
    assert res.success is False
    assert res.failures == ["test.x: 3 rows"]


def test_entrypoint_exit_codes(etl_env, fake_dbt):
    fake_dbt.result = SimpleNamespace(
        success=True, exception=None, result=SimpleNamespace(results=[])
    )
    assert entrypoint.main() == 0
    fake_dbt.result = SimpleNamespace(success=False, exception=RuntimeError("boom"), result=None)
    assert entrypoint.main() == 1
