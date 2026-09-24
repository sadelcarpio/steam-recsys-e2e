import boto3
import pytest
from moto import mock_aws
from pydantic import ValidationError

from steam_etl.config import DEFAULT_DBT_PROJECT_DIR, EtlSettings


def test_from_env(etl_env):
    s = EtlSettings()
    assert s.athena_work_group == "wg"
    assert s.dbt_schema == "steam"
    assert s.full_refresh is False
    assert s.dbt_project_dir == DEFAULT_DBT_PROJECT_DIR
    assert (DEFAULT_DBT_PROJECT_DIR / "dbt_project.yml").is_file()


def test_rejects_bad_values(etl_env, monkeypatch):
    monkeypatch.setenv("ICEBERG_S3_DATA_DIR", "processed-bucket/iceberg/")
    with pytest.raises(ValidationError):
        EtlSettings()
    monkeypatch.setenv("ICEBERG_S3_DATA_DIR", "s3://processed-bucket/iceberg/")
    monkeypatch.setenv("DBT_SCHEMA", "Steam-Prod")
    with pytest.raises(ValidationError):
        EtlSettings()


@mock_aws
def test_ssm_source_when_enabled(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    ssm = boto3.client("ssm", region_name="us-east-1")
    params = {
        "ATHENA_WORK_GROUP": "steam-recsys-etl",
        "ATHENA_S3_STAGING_DIR": "s3://p/athena-results/",
        "ICEBERG_S3_DATA_DIR": "s3://p/iceberg/",
        "DBT_THREADS": "8",
    }
    for name, value in params.items():
        ssm.put_parameter(Name=f"/etl/{name}", Value=value, Type="String")
    ssm.put_parameter(Name="/data-ingestion/RAW_BUCKET", Value="ignored", Type="String")

    monkeypatch.setenv("USE_SSM", "true")
    monkeypatch.setenv("DBT_THREADS", "2")  # env wins over SSM
    s = EtlSettings()
    assert s.athena_work_group == "steam-recsys-etl"
    assert s.iceberg_s3_data_dir == "s3://p/iceberg/"
    assert s.dbt_threads == 2


def test_ssm_not_called_by_default(etl_env, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("SSM must not be used without USE_SSM=true")

    monkeypatch.setattr(boto3, "client", boom)
    assert EtlSettings().athena_work_group == "wg"
