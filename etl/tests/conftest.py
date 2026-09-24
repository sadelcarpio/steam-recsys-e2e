import pytest

ETL_ENV = {
    "ATHENA_WORK_GROUP": "wg",
    "ATHENA_S3_STAGING_DIR": "s3://results-bucket/athena-results/",
    "ICEBERG_S3_DATA_DIR": "s3://processed-bucket/iceberg/",
}


@pytest.fixture
def etl_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.delenv("USE_SSM", raising=False)
    for key, value in ETL_ENV.items():
        monkeypatch.setenv(key, value)
    return ETL_ENV
