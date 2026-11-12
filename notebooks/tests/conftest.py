from __future__ import annotations

import pytest

from steam_eda import data
from tests.synthetic import write_dataset


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("dataset"))


@pytest.fixture
def env(dataset, tmp_path, monkeypatch):
    """Settings pointing at the synthetic dataset, every user, no text models."""
    monkeypatch.setenv("EDA_DATA_DIR", str(dataset))
    monkeypatch.setenv("EDA_SAMPLE_USERS", "1")
    monkeypatch.setenv("EDA_TEXT_MODELS", "lexical")
    monkeypatch.setenv("EDA_CACHE_DIR", str(tmp_path / "cache"))
    data.settings.cache_clear()
    yield
    data.settings.cache_clear()
