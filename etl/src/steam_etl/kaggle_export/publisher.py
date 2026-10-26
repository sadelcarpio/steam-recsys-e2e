"""Kaggle side of the export: a small protocol (faked in tests) and its `kaggle` client."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetStatus:
    status: str  # "ready", "pending" or "error"
    version: int


class KagglePublishError(RuntimeError):
    pass


class Publisher(Protocol):
    def status(self) -> DatasetStatus | None:
        """None when the dataset doesn't exist yet."""
        ...

    def create(self, folder: Path) -> None:
        """A new private dataset from `folder` (data files + dataset-metadata.json)."""
        ...

    def version(self, folder: Path, note: str) -> None: ...


class KagglePublisher:
    """`kaggle` 2.x API. The token is only put in this process's environment, never logged."""

    def __init__(self, dataset: str, token: str, *, config_dir: str) -> None:
        self.dataset = dataset
        os.environ["KAGGLE_API_TOKEN"] = token
        os.environ.setdefault("KAGGLE_CONFIG_DIR", config_dir)  # `kaggle` writes it on import
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()
        self.api.authenticate()

    def status(self) -> DatasetStatus | None:
        try:
            payload = json.loads(self.api.dataset_status(self.dataset, format="json"))
        except Exception as err:
            if getattr(getattr(err, "response", None), "status_code", None) == 404:
                return None
            # `from None`: the client's error can carry request details
            raise KagglePublishError(
                f"status of {self.dataset} failed: {type(err).__name__}"
            ) from None
        return DatasetStatus(
            status=str(payload["status"]), version=int(payload["current_version_number"] or 0)
        )

    def create(self, folder: Path) -> None:
        response = self.api.dataset_create_new(
            str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip"
        )
        self._check(response, "create")

    def version(self, folder: Path, note: str) -> None:
        response = self.api.dataset_create_version(
            str(folder), note, quiet=True, convert_to_csv=False, dir_mode="skip"
        )
        self._check(response, "version")

    def _check(self, response, action: str) -> None:
        error = getattr(response, "error", None)
        if error:
            raise KagglePublishError(f"{action} of {self.dataset} failed: {error}")
        log.info("kaggle %s accepted: %s", action, getattr(response, "url", ""))
