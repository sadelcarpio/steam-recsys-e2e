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
        # Existence from the owner's own datasets (private ones included): Kaggle doesn't answer
        # the status of a dataset that was never created with a plain 404.
        slug = self.dataset.split("/")[-1]
        mine = self._call("list", lambda: self.api.dataset_list(mine=True, search=slug)) or []
        if not any(getattr(d, "ref", None) == self.dataset for d in mine):
            return None
        payload = json.loads(
            self._call("status", lambda: self.api.dataset_status(self.dataset, format="json"))
        )
        return DatasetStatus(
            status=str(payload["status"]), version=int(payload["current_version_number"] or 0)
        )

    def create(self, folder: Path) -> None:
        response = self._call(
            "create",
            lambda: self.api.dataset_create_new(
                str(folder), public=False, quiet=True, convert_to_csv=False, dir_mode="skip"
            ),
        )
        self._check(response, "create")

    def version(self, folder: Path, note: str) -> None:
        response = self._call(
            "version",
            lambda: self.api.dataset_create_version(
                str(folder), note, quiet=True, convert_to_csv=False, dir_mode="skip"
            ),
        )
        self._check(response, "version")

    def _call(self, action: str, request):
        try:
            return request()
        except Exception as err:
            # `from None`: the client's error can carry request details. Only the HTTP codes
            # are kept (the response's, and the one in a JSON error body, which can differ).
            raise KagglePublishError(
                f"{action} of {self.dataset} failed: {type(err).__name__}{_http_codes(err)}"
            ) from None

    def _check(self, response, action: str) -> None:
        error = getattr(response, "error", None)
        if error:
            raise KagglePublishError(f"{action} of {self.dataset} failed: {error}")
        log.info("kaggle %s accepted: %s", action, getattr(response, "url", ""))


def _http_codes(err: Exception) -> str:
    response = getattr(err, "response", None)
    if response is None:
        return ""
    codes = [str(getattr(response, "status_code", "?"))]
    try:
        body_code = response.json().get("code")
    except Exception:
        body_code = None
    if body_code is not None and str(body_code) != codes[0]:
        codes.append(f"body {body_code}")
    return f" (HTTP {', '.join(codes)})"
