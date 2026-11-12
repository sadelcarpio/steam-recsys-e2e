"""Runs every notebook end to end on the synthetic dataset (lexical text models, no AWS, no
downloads), so a broken cell fails CI."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].glob("[0-9]*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path: Path, env, tmp_path) -> None:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(tmp_path)}}).execute()  # fmt: skip
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs
              if o.output_type == "error"]  # fmt: skip
    assert not errors
