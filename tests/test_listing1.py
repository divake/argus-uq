"""Quickstart contract test.

This test
asserts that the example imports cleanly and that the API surface it
relies on (load / score / conformal_disk) is wired up. The test pins
the surface shape so any future drift is caught.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
QUICKSTART = REPO_ROOT / "examples" / "quickstart.py"


def test_quickstart_file_exists():
    assert QUICKSTART.is_file(), f"missing {QUICKSTART}"


def test_quickstart_loads_argus_uq():
    """The example imports argus_uq at module level."""
    text = QUICKSTART.read_text()
    assert "import argus_uq" in text


def test_quickstart_invokes_three_entry_points():
    """The example must call load, score, and conformal_disk."""
    text = QUICKSTART.read_text()
    assert "argus_uq.load(" in text
    assert "argus_uq.score(" in text
    assert "argus_uq.conformal_disk(" in text


def test_quickstart_runs_end_to_end():
    """The full quickstart should execute and print finite numbers.

    Skips gracefully if the data_release artifacts are not yet present
    (e.g. on a fresh CI machine before ``argus_uq.cli.prepare_data``).
    """

    import argus_uq

    try:
        cell = argus_uq.load("PTxOSG")
    except FileNotFoundError:
        pytest.skip("data_release/per_item_full/PTxOSG.parquet not present yet")

    result = argus_uq.score(cell, method="saplma", seeds=10)
    assert 0.0 <= result.auroc <= 1.0
    assert result.method == "saplma"
    assert result.cell_id == "PTxOSG"

    disk = argus_uq.conformal_disk(
        cell, plug_in="saplma", alpha=0.10, variant="normalized", seeds=5
    )
    assert disk.radius_pixels > 0
    assert 0.0 <= disk.coverage <= 1.0


def test_quickstart_runs_via_module_load():
    """Run examples/quickstart.py as a module - ensures the file is valid."""

    import argus_uq

    try:
        argus_uq.load("PTxOSG")
    except FileNotFoundError:
        pytest.skip("data_release not populated; run argus_uq.cli.prepare_data")

    spec = importlib.util.spec_from_file_location("quickstart", QUICKSTART)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
