"""Integration tests for argus_uq.{load, score, conformal_disk} on real data.

These tests use the data_release/ artifacts. They skip gracefully when
``data_release/per_item_full/<cell>.parquet`` is missing (e.g. on a
fresh CI machine before ``argus_uq.cli.prepare_data`` has been run).
"""

from __future__ import annotations

import math

import pytest

import argus_uq
from argus_uq.data import ALL_CELLS, CLOSED_CELLS, OPEN_CELLS


def _maybe_skip(cell_id: str):
    """Skip if the parquet for ``cell_id`` is not yet on disk."""

    try:
        return argus_uq.load(cell_id)
    except FileNotFoundError:
        pytest.skip(f"data_release/per_item_full/{cell_id}.parquet not present")


def test_load_unknown_cell_raises_keyerror():
    with pytest.raises(KeyError):
        argus_uq.load("not_a_cell")


@pytest.mark.parametrize("cell_id", OPEN_CELLS)
def test_load_each_open_cell(cell_id):
    cell = _maybe_skip(cell_id)
    assert cell.meta.cell_id == cell_id
    assert cell.meta.tier == "open_source"
    assert cell.meta.n_items > 0
    df = cell.records
    # Open-source parquets carry 27 method columns.
    method_cols = [c for c in df.columns if c not in ("item_id", "correct", "dist")]
    assert len(method_cols) == 27


@pytest.mark.parametrize("cell_id", CLOSED_CELLS)
def test_load_each_closed_cell(cell_id):
    cell = _maybe_skip(cell_id)
    assert cell.meta.cell_id == cell_id
    assert cell.meta.tier == "closed_source"
    assert cell.meta.n_items > 0
    df = cell.records
    # Closed-source parquets carry 8 OS-identical method columns.
    method_cols = [c for c in df.columns if c not in ("item_id", "correct", "dist")]
    assert len(method_cols) == 8
    expected = {
        "self_consistency",
        "semantic_entropy",
        "lexical_similarity",
        "ccp",
        "verbalized_1s",
        "verbalized_2s",
        "hedge",
        "imghedge",
    }
    assert set(method_cols) == expected


def test_score_listing_1():
    cell = _maybe_skip("PTxOSG")
    res = argus_uq.score(cell, method="saplma", seeds=10)
    assert 0.0 <= res.auroc <= 1.0
    assert res.auroc_std >= 0.0
    assert res.auroc_ci_lo <= res.auroc <= res.auroc_ci_hi
    assert res.method == "saplma"
    assert res.cell_id == "PTxOSG"
    assert res.n_seeds == 10


def test_score_unknown_method_raises():
    cell = _maybe_skip("PTxOSG")
    with pytest.raises(KeyError):
        argus_uq.score(cell, method="not_a_method")


def test_score_fixed_seed_reproducible():
    cell = _maybe_skip("PTxOSG")
    r1 = argus_uq.score(cell, method="saplma", seeds=1, seed=42)
    r2 = argus_uq.score(cell, method="saplma", seeds=1, seed=42)
    assert r1.auroc == r2.auroc


def test_conformal_disk_fixed():
    cell = _maybe_skip("PTxOSG")
    disk = argus_uq.conformal_disk(cell, alpha=0.10, variant="fixed", seeds=5)
    assert disk.variant == "fixed"
    assert disk.radius_pixels > 0
    assert 0.0 <= disk.coverage <= 1.0


def test_conformal_disk_normalized():
    cell = _maybe_skip("PTxOSG")
    disk = argus_uq.conformal_disk(
        cell, plug_in="saplma", alpha=0.10, variant="normalized", seeds=5
    )
    assert disk.variant == "normalized"
    assert disk.plug_in == "saplma"
    assert disk.radius_pixels > 0


def test_conformal_disk_cqr():
    cell = _maybe_skip("PTxOSG")
    disk = argus_uq.conformal_disk(
        cell, plug_in="saplma", alpha=0.10, variant="cqr", seeds=5
    )
    assert disk.variant == "cqr"


def test_conformal_disk_alpha_sweep():
    """Looser coverage budgets give smaller disks."""

    cell = _maybe_skip("PTxOSG")
    r05 = argus_uq.conformal_disk(cell, alpha=0.05, variant="fixed", seeds=3)
    r20 = argus_uq.conformal_disk(cell, alpha=0.20, variant="fixed", seeds=3)
    # Higher alpha => looser coverage => smaller (or equal) radius.
    assert r05.radius_pixels >= r20.radius_pixels


def test_conformal_disk_unknown_variant():
    cell = _maybe_skip("PTxOSG")
    with pytest.raises(ValueError):
        argus_uq.conformal_disk(cell, variant="not_a_variant")


def test_score_closed_source_8_methods():
    """All 8 closed-source methods score on the closed-source matrix."""

    cell = _maybe_skip("OAIxSP")
    closed_methods = [
        "self_consistency",
        "semantic_entropy",
        "lexical_similarity",
        "ccp",
        "verbalized_1s",
        "verbalized_2s",
        "hedge",
        "imghedge",
    ]
    for m in closed_methods:
        r = argus_uq.score(cell, method=m, seeds=5)
        assert math.isnan(r.auroc) or 0.0 <= r.auroc <= 1.0


def test_score_open_only_method_unavailable_on_closed_cell():
    """SAPLMA is open-only; scoring it on a closed-source cell raises."""

    cell = _maybe_skip("OAIxSP")
    with pytest.raises(KeyError):
        argus_uq.score(cell, method="saplma", seeds=1)
