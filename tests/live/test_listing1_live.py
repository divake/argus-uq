"""Listing 1 end-to-end on real cached records (no live model load).

This test exercises the full ``argus_uq.{load, score, conformal_disk}``
path on the released ``data_release/`` artifacts. Unlike the live model
tests, this runs anywhere with the data release present.
"""

from __future__ import annotations

import pytest

import argus_uq


def test_listing1_paper_headline():
    try:
        cell = argus_uq.load("PTxOSG")
    except FileNotFoundError:
        pytest.skip("data_release/per_item_full/PTxOSG.parquet not present")

    result = argus_uq.score(cell, method="saplma", seeds=50)
    # Paper expectation: SAPLMA on PTxOSG AUROC ~0.80 (range 0.78-0.82
    # over 50 seeds).
    assert 0.7 <= result.auroc <= 0.9
    assert result.n_seeds == 50

    disk = argus_uq.conformal_disk(
        cell, plug_in="saplma", alpha=0.10, variant="normalized", seeds=10
    )
    # Coverage should be at or above 1 - alpha = 0.90 within tolerance.
    assert disk.coverage >= 0.85
    assert disk.radius_pixels > 0
