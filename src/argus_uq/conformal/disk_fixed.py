"""Disk-Fixed (vanilla inductive split conformal prediction).

Reference: Vovk et al. 2005, *Algorithmic Learning in a Random World*;
Angelopoulos & Bates 2023, *Conformal Prediction: A Gentle Introduction*
section 2.2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from argus_uq.conformal.base import CoordCPResult, point_to_bbox_distance


def disk_fixed(
    calib_clicks: Sequence[tuple[int, int] | None],
    calib_bboxes: Sequence[Sequence[float]],
    test_clicks: Sequence[tuple[int, int] | None],
    test_bboxes: Sequence[Sequence[float]],
    alpha: float,
) -> CoordCPResult:
    """Inductive split-CP on click-to-bbox distance.

    Calibration: compute ``d_i = distance(click_i, bbox_i)``. Report
    the conformal quantile ``q_hat`` at level ``1 - alpha`` with the
    finite-sample correction
    ``ceil((n+1)(1-alpha)) / n``.

    Test: the method is "covered" for item ``i`` iff
    ``distance(click_i, bbox_i) <= q_hat``. Marginal coverage
    ``>= 1 - alpha`` in expectation over the split. Infinite distances
    (``None`` clicks) count as failures.
    """

    d_calib = np.asarray(
        [
            point_to_bbox_distance(c, b)
            for c, b in zip(calib_clicks, calib_bboxes, strict=False)
        ]
    )
    finite = np.isfinite(d_calib)
    if finite.sum() < 2:
        q_hat = float("inf")
    else:
        n_c = int(finite.sum())
        level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
        q_hat = float(np.quantile(d_calib[finite], level, method="higher"))

    d_test = np.asarray(
        [
            point_to_bbox_distance(c, b)
            for c, b in zip(test_clicks, test_bboxes, strict=False)
        ]
    )
    covered = (d_test <= q_hat).astype(float)
    coverage = float(covered.mean()) if covered.size else 0.0
    gap = float(max(0.0, (1 - alpha) - coverage))
    median_dist = (
        float(np.median(d_test[np.isfinite(d_test)]))
        if np.isfinite(d_test).any()
        else float("inf")
    )
    return CoordCPResult(
        alpha=alpha,
        q_hat=q_hat,
        test_coverage=coverage,
        coverage_gap=gap,
        n_calib=int(len(calib_clicks)),
        n_test=int(len(test_clicks)),
        median_distance=median_dist,
        variant="fixed",
        plug_in="none",
    )


# Back-compat alias.
inductive_cp_coords = disk_fixed
