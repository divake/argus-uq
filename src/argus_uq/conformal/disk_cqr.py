"""Disk-CQR (conformalised quantile regression on click distance).

Reference: Romano et al. 2019, *Conformalized Quantile Regression*
(NeurIPS 2019). Fits a quantile regressor on calibration distances
using the plug-in UQ score as the feature, then conformalises the
residuals.

Implementation: a simple isotonic regressor of distance against the
plug-in UQ score (monotone non-decreasing) provides the predicted
quantile ``q_hat(u_i)``; the conformal correction adds a
distribution-free residual quantile.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from argus_uq.conformal.base import CoordCPResult, point_to_bbox_distance


def disk_cqr(
    calib_clicks: Sequence[tuple[int, int] | None],
    calib_bboxes: Sequence[Sequence[float]],
    calib_uq: Sequence[float],
    test_clicks: Sequence[tuple[int, int] | None],
    test_bboxes: Sequence[Sequence[float]],
    test_uq: Sequence[float],
    alpha: float,
    plug_in: str = "saplma",
) -> CoordCPResult:
    """Conformalised quantile regression on click-to-bbox distance.

    Steps:
    1. Fit a monotone-increasing regressor (isotonic) of
       ``d_calib`` against ``u_calib``: ``q(u_i) = isotonic(u_i)``.
    2. Compute calibration residuals ``r_i = max(0, d_i - q(u_i))``.
    3. Conformal quantile ``q_hat`` of ``r`` at level ``1 - alpha``.
    4. Per-item radius at test: ``q(u_test_i) + q_hat``.
    """

    from sklearn.isotonic import IsotonicRegression

    u_calib = np.asarray(calib_uq, dtype=np.float64)
    u_test = np.asarray(test_uq, dtype=np.float64)

    d_calib = np.asarray(
        [
            point_to_bbox_distance(c, b)
            for c, b in zip(calib_clicks, calib_bboxes, strict=False)
        ]
    )
    d_test = np.asarray(
        [
            point_to_bbox_distance(c, b)
            for c, b in zip(test_clicks, test_bboxes, strict=False)
        ]
    )

    finite = np.isfinite(d_calib) & np.isfinite(u_calib)
    if finite.sum() < 5:
        # Fall back to fixed-CP (constant zero-quantile + residual = full distance).
        if finite.sum() == 0:
            q_hat = float("inf")
        else:
            n_c = int(finite.sum())
            level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
            q_hat = float(np.quantile(d_calib[finite], level, method="higher"))
        radii = np.full_like(d_test, q_hat)
        covered = (d_test <= radii).astype(float)
        coverage = float(covered.mean()) if covered.size else 0.0
        gap = float(max(0.0, (1 - alpha) - coverage))
        return CoordCPResult(
            alpha=alpha,
            q_hat=q_hat,
            test_coverage=coverage,
            coverage_gap=gap,
            n_calib=int(len(calib_clicks)),
            n_test=int(len(test_clicks)),
            median_distance=float(np.median(radii)),
            variant="cqr",
            plug_in=plug_in,
            radii=radii,
        )

    # Fit isotonic regressor.
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(u_calib[finite], d_calib[finite])
    q_calib = iso.transform(u_calib[finite])
    residuals = np.maximum(0.0, d_calib[finite] - q_calib)
    n_c = int(finite.sum())
    level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
    q_hat = float(np.quantile(residuals, level, method="higher"))

    finite_test = np.isfinite(u_test)
    q_test = np.where(finite_test, iso.transform(np.where(finite_test, u_test, 0.0)), 0.0)
    radii = q_test + q_hat
    covered = (d_test <= radii).astype(float)
    coverage = float(covered.mean()) if covered.size else 0.0
    gap = float(max(0.0, (1 - alpha) - coverage))
    median_radius = float(np.median(radii[np.isfinite(radii)])) if np.isfinite(radii).any() else float("inf")
    return CoordCPResult(
        alpha=alpha,
        q_hat=q_hat,
        test_coverage=coverage,
        coverage_gap=gap,
        n_calib=int(len(calib_clicks)),
        n_test=int(len(test_clicks)),
        median_distance=median_radius,
        variant="cqr",
        plug_in=plug_in,
        radii=radii,
    )
