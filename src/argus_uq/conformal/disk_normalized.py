"""Disk-Normalized (Lei et al. 2018 locally-weighted conformal).

Per-item radius is modulated by a plug-in UQ score:
``radius_i = q_hat * f(u_i, u_calib)``, where ``f`` is the local
weighting (here: linear in min-max-normalized uncertainty rank).

Reference: Lei et al. 2018, *Distribution-Free Predictive Inference
for Regression* (JASA), section 4.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from argus_uq.conformal.base import CoordCPResult, point_to_bbox_distance


def disk_normalized(
    calib_clicks: Sequence[tuple[int, int] | None],
    calib_bboxes: Sequence[Sequence[float]],
    calib_uq: Sequence[float],
    test_clicks: Sequence[tuple[int, int] | None],
    test_bboxes: Sequence[Sequence[float]],
    test_uq: Sequence[float],
    alpha: float,
    plug_in: str = "saplma",
) -> CoordCPResult:
    """Locally-weighted CP click-disk.

    The per-item radius is
    ``r_i = q_hat * (1 + lambda * normalized_rank(u_i))``
    where ``lambda = 1.0`` and ``normalized_rank`` is in ``[0, 1]``.
    Calibration solves for ``q_hat`` so the rescaled distances
    ``d_i / r_i`` satisfy the standard CP coverage at level ``1 - alpha``.
    """

    u_calib = np.asarray(calib_uq, dtype=np.float64)
    u_test = np.asarray(test_uq, dtype=np.float64)

    # Local weighting factor: rank-normalised uncertainty in [0, 1].
    def _rank_normalize(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if ref.size == 0:
            return np.zeros_like(u)
        finite_ref = ref[np.isfinite(ref)]
        if finite_ref.size == 0:
            return np.zeros_like(u)
        sorted_ref = np.sort(finite_ref)
        return np.searchsorted(sorted_ref, u) / max(len(sorted_ref), 1)

    w_calib = 1.0 + _rank_normalize(u_calib, u_calib)
    w_test = 1.0 + _rank_normalize(u_test, u_calib)

    d_calib = np.asarray(
        [
            point_to_bbox_distance(c, b)
            for c, b in zip(calib_clicks, calib_bboxes, strict=False)
        ]
    )
    finite = np.isfinite(d_calib) & np.isfinite(w_calib) & (w_calib > 0)
    if finite.sum() < 2:
        q_hat = float("inf")
    else:
        scaled = d_calib[finite] / w_calib[finite]
        n_c = int(finite.sum())
        level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
        q_hat = float(np.quantile(scaled, level, method="higher"))

    d_test = np.asarray(
        [
            point_to_bbox_distance(c, b)
            for c, b in zip(test_clicks, test_bboxes, strict=False)
        ]
    )
    radii = q_hat * w_test
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
        variant="normalized",
        plug_in=plug_in,
        radii=radii,
    )
