"""Conformal Risk Control on selective prediction.

Reference: Bates et al. 2024, *Conformal Risk Control* (ICLR 2024).

Given any UQ score, fit a threshold ``tau`` on calibration so that the
expected selective error rate among kept items is ``<= alpha``. At
test, items with ``u <= tau`` are kept; the rest are abstained on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from argus_uq.conformal.base import ClassConditionalCRCResult, CRCResult


def _fit_crc_threshold(
    u_calib: np.ndarray, c_calib: np.ndarray, alpha: float
) -> float:
    """Smallest tau such that calibration selective error <= alpha (with finite-sample correction)."""

    finite = np.isfinite(u_calib)
    u = u_calib[finite]
    c = c_calib[finite]
    if u.size == 0:
        return float("-inf")
    order = np.argsort(u, kind="mergesort")
    u_sorted = u[order]
    c_sorted = c[order]
    cum_correct = np.cumsum(c_sorted)
    k = np.arange(1, len(u) + 1)
    sel_err = 1.0 - (cum_correct / k)
    tolerance = 1.0 / (len(u) + 1)
    target = max(alpha - tolerance, 0.0)
    feasible = np.where(sel_err <= target)[0]
    if feasible.size == 0:
        return float("-inf")
    k_star = feasible[-1]
    return float(u_sorted[k_star])


def conformal_risk_control(
    u_calib: Sequence[float],
    c_calib: Sequence[int],
    u_test: Sequence[float],
    c_test: Sequence[int],
    alpha: float,
) -> CRCResult:
    """CRC on selective prediction (Bates et al. 2024)."""

    u_c = np.asarray(u_calib, dtype=np.float64)
    c_c = np.asarray(c_calib, dtype=np.float64)
    u_t = np.asarray(u_test, dtype=np.float64)
    c_t = np.asarray(c_test, dtype=np.float64)
    tau = _fit_crc_threshold(u_c, c_c, alpha)
    finite_t = np.isfinite(u_t)
    kept_mask = (u_t <= tau) & finite_t
    n_kept = int(kept_mask.sum())
    n_test = int(len(u_t))
    coverage = n_kept / max(n_test, 1)
    if n_kept == 0:
        sel_err = 0.0
    else:
        sel_err = float(1.0 - c_t[kept_mask].mean())
    risk_gap = max(0.0, sel_err - alpha)
    return CRCResult(
        alpha=alpha,
        tau=tau,
        test_coverage=coverage,
        selective_error=sel_err,
        risk_gap=risk_gap,
        n_kept=n_kept,
        n_calib=int(len(u_calib)),
        n_test=n_test,
    )


def class_conditional_crc(
    u_calib: Sequence[float],
    c_calib: Sequence[int],
    classes_calib: Sequence[str],
    u_test: Sequence[float],
    c_test: Sequence[int],
    classes_test: Sequence[str],
    alpha: float,
) -> ClassConditionalCRCResult:
    """Per-class CRC with one threshold per class label."""

    classes_calib_arr = np.asarray(classes_calib)
    classes_test_arr = np.asarray(classes_test)
    unique_classes = sorted(set(classes_calib_arr.tolist()) | set(classes_test_arr.tolist()))

    u_t = np.asarray(u_test, dtype=np.float64)
    c_t = np.asarray(c_test, dtype=np.float64)

    per_class: dict[str, CRCResult] = {}
    total_kept = 0
    total_kept_correct = 0
    total_test = 0
    for cls in unique_classes:
        calib_mask = classes_calib_arr == cls
        test_mask = classes_test_arr == cls
        if calib_mask.sum() < 5 or test_mask.sum() < 5:
            continue
        result = conformal_risk_control(
            u_calib=np.asarray(u_calib)[calib_mask],
            c_calib=np.asarray(c_calib)[calib_mask],
            u_test=u_t[test_mask],
            c_test=c_t[test_mask],
            alpha=alpha,
        )
        per_class[cls] = result
        total_kept += result.n_kept
        total_kept_correct += int(round(result.n_kept * (1.0 - result.selective_error)))
        total_test += int(test_mask.sum())

    aggregate_coverage = total_kept / max(total_test, 1)
    aggregate_sel_err = (
        1.0 - (total_kept_correct / total_kept) if total_kept else 0.0
    )
    aggregate_risk_gap = max(0.0, aggregate_sel_err - alpha)
    return ClassConditionalCRCResult(
        alpha=alpha,
        per_class=per_class,
        aggregate_coverage=aggregate_coverage,
        aggregate_selective_error=aggregate_sel_err,
        aggregate_risk_gap=aggregate_risk_gap,
    )
