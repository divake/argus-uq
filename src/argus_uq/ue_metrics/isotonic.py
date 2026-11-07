"""Isotonic Performance-Calibrated Confidence (PCC) calibrator.

The AAAI 2026 tutorial recommends isotonic regression as the default
mapping from a raw UQ score to a calibrated probability of correctness.
It preserves the ranking (so AUROC and PRR are unchanged by definition)
but produces confidences whose absolute values are meaningful for ECE,
Brier, and reliability diagrams.

Reference: AAAI 2026 Tutorial, "Uncertainty Quantification for Large
Language Models", Shelmanov / Rvanova, section 6 (slides 136-141).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from argus_uq.ue_metrics._helpers import binned_calibration_errors, to_arrays


@dataclass
class IsotonicCalibrator:
    """Isotonic-PCC calibrator: maps raw uncertainty -> P(correct).

    Fit on a calibration subset of ``(uncertainty, correctness)`` pairs.
    Internally uses ``sklearn.isotonic.IsotonicRegression`` with
    ``increasing=False`` so output is monotonically decreasing in
    uncertainty.

    For heavy-tailed UQ scores (e.g. raw Mahalanobis-RMD with values in
    ``[1e4, 1e6]``), an optional ``QuantileTransformer`` pre-transform
    converts the calibration distribution to uniform-[0, 1] before
    fitting the isotonic regressor.
    """

    iso: IsotonicRegression | None = field(default=None, repr=False)
    quantile: Any | None = field(default=None, repr=False)
    use_quantile: bool = True

    def fit(self, uncertainty: Any, correctness: Any) -> "IsotonicCalibrator":
        from sklearn.preprocessing import QuantileTransformer

        c = np.asarray(correctness, dtype=np.float64).ravel()
        u = np.asarray(uncertainty, dtype=np.float64).ravel()
        mask = np.isfinite(u)
        c, u = c[mask], u[mask]
        if u.size == 0 or c.size == 0:
            self.iso = None
            self.quantile = None
            return self

        if self.use_quantile and u.size >= 10:
            self.quantile = QuantileTransformer(
                n_quantiles=min(1000, u.size),
                output_distribution="uniform",
            )
            u_t = self.quantile.fit_transform(u.reshape(-1, 1)).ravel()
        else:
            self.quantile = None
            u_t = u

        if np.unique(c).size < 2:
            self.iso = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=False
            )
            base = float(c.mean())
            self.iso.fit(
                np.asarray([float(u_t.min()), float(u_t.max())]),
                np.asarray([base, base]),
            )
            return self
        self.iso = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=False
        )
        self.iso.fit(u_t, c)
        return self

    def transform(self, uncertainty: Any) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        u = np.asarray(uncertainty, dtype=np.float64).ravel()
        finite_mask = np.isfinite(u)
        out = np.full(u.shape, 0.5, dtype=np.float64)
        if finite_mask.any():
            u_finite = u[finite_mask]
            if self.quantile is not None:
                u_finite = self.quantile.transform(u_finite.reshape(-1, 1)).ravel()
            out[finite_mask] = self.iso.transform(u_finite)
        return out


def _ece_from_confidence(
    correctness: np.ndarray, confidences: np.ndarray, n_bins: int
) -> float:
    bins = binned_calibration_errors(confidences, correctness, n_bins)
    return float(sum(w * abs(conf - acc) for w, conf, acc in bins))


def _mce_from_confidence(
    correctness: np.ndarray, confidences: np.ndarray, n_bins: int
) -> float:
    bins = binned_calibration_errors(confidences, correctness, n_bins)
    return float(max((abs(conf - acc) for _w, conf, acc in bins), default=0.0))


def ece_isotonic(
    correctness_test: Any,
    uncertainty_test: Any,
    correctness_calib: Any,
    uncertainty_calib: Any,
    n_bins: int = 20,
) -> float:
    """ECE with isotonic-PCC calibrated confidences."""

    calib = IsotonicCalibrator().fit(uncertainty_calib, correctness_calib)
    c_test, u_test = to_arrays(correctness_test, uncertainty_test)
    if c_test.size == 0:
        return 0.0
    conf = calib.transform(u_test)
    return _ece_from_confidence(c_test, conf, n_bins)


def mce_isotonic(
    correctness_test: Any,
    uncertainty_test: Any,
    correctness_calib: Any,
    uncertainty_calib: Any,
    n_bins: int = 20,
) -> float:
    """MCE with isotonic-PCC calibrated confidences."""

    calib = IsotonicCalibrator().fit(uncertainty_calib, correctness_calib)
    c_test, u_test = to_arrays(correctness_test, uncertainty_test)
    if c_test.size == 0:
        return 0.0
    conf = calib.transform(u_test)
    return _mce_from_confidence(c_test, conf, n_bins)


def brier_isotonic(
    correctness_test: Any,
    uncertainty_test: Any,
    correctness_calib: Any,
    uncertainty_calib: Any,
) -> float:
    """Brier score with isotonic-PCC calibrated confidences."""

    calib = IsotonicCalibrator().fit(uncertainty_calib, correctness_calib)
    c_test, u_test = to_arrays(correctness_test, uncertainty_test)
    if c_test.size == 0:
        return 0.0
    conf = calib.transform(u_test)
    return float(np.mean((conf - c_test) ** 2))
