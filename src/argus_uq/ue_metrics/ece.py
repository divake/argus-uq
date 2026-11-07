"""Expected and Maximum Calibration Error.

Reference: Guo et al. 2017, *On Calibration of Modern Neural Networks*
(ICML 2017). Implementation mirrors lm-polygraph ``ue_metrics/ece.py``
with per-method min-max normalization of the uncertainty scalar so
methods whose uncertainty scale is not in ``[0, 1]`` (perplexity,
Mahalanobis, etc.) are calibrated fairly. Default ``n_bins=20`` matches
lm-polygraph.
"""

from __future__ import annotations

from typing import Any

from argus_uq.ue_metrics._helpers import binned_calibration_errors, min_max, to_arrays


def ece(correctness: Any, uncertainty: Any, n_bins: int = 20) -> float:
    """Expected Calibration Error (Guo et al. 2017)."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0:
        return 0.0
    confidences = min_max(-u)
    bins = binned_calibration_errors(confidences, c, n_bins)
    return float(sum(w * abs(conf - acc) for w, conf, acc in bins))


def mce(correctness: Any, uncertainty: Any, n_bins: int = 20) -> float:
    """Maximum Calibration Error using the same binning as ECE."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0:
        return 0.0
    confidences = min_max(-u)
    bins = binned_calibration_errors(confidences, c, n_bins)
    return float(max((abs(conf - acc) for _w, conf, acc in bins), default=0.0))
