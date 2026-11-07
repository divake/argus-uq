"""Risk-Coverage AUC metrics.

Reference: lm-polygraph ``ue_metrics/risk_cov_curve.py:RiskCoverageCurveAUC``.
Lower values indicate better uncertainty (risk accumulates more slowly
under low-uncertainty selection).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.ue_metrics._helpers import to_arrays


def rcc_auc(correctness: Any, uncertainty: Any) -> float:
    """Area under the Risk-Coverage curve. Lower is better."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0:
        return 0.0
    risk = 1.0 - c
    order = np.argsort(u, kind="mergesort")
    cum_risk = np.cumsum(risk[order])
    cum_risk = cum_risk / np.arange(1, len(u) + 1)
    return float(cum_risk.mean())


def auarc(correctness: Any, uncertainty: Any) -> float:
    """Area under the Accuracy-Rejection curve. ``1 - rcc_auc``."""

    return 1.0 - rcc_auc(correctness, uncertainty)


# Alias for the paper's preferred name.
aurc = rcc_auc
