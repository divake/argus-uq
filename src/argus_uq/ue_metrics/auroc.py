"""AUROC + PR-AUC metrics.

Reference: sklearn ``roc_auc_score`` and ``average_precision_score``,
wrapped to handle ``inf`` the same way lm-polygraph does
(``ue_metrics/risk_cov_curve.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from argus_uq.ue_metrics._helpers import preprocess_inf, to_arrays


def auroc_incorrect(correctness: Any, uncertainty: Any) -> float:
    """AUROC with uncertainty as the score for predicting incorrectness."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0 or c.sum() in (0, len(c)):
        return 0.5
    cleaned = np.asarray([preprocess_inf(x, u) for x in u], dtype=np.float64)
    labels = 1.0 - c
    return float(roc_auc_score(labels, cleaned))


def pr_auc_incorrect(correctness: Any, uncertainty: Any) -> float:
    """Average-precision (PR-AUC) for predicting incorrectness.

    Companion to AUROC for class-imbalanced benchmarks.
    """

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0 or c.sum() in (0, len(c)):
        return float("nan")
    cleaned = np.asarray([preprocess_inf(x, u) for x in u], dtype=np.float64)
    labels = 1.0 - c
    return float(average_precision_score(labels, cleaned))


def pr_auc_correct(correctness: Any, uncertainty: Any) -> float:
    """Average-precision (PR-AUC) for predicting correctness."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0 or c.sum() in (0, len(c)):
        return float("nan")
    cleaned = np.asarray([preprocess_inf(x, u) for x in u], dtype=np.float64)
    return float(average_precision_score(c, -cleaned))
