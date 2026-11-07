"""Selective accuracy at fixed coverage.

Reference: Geifman & El-Yaniv 2017, *Selective Classification for Deep
Neural Networks* (NeurIPS 2017).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.ue_metrics._helpers import to_arrays


def selective_accuracy_at_coverage(
    correctness: Any, uncertainty: Any, coverage: float
) -> float:
    """Accuracy on the ``coverage`` fraction of least-uncertain items."""

    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    c, u = to_arrays(correctness, uncertainty)
    n = len(c)
    if n == 0:
        return 0.0
    k = max(1, int(round(coverage * n)))
    order = np.argsort(u, kind="mergesort")
    return float(c[order[:k]].mean())
