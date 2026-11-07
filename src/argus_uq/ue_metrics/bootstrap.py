"""Generic bootstrap CI helper for UE metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from argus_uq.ue_metrics._helpers import to_arrays


def bootstrap_ci(
    metric_fn: Callable[[Any, Any], float],
    correctness: Any,
    uncertainty: Any,
    n_resamples: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` bootstrap estimates for ``metric_fn``."""

    c, u = to_arrays(correctness, uncertainty)
    n = len(c)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[k] = metric_fn(c[idx], u[idx])
    point = metric_fn(c, u)
    lo = float(np.quantile(samples, (1 - ci) / 2))
    hi = float(np.quantile(samples, 1 - (1 - ci) / 2))
    return float(point), lo, hi
