"""Spearman rank correlation between uncertainty and -correctness.

Reference: scipy ``spearmanr``. Matches lm-polygraph
``ue_metrics/spearmanr.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from argus_uq.ue_metrics._helpers import to_arrays


def spearman_corr(correctness: Any, uncertainty: Any) -> float:
    """Spearman rho between uncertainty and -correctness."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size < 3 or np.unique(c).size < 2 or np.unique(u).size < 2:
        return 0.0
    rho = stats.spearmanr(u, -c).correlation
    return float(rho) if np.isfinite(rho) else 0.0
