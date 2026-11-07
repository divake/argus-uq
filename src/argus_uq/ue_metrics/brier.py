"""Brier score (1950).

Mean squared error between min-max-normalized confidence
``1 - normalize(uncertainty)`` and the 0/1 correctness label.
Normalization matches :func:`ece`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.ue_metrics._helpers import min_max, to_arrays


def brier(correctness: Any, uncertainty: Any) -> float:
    """Brier score using min-max-normalized confidence."""

    c, u = to_arrays(correctness, uncertainty)
    if c.size == 0:
        return 0.0
    confidences = min_max(-u)
    return float(np.mean((confidences - c) ** 2))
