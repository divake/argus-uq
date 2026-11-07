"""Area Under the Sparsification Error curve (AUSE).

AUSE measures how well the uncertainty score ranks items by error
*magnitude* (not just by binary correctness). Lower is better.

Reference: Ilg et al. 2018, *Uncertainty Estimates and Multi-Hypotheses
Networks for Optical Flow* (ECCV 2018). Adapted to GUI grounding by
using ``log(1 + d_norm)`` as the per-item error target, where
``d_norm = distance_to_gold_bbox / image_diagonal``.

Implementation: at each rejection threshold ``k``, compute the mean
error of the kept items; the sparsification curve is the model's mean
error vs the oracle mean error (rejecting by true error). AUSE is the
area between the two curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.ue_metrics._helpers import to_arrays


def ause(error: Any, uncertainty: Any) -> float:
    """Area Under the Sparsification Error curve.

    Parameters
    ----------
    error:
        Per-item error magnitude (not 0/1 correctness). Higher = more
        wrong. The Argus convention is ``log(1 + d_norm)`` for grounding.
    uncertainty:
        Per-item uncertainty score. Higher = more uncertain.

    Returns
    -------
    float:
        AUSE in ``[0, max_error]``. Lower is better; 0 = perfect ranking.
    """

    e, u = to_arrays(error, uncertainty)
    n = len(e)
    if n < 2:
        return 0.0

    # Sparsification: sort by uncertainty (ascending), report mean
    # error of kept items.
    order_unc = np.argsort(u, kind="mergesort")
    e_unc = e[order_unc]
    keep = np.arange(n - 1, 0, -1)  # kept counts: n-1, n-2, ..., 1
    cum_unc = np.cumsum(e_unc)
    mean_unc = cum_unc[: n - 1] / np.arange(1, n)
    # Reverse so we get mean-of-kept as we drop highest-uncertainty.
    mean_unc = mean_unc[::-1]

    # Oracle: sort by error (ascending) - oracle drops highest-error first.
    order_orc = np.argsort(e, kind="mergesort")
    e_orc = e[order_orc]
    cum_orc = np.cumsum(e_orc)
    mean_orc = cum_orc[: n - 1] / np.arange(1, n)
    mean_orc = mean_orc[::-1]

    # Sparsification error = (mean_unc - mean_orc) at each rejection
    # threshold. AUSE = mean of the curve.
    return float(np.mean(mean_unc - mean_orc))
