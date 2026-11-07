"""Prediction-Rejection Ratio metrics.

Reference: Malinin & Gales 2018, *Predictive Uncertainty Estimation via
Prior Networks*. Implementation mirrors lm-polygraph
``ue_metrics/pred_rej_area.py:PredictionRejectionArea``. The
``_normalized`` variants follow the AAAI 2026 tutorial slide 148:

    PRR_norm = (AUC_unc - AUC_rnd) / (AUC_oracle - AUC_rnd)

The headline paper metric is ``prr_normalized_at_05`` (TACL 2025
convention; truncates the noisy tail).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.ue_metrics._helpers import to_arrays


def prr(correctness: Any, uncertainty: Any, max_rejection: float = 1.0) -> float:
    """Raw area under the prediction-rejection curve (lm-polygraph form)."""

    c, u = to_arrays(correctness, uncertainty)
    n = len(c)
    if n == 0:
        return 0.0
    num_rej = max(1, int(max_rejection * n))
    order = np.argsort(u, kind="mergesort")
    sorted_correct = c[order]
    cumsum = np.cumsum(sorted_correct)[-num_rej:]
    denom = np.arange(n - num_rej + 1, n + 1)
    scores = (cumsum / denom)[::-1]
    return float(np.sum(scores) / num_rej)


def prr_normalized(
    correctness: Any,
    uncertainty: Any,
    max_rejection: float = 1.0,
    seed: int = 0,
    n_random: int = 200,
) -> float:
    """Oracle-normalized Prediction-Rejection Ratio (Malinin & Gales 2018)."""

    c, u = to_arrays(correctness, uncertainty)
    n = len(c)
    if n == 0:
        return 0.0
    if c.sum() in (0, n):
        return 0.0
    auc_unc = prr(c, u, max_rejection=max_rejection)
    auc_oracle = prr(c, 1.0 - c, max_rejection=max_rejection)
    rng = np.random.default_rng(seed)
    rand_aucs = []
    for _ in range(n_random):
        rand_u = rng.random(n)
        rand_aucs.append(prr(c, rand_u, max_rejection=max_rejection))
    auc_rand = float(np.mean(rand_aucs))
    denom = auc_oracle - auc_rand
    if abs(denom) < 1e-12:
        return 0.0
    return float((auc_unc - auc_rand) / denom)


def prr_at_05(correctness: Any, uncertainty: Any) -> float:
    """Raw PRR truncated at 50% rejection (avoids noisy tail)."""

    return prr(correctness, uncertainty, max_rejection=0.5)


def prr_normalized_at_05(correctness: Any, uncertainty: Any) -> float:
    """Oracle-normalized PRR at 50% rejection (paper headline metric)."""

    return prr_normalized(correctness, uncertainty, max_rejection=0.5)
