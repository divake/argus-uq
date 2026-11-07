"""Score using pre-computed per-item UQ scores from a Cell parquet.

This is the fast path used by :func:`argus_uq.score`. The released
per-item parquets carry one column per method already; we only need
to evaluate metrics across N random calibration/test seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from argus_uq.scoring.multi_seed import SEED_LIST_CANONICAL, multi_seed_metrics


@dataclass
class CachedScoreResult:
    """Multi-seed metrics for one (cell, method)."""

    cell_id: str
    method: str
    n_seeds: int
    auroc: float
    auroc_std: float
    auroc_ci_lo: float
    auroc_ci_hi: float
    prr_norm_at_05: float
    prr_norm_at_05_std: float
    ause: float
    ause_std: float
    aurc: float
    aurc_std: float
    ece_iso: float
    ece_iso_std: float
    brier_iso: float
    brier_iso_std: float


def score_cached(
    cell_id: str,
    method: str,
    *,
    correctness: np.ndarray,
    uq_scores: np.ndarray,
    errors: np.ndarray | None = None,
    seeds: int = 50,
    fixed_seed: int | None = None,
) -> CachedScoreResult:
    """Compute multi-seed metrics from pre-cached per-item UQ scores.

    Parameters
    ----------
    cell_id:
        Cell identifier (for the result label).
    method:
        Method name (for the result label).
    correctness:
        Per-item correctness array (0/1), shape (n_items,).
    uq_scores:
        Per-item UQ score array, shape (n_items,).
    errors:
        Per-item error magnitude for AUSE; defaults to
        ``log(1 + correctness_inverted)`` if not given. For real
        grounding cells, callers should pass ``log(1 + dist_norm)``.
    seeds:
        Number of random calibration/test splits to average over.
        Default 50 matches the paper.
    fixed_seed:
        If set and seeds=1, use exactly this seed for bit-exact
        reproducibility.

    Returns
    -------
    CachedScoreResult
    """

    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    if seeds == 1 and fixed_seed is not None:
        seed_list = [fixed_seed]
    else:
        seed_list = list(SEED_LIST_CANONICAL[:seeds])

    if errors is None:
        # Crude fallback: error = 1 if incorrect else 0.
        errors = (1.0 - np.asarray(correctness, dtype=np.float64))

    metrics = multi_seed_metrics(
        correctness=correctness,
        uq_scores=uq_scores,
        errors=errors,
        seeds=seed_list,
    )

    return CachedScoreResult(
        cell_id=cell_id,
        method=method,
        n_seeds=len(seed_list),
        auroc=float(metrics["auroc"].mean()),
        auroc_std=float(metrics["auroc"].std()),
        auroc_ci_lo=float(np.quantile(metrics["auroc"], 0.025)),
        auroc_ci_hi=float(np.quantile(metrics["auroc"], 0.975)),
        prr_norm_at_05=float(metrics["prr_norm_at_05"].mean()),
        prr_norm_at_05_std=float(metrics["prr_norm_at_05"].std()),
        ause=float(metrics["ause"].mean()),
        ause_std=float(metrics["ause"].std()),
        aurc=float(metrics["aurc"].mean()),
        aurc_std=float(metrics["aurc"].std()),
        ece_iso=float(metrics["ece_iso"].mean()),
        ece_iso_std=float(metrics["ece_iso"].std()),
        brier_iso=float(metrics["brier_iso"].mean()),
        brier_iso_std=float(metrics["brier_iso"].std()),
    )
