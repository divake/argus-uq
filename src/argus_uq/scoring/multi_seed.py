"""Multi-seed metric aggregation across stratified cal/test splits.

Each (cell, method) gets ``len(seeds)`` evaluations of 6 metrics
(AUROC, PRR_norm@0.5, AUSE, AURC, ECE_iso, Brier_iso). The canonical
seed list is ``[42, 0, 1, ..., 48]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.manager import stratified_split
from argus_uq.ue_metrics import (
    auroc_incorrect,
    ause,
    brier_isotonic,
    ece_isotonic,
    prr_normalized_at_05,
    rcc_auc,
)


# Canonical seed list: 42 first, then 0..48.
SEED_LIST_CANONICAL: list[int] = [42] + list(range(49))


METRIC_NAMES = ("auroc", "prr_norm_at_05", "ause", "aurc", "ece_iso", "brier_iso")


def multi_seed_metrics(
    correctness: np.ndarray,
    uq_scores: np.ndarray,
    errors: np.ndarray,
    seeds: list[int],
    test_frac: float = 0.8,
) -> dict[str, np.ndarray]:
    """Run the (calib, test) split N times and aggregate 6 metrics.

    Returns a dict keyed by metric name, value = array of length
    ``len(seeds)``.
    """

    correctness = np.asarray(correctness, dtype=np.float64)
    uq_scores = np.asarray(uq_scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)

    out: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for seed in seeds:
        calib_idx, test_idx = stratified_split(correctness, test_frac=test_frac, seed=seed)
        c_calib = correctness[calib_idx]
        c_test = correctness[test_idx]
        u_calib = uq_scores[calib_idx]
        u_test = uq_scores[test_idx]
        e_test = errors[test_idx]
        out["auroc"].append(auroc_incorrect(c_test, u_test))
        out["prr_norm_at_05"].append(prr_normalized_at_05(c_test, u_test))
        out["ause"].append(ause(e_test, u_test))
        out["aurc"].append(rcc_auc(c_test, u_test))
        out["ece_iso"].append(ece_isotonic(c_test, u_test, c_calib, u_calib))
        out["brier_iso"].append(brier_isotonic(c_test, u_test, c_calib, u_calib))

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


class MultiSeedScorer:
    """Run multi-seed scoring on a per-method per-cell basis."""

    def __init__(self, seeds: list[int] | None = None, test_frac: float = 0.8) -> None:
        self.seeds = seeds if seeds is not None else SEED_LIST_CANONICAL
        self.test_frac = test_frac

    def score(
        self,
        correctness: np.ndarray,
        uq_scores: np.ndarray,
        errors: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if errors is None:
            errors = 1.0 - np.asarray(correctness, dtype=np.float64)
        return multi_seed_metrics(
            correctness, uq_scores, errors, self.seeds, self.test_frac
        )
