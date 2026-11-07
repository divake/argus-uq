"""ScoringManager - orchestrates per-cell scoring of UQ methods.

The manager:
1. Takes a list of estimators (or method names).
2. For each method, fits trainable estimators on the calibration split.
3. Scores every record with every estimator.
4. Computes UE metrics (AUROC, PRR, AUSE, AURC, ECE_iso, Brier_iso).
5. Returns a per-method results table.

This is the core of :func:`argus_uq.score`. The Listing-1 ``score()``
function is a thin wrapper that uses cached per-method scores from the
parquet (Mode A); ``ScoringManager.score_records()`` is for users who
want to score new records (Mode B; reruns each estimator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from argus_uq.estimators.base import Estimator
from argus_uq.factory import FactoryEstimator
from argus_uq.ue_metrics import (
    auroc_incorrect,
    ause,
    brier_isotonic,
    ece_isotonic,
    prr_normalized_at_05,
    rcc_auc,
)


@dataclass
class MethodResult:
    """Per-method metrics on one (calib, test) split."""

    method: str
    family: str
    auroc: float
    prr_norm_at_05: float
    ause: float
    aurc: float
    ece_iso: float
    brier_iso: float
    n_calib: int
    n_test: int


def stratified_split(
    labels: np.ndarray,
    test_frac: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified ``test_frac/(1-test_frac)`` split conditioned on labels.

    Returns ``(calib_idx, test_idx)`` numpy arrays. The smaller (calib)
    split is the (1 - test_frac) fraction.
    """

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    test_idx_list: list[int] = []
    calib_idx_list: list[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_frac))
        # Guarantee at least one item per class in each split if the class size allows.
        n_test = max(1, min(n_test, len(cls_idx) - 1)) if len(cls_idx) >= 2 else len(cls_idx)
        test_idx_list.extend(cls_idx[:n_test].tolist())
        calib_idx_list.extend(cls_idx[n_test:].tolist())
    test_idx = np.asarray(sorted(test_idx_list), dtype=np.int64)
    calib_idx = np.asarray(sorted(calib_idx_list), dtype=np.int64)
    assert len(test_idx) + len(calib_idx) == n
    return calib_idx, test_idx


class ScoringManager:
    """Per-cell scorer: fits trainable methods, scores all methods, runs metrics."""

    def __init__(
        self,
        estimators: list[Estimator] | list[str] | None = None,
        *,
        seed: int = 42,
        ignore_exceptions: bool = True,
    ) -> None:
        if estimators is None:
            estimators = FactoryEstimator.list_names()
        self.estimators: list[Estimator] = [
            FactoryEstimator.build(e) if isinstance(e, str) else e
            for e in estimators
        ]
        self.seed = seed
        self.ignore_exceptions = ignore_exceptions

    def score_records(
        self,
        records: list[dict[str, Any]],
        labels: np.ndarray,
        errors: np.ndarray | None = None,
        test_frac: float = 0.8,
    ) -> list[MethodResult]:
        """Score every estimator on a (calib, test) split of records.

        Parameters
        ----------
        records:
            List of per-item record dicts.
        labels:
            Per-item correctness (0/1).
        errors:
            Per-item error magnitude for AUSE; defaults to ``log(1 + dist_norm)``
            extracted from each record.
        test_frac:
            Fraction of items going into the test split.

        Returns
        -------
        list[MethodResult]
        """

        labels = np.asarray(labels)
        if errors is None:
            errors = np.asarray(
                [np.log1p(r.get("dist_norm", 0.0)) for r in records],
                dtype=np.float64,
            )
        else:
            errors = np.asarray(errors, dtype=np.float64)

        calib_idx, test_idx = stratified_split(labels, test_frac=test_frac, seed=self.seed)
        calib_recs = [records[i] for i in calib_idx]
        test_recs = [records[i] for i in test_idx]
        calib_labels = labels[calib_idx]
        test_labels = labels[test_idx]
        test_errors = errors[test_idx]

        out: list[MethodResult] = []
        for est in self.estimators:
            try:
                if est.trainable:
                    est.fit(calib_recs, calib_labels)
                u_calib = est.score_batch(calib_recs)
                u_test = est.score_batch(test_recs)
            except Exception:
                if not self.ignore_exceptions:
                    raise
                out.append(
                    MethodResult(
                        method=est.name,
                        family=est.family,
                        auroc=float("nan"),
                        prr_norm_at_05=float("nan"),
                        ause=float("nan"),
                        aurc=float("nan"),
                        ece_iso=float("nan"),
                        brier_iso=float("nan"),
                        n_calib=int(len(calib_idx)),
                        n_test=int(len(test_idx)),
                    )
                )
                continue
            out.append(
                MethodResult(
                    method=est.name,
                    family=est.family,
                    auroc=auroc_incorrect(test_labels, u_test),
                    prr_norm_at_05=prr_normalized_at_05(test_labels, u_test),
                    ause=ause(test_errors, u_test),
                    aurc=rcc_auc(test_labels, u_test),
                    ece_iso=ece_isotonic(test_labels, u_test, calib_labels, u_calib),
                    brier_iso=brier_isotonic(test_labels, u_test, calib_labels, u_calib),
                    n_calib=int(len(calib_idx)),
                    n_test=int(len(test_idx)),
                )
            )
        return out

    def __repr__(self) -> str:
        return f"ScoringManager(n_estimators={len(self.estimators)}, seed={self.seed})"
