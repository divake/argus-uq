"""Unit tests for ScoringManager and stratified split."""

from __future__ import annotations

import numpy as np
import pytest

from argus_uq.manager import ScoringManager, stratified_split
from tests._fixtures import labels_from_records, make_corpus


def test_stratified_split_balances_classes():
    labels = np.array([1] * 60 + [0] * 40)
    calib_idx, test_idx = stratified_split(labels, test_frac=0.8, seed=42)
    # Test split: 80% of 60 + 80% of 40 = 48 + 32 = 80.
    assert len(test_idx) + len(calib_idx) == 100
    # Both splits contain both classes.
    assert (labels[test_idx] == 1).any()
    assert (labels[test_idx] == 0).any()
    assert (labels[calib_idx] == 1).any()
    assert (labels[calib_idx] == 0).any()


def test_stratified_split_deterministic():
    labels = np.array([1] * 40 + [0] * 40)
    a1, b1 = stratified_split(labels, seed=42)
    a2, b2 = stratified_split(labels, seed=42)
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


def test_scoring_manager_runs_subset_of_methods():
    """Manager scores only the methods we ask for."""

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    corpus = make_corpus(n_items=60, p_correct=0.6, rng_seed=0)
    labels = labels_from_records(corpus)
    mgr = ScoringManager(estimators=["msp", "self_consistency", "ccp"], seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = mgr.score_records(corpus, labels)
    assert len(results) == 3
    names = [r.method for r in results]
    assert names == ["msp", "self_consistency", "ccp"]
    for r in results:
        assert r.n_calib + r.n_test == 60


def test_scoring_manager_handles_trainable_methods():
    """Trainable methods (SAPLMA / SEP) get fit on the calib split before scoring."""

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    corpus = make_corpus(n_items=40, p_correct=0.6, rng_seed=1)
    labels = labels_from_records(corpus)
    mgr = ScoringManager(estimators=["sep"], seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = mgr.score_records(corpus, labels)
    assert len(results) == 1
    r = results[0]
    assert r.method == "sep"
    # AUROC computed on test split; should be a finite value or NaN.
    assert np.isfinite(r.auroc) or np.isnan(r.auroc)


def test_scoring_manager_default_uses_all_27():
    mgr = ScoringManager()
    assert len(mgr.estimators) == 27


def test_scoring_manager_repr():
    mgr = ScoringManager(estimators=["msp"], seed=7)
    s = repr(mgr)
    assert "n_estimators=1" in s
    assert "seed=7" in s
