"""Unit tests for density-family estimators (all trainable)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from argus_uq.estimators.density import (
    SAPLMA,
    Mahalanobis,
    MahalanobisRDE,
    MahalanobisRMD,
    SemanticEntropyProbe,
)
from tests._fixtures import labels_from_records, make_corpus, make_record


@pytest.fixture(scope="module")
def calib_corpus():
    return make_corpus(n_items=50, p_correct=0.6, rng_seed=0)


@pytest.fixture(scope="module")
def test_record():
    return make_record(rng_seed=999)


def test_mahalanobis_must_be_fit_before_call(test_record):
    m = Mahalanobis()
    with pytest.raises(RuntimeError):
        m.score_item(test_record)


def test_mahalanobis_fits_and_scores(calib_corpus, test_record):
    labels = labels_from_records(calib_corpus)
    m = Mahalanobis()
    m.fit(calib_corpus, labels)
    score = m.score_item(test_record)
    assert math.isfinite(score)
    assert score >= 0.0


def test_mahalanobis_rmd_can_be_negative(calib_corpus, test_record):
    """RMD subtracts background -> result can be negative."""

    labels = labels_from_records(calib_corpus)
    m = MahalanobisRMD()
    m.fit(calib_corpus, labels)
    score = m.score_item(test_record)
    assert math.isfinite(score)


def test_mahalanobis_rde_returns_finite_or_inf(calib_corpus, test_record):
    """RDE returns inf when MCD fitting fails on tiny corpora."""

    labels = labels_from_records(calib_corpus)
    m = MahalanobisRDE(n_components=10)
    m.fit(calib_corpus, labels)
    score = m.score_item(test_record)
    assert math.isfinite(score) or math.isinf(score)


def test_sep_returns_unit_interval(calib_corpus, test_record):
    labels = labels_from_records(calib_corpus)
    m = SemanticEntropyProbe()
    m.fit(calib_corpus, labels)
    score = m.score_item(test_record)
    assert 0.0 <= score <= 1.0


def test_saplma_returns_unit_interval(calib_corpus, test_record):
    """SAPLMA: 256-128-64 MLP probe; score in [0, 1]."""

    import warnings

    from sklearn.exceptions import ConvergenceWarning

    labels = labels_from_records(calib_corpus)
    # Tiny MLP on tiny corpus; ignore the inevitable convergence warning.
    m = SAPLMA(hidden_sizes=(16, 8), max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(calib_corpus, labels)
    score = m.score_item(test_record)
    assert 0.0 <= score <= 1.0


def test_density_handles_empty_hidden_states():
    """Records with empty hidden states should not crash."""

    rec = make_record()
    rec["greedy_last_hidden"] = []
    m = SemanticEntropyProbe()
    m.fit(make_corpus(20), labels_from_records(make_corpus(20)))
    score = m.score_item(rec)
    assert score == 1.0


def test_saplma_falls_back_when_single_class():
    """If calib has only one class, SAPLMA returns the base rate."""

    records = [make_record(correct=1, rng_seed=i) for i in range(20)]
    labels = np.ones(20, dtype=np.int64)
    m = SAPLMA(hidden_sizes=(8,))
    m.fit(records, labels)
    score = m.score_item(make_record(rng_seed=99))
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "method_cls",
    [Mahalanobis, MahalanobisRMD, MahalanobisRDE, SemanticEntropyProbe, SAPLMA],
)
def test_density_methods_open_source_only(method_cls):
    m = method_cls()
    assert m.available_on_closed_source is False
    assert m.trainable is True
    assert m.family == "density"
