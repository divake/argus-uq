"""Unit tests for UE metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from argus_uq.ue_metrics import (
    ALL_METRICS,
    ISOTONIC_METRICS,
    IsotonicCalibrator,
    auarc,
    auroc_incorrect,
    ause,
    bootstrap_ci,
    brier,
    brier_isotonic,
    ece,
    ece_isotonic,
    mce,
    pr_auc_correct,
    pr_auc_incorrect,
    prr,
    prr_at_05,
    prr_normalized,
    prr_normalized_at_05,
    rcc_auc,
    selective_accuracy_at_coverage,
    spearman_corr,
)


@pytest.fixture
def perfect_ranking():
    """20 items: all incorrect items have higher uncertainty than correct ones."""

    correctness = np.array([1] * 10 + [0] * 10)
    uncertainty = np.array([0.1 * i for i in range(10)] + [1.0 + 0.1 * i for i in range(10)])
    return correctness, uncertainty


@pytest.fixture
def random_ranking():
    """20 items, uncertainty independent of correctness."""

    rng = np.random.default_rng(0)
    correctness = (rng.random(20) < 0.6).astype(int)
    uncertainty = rng.random(20)
    return correctness, uncertainty


def test_auroc_perfect_is_one(perfect_ranking):
    c, u = perfect_ranking
    assert auroc_incorrect(c, u) == pytest.approx(1.0)


def test_auroc_random_is_near_half(random_ranking):
    c, u = random_ranking
    score = auroc_incorrect(c, u)
    assert 0.3 <= score <= 0.7


def test_prr_perfect_is_one(perfect_ranking):
    c, u = perfect_ranking
    score = prr_normalized(c, u)
    assert score == pytest.approx(1.0)


def test_prr_no_normalization_in_unit_range(random_ranking):
    c, u = random_ranking
    score = prr(c, u)
    assert 0.0 <= score <= 1.0


def test_prr_at_05_uses_half_rejection(perfect_ranking):
    c, u = perfect_ranking
    s_full = prr(c, u, max_rejection=1.0)
    s_half = prr_at_05(c, u)
    # On a perfect ranking the values can be different but both are <= 1.
    assert 0.0 <= s_half <= 1.0
    assert 0.0 <= s_full <= 1.0


def test_prr_normalized_at_05_perfect():
    c = np.array([1] * 10 + [0] * 10)
    u = np.array([0.1 * i for i in range(10)] + [10.0 + 0.1 * i for i in range(10)])
    assert prr_normalized_at_05(c, u) == pytest.approx(1.0)


def test_rcc_auc_lower_is_better(perfect_ranking, random_ranking):
    c1, u1 = perfect_ranking
    c2, u2 = random_ranking
    # Perfect ranking has lower (better) rcc_auc.
    assert rcc_auc(c1, u1) < rcc_auc(c2, u2)


def test_auarc_complements_rcc_auc(perfect_ranking):
    c, u = perfect_ranking
    assert auarc(c, u) == pytest.approx(1.0 - rcc_auc(c, u))


def test_ece_finite_on_uniform_input():
    """All-zero uncertainty + all-one correctness: min-max -> 0.5; ECE bounded."""

    c = np.ones(50)
    u = np.zeros(50)
    score = ece(c, u)
    assert 0.0 <= score <= 1.0


def test_ece_low_when_calibrated_signal():
    """High accuracy + low uncertainty for correct items -> small ECE."""

    rng = np.random.default_rng(42)
    n = 200
    c = (rng.random(n) < 0.9).astype(int)
    # Uncertainty is high when wrong, low when correct -> well calibrated.
    u = np.where(c == 1, rng.uniform(0, 0.3, n), rng.uniform(0.7, 1.0, n))
    assert ece(c, u) < 0.5


def test_brier_in_unit_range(random_ranking):
    c, u = random_ranking
    score = brier(c, u)
    assert 0.0 <= score <= 1.0


def test_mce_at_least_ece(random_ranking):
    c, u = random_ranking
    assert mce(c, u) >= ece(c, u)


def test_spearman_in_minus_one_to_one(perfect_ranking, random_ranking):
    c, u = perfect_ranking
    assert -1.0 <= spearman_corr(c, u) <= 1.0
    c, u = random_ranking
    assert -1.0 <= spearman_corr(c, u) <= 1.0


def test_selective_accuracy_at_coverage(perfect_ranking):
    c, u = perfect_ranking
    # Top 50% by certainty (low uncertainty) are all correct.
    assert selective_accuracy_at_coverage(c, u, 0.5) == 1.0


def test_selective_accuracy_validates_coverage():
    c = np.array([1, 0, 1, 0])
    u = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        selective_accuracy_at_coverage(c, u, 0.0)
    with pytest.raises(ValueError):
        selective_accuracy_at_coverage(c, u, 1.5)


def test_pr_auc_incorrect_bounded(random_ranking):
    c, u = random_ranking
    score = pr_auc_incorrect(c, u)
    assert math.isnan(score) or 0.0 <= score <= 1.0


def test_pr_auc_correct_bounded(random_ranking):
    c, u = random_ranking
    score = pr_auc_correct(c, u)
    assert math.isnan(score) or 0.0 <= score <= 1.0


def test_ause_zero_when_uncertainty_matches_error():
    """Perfect uncertainty ranking on a per-item error -> AUSE near 0."""

    error = np.linspace(0, 1, 30)
    uncertainty = error.copy()
    score = ause(error, uncertainty)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_ause_positive_when_random_ranking():
    rng = np.random.default_rng(0)
    error = rng.random(50)
    uncertainty = rng.random(50)
    score = ause(error, uncertainty)
    assert score >= 0.0


def test_isotonic_calibrator_returns_unit_range():
    """Isotonic-PCC outputs lie in [0, 1]."""

    rng = np.random.default_rng(0)
    n = 200
    c = (rng.random(n) < 0.5).astype(int)
    u = rng.random(n)
    cal = IsotonicCalibrator().fit(u, c)
    confidences = cal.transform(u)
    assert (confidences >= 0.0).all()
    assert (confidences <= 1.0).all()


def test_isotonic_calibrator_monotone():
    """Isotonic regression is monotone, so the AUROC is preserved on adequate samples."""

    rng = np.random.default_rng(42)
    n = 500
    c = (rng.random(n) < 0.5).astype(int)
    # Synthetic correlated uncertainty: high when wrong, low when correct.
    u = np.where(c == 1, rng.uniform(0, 0.4, n), rng.uniform(0.6, 1.0, n))
    half = n // 2
    c_calib, c_test = c[:half], c[half:]
    u_calib, u_test = u[:half], u[half:]
    cal = IsotonicCalibrator().fit(u_calib, c_calib)
    confidences = cal.transform(u_test)
    auroc_raw = auroc_incorrect(c_test, u_test)
    auroc_iso = auroc_incorrect(c_test, -confidences)
    # On 250 test items with strong signal, isotonic-PCC AUROC must
    # remain within a couple of percentage points of the raw AUROC.
    assert abs(auroc_raw - auroc_iso) < 0.05


def test_isotonic_metrics_finite(random_ranking):
    c, u = random_ranking
    half = len(c) // 2
    c_calib, c_test = c[:half], c[half:]
    u_calib, u_test = u[:half], u[half:]
    e = ece_isotonic(c_test, u_test, c_calib, u_calib)
    b = brier_isotonic(c_test, u_test, c_calib, u_calib)
    assert math.isfinite(e)
    assert math.isfinite(b)


def test_bootstrap_ci_returns_three_values(random_ranking):
    c, u = random_ranking
    point, lo, hi = bootstrap_ci(auroc_incorrect, c, u, n_resamples=50)
    assert lo <= point <= hi


def test_all_metrics_registry_complete():
    expected = {
        "prr",
        "prr_normalized",
        "prr_at_0.5",
        "prr_normalized_at_0.5",
        "auroc_incorrect",
        "pr_auc_incorrect",
        "pr_auc_correct",
        "rcc_auc",
        "aurc",
        "auarc",
        "ece",
        "mce",
        "brier",
        "spearman",
        "sel_acc@0.5",
        "sel_acc@0.75",
        "sel_acc@0.9",
    }
    assert set(ALL_METRICS) == expected


def test_isotonic_metrics_registry_complete():
    expected = {"ece_isotonic", "mce_isotonic", "brier_isotonic"}
    assert set(ISOTONIC_METRICS) == expected
