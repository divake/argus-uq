"""Unit tests for cached + multi-seed scoring."""

from __future__ import annotations

import numpy as np
import pytest

from argus_uq.scoring import (
    SEED_LIST_CANONICAL,
    MultiSeedScorer,
    multi_seed_metrics,
    score_cached,
)


def test_seed_list_canonical_starts_with_42():
    assert SEED_LIST_CANONICAL[0] == 42
    assert len(SEED_LIST_CANONICAL) == 50


def test_multi_seed_metrics_returns_six_metrics():
    rng = np.random.default_rng(0)
    n = 200
    c = (rng.random(n) < 0.6).astype(int)
    u = rng.random(n)
    e = (1.0 - c) + rng.random(n) * 0.1
    out = multi_seed_metrics(c, u, e, seeds=list(range(5)))
    assert set(out.keys()) == {"auroc", "prr_norm_at_05", "ause", "aurc", "ece_iso", "brier_iso"}
    for arr in out.values():
        assert arr.shape == (5,)


def test_score_cached_returns_score_result():
    rng = np.random.default_rng(0)
    n = 200
    c = (rng.random(n) < 0.6).astype(int)
    u = np.where(c == 1, rng.uniform(0, 0.4, n), rng.uniform(0.6, 1.0, n))
    res = score_cached("PTxOSG", "saplma", correctness=c, uq_scores=u, seeds=10)
    assert res.cell_id == "PTxOSG"
    assert res.method == "saplma"
    assert res.n_seeds == 10
    assert 0.0 <= res.auroc <= 1.0
    # Strong signal -> AUROC well above 0.5.
    assert res.auroc > 0.7


def test_score_cached_fixed_seed_reproducible():
    rng = np.random.default_rng(0)
    n = 100
    c = (rng.random(n) < 0.6).astype(int)
    u = rng.random(n)
    res1 = score_cached("X", "method", correctness=c, uq_scores=u, seeds=1, fixed_seed=42)
    res2 = score_cached("X", "method", correctness=c, uq_scores=u, seeds=1, fixed_seed=42)
    assert res1.auroc == res2.auroc
    assert res1.auroc_std == 0.0


def test_score_cached_validates_seeds():
    with pytest.raises(ValueError):
        score_cached("X", "y", correctness=np.array([1, 0]), uq_scores=np.array([0.5, 0.5]), seeds=0)


def test_multi_seed_scorer_class():
    rng = np.random.default_rng(0)
    n = 100
    c = (rng.random(n) < 0.5).astype(int)
    u = rng.random(n)
    scorer = MultiSeedScorer(seeds=[42, 0, 1])
    out = scorer.score(c, u)
    assert out["auroc"].shape == (3,)
