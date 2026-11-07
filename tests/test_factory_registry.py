"""Tests for the FactoryEstimator registry."""

from __future__ import annotations

import pytest

from argus_uq.estimators import (  # noqa: F401  -- triggers auto-registration
    attention,
    consistency,
    density,
    logit,
    prompt,
    sampling,
    vlm_native,
)
from argus_uq.factory import FactoryEstimator


def test_27_methods_registered():
    assert len(FactoryEstimator.list_names()) == 27


def test_canonical_names_present():
    expected_open = {
        # logit (5)
        "msp",
        "perplexity",
        "perplexity_exp",
        "sequence_probability",
        "mean_token_entropy",
        # sampling (6)
        "self_consistency",
        "semantic_entropy",
        "semantic_entropy_weighted",
        "mcse",
        "mcnse",
        "lexical_similarity",
        # consistency (3)
        "ccp",
        "cocoa",
        "cocoa_1mca",
        # density (5)
        "mahalanobis",
        "mahalanobis_rmd",
        "mahalanobis_rde",
        "saplma",
        "sep",
        # attention (3)
        "focus",
        "rauq_proxy_full",
        "uqac",
        # prompt (3)
        "p_true",
        "verbalized_1s",
        "verbalized_2s",
        # vlm_native (2)
        "hedge",
        "imghedge",
    }
    assert set(FactoryEstimator.list_names()) == expected_open


def test_family_counts():
    by_fam = FactoryEstimator.list_by_family()
    assert len(by_fam["logit"]) == 5
    assert len(by_fam["sampling"]) == 6
    assert len(by_fam["consistency"]) == 3
    assert len(by_fam["density"]) == 5
    assert len(by_fam["attention"]) == 3
    assert len(by_fam["prompt"]) == 3
    assert len(by_fam["vlm_native"]) == 2


def test_8_closed_source_methods():
    """Closed-source-compatible method count must be exactly 8."""

    closed = [
        n
        for n in FactoryEstimator.list_names()
        if FactoryEstimator.build(n).available_on_closed_source
    ]
    expected_closed = {
        "self_consistency",
        "semantic_entropy",
        "lexical_similarity",
        "ccp",
        "verbalized_1s",
        "verbalized_2s",
        "hedge",
        "imghedge",
    }
    assert set(closed) == expected_closed
    assert len(closed) == 8


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        FactoryEstimator.build("not_a_method")


def test_build_returns_callable_with_score_item():
    est = FactoryEstimator.build("msp")
    assert hasattr(est, "score_item")
    assert callable(est.score_item)


def test_demoted_methods_not_registered():
    """Five demoted attention methods must not be in the registry."""

    demoted = {
        "rauq_proxy",
        "attention_rollout",
        "llm_check",
        "attention_entropy",
        "lookback_lens_unsupervised",
    }
    registered = set(FactoryEstimator.list_names())
    assert demoted.isdisjoint(registered)
