"""Unit tests for logit-family estimators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from argus_uq.estimators.logit import (
    MSP,
    MeanTokenEntropy,
    Perplexity,
    PerplexityExp,
    SequenceProbability,
)
from tests._fixtures import make_record


def test_msp_returns_finite_in_unit_range():
    rec = make_record()
    score = MSP().score_item(rec)
    assert 0.0 <= score <= 1.0
    assert math.isfinite(score)


def test_msp_handles_empty_logprobs():
    rec = make_record()
    rec["greedy_logprobs"] = []
    assert MSP().score_item(rec) == 1.0


def test_perplexity_finite_and_positive():
    rec = make_record()
    score = Perplexity().score_item(rec)
    assert math.isfinite(score)
    assert score > 0


def test_perplexity_rank_invariant_with_perplexity_exp():
    """exp is monotone, so the two should produce the same order."""

    rng = np.random.default_rng(0)
    p = Perplexity()
    pe = PerplexityExp()
    scores_p, scores_pe = [], []
    for i in range(20):
        rec = make_record(rng_seed=i)
        scores_p.append(p.score_item(rec))
        scores_pe.append(pe.score_item(rec))
    order_p = np.argsort(scores_p)
    order_pe = np.argsort(scores_pe)
    assert np.array_equal(order_p, order_pe)


def test_sequence_probability_no_length_norm():
    """SP grows with sequence length when logprobs are constant."""

    short = make_record(n_tokens=4)
    long = make_record(n_tokens=12)
    sp = SequenceProbability()
    assert sp.score_item(long) > sp.score_item(short)


def test_mean_token_entropy_zero_when_empty():
    rec = make_record()
    rec["greedy_entropies"] = []
    assert MeanTokenEntropy().score_item(rec) == 0.0


@pytest.mark.parametrize(
    "method",
    [MSP(), Perplexity(), PerplexityExp(), SequenceProbability(), MeanTokenEntropy()],
)
def test_logit_methods_closed_source_unavailable(method):
    """All logit methods must be flagged closed-source-incompatible."""

    assert method.available_on_closed_source is False
    assert method.family == "logit"


def test_score_batch_loops_through_records():
    rec = make_record()
    out = MSP().score_batch([rec, rec, rec])
    assert out.shape == (3,)
    assert (out == out[0]).all()
