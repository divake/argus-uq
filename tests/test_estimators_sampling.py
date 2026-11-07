"""Unit tests for sampling-family estimators."""

from __future__ import annotations

import math

import pytest

from argus_uq.estimators.sampling import (
    MCNSE,
    MCSE,
    LexicalSimilarity,
    SelfConsistency,
    SemanticEntropy,
    SemanticEntropyWeighted,
)
from tests._fixtures import make_record


def test_self_consistency_zero_when_all_agree():
    """All five samples at same point -> consistency = 1, return 0."""

    clicks = [(100, 200)] * 5
    rec = make_record(samples_clicks=clicks)
    assert SelfConsistency().score_item(rec) == 0.0


def test_self_consistency_high_when_scattered():
    """Five samples at five distinct distant points -> all singletons."""

    clicks = [(0, 0), (500, 0), (0, 500), (500, 500), (250, 250)]
    rec = make_record(samples_clicks=clicks)
    score = SelfConsistency().score_item(rec)
    assert score == pytest.approx(0.8)  # 1 - 1/5


def test_self_consistency_handles_missing_samples():
    rec = make_record()
    rec["samples_clicks"] = []
    assert SelfConsistency().score_item(rec) == 1.0


def test_semantic_entropy_zero_when_one_cluster():
    clicks = [(100, 200), (101, 201), (99, 200), (102, 199), (100, 198)]
    rec = make_record(samples_clicks=clicks)
    assert SemanticEntropy().score_item(rec) == 0.0


def test_semantic_entropy_max_when_all_singletons():
    """5 singleton clusters -> entropy = log(5)."""

    clicks = [(0, 0), (500, 0), (0, 500), (500, 500), (250, 250)]
    rec = make_record(samples_clicks=clicks)
    score = SemanticEntropy().score_item(rec)
    assert score == pytest.approx(math.log(5.0))


def test_semantic_entropy_weighted_falls_back_when_no_logprobs():
    """No sample logprobs -> fall back to cluster-frequency entropy."""

    rec = make_record()
    rec["samples_mean_seq_logprobs"] = []
    se = SemanticEntropy().score_item(rec)
    sew = SemanticEntropyWeighted().score_item(rec)
    assert sew == pytest.approx(se)


def test_mcse_returns_negative_mean_logprob():
    rec = make_record(samples_logprobs=[-1.0, -2.0, -3.0, -4.0, -5.0])
    score = MCSE().score_item(rec)
    assert score == pytest.approx(3.0)  # -mean([-1,-2,-3,-4,-5])


def test_mcse_inf_when_all_logprobs_inf():
    rec = make_record()
    rec["samples_mean_seq_logprobs"] = [float("-inf")] * 5
    assert MCSE().score_item(rec) == float("inf")


def test_mcnse_equals_mcse_in_pipeline():
    """Per pipeline, both consume the same length-normalised logprobs."""

    rec = make_record()
    assert MCNSE().score_item(rec) == pytest.approx(MCSE().score_item(rec))


def test_lexical_similarity_zero_for_identical_texts():
    rec = make_record(samples_texts=["click here"] * 5)
    assert LexicalSimilarity().score_item(rec) == 0.0


def test_lexical_similarity_high_for_distinct_texts():
    rec = make_record(samples_texts=["aaaa", "bbbb", "cccc", "dddd", "eeee"])
    score = LexicalSimilarity().score_item(rec)
    assert score > 0.5


@pytest.mark.parametrize(
    "method",
    [SelfConsistency(), SemanticEntropy(), LexicalSimilarity()],
)
def test_sampling_methods_closed_source_compatible(method):
    """SC, SE-frequency, lexical_sim work on both panels."""

    assert method.available_on_closed_source is True
    assert method.family == "sampling"


@pytest.mark.parametrize(
    "method",
    [SemanticEntropyWeighted(), MCSE(), MCNSE()],
)
def test_logprob_methods_open_source_only(method):
    assert method.available_on_closed_source is False
    assert method.family == "sampling"
