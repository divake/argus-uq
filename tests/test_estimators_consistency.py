"""Unit tests for consistency-family estimators."""

from __future__ import annotations

import math

import pytest

from argus_uq.estimators.consistency import CCP, CoCoA, CoCoA1MCA
from tests._fixtures import make_record


def test_ccp_zero_when_all_samples_match_greedy():
    clicks = [(100, 200), (101, 200), (100, 201), (102, 199), (99, 200)]
    rec = make_record(greedy_click=(100, 200), samples_clicks=clicks)
    assert CCP().score_item(rec) == 0.0


def test_ccp_one_when_no_samples_match():
    clicks = [(0, 0), (500, 0), (0, 500), (500, 500), (300, 300)]
    rec = make_record(greedy_click=(100, 200), samples_clicks=clicks)
    assert CCP().score_item(rec) == 1.0


def test_ccp_partial_match():
    """3 of 5 within tolerance -> 1 - 3/5 = 0.4."""

    clicks = [(100, 200), (101, 200), (102, 201), (500, 500), (300, 300)]
    rec = make_record(greedy_click=(100, 200), samples_clicks=clicks)
    assert CCP().score_item(rec) == pytest.approx(0.4)


def test_ccp_handles_none_greedy():
    rec = make_record(greedy_click=None)
    score = CCP().score_item(rec)
    assert math.isfinite(score)


def test_cocoa_canonical_form():
    """U_inf * U_cons. With logprobs=[-0.5]*8 and 0 disagreements -> 0."""

    rec = make_record(greedy_click=(100, 200))
    rec["greedy_logprobs"] = [-0.5] * 8
    rec["samples_clicks"] = [(100, 200)] * 5
    # u_inf = -sum(logprobs) = 4.0; u_cons = 0/5 = 0; product = 0.
    assert CoCoA().score_item(rec) == 0.0


def test_cocoa_positive_when_disagree():
    rec = make_record(greedy_click=(100, 200))
    rec["greedy_logprobs"] = [-0.5] * 8
    rec["samples_clicks"] = [(0, 0), (500, 500), (300, 300), (200, 200), (100, 100)]
    # u_inf = 4.0; u_cons = 5/5 = 1.0; product = 4.0.
    assert CoCoA().score_item(rec) == pytest.approx(4.0)


def test_cocoa_1mca_form():
    """1 - C * A where C = exp(mean logprob)."""

    rec = make_record(greedy_click=(100, 200))
    rec["greedy_logprobs"] = [0.0] * 8  # confidence = 1.0
    rec["samples_clicks"] = [(100, 200)] * 5
    # C = 1.0; A = 1.0; 1 - 1*1 = 0.
    assert CoCoA1MCA().score_item(rec) == 0.0


def test_cocoa_1mca_zero_confidence_returns_one():
    rec = make_record()
    rec["greedy_logprobs"] = []
    rec["samples_clicks"] = []
    rec["greedy_click"] = None
    assert CoCoA1MCA().score_item(rec) == 1.0


def test_ccp_is_closed_source_compatible():
    assert CCP().available_on_closed_source is True


@pytest.mark.parametrize("method", [CoCoA(), CoCoA1MCA()])
def test_cocoa_variants_open_source_only(method):
    assert method.available_on_closed_source is False
