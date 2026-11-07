"""Unit tests for attention-family estimators."""

from __future__ import annotations

import math

import pytest

from argus_uq.estimators.attention import (
    UQAC,
    Focus,
    RAUQProxyFull,
    attn_all_scores,
)
from tests._fixtures import make_record


def test_focus_returns_finite():
    rec = make_record()
    score = Focus().score_item(rec)
    assert math.isfinite(score)


def test_focus_handles_missing_attention():
    rec = make_record(with_attention=False)
    assert Focus().score_item(rec) == 0.0


def test_rauq_proxy_full_returns_unit_interval_ish():
    rec = make_record()
    score = RAUQProxyFull().score_item(rec)
    assert math.isfinite(score)


def test_rauq_proxy_full_default_when_no_attention():
    rec = make_record(with_attention=False)
    assert RAUQProxyFull().score_item(rec) == 0.5


def test_uqac_returns_finite():
    rec = make_record()
    score = UQAC().score_item(rec)
    assert math.isfinite(score)


def test_uqac_zero_when_no_attention():
    rec = make_record(with_attention=False)
    assert UQAC().score_item(rec) == 0.0


def test_attn_all_scores_returns_three_methods():
    rec = make_record()
    out = attn_all_scores(rec["attention"])
    assert set(out.keys()) == {"focus", "rauq_proxy_full", "uqac"}
    for v in out.values():
        assert math.isfinite(v)


@pytest.mark.parametrize("method_cls", [Focus, RAUQProxyFull, UQAC])
def test_attention_methods_open_source_only(method_cls):
    m = method_cls()
    assert m.available_on_closed_source is False
    assert m.family == "attention"
