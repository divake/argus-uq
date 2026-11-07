"""Unit tests for prompt-family estimators."""

from __future__ import annotations

import pytest

from argus_uq.estimators.prompt import PTrueMethod, Verbalized1S, Verbalized2S
from tests._fixtures import make_record


def test_p_true_inverts_confidence():
    rec = make_record(p_true=0.7)
    assert PTrueMethod().score_item(rec) == pytest.approx(0.3)


def test_p_true_returns_one_when_missing():
    rec = make_record(p_true=None)
    assert PTrueMethod().score_item(rec) == 1.0


def test_verbalized_1s_inverts():
    rec = make_record(verbalized_1s=0.8)
    assert Verbalized1S().score_item(rec) == pytest.approx(0.2)


def test_verbalized_2s_inverts():
    rec = make_record(verbalized_2s=0.75)
    assert Verbalized2S().score_item(rec) == pytest.approx(0.25)


def test_p_true_open_source_only():
    assert PTrueMethod().available_on_closed_source is False


@pytest.mark.parametrize("method_cls", [Verbalized1S, Verbalized2S])
def test_verbalised_closed_source_compatible(method_cls):
    assert method_cls().available_on_closed_source is True
    assert method_cls().family == "prompt"
