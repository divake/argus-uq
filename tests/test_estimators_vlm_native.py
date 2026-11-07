"""Unit tests for VLM-native estimators (HEDGE, IMG-HEDGE)."""

from __future__ import annotations

import math

import pytest

from argus_uq.estimators.vlm_native import (
    HEDGE,
    IMGHEDGE,
    HEDGE_PARAPHRASE_TEMPLATES,
    IMGHEDGE_PERTURBATIONS,
)
from tests._fixtures import make_record


def test_hedge_zero_when_all_paraphrase_clicks_agree():
    samples = [(f"para_{i}", (100, 200)) for i in range(3)]
    rec = make_record()
    rec["hedge_samples"] = samples
    assert HEDGE().score_item(rec) == 0.0


def test_hedge_max_when_clicks_scatter():
    samples = [
        ("para_0", (0, 0)),
        ("para_1", (500, 500)),
        ("para_2", (1000, 0)),
    ]
    rec = make_record()
    rec["hedge_samples"] = samples
    assert HEDGE().score_item(rec) == pytest.approx(math.log(3.0))


def test_imghedge_uses_baseline_plus_perturbed():
    """IMG-HEDGE includes the greedy click in the entropy computation."""

    rec = make_record(greedy_click=(100, 200))
    rec["imghedge_samples"] = [
        ("sat_up_30", (100, 200)),
        ("gauss_noise_8", (100, 200)),
        ("jpeg_q50", (100, 200)),
    ]
    # All 4 (1 baseline + 3 perturbed) at same point -> entropy = 0.
    assert IMGHEDGE().score_item(rec) == 0.0


def test_imghedge_handles_no_perturbed_clicks():
    rec = make_record()
    rec["imghedge_samples"] = []
    rec["greedy_click"] = (100, 200)
    # Only the baseline click; len < 2 -> 0.
    assert IMGHEDGE().score_item(rec) == 0.0


def test_paraphrase_templates_are_three():
    assert len(HEDGE_PARAPHRASE_TEMPLATES) == 3
    assert all("{instruction}" in t for t in HEDGE_PARAPHRASE_TEMPLATES)


def test_perturbations_are_three():
    assert len(IMGHEDGE_PERTURBATIONS) == 3
    names = [name for name, _ in IMGHEDGE_PERTURBATIONS]
    assert names == ["sat_up_30", "gauss_noise_8", "jpeg_q50"]


@pytest.mark.parametrize("method_cls", [HEDGE, IMGHEDGE])
def test_vlm_native_closed_source_compatible(method_cls):
    m = method_cls()
    assert m.available_on_closed_source is True
    assert m.family == "vlm_native"
