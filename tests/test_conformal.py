"""Unit tests for conformal-prediction click-disks + CRC."""

from __future__ import annotations

import numpy as np
import pytest

from argus_uq.conformal import (
    class_conditional_crc,
    conformal_risk_control,
    disk_cqr,
    disk_fixed,
    disk_normalized,
    dispatch,
    inductive_cp_coords,
    point_to_bbox_distance,
)


def test_point_to_bbox_inside_returns_zero():
    assert point_to_bbox_distance((50, 50), (10, 10, 100, 100)) == 0.0


def test_point_to_bbox_outside_positive():
    d = point_to_bbox_distance((200, 50), (10, 10, 100, 100))
    assert d == pytest.approx(100.0, abs=1e-6)


def test_point_to_bbox_none_returns_inf():
    assert point_to_bbox_distance(None, (10, 10, 100, 100)) == float("inf")


@pytest.fixture
def cp_fixture():
    """50 calib + 50 test items; clicks are centered on bbox with noise."""

    rng = np.random.default_rng(0)
    n = 50
    bboxes = [(0, 0, 100, 100)] * (2 * n)
    # Calib clicks: center +/- small noise.
    calib_clicks = [
        (50 + int(rng.normal(0, 10)), 50 + int(rng.normal(0, 10))) for _ in range(n)
    ]
    test_clicks = [
        (50 + int(rng.normal(0, 10)), 50 + int(rng.normal(0, 10))) for _ in range(n)
    ]
    return calib_clicks, bboxes[:n], test_clicks, bboxes[n:]


def test_disk_fixed_meets_target_coverage(cp_fixture):
    calib_clicks, calib_b, test_clicks, test_b = cp_fixture
    res = disk_fixed(calib_clicks, calib_b, test_clicks, test_b, alpha=0.10)
    # Most calibration distances are 0 (clicks inside bbox), so coverage = 1.
    assert res.test_coverage >= 0.85
    assert res.coverage_gap <= 0.05
    assert res.variant == "fixed"
    assert res.q_hat >= 0


def test_inductive_cp_coords_alias():
    """Back-compat alias matches disk_fixed."""

    assert inductive_cp_coords is disk_fixed


def test_disk_normalized_uses_per_item_radii(cp_fixture):
    calib_clicks, calib_b, test_clicks, test_b = cp_fixture
    rng = np.random.default_rng(0)
    calib_uq = rng.random(len(calib_clicks))
    test_uq = rng.random(len(test_clicks))
    res = disk_normalized(
        calib_clicks, calib_b, calib_uq, test_clicks, test_b, test_uq, alpha=0.10
    )
    assert res.variant == "normalized"
    assert res.radii is not None
    assert res.radii.shape == (len(test_clicks),)


def test_disk_cqr_falls_back_when_calib_too_small():
    """CQR with <5 finite calibration items falls back to fixed."""

    res = disk_cqr(
        calib_clicks=[(50, 50)] * 3,
        calib_bboxes=[(0, 0, 100, 100)] * 3,
        calib_uq=[0.1, 0.2, 0.3],
        test_clicks=[(60, 60)] * 5,
        test_bboxes=[(0, 0, 100, 100)] * 5,
        test_uq=[0.1, 0.2, 0.3, 0.4, 0.5],
        alpha=0.10,
    )
    assert res.variant == "cqr"
    assert res.radii is not None


def test_conformal_risk_control(cp_fixture):
    rng = np.random.default_rng(0)
    n = 100
    correctness = (rng.random(n) < 0.7).astype(int)
    # Uncertainty is low when correct, high when incorrect (so threshold works).
    uncertainty = np.where(correctness == 1, rng.uniform(0, 0.3, n), rng.uniform(0.7, 1.0, n))
    half = n // 2
    res = conformal_risk_control(
        u_calib=uncertainty[:half],
        c_calib=correctness[:half],
        u_test=uncertainty[half:],
        c_test=correctness[half:],
        alpha=0.10,
    )
    assert 0 <= res.test_coverage <= 1
    assert 0 <= res.selective_error <= 1


def test_class_conditional_crc_per_class():
    rng = np.random.default_rng(0)
    n = 100
    correctness = (rng.random(n) < 0.7).astype(int)
    uncertainty = rng.random(n)
    classes = ["icon" if i % 2 == 0 else "text" for i in range(n)]
    half = n // 2
    res = class_conditional_crc(
        u_calib=uncertainty[:half],
        c_calib=correctness[:half],
        classes_calib=classes[:half],
        u_test=uncertainty[half:],
        c_test=correctness[half:],
        classes_test=classes[half:],
        alpha=0.20,
    )
    # Both classes should produce a per-class result.
    assert "icon" in res.per_class
    assert "text" in res.per_class


def test_dispatch_routes_correctly(cp_fixture):
    calib_clicks, calib_b, test_clicks, test_b = cp_fixture
    res = dispatch(
        calib_clicks=calib_clicks,
        calib_bboxes=calib_b,
        test_clicks=test_clicks,
        test_bboxes=test_b,
        alpha=0.10,
        variant="fixed",
    )
    assert res.variant == "fixed"


def test_dispatch_normalized_requires_uq(cp_fixture):
    calib_clicks, calib_b, test_clicks, test_b = cp_fixture
    with pytest.raises(ValueError):
        dispatch(
            calib_clicks=calib_clicks,
            calib_bboxes=calib_b,
            test_clicks=test_clicks,
            test_bboxes=test_b,
            alpha=0.10,
            variant="normalized",
        )


def test_dispatch_unknown_variant():
    with pytest.raises(ValueError):
        dispatch(
            calib_clicks=[],
            calib_bboxes=[],
            test_clicks=[],
            test_bboxes=[],
            alpha=0.10,
            variant="not_a_variant",
        )
