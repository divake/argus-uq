"""Unit tests for grounding environments.

These tests do NOT load real datasets - they verify the class
contracts, parameter validation, and filtering helpers using a tiny
fixture dataset on a temp directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus_uq.envs import OSWorldG, ScreenSpotPro, ScreenSpotV2, UIVision
from argus_uq.envs.base import Env, GroundingItem


def test_grounding_item_basic():
    item = GroundingItem(
        id="x1",
        image_path=Path("/tmp/x.png"),
        instruction="click here",
        bbox=(10, 20, 30, 40),
        img_size=(100, 100),
    )
    assert item.point_in_bbox(15, 25) is True
    assert item.point_in_bbox(0, 0) is False


@pytest.mark.parametrize(
    "cls,expected_tag,expected_name",
    [
        (ScreenSpotV2, "V2", "screenspot_v2"),
        (ScreenSpotPro, "SP", "screenspot_pro"),
        (OSWorldG, "OSG", "osworld_g"),
        (UIVision, "UIV", "ui_vision"),
    ],
)
def test_env_class_attributes(cls, expected_tag, expected_name):
    assert cls.cell_tag == expected_tag
    assert cls.name == expected_name
    assert issubclass(cls, Env)


def test_screenspot_pro_loads_from_fixture(tmp_path):
    """Build a minimal SS-Pro fixture and load it."""

    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    (annotations_dir / "fixture.json").write_text(
        json.dumps([
            {
                "id": "fix_001",
                "img_filename": "fix_001.png",
                "instruction": "click the menu",
                "bbox": [10, 20, 50, 60],
                "img_size": [800, 600],
                "application": "android_studio",
                "platform": "linux",
                "ui_type": "icon",
                "group": "ide",
            }
        ])
    )
    (images_dir / "fix_001.png").write_bytes(b"")  # empty placeholder

    env = ScreenSpotPro(root=tmp_path)
    assert len(env) == 1
    item = env[0]
    assert item.id == "fix_001"
    assert item.bbox == (10, 20, 50, 60)
    assert item.application == "android_studio"


def test_env_filter_returns_subset(tmp_path):
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    (annotations_dir / "fixture.json").write_text(
        json.dumps([
            {
                "id": f"fix_{i}",
                "img_filename": f"fix_{i}.png",
                "instruction": "click",
                "bbox": [0, 0, 10, 10],
                "img_size": [100, 100],
                "application": "app_a" if i % 2 == 0 else "app_b",
                "platform": "linux",
                "ui_type": "icon",
                "group": "ide",
            }
            for i in range(6)
        ])
    )
    env = ScreenSpotPro(root=tmp_path)
    assert len(env) == 6
    filtered = env.filter(applications=["app_a"])
    assert len(filtered) == 3
    assert all(it.application == "app_a" for it in filtered)


def test_env_limit_truncates(tmp_path):
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    (annotations_dir / "fixture.json").write_text(
        json.dumps([
            {
                "id": f"fix_{i}",
                "img_filename": f"fix_{i}.png",
                "instruction": "click",
                "bbox": [0, 0, 10, 10],
                "img_size": [100, 100],
                "application": "x",
                "platform": "linux",
                "ui_type": "icon",
                "group": "ide",
            }
            for i in range(20)
        ])
    )
    env = ScreenSpotPro(root=tmp_path, limit=5)
    assert len(env) == 5


def test_env_iter_protocol(tmp_path):
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    (annotations_dir / "fix.json").write_text(
        json.dumps([
            {
                "id": "a",
                "img_filename": "a.png",
                "instruction": "click",
                "bbox": [0, 0, 10, 10],
                "img_size": [100, 100],
                "application": "x",
                "platform": "linux",
                "ui_type": "icon",
                "group": "g",
            }
        ])
    )
    env = ScreenSpotPro(root=tmp_path)
    items = list(env)
    assert len(items) == 1
    assert items[0].id == "a"
