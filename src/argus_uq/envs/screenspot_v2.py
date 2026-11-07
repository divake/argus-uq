"""ScreenSpot-v2 dataset wrapper.

ScreenSpot-v2 is the commodity-software grounding benchmark
predecessor of ScreenSpot-Pro. 1,272 items across mobile / web /
desktop with simpler app distribution.

License: Apache 2.0.
HuggingFace: https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2

The v2 schema differs from v1: bboxes are ``[x, y, w, h]`` rather than
``[x1, y1, x2, y2]``; the loader normalises to xyxy. Application
metadata is sparse - we synthesise ``application`` / ``platform`` /
``group`` from the ``data_source`` field.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from argus_uq.envs.base import Env, GroundingItem


class ScreenSpotV2(Env):
    """ScreenSpot-v2 grounding dataset (1,272 items)."""

    name = "screenspot_v2"
    cell_tag = "V2"

    def _discover_root(self) -> Path:
        override = os.environ.get("SCREENSPOT_V2_DIR")
        if override:
            p = Path(override)
            if not p.is_dir():
                raise FileNotFoundError(f"SCREENSPOT_V2_DIR={override} not found")
            return p
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ScreenSpot-v2 auto-download requires huggingface_hub; "
                "either install it or set SCREENSPOT_V2_DIR."
            ) from exc
        cached = snapshot_download(
            repo_id="HongxinLi/ScreenSpot_v2",
            repo_type="dataset",
        )
        return Path(cached)

    def _load_all(self) -> Iterator[GroundingItem]:
        # v2 ships per-split JSON files (mobile / web / desktop / ...).
        for ann_path in sorted(self.root.glob("**/*.json")):
            try:
                with open(ann_path) as fh:
                    raw_items = json.load(fh)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw_items, list):
                continue
            for i, raw in enumerate(raw_items):
                if not isinstance(raw, dict) or "instruction" not in raw:
                    continue
                bbox_raw = raw.get("bbox") or raw.get("ground_truth_bbox")
                if not bbox_raw or len(bbox_raw) != 4:
                    continue
                x, y, w, h = bbox_raw
                # v2 stores [x, y, w, h]; convert to [x1, y1, x2, y2].
                if w < 1.0 and h < 1.0:
                    # Already xyxy with some 0-1 normalisation; skip.
                    bbox = (int(x), int(y), int(w), int(h))
                else:
                    bbox = (int(x), int(y), int(x + w), int(y + h))
                data_source = raw.get("data_source", ann_path.stem)
                platform = self._infer_platform(data_source)
                yield GroundingItem(
                    id=raw.get("id", f"{ann_path.stem}_{i}"),
                    image_path=self.root / raw.get("img_filename", ""),
                    instruction=raw["instruction"],
                    bbox=bbox,
                    img_size=tuple(raw.get("img_size", (0, 0))),
                    application=raw.get("application", ""),
                    platform=platform,
                    ui_type=raw.get("data_type", raw.get("ui_type", "")),
                    group=data_source,
                )

    @staticmethod
    def _infer_platform(data_source: str) -> str:
        ds = (data_source or "").lower()
        if "mobile" in ds or "android" in ds or "ios" in ds:
            return "mobile"
        if "web" in ds:
            return "web"
        if "desktop" in ds:
            return "desktop"
        return ""
