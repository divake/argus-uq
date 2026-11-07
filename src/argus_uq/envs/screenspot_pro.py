"""ScreenSpot-Pro dataset wrapper.

ScreenSpot-Pro is a single-step grounding benchmark: each item is a
screenshot of a professional software interface plus a natural-language
instruction; the agent predicts a click coordinate and is scored by
whether the predicted point falls inside the gold bounding box. 1,581
items across 26 applications.

License: Apache 2.0.
HuggingFace: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
GitHub: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding

Discovery order:
1. ``SCREENSPOT_PRO_DIR`` environment variable.
2. HuggingFace cache (auto-downloads if missing).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from argus_uq.envs.base import Env, GroundingItem


class ScreenSpotPro(Env):
    """ScreenSpot-Pro grounding dataset (1,581 items)."""

    name = "screenspot_pro"
    cell_tag = "SP"

    def _discover_root(self) -> Path:
        override = os.environ.get("SCREENSPOT_PRO_DIR")
        if override:
            p = Path(override)
            if not (p / "annotations").is_dir():
                raise FileNotFoundError(
                    f"SCREENSPOT_PRO_DIR={override} does not contain 'annotations/'"
                )
            return p
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ScreenSpot-Pro auto-download requires huggingface_hub; "
                "either install it or set SCREENSPOT_PRO_DIR."
            ) from exc
        cached = snapshot_download(
            repo_id="likaixin/ScreenSpot-Pro",
            repo_type="dataset",
            allow_patterns=["annotations/*.json", "images/**/*.png"],
        )
        return Path(cached)

    def _load_all(self) -> Iterator[GroundingItem]:
        annotations_dir = self.root / "annotations"
        images_dir = self.root / "images"
        for ann_path in sorted(annotations_dir.glob("*.json")):
            with open(ann_path) as fh:
                raw_items = json.load(fh)
            for raw in raw_items:
                yield GroundingItem(
                    id=raw["id"],
                    image_path=images_dir / raw["img_filename"],
                    instruction=raw["instruction"],
                    bbox=tuple(raw["bbox"]),
                    img_size=tuple(raw["img_size"]),
                    application=raw.get("application", ""),
                    platform=raw.get("platform", ""),
                    ui_type=raw.get("ui_type", ""),
                    group=raw.get("group", ""),
                )
