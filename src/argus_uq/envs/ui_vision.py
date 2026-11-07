"""UI-Vision Element Grounding dataset wrapper.

UI-Vision is a 2026 GUI grounding benchmark with 5,479 items across
desktop applications. The Argus paper uses a curated 501-item subset
stratified across Basic / Functional / Spatial sub-splits with
35/35/30% easy/medium/hard difficulty distribution.

License: MIT.

Discovery: ``UI_VISION_DIR`` env var or HF cache. Set
``UI_VISION_FULL=1`` to load the full 5,479-item benchmark instead of
the 501-item subset.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from argus_uq.envs.base import Env, GroundingItem


class UIVision(Env):
    """UI-Vision Element Grounding dataset (501-item subset by default)."""

    name = "ui_vision"
    cell_tag = "UIV"

    def _discover_root(self) -> Path:
        override = os.environ.get("UI_VISION_DIR")
        if override:
            return Path(override)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "UI-Vision auto-download requires huggingface_hub; "
                "either install it or set UI_VISION_DIR."
            ) from exc
        cached = snapshot_download(
            repo_id="UI-Vision/UI-Vision",
            repo_type="dataset",
        )
        return Path(cached)

    def _load_all(self) -> Iterator[GroundingItem]:
        full = bool(int(os.environ.get("UI_VISION_FULL", "0")))
        target = "full" if full else "subset_501"
        # Each subsplit has its own annotations file.
        for ann_path in sorted(self.root.glob(f"**/{target}*.json")):
            try:
                with open(ann_path) as fh:
                    raw_items = json.load(fh)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw_items, list):
                continue
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                bbox_raw = raw.get("bbox") or raw.get("ground_truth_bbox")
                if not bbox_raw or len(bbox_raw) != 4:
                    continue
                yield GroundingItem(
                    id=raw.get("id", "uiv_?"),
                    image_path=self.root / raw.get("image", ""),
                    instruction=raw.get("instruction", ""),
                    bbox=tuple(int(v) for v in bbox_raw),
                    img_size=tuple(raw.get("img_size", (0, 0))),
                    application=raw.get("application", ""),
                    platform=raw.get("platform", "desktop"),
                    ui_type=raw.get("ui_type", ""),
                    group=raw.get("subsplit", ""),
                    difficulty=raw.get("difficulty", ""),
                )
