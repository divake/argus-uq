"""OSWorld-G dataset wrapper.

OSWorld-G is a desktop-grounding benchmark with refusal-aware items
(some items have no clickable target; the agent must abstain). The
non-refusal subset contains 564 items.

License: research (see upstream).
HuggingFace: https://huggingface.co/datasets/xlangai/OSWorld-G
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from argus_uq.envs.base import Env, GroundingItem


class OSWorldG(Env):
    """OSWorld-G non-refusal grounding subset (564 items)."""

    name = "osworld_g"
    cell_tag = "OSG"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        include_refusal: bool = False,
        limit: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.include_refusal = include_refusal
        super().__init__(root=root, limit=limit, shuffle_seed=shuffle_seed)

    def _discover_root(self) -> Path:
        override = os.environ.get("OSWORLD_G_DIR")
        if override:
            return Path(override)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OSWorld-G auto-download requires huggingface_hub; "
                "either install it or set OSWORLD_G_DIR."
            ) from exc
        cached = snapshot_download(
            repo_id="xlangai/OSWorld-G",
            repo_type="dataset",
        )
        return Path(cached)

    def _load_all(self) -> Iterator[GroundingItem]:
        # Look for any JSON file with grounding items.
        for ann_path in sorted(self.root.glob("**/*.json")):
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
                refusal = bool(raw.get("refusal", bbox_raw is None))
                if refusal and not self.include_refusal:
                    continue
                if not bbox_raw or len(bbox_raw) != 4:
                    continue
                yield GroundingItem(
                    id=raw.get("id", f"osg_{raw.get('image_id', '?')}"),
                    image_path=self.root / raw.get("image", ""),
                    instruction=raw.get("instruction", ""),
                    bbox=tuple(int(v) for v in bbox_raw),
                    img_size=tuple(raw.get("img_size", (0, 0))),
                    application=raw.get("application", ""),
                    platform=raw.get("platform", "desktop"),
                    ui_type=raw.get("ui_type", ""),
                    group=raw.get("category", ""),
                )
