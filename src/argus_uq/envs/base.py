"""Grounding environment base.

A grounding env is an iterable collection of ``GroundingItem`` records.
Each item bundles an image path, an instruction, a gold bounding box,
and metadata for slicing / stratification (application / platform /
ui_type / group / difficulty).

The env subclass discovers the dataset on disk (env-var override or
HuggingFace cache fallback) and yields items lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroundingItem:
    """One grounding example.

    Fields:

    - ``id``: stable identifier for the item (used as parquet row key).
    - ``image_path``: absolute path to the screenshot.
    - ``instruction``: natural-language click target.
    - ``bbox``: gold bounding box ``(x1, y1, x2, y2)`` in pixel coords.
    - ``img_size``: ``(w, h)`` original image dimensions.
    - ``application``: app name (e.g. "android_studio").
    - ``platform``: ``"linux" | "macos" | "windows" | "web"``.
    - ``ui_type``: ``"text" | "icon" | "image" | ...``.
    - ``group``: high-level cluster (e.g. "office" | "creative").
    - ``difficulty``: optional ``"easy" | "medium" | "hard"``.
    """

    id: str
    image_path: Path
    instruction: str
    bbox: tuple[int, int, int, int]
    img_size: tuple[int, int]
    application: str = ""
    platform: str = ""
    ui_type: str = ""
    group: str = ""
    difficulty: str = ""
    extras: dict[str, Any] | None = None

    def load_image(self) -> Any:
        """Load the PIL image from disk. Converts to RGB."""

        from PIL import Image

        return Image.open(self.image_path).convert("RGB")

    def point_in_bbox(self, x: float, y: float) -> bool:
        """Return True if ``(x, y)`` falls inside the gold bounding box."""

        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class Env(ABC):
    """Abstract base for a grounding dataset wrapper.

    Subclasses implement :meth:`_load_all` to enumerate
    :class:`GroundingItem` instances. The ABC provides filtering,
    iteration, and length.
    """

    name: str = ""
    cell_tag: str = ""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        limit: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else self._discover_root()
        self._items: list[GroundingItem] = list(self._load_all())
        if shuffle_seed is not None:
            import random as _random

            rng = _random.Random(shuffle_seed)
            rng.shuffle(self._items)
        if limit is not None:
            self._items = self._items[:limit]

    @abstractmethod
    def _load_all(self) -> Iterator[GroundingItem]:
        """Enumerate every item in the dataset."""

    @abstractmethod
    def _discover_root(self) -> Path:
        """Locate the dataset on disk (env-var or HF cache fallback)."""

    def __iter__(self) -> Iterator[GroundingItem]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> GroundingItem:
        return self._items[idx]

    def filter(
        self,
        *,
        applications: list[str] | None = None,
        platforms: list[str] | None = None,
        ui_types: list[str] | None = None,
        groups: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> "Env":
        """Return a new env subset (in-place filter on a shallow copy)."""

        items = self._items
        if applications:
            items = [it for it in items if it.application in applications]
        if platforms:
            items = [it for it in items if it.platform in platforms]
        if ui_types:
            items = [it for it in items if it.ui_type in ui_types]
        if groups:
            items = [it for it in items if it.group in groups]
        if difficulties:
            items = [it for it in items if it.difficulty in difficulties]
        new = self.__class__.__new__(self.__class__)
        new.root = self.root
        new._items = items
        return new

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={self.root}, n_items={len(self._items)})"
