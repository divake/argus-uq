"""Conformal prediction base types.

Three disk variants are exposed in :mod:`argus_uq.conformal`:

- ``Disk-Fixed`` (Vovk et al. 2005; Angelopoulos & Bates 2023):
  vanilla split-CP on the per-item click-to-bbox distance; one
  global radius ``q_hat`` for all items.
- ``Disk-Normalized`` (Lei et al. 2018): per-item radius modulated by
  a plug-in UQ score; tighter disks on confident items.
- ``Disk-CQR`` (Romano et al. 2019): conformalised quantile
  regression with a plug-in regressor on per-item distance.

Plus two CRC variants (Bates et al. 2024) for selective-prediction
risk control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def point_to_bbox_distance(
    xy: tuple[float, float] | None, bbox: Sequence[float]
) -> float:
    """Euclidean distance from a point to the nearest edge of a bbox.

    Returns 0 if the point is inside the bbox, ``+inf`` if the point is
    ``None`` (unparseable agent response).
    """

    if xy is None:
        return float("inf")
    x, y = xy
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(np.sqrt(dx * dx + dy * dy))


@dataclass
class CoordCPResult:
    """Result of an inductive CP click-disk."""

    alpha: float
    q_hat: float
    test_coverage: float
    coverage_gap: float
    n_calib: int
    n_test: int
    median_distance: float
    variant: str = "fixed"
    plug_in: str = "none"
    radii: np.ndarray | None = None  # per-item radii for non-fixed variants


@dataclass
class CRCResult:
    """Result of conformal risk control on selective prediction."""

    alpha: float
    tau: float
    test_coverage: float
    selective_error: float
    risk_gap: float
    n_kept: int
    n_calib: int
    n_test: int


@dataclass
class ClassConditionalCRCResult:
    """Result of class-conditional CRC."""

    alpha: float
    per_class: dict[str, CRCResult]
    aggregate_coverage: float
    aggregate_selective_error: float
    aggregate_risk_gap: float
