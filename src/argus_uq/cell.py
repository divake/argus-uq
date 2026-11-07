"""Cell - the unit of evaluation in argus-uq.

A Cell is one (model, dataset) pair scored on the canonical 27-method
panel (or the 9-method closed-source panel). It bundles per-item records
plus per-seed metric matrices into a single object that the public API
returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


CellTier = Literal["open_source", "closed_source"]


@dataclass(frozen=True)
class CellMetadata:
    """Static metadata for a cell - not the records themselves.

    Populated at load time from data_release/cells.json.
    """

    cell_id: str  # e.g. "PTxOSG"
    tier: CellTier
    model_tag: str  # e.g. "PT"
    model_name: str  # e.g. "POINTS-GUI-G-8B"
    model_hf_id: str | None  # None for closed-source
    dataset_tag: str  # e.g. "OSG"
    dataset_name: str  # e.g. "OSWorld-G non-refusal subset"
    n_items: int
    n_test: int  # 80% of n_items
    n_calib: int  # 20% of n_items
    test_acc: float  # greedy accuracy on test split (multi-seed mean)
    license_code: str  # MIT, CC-BY-4.0, Apache-2.0, ...


@dataclass
class Cell:
    """One model x dataset cell with all the per-item and per-seed scores.

    Returned by `argus_uq.load(cell_id)`. Consumed by `argus_uq.score()`,
    `argus_uq.conformal_disk()`, and any user-side analysis.

    Attributes
    ----------
    meta:
        Static metadata (model, dataset, sizes, license).
    records:
        Per-item records as a pandas DataFrame. Schema:
        - item_id: str
        - correct: int (0 / 1)
        - dist: float (Euclidean distance from predicted click to gold
          bbox edge in pixels; 0 if click inside bbox)
        - dist_norm: float (dist / image diagonal)
        - greedy_x, greedy_y: int (predicted click in original image space)
        - gold_x1, gold_y1, gold_x2, gold_y2: int (gold bbox)
        - image_w, image_h: int (image dimensions)
        - <method-name>: float (per-item UQ score for each of the 27 or 9
          methods; higher = more uncertain)
        - p_true: float (P(model says greedy click is correct))
    multi_seed_metrics:
        Long-format DataFrame of per-(method, seed) metric values.
        Schema: cell_id, model, dataset, method, family, metric, seed_idx,
        value. 50 seeds x 27 methods x 6 metrics = ~8,100 rows per cell.
    derived:
        Pre-computed cell-level summaries from the released artifact pack.
        Keys: "winners", "family_best", "ause_winners", "tuning_deltas",
        "bootstrap_ci", "conformal_disks_summary".
    artifacts_root:
        Path to the data_release directory the cell was loaded from. Used
        by downstream functions that need to find sibling cells (e.g.
        cross-cell rho).
    """

    meta: CellMetadata
    records: Any  # pandas.DataFrame; lazy import to keep core install snappy
    multi_seed_metrics: Any | None = None  # pandas.DataFrame
    derived: dict[str, Any] = field(default_factory=dict)
    artifacts_root: Path | None = None

    def __repr__(self) -> str:
        return (
            f"Cell(id={self.meta.cell_id!r}, tier={self.meta.tier!r}, "
            f"model={self.meta.model_tag!r}, dataset={self.meta.dataset_tag!r}, "
            f"n_test={self.meta.n_test}, test_acc={self.meta.test_acc:.3f})"
        )
