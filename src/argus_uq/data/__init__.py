"""argus_uq.data - released-artifact loaders.

- :func:`load_cell` returns a fully-populated ``Cell`` for the given
  cell id. Used by ``argus_uq.load``.
- :func:`load_per_item` returns just the per-item parquet DataFrame.
- :func:`load_derived` returns one of the canonical paper CSVs.
- :data:`CATALOG`, :data:`ALL_CELLS`, :data:`OPEN_CELLS`,
  :data:`CLOSED_CELLS` enumerate the released cells.
"""

from argus_uq.data.cells import (
    ALL_CELLS,
    CATALOG,
    CLOSED_CELLS,
    OPEN_CELLS,
)
from argus_uq.data.derived import (
    DERIVED_FILES,
    derived_path,
    list_available_derived,
    load_derived,
)
from argus_uq.data.per_item import (
    has_method,
    list_available_cells,
    load_per_item,
    per_item_path,
)

__all__ = [
    "CATALOG",
    "ALL_CELLS",
    "OPEN_CELLS",
    "CLOSED_CELLS",
    "DERIVED_FILES",
    "load_cell",
    "load_per_item",
    "load_derived",
    "per_item_path",
    "derived_path",
    "list_available_cells",
    "list_available_derived",
    "has_method",
]


def _closed_records(cell_id: str, df):
    """Project a closed-source parquet onto the ``Cell.records`` schema.

    Closed-source parquets store the raw click distance in pixels
    (``d_pix``) and no image size. ``dist`` is ``d_pix`` over the same
    per-dataset diagonal :func:`argus_uq.conformal_disk` multiplies by,
    so conformal radii come out in true pixels.
    """

    from argus_uq.api import _image_diagonal_for_cell

    dropped = ["greedy_x", "greedy_y", "gold_x", "gold_y", "image_w", "image_h", "d_pix"]
    methods = [c for c in df.columns if c not in ("item_id", "correct", *dropped)]
    out = df[["item_id", "correct", *methods]].copy()
    out.insert(2, "dist", df["d_pix"] / _image_diagonal_for_cell(cell_id))
    return out


def load_cell(cell_id: str):
    """Resolve a cell id and return a populated :class:`argus_uq.cell.Cell`.

    Raises ``KeyError`` if the cell id is not in the catalog.
    Raises ``FileNotFoundError`` if the parquet is missing.
    """

    from dataclasses import replace

    from argus_uq.cell import Cell
    from argus_uq.utils.paths import DATA_RELEASE_DIR

    if cell_id not in CATALOG:
        raise KeyError(
            f"Unknown cell id {cell_id!r}. "
            f"Available open-source cells: {OPEN_CELLS}. "
            f"Available closed-source cells: {CLOSED_CELLS}."
        )

    df = load_per_item(cell_id)
    base_meta = CATALOG[cell_id]
    if base_meta.tier == "closed_source":
        df = _closed_records(cell_id, df)
    meta = replace(
        base_meta,
        n_items=len(df),
        n_test=int(round(len(df) * 0.8)),
        n_calib=int(round(len(df) * 0.2)),
    )
    return Cell(
        meta=meta,
        records=df,
        artifacts_root=DATA_RELEASE_DIR,
    )
