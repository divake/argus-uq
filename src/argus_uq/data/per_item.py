"""Per-item parquet loader.

Each open-source per-item parquet has 30 columns: 3 metadata (item_id,
correct, dist) + 27 per-method UQ scores. Closed-source parquets carry
click and gold coordinates, the pixel distance ``d_pix``, and the 8
OS-identical method columns; ``argus_uq.load`` maps them to the same
record schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from argus_uq.utils.paths import PER_ITEM_DIR


def per_item_path(cell_id: str) -> Path:
    """Resolve the parquet path for a cell."""

    return PER_ITEM_DIR / f"{cell_id}.parquet"


def load_per_item(cell_id: str) -> Any:
    """Load the per-item parquet for ``cell_id``.

    Returns a pandas DataFrame.
    """

    import pandas as pd

    path = per_item_path(cell_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Per-item parquet not found for cell {cell_id!r} at {path}. "
            "Run `python -m argus_uq.cli.prepare_data` to populate "
            "data_release/, or set ARGUS_DATA_DIR to your local copy."
        )
    return pd.read_parquet(path)


def list_available_cells() -> list[str]:
    """Cell IDs that currently have a parquet on disk."""

    if not PER_ITEM_DIR.is_dir():
        return []
    return sorted(p.stem for p in PER_ITEM_DIR.glob("*.parquet"))


def has_method(df: Any, method: str) -> bool:
    """Return True if the DataFrame has a column for ``method``."""

    return method in df.columns
