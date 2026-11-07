"""Derived-CSV loader.

The derived/ directory holds the canonical paper-numbers CSVs computed
from per-item parquets via 50 random calibration/test seeds. See
``data_release/README.md`` for the catalogue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from argus_uq.utils.paths import DERIVED_DIR


DERIVED_FILES = (
    "cell_winners.csv",
    "family_best_per_cell.csv",
    "method_leaderboard.csv",
    "family_means.csv",
    "cross_cell_rho.csv",
    "cross_cell_rho_pairs.csv",
    "cross_cell_rho_stats.csv",
    "ause_winners.csv",
    "tuning_deltas.csv",
    "nan_audit.csv",
    "significance_pairs.csv",
    "bootstrap_ci.csv",
    "conformal_disks.csv",
    "conformal_disks_summary.csv",
    "cell_winners_closed.csv",
    "family_best_per_cell_closed.csv",
    "method_leaderboard_closed.csv",
    "family_means_closed.csv",
    "cross_cell_rho_closed.csv",
    "cross_cell_rho_pairs_closed.csv",
    "cross_cell_rho_stats_closed.csv",
    "cross_open_closed_rho.csv",
    "ause_winners_closed.csv",
    "tuning_deltas_closed.csv",
    "nan_audit_closed.csv",
    "significance_pairs_closed.csv",
    "bootstrap_ci_closed.csv",
    "conformal_disks_closed.csv",
    "conformal_disks_summary_closed.csv",
)


def load_derived(name: str) -> Any:
    """Load a derived CSV by name (with or without ``.csv`` extension)."""

    import pandas as pd

    if not name.endswith(".csv"):
        name = f"{name}.csv"
    path = DERIVED_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Derived CSV {name!r} not found at {path}. "
            "Run `python -m argus_uq.cli.prepare_data` to populate "
            "data_release/derived/."
        )
    return pd.read_csv(path)


def derived_path(name: str) -> Path:
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    return DERIVED_DIR / name


def list_available_derived() -> list[str]:
    if not DERIVED_DIR.is_dir():
        return []
    return sorted(p.name for p in DERIVED_DIR.glob("*.csv"))
