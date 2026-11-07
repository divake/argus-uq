"""argus_reproduce CLI - regenerate paper tables from data_release/derived/.

Usage:
    argus_reproduce --table leaderboard
    argus_reproduce --table all
    argus_reproduce --list

Tables map to the canonical paper artefacts:

- ``cells`` -> the model x dataset matrix (loaded from CATALOG)
- ``leaderboard`` -> method_leaderboard.csv (mean AUROC across cells)
- ``family`` -> family_best_per_cell.csv
- ``rho`` -> cross_cell_rho.csv (16x16 Spearman matrix)
- ``ause`` -> ause_winners.csv (AUROC top-1 vs AUSE top-1)
- ``conformal`` -> conformal_disks_summary.csv
- ``tuning`` -> tuning_deltas.csv (Q7 -> UI per-method delta)
- ``all`` -> every table

This is a thin reader/printer; the canonical numbers were computed
once by the multi-seed pass and ship in data_release/derived/.
"""

from __future__ import annotations

import argparse
import sys


TABLE_MAP = {
    "leaderboard": "method_leaderboard.csv",
    "family": "family_best_per_cell.csv",
    "rho": "cross_cell_rho.csv",
    "rho_pairs": "cross_cell_rho_pairs.csv",
    "rho_stats": "cross_cell_rho_stats.csv",
    "ause": "ause_winners.csv",
    "winners": "cell_winners.csv",
    "tuning": "tuning_deltas.csv",
    "conformal": "conformal_disks_summary.csv",
    "family_means": "family_means.csv",
    "bootstrap": "bootstrap_ci.csv",
    "significance": "significance_pairs.csv",
    "nan_audit": "nan_audit.csv",
    "leaderboard_closed": "method_leaderboard_closed.csv",
    "family_closed": "family_best_per_cell_closed.csv",
    "rho_closed": "cross_cell_rho_closed.csv",
    "rho_open_closed": "cross_open_closed_rho.csv",
    "ause_closed": "ause_winners_closed.csv",
    "winners_closed": "cell_winners_closed.csv",
    "conformal_closed": "conformal_disks_summary_closed.csv",
    "family_means_closed": "family_means_closed.csv",
    "bootstrap_closed": "bootstrap_ci_closed.csv",
    "significance_closed": "significance_pairs_closed.csv",
}


def _print_cells_table() -> None:
    """Print the cell catalog (16 open + 12 closed)."""

    from argus_uq.data import CATALOG, CLOSED_CELLS, OPEN_CELLS

    print("=== Cell catalog ===")
    print()
    print(f"Open-source ({len(OPEN_CELLS)}):")
    for cid in OPEN_CELLS:
        m = CATALOG[cid]
        print(
            f"  {cid:8s}  {m.model_name:32s}  {m.dataset_name:48s}  "
            f"acc={m.test_acc:.3f}  ({m.license_code})"
        )
    print()
    print(f"Closed-source ({len(CLOSED_CELLS)}):")
    for cid in CLOSED_CELLS:
        m = CATALOG[cid]
        print(
            f"  {cid:8s}  {m.model_name:32s}  {m.dataset_name:48s}  "
            f"acc={m.test_acc:.3f}"
        )


def _print_table(name: str) -> int:
    if name == "cells":
        _print_cells_table()
        return 0
    if name not in TABLE_MAP:
        print(f"ERROR: unknown table {name!r}.", file=sys.stderr)
        print(f"Known tables: cells, {', '.join(TABLE_MAP)}.", file=sys.stderr)
        return 1
    from argus_uq.data import load_derived

    try:
        df = load_derived(TABLE_MAP[name])
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"=== {name} ({TABLE_MAP[name]}) ===")
    print()
    with_pd_options = ("display.max_rows", 100, "display.max_columns", 20, "display.width", 200)
    import pandas as pd

    with pd.option_context(*with_pd_options):
        print(df.to_string(index=False))
    print()
    print(f"shape: {df.shape}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default=None,
        help="Table name to print (cells, leaderboard, family, rho, ause, ...).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tables and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every table.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("cells")
        for name in sorted(TABLE_MAP):
            print(name)
        return 0

    if args.all or (args.table == "all"):
        ec = 0
        for name in ["cells"] + sorted(TABLE_MAP):
            ec |= _print_table(name)
            print()
        return ec

    if not args.table:
        parser.print_help()
        return 1

    return _print_table(args.table)


if __name__ == "__main__":
    raise SystemExit(main())
