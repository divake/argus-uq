"""Score multiple methods on a single cell, sweep across cells.

Demonstrates the public API beyond the quickstart: iterate over the cell
catalog, score a fixed method on each, and print a leaderboard-style
table.
"""

import argus_uq
from argus_uq.data import OPEN_CELLS


def main():
    rows = []
    for cell_id in OPEN_CELLS[:6]:  # first 6 to keep the demo quick
        try:
            cell = argus_uq.load(cell_id)
        except FileNotFoundError:
            print(f"  skip {cell_id} (not in data_release)")
            continue
        result = argus_uq.score(cell, method="saplma", seeds=10)
        rows.append((cell_id, result.auroc, result.auroc_std))
    print(f"{'cell':10s} {'AUROC':>8s}  {'+/- SD':>8s}")
    for cid, auroc, sd in rows:
        print(f"{cid:10s} {auroc:>8.3f}  {sd:>8.3f}")


if __name__ == "__main__":
    main()
