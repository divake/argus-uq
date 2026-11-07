"""Compare an OS-identical method across open-weight and closed-API cells.

The 8 closed-source
methods use formulas bit-for-bit identical to their open-source
counterparts. Cross-tier comparisons of these 8 methods are
apples-to-apples.

This example scores ``self_consistency`` on every cell where it is
runnable.
"""

import argus_uq
from argus_uq.data import ALL_CELLS


def main(method: str = "self_consistency") -> None:
    print(f"=== {method} across cells ===\n")
    print(f"{'cell':10s} {'tier':14s} {'AUROC':>8s} {'+/- SD':>8s}")
    for cell_id in ALL_CELLS:
        try:
            cell = argus_uq.load(cell_id)
        except FileNotFoundError:
            continue
        if method not in cell.records.columns:
            continue
        r = argus_uq.score(cell, method=method, seeds=10)
        print(f"{cell_id:10s} {cell.meta.tier:14s} {r.auroc:>8.3f} {r.auroc_std:>8.3f}")


if __name__ == "__main__":
    main()
