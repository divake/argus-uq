"""argus_score CLI - score a single (cell, method) and print metrics.

Usage:
    argus_score --cell PTxOSG --method saplma --seeds 50
    argus_score --cell OAIxSP --method ccp --seeds 50 --json
"""

from __future__ import annotations

import argparse
import json
import sys

import argus_uq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="Cell id (e.g. PTxOSG, OAIxSP).")
    parser.add_argument("--method", required=True, help="UQ method name (e.g. saplma).")
    parser.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of random calibration/test splits (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed for bit-exact reproducibility (only when --seeds=1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON dict instead of a human-readable summary.",
    )
    args = parser.parse_args(argv)

    try:
        cell = argus_uq.load(args.cell)
    except (KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        result = argus_uq.score(
            cell, method=args.method, seeds=args.seeds, seed=args.seed
        )
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.__dict__, indent=2))
    else:
        print(f"cell:   {result.cell_id}")
        print(f"method: {result.method}")
        print(f"seeds:  {result.n_seeds}")
        print(f"AUROC:           {result.auroc:.4f} +/- {result.auroc_std:.4f}")
        print(f"  95% CI:        [{result.auroc_ci_lo:.4f}, {result.auroc_ci_hi:.4f}]")
        print(f"PRR_norm@0.5:    {result.prr_norm_at_05:.4f} +/- {result.prr_norm_at_05_std:.4f}")
        print(f"AUSE:            {result.ause:.4f} +/- {result.ause_std:.4f}")
        print(f"AURC:            {result.aurc:.4f} +/- {result.aurc_std:.4f}")
        print(f"ECE_iso:         {result.ece_iso:.4f} +/- {result.ece_iso_std:.4f}")
        print(f"Brier_iso:       {result.brier_iso:.4f} +/- {result.brier_iso_std:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
