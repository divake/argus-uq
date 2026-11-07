"""argus_capture CLI - score on user-supplied per-item records.

Usage:
    argus_capture --records my_records.parquet --methods msp,saplma,ccp

Each row of ``my_records.parquet`` must follow the
:class:`argus_uq.records.Record` schema. Columns ``correct``, ``dist``
are required; the rest depend on the methods being scored. The released
per-item parquets hold finished scores, not these raw signals, so this
command returns NaN or chance-level metrics on them; use ``argus_score``
for released cells.

Inference from scratch is not part of this release.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        required=True,
        help="Path to a parquet of per-item records.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names, or 'all' for the full registered set.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write a CSV of per-method results.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.8,
        help="Fraction of items going into the test split (default: 0.8).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.records)
    if "correct" not in df.columns:
        print("ERROR: records must have a 'correct' column.", file=sys.stderr)
        return 1
    labels = df["correct"].to_numpy()

    from argus_uq.factory import FactoryEstimator
    from argus_uq.manager import ScoringManager

    if args.methods == "all":
        method_names = FactoryEstimator.list_names()
    else:
        method_names = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Convert each row to a record dict for the manager. Only fields
    # listed in argus_uq.records.Record are used; missing fields stay
    # None and methods that need them will raise / NaN gracefully.
    records = df.to_dict(orient="records")

    mgr = ScoringManager(estimators=method_names, seed=args.seed)
    results = mgr.score_records(records, labels, test_frac=args.test_frac)

    out_df = pd.DataFrame([r.__dict__ for r in results])
    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(out_df)} methods x {len(out_df.columns)} columns)")
    else:
        print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
