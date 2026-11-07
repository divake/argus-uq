"""argus_normalize CLI - apply isotonic-PCC calibration to per-item UQ scores.

Usage:
    argus_normalize --cell PTxOSG --method saplma

Reads the per-item parquet for the cell, fits an isotonic-PCC
calibrator on a calibration split, and prints calibration metrics
(ECE_iso, MCE_iso, Brier_iso) plus the calibrated confidence range.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import argus_uq
from argus_uq.manager import stratified_split
from argus_uq.ue_metrics import IsotonicCalibrator, brier_isotonic, ece_isotonic, mce_isotonic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        cell = argus_uq.load(args.cell)
    except (KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    df = cell.records
    if args.method not in df.columns:
        print(f"ERROR: method {args.method!r} not in cell {args.cell!r}.", file=sys.stderr)
        return 1

    correctness = df["correct"].to_numpy()
    uq = df[args.method].to_numpy()

    calib_idx, test_idx = stratified_split(correctness, test_frac=0.8, seed=args.seed)
    c_calib, c_test = correctness[calib_idx], correctness[test_idx]
    u_calib, u_test = uq[calib_idx], uq[test_idx]

    calibrator = IsotonicCalibrator().fit(u_calib, c_calib)
    confidences = calibrator.transform(u_test)
    e_iso = ece_isotonic(c_test, u_test, c_calib, u_calib)
    m_iso = mce_isotonic(c_test, u_test, c_calib, u_calib)
    b_iso = brier_isotonic(c_test, u_test, c_calib, u_calib)

    print(f"cell:        {args.cell}")
    print(f"method:      {args.method}")
    print(f"n_calib:     {len(calib_idx)}")
    print(f"n_test:      {len(test_idx)}")
    print(f"ECE_iso:     {e_iso:.4f}")
    print(f"MCE_iso:     {m_iso:.4f}")
    print(f"Brier_iso:   {b_iso:.4f}")
    print(f"P(correct) range on test: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print(f"  median:    {np.median(confidences):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
