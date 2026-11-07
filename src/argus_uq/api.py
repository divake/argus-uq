"""argus_uq public API.

Three entry points:
    load(cell_id)
    score(cell, method, *, seeds=50, seed=None)
    conformal_disk(cell, *, plug_in, alpha, variant)

These are the only symbols intended for public use. Internal modules
(argus_uq.estimators, argus_uq.scoring, etc.) may evolve between
versions; the surface here is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from argus_uq.cell import Cell


ConformalVariant = Literal["fixed", "normalized", "cqr"]


@dataclass
class ScoreResult:
    """Result of :func:`score`. Multi-seed metrics on the test split."""

    method: str
    cell_id: str
    auroc: float
    auroc_std: float
    auroc_ci_lo: float
    auroc_ci_hi: float
    prr_norm_at_05: float
    prr_norm_at_05_std: float
    ause: float
    ause_std: float
    aurc: float
    aurc_std: float
    ece_iso: float
    ece_iso_std: float
    brier_iso: float
    brier_iso_std: float
    n_seeds: int


@dataclass
class ConformalDisk:
    """Result of :func:`conformal_disk`."""

    cell_id: str
    plug_in: str
    alpha: float
    variant: ConformalVariant
    radius_pixels: float
    radius_pixels_std: float
    coverage: float
    coverage_gap: float
    n_test: int
    n_calib: int
    n_seeds: int


def load(cell_id: str) -> Cell:
    """Load a cell from the released artifact pack.

    Parameters
    ----------
    cell_id:
        One of the 16 open-source cells (Q7xV2, Q7xSP, Q7xOSG, Q7xUIV,
        Q72xV2, Q72xSP, Q72xOSG, Q72xUIV, UIxV2, UIxSP, UIxOSG, UIxUIV,
        PTxV2, PTxSP, PTxOSG, PxUIV) or 12 closed-source cells (OAIxV2,
        OAIxSP, OAIxOSG, OAIxUIV, SonxV2, SonxSP, SonxOSG, SonxUIV,
        GemxV2, GemxSP, GemxOSG, GemxUIV).

    Returns
    -------
    Cell:
        Loaded cell with per-item records (DataFrame) and metadata.
    """

    from argus_uq.data import load_cell

    return load_cell(cell_id)


def score(
    cell: Cell,
    method: str,
    *,
    seeds: int = 50,
    seed: int | None = None,
) -> ScoreResult:
    """Score a UQ method on a cell.

    Reads the cached per-method UQ scores from the cell's per-item
    parquet (no re-inference) and computes multi-seed metrics over
    random 80/20 calibration/test splits.

    Parameters
    ----------
    cell:
        From :func:`load`.
    method:
        One of the 27 (or 8 closed-source) registered methods. See
        ``argus_uq.factory.FactoryEstimator.list_names()``.
    seeds:
        Number of random splits to average over. Default 50 (paper).
    seed:
        Optional fixed seed (only when ``seeds=1``) for bit-exact
        reproducibility.

    Returns
    -------
    ScoreResult
    """

    from argus_uq.scoring import score_cached

    df = cell.records
    if method not in df.columns:
        available = sorted(c for c in df.columns if c not in ("item_id", "correct", "dist"))
        raise KeyError(
            f"Method {method!r} not in cell {cell.meta.cell_id!r}. "
            f"Available methods: {available}"
        )

    correctness = df["correct"].to_numpy()
    uq = df[method].to_numpy()
    if "dist" in df.columns and df["dist"].notna().any():
        dist = df["dist"].fillna(0.0).to_numpy()
        errors = np.log1p(np.abs(dist))
    else:
        errors = (1.0 - correctness).astype(np.float64)

    cached = score_cached(
        cell.meta.cell_id,
        method,
        correctness=correctness,
        uq_scores=uq,
        errors=errors,
        seeds=seeds,
        fixed_seed=seed,
    )
    return ScoreResult(
        method=method,
        cell_id=cell.meta.cell_id,
        auroc=cached.auroc,
        auroc_std=cached.auroc_std,
        auroc_ci_lo=cached.auroc_ci_lo,
        auroc_ci_hi=cached.auroc_ci_hi,
        prr_norm_at_05=cached.prr_norm_at_05,
        prr_norm_at_05_std=cached.prr_norm_at_05_std,
        ause=cached.ause,
        ause_std=cached.ause_std,
        aurc=cached.aurc,
        aurc_std=cached.aurc_std,
        ece_iso=cached.ece_iso,
        ece_iso_std=cached.ece_iso_std,
        brier_iso=cached.brier_iso,
        brier_iso_std=cached.brier_iso_std,
        n_seeds=cached.n_seeds,
    )


def conformal_disk(
    cell: Cell,
    *,
    plug_in: str = "none",
    alpha: float = 0.10,
    variant: ConformalVariant = "fixed",
    seeds: int = 10,
) -> ConformalDisk:
    """Construct a deployable inductive-conformal click-disk for a cell.

    The cell's per-item parquet records the per-item normalised
    distance to the gold bbox (``dist``) but not the per-item raw
    pixel coordinates. The conformal disk is therefore expressed in
    normalised-distance units and we convert to a representative
    pixel radius using the dataset's median image diagonal (~1500
    pixels for desktop datasets). For exact pixel disks on a specific
    image, use :mod:`argus_uq.conformal` directly with raw clicks.

    Parameters
    ----------
    cell:
        From :func:`load`.
    plug_in:
        Method name used as a per-item radius modulator. For
        ``variant="fixed"`` this is ignored.
    alpha:
        Marginal-miscoverage budget in ``(0, 1)``.
    variant:
        ``"fixed"`` | ``"normalized"`` | ``"cqr"``.
    seeds:
        Number of random calibration/test splits. The disk radius is
        averaged across seeds; ``radius_pixels_std`` is the
        across-seed SD.

    Returns
    -------
    ConformalDisk
    """

    from argus_uq.manager import stratified_split
    from argus_uq.scoring.multi_seed import SEED_LIST_CANONICAL

    df = cell.records
    if "dist" not in df.columns:
        raise ValueError(f"cell {cell.meta.cell_id!r} has no 'dist' column.")

    dist_norm = df["dist"].fillna(0.0).to_numpy()
    correctness = df["correct"].to_numpy()
    diag = float(_image_diagonal_for_cell(cell.meta.cell_id))
    dist_pix = dist_norm * diag

    radii: list[float] = []
    coverages: list[float] = []
    seed_list = SEED_LIST_CANONICAL[:seeds]
    for s in seed_list:
        calib_idx, test_idx = stratified_split(correctness, test_frac=0.8, seed=s)
        d_calib = dist_pix[calib_idx]
        d_test = dist_pix[test_idx]
        if variant == "fixed":
            r = _conformal_quantile(d_calib, alpha)
            r_per_item = np.full_like(d_test, r)
        elif variant in ("normalized", "cqr"):
            if plug_in not in df.columns:
                raise KeyError(
                    f"plug-in method {plug_in!r} not in cell {cell.meta.cell_id!r}."
                )
            u = df[plug_in].to_numpy()
            u_calib = u[calib_idx]
            u_test = u[test_idx]
            r_per_item = _normalized_or_cqr_radii(
                d_calib, u_calib, u_test, alpha, variant
            )
        else:
            raise ValueError(f"unknown variant {variant!r}")
        radii.append(float(np.nanmedian(r_per_item)))
        coverages.append(float(np.mean(d_test <= r_per_item)) if len(d_test) else 0.0)

    radius_mean = float(np.mean(radii))
    radius_std = float(np.std(radii))
    coverage_mean = float(np.mean(coverages))
    return ConformalDisk(
        cell_id=cell.meta.cell_id,
        plug_in=plug_in,
        alpha=alpha,
        variant=variant,
        radius_pixels=radius_mean,
        radius_pixels_std=radius_std,
        coverage=coverage_mean,
        coverage_gap=max(0.0, (1 - alpha) - coverage_mean),
        n_test=int(round(len(df) * 0.8)),
        n_calib=int(round(len(df) * 0.2)),
        n_seeds=len(seed_list),
    )


def _image_diagonal_for_cell(cell_id: str) -> float:
    """Approximate median image diagonal (pixels) for a dataset.

    Used to convert normalised distance back to pixels in
    :func:`conformal_disk`.
    """

    if "V2" in cell_id:
        return 1300.0
    if "SP" in cell_id:
        return 1900.0
    if "OSG" in cell_id:
        return 1700.0
    return 1600.0  # UI-Vision and fallback


def _conformal_quantile(d_calib: np.ndarray, alpha: float) -> float:
    finite = np.isfinite(d_calib)
    if finite.sum() < 2:
        return float("inf")
    n_c = int(finite.sum())
    level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
    return float(np.quantile(d_calib[finite], level, method="higher"))


def _normalized_or_cqr_radii(
    d_calib: np.ndarray,
    u_calib: np.ndarray,
    u_test: np.ndarray,
    alpha: float,
    variant: str,
) -> np.ndarray:
    """Per-item radii for normalized / CQR variants on cached scores."""

    finite_c = np.isfinite(d_calib) & np.isfinite(u_calib)
    finite_t = np.isfinite(u_test)
    if finite_c.sum() < 5:
        r = _conformal_quantile(d_calib, alpha)
        return np.full_like(u_test, r, dtype=np.float64)

    if variant == "normalized":
        sorted_u = np.sort(u_calib[finite_c])
        # Local weighting: 1 + rank(u_i)/n_calib, in [1, 2].
        def rank_norm(u):
            return np.searchsorted(sorted_u, u) / max(len(sorted_u), 1)

        w_calib = 1.0 + rank_norm(u_calib[finite_c])
        scaled = d_calib[finite_c] / w_calib
        n_c = int(finite_c.sum())
        level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
        q_hat = float(np.quantile(scaled, level, method="higher"))
        w_test = np.where(finite_t, 1.0 + rank_norm(u_test), 1.0)
        return q_hat * w_test

    # CQR: isotonic regression of d on u, then conformalised residual.
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(u_calib[finite_c], d_calib[finite_c])
    q_calib = iso.transform(u_calib[finite_c])
    residuals = np.maximum(0.0, d_calib[finite_c] - q_calib)
    n_c = int(finite_c.sum())
    level = min(1.0, np.ceil((n_c + 1) * (1 - alpha)) / n_c)
    q_hat = float(np.quantile(residuals, level, method="higher"))
    q_test = np.where(finite_t, iso.transform(np.where(finite_t, u_test, 0.0)), 0.0)
    return q_test + q_hat


__all__ = [
    "ScoreResult",
    "ConformalDisk",
    "ConformalVariant",
    "load",
    "score",
    "conformal_disk",
]
