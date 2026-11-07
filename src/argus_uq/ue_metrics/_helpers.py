"""Shared helpers for UE metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_arrays(correctness: Any, uncertainty: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D float arrays and drop ``inf`` from uncertainty."""

    c = np.asarray(correctness, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if c.shape != u.shape:
        raise ValueError(
            f"shape mismatch: correctness {c.shape} vs uncertainty {u.shape}"
        )
    mask = np.isfinite(u)
    return c[mask], u[mask]


def min_max(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to ``[0, 1]``. Matches lm-polygraph's ``normalize``."""

    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def preprocess_inf(x: float, array: np.ndarray) -> float:
    """Replace ``+/-inf`` with slightly-outside-the-observed-range values."""

    if not np.isinf(x):
        return float(x)
    if x > 0:
        return float(array[np.isfinite(array)].max()) + 1.0 if np.isfinite(array).any() else 1.0
    return float(array[np.isfinite(array)].min()) - 1.0 if np.isfinite(array).any() else -1.0


def binned_calibration_errors(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int
) -> list[tuple[float, float, float]]:
    """Return ``(bin_weight, bin_confidence, bin_accuracy)`` per non-empty bin."""

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[tuple[float, float, float]] = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        w = float(in_bin.sum()) / n
        conf = float(confidences[in_bin].mean())
        acc = float(correctness[in_bin].mean())
        out.append((w, conf, acc))
    return out
