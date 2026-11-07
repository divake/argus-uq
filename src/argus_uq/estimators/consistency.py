"""Consistency-family UQ methods (3 methods).

References:

- ``CCP`` (formerly ``CoordAgreement``): coordinate-tolerance frequency
  scoring. ``1 - p_modal`` where ``p_modal`` is the fraction of samples
  whose click is within ``tolerance_px`` of the greedy click. Closer in
  form to Frequency Scoring (Mohri & Hashimoto 2024) than to the
  canonical Fadeeva et al. 2024 CCP, which requires NLI-token-level
  claim decomposition that does not transfer to a 2D-point output.
  Serialised name kept as ``ccp`` for back-compat with v1-v5 csvs.
  Available on both panels.

- ``CoCoA`` (canonical, Vashurin et al. 2025; AAAI 2026 tutorial slide
  73): ``U_CoCoA = U_inf * U_cons`` where
  ``U_inf = -log P(y*|x) = -sum(logprob_t)`` and
  ``U_cons = (1/N) * sum_i (1 - s(y*, y^i))`` with the sample-greedy
  similarity ``s = 1`` if within ``tolerance_px``, else ``0``.
  Open-source only - closed APIs don't expose sequence NLL.

- ``CoCoA1MCA``: legacy ``1 - C * A`` form with confidence
  ``C = exp(mean logprob)`` and agreement ``A`` = fraction within
  tolerance. Kept for back-compat. Open-source only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from argus_uq.estimators.base import Estimator, FamilyName
from argus_uq.records import coordinate_clusters


class CCP(Estimator):
    """Coordinate-agreement consistency score (Mohri & Hashimoto 2024).

    Returns ``1 - p_modal`` where ``p_modal`` is the fraction of
    samples whose click matches the greedy click within
    ``tolerance_px``. Unparseable samples count as non-matching.

    Renamed from the Fadeeva et al. 2024 CCP for honesty - the
    canonical CCP requires NLI-token-level claim decomposition that
    does not apply to a 2D-point output. This is closer to Frequency
    Scoring (Mohri & Hashimoto 2024). The serialised name ``ccp`` is
    kept for back-compat with v1-v5 csvs and the closed-source
    rescore (which renames the historical ``coord_agreement`` column
    to ``ccp``).

    Available on both panels.
    """

    name = "ccp"
    family: FamilyName = "consistency"
    available_on_closed_source = True

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        greedy_click = record.get("greedy_click")
        clicks = record.get("samples_clicks") or []
        if not clicks:
            return 1.0
        if greedy_click is None:
            labels = coordinate_clusters(clicks, self.tolerance_px)
            counts = Counter(labels)
            p_modal = max(counts.values()) / len(labels)
            return float(1.0 - p_modal)
        matches = 0
        for c in clicks:
            if c is None:
                continue
            dx = greedy_click[0] - c[0]
            dy = greedy_click[1] - c[1]
            if (dx * dx + dy * dy) ** 0.5 <= self.tolerance_px:
                matches += 1
        return float(1.0 - matches / len(clicks))


class CoCoA(Estimator):
    """Canonical CoCoA: ``U_CoCoA = U_inf * U_cons`` (open-source only).

    Vashurin et al. 2025 (arXiv 2502.04964); AAAI 2026 tutorial
    slide 73.

    - ``U_inf = -log P(y*|x) = -sum(logprob_t)`` - sequence NLL of the
      greedy generation, in ``[0, inf)``.
    - ``U_cons = (1/N) * sum_i (1 - s(y*, y^i))`` - disagreement
      fraction in ``[0, 1]``. For grounding ``s = 1`` if the sample
      click is within ``tolerance_px`` of the greedy click, else ``0``.
      Unparseable samples count as full disagreement.

    Open-source only because the canonical formula requires sequence
    NLL, which closed APIs don't expose for the latest snapshots.
    """

    name = "cocoa"
    family: FamilyName = "consistency"
    available_on_closed_source = False

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        logprobs = record.get("greedy_logprobs") or []
        u_inf = -float(np.sum(logprobs)) if logprobs else float("inf")

        greedy_click = record.get("greedy_click")
        clicks = record.get("samples_clicks") or []
        if not clicks or greedy_click is None:
            return float(u_inf)
        n = len(clicks)
        if n == 0:
            return float(u_inf)
        disagreements = 0
        for c in clicks:
            if c is None:
                disagreements += 1
                continue
            dx = greedy_click[0] - c[0]
            dy = greedy_click[1] - c[1]
            if (dx * dx + dy * dy) ** 0.5 > self.tolerance_px:
                disagreements += 1
        u_cons = disagreements / n
        return float(u_inf * u_cons)


class CoCoA1MCA(Estimator):
    """Legacy CoCoA in ``1 - C * A`` form (open-source only).

    Kept for back-compat with v1-v5 csvs. ``C = exp(mean logprob)``;
    ``A`` = fraction of samples within tolerance of greedy. Returns
    ``1 - C * A``.

    Open-source only - same reason as :class:`CoCoA`.
    """

    name = "cocoa_1mca"
    family: FamilyName = "consistency"
    available_on_closed_source = False

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        greedy_click = record.get("greedy_click")
        clicks = record.get("samples_clicks") or []
        logprobs = record.get("greedy_logprobs") or []
        confidence = float(np.exp(np.mean(logprobs))) if logprobs else 0.0
        if not clicks or greedy_click is None:
            return float(1.0 - confidence)
        agreements = 0
        for c in clicks:
            if c is None:
                continue
            dx = greedy_click[0] - c[0]
            dy = greedy_click[1] - c[1]
            if (dx * dx + dy * dy) ** 0.5 <= self.tolerance_px:
                agreements += 1
        agreement = agreements / max(len(clicks), 1)
        return float(1.0 - confidence * agreement)
