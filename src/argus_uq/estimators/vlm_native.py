"""VLM-native UQ methods (2 methods).

These probe the vision-language alignment by perturbing the inputs and
measuring how the model's predictions respond.

References:

- ``HEDGE``: Gautam et al. 2025, *HEDGE: Hallucination Evaluation via
  Decoding-time Gaze and Entropy* (adapted). K paraphrased
  instructions, cluster entropy on resulting clicks. The 3 paraphrase
  templates are documented in :data:`HEDGE_PARAPHRASE_TEMPLATES`.

- ``IMGHEDGE``: vision-side twin of HEDGE. Applies K pixel-value
  perturbations to the image (saturation +30%, Gaussian noise sigma=8,
  JPEG quality=50) with the instruction held fixed. The 3
  perturbations were selected from a 20-item pilot of 8 candidates
  (AUROC >= 0.69 winners; AUROC <= 0.55 losers excluded).

Both methods are available on both panels - they only require black-box
input perturbation + click parsing.
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from argus_uq.estimators.base import Estimator, FamilyName
from argus_uq.records import coordinate_clusters


# Three fixed paraphrase templates including the original verbatim as
# the first template. Each ``{instruction}`` substitutes the task's raw
# English instruction. Templates are deliberately *semantic* - they
# preserve the action (click) and the target (instruction noun phrase)
# while varying surface form, so any dispersion in predicted clicks
# reflects vision-language alignment weakness, not semantic drift.
HEDGE_PARAPHRASE_TEMPLATES: tuple[str, ...] = (
    "{instruction}",
    "Please click on: {instruction}",
    "Tap the UI element: {instruction}",
)


class HEDGE(Estimator):
    """HEDGE cluster entropy over paraphrased-instruction clicks.

    Consumes ``record['hedge_samples']``: a list of
    ``(paraphrase_id, click_xy_or_None)`` tuples produced by the
    capture phase using :data:`HEDGE_PARAPHRASE_TEMPLATES`. Clusters
    the clicks by 50-px coordinate tolerance and returns the Shannon
    entropy in nats.

    Available on both panels.
    """

    name = "hedge"
    family: FamilyName = "vlm_native"
    available_on_closed_source = True

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        samples = record.get("hedge_samples") or []
        clicks = [c for _, c in samples] if samples else []
        if not clicks:
            return 0.0
        labels = coordinate_clusters(clicks, self.tolerance_px)
        counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-(probs * np.log(probs)).sum())


# ----------------------------------------------------------------------
# IMGHEDGE - image-perturbation twin of HEDGE
# ----------------------------------------------------------------------


def pb_saturation_up_30(image: Image.Image) -> Image.Image:
    """Saturation +30% (winner; AUROC 0.725 in pilot)."""

    return ImageEnhance.Color(image).enhance(1.30)


def pb_gauss_noise_8(image: Image.Image) -> Image.Image:
    """Gaussian pixel noise sigma=8 (winner; AUROC 0.700 in pilot)."""

    arr = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 8.0, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def pb_jpeg_q50(image: Image.Image) -> Image.Image:
    """JPEG compression quality 50 (winner; AUROC 0.690 in pilot)."""

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


IMGHEDGE_PERTURBATIONS: tuple[
    tuple[str, Callable[[Image.Image], Image.Image]], ...
] = (
    ("sat_up_30", pb_saturation_up_30),
    ("gauss_noise_8", pb_gauss_noise_8),
    ("jpeg_q50", pb_jpeg_q50),
)


class IMGHEDGE(Estimator):
    """IMGHEDGE cluster entropy over image-perturbed-grounding clicks.

    Consumes ``record['imghedge_samples']``: a list of
    ``(perturbation_name, click_xy_or_None)`` tuples produced by the
    capture phase using :data:`IMGHEDGE_PERTURBATIONS`, plus the
    original greedy click from ``record['greedy_click']``. Clusters
    the N+1 clicks with the same coordinate tolerance as :class:`HEDGE`
    and returns the Shannon entropy in nats.

    Vision-side twin of :class:`HEDGE`: same aggregation, opposite
    perturbation modality.

    Available on both panels.
    """

    name = "imghedge"
    family: FamilyName = "vlm_native"
    available_on_closed_source = True

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        samples = record.get("imghedge_samples") or []
        perturbed_clicks = [c for _, c in samples] if samples else []
        baseline_click = record.get("greedy_click")
        clicks = [baseline_click] + perturbed_clicks
        if len(clicks) < 2 or all(c is None for c in clicks):
            return 0.0
        labels = coordinate_clusters(clicks, self.tolerance_px)
        counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-(probs * np.log(probs)).sum())
