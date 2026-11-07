"""Sampling-family UQ methods (6 methods).

All methods here consume the N stochastic samples produced by the
capture phase. For grounding tasks the natural "answer" is the
predicted click coordinate, so semantic clustering is done in
coordinate space rather than via NLI - two samples belong to the same
cluster iff their clicks fall within ``tolerance_px`` of each other.
Unparsable samples each get their own singleton cluster.

References:

- ``SelfConsistency``: Wang et al. 2023, *Self-Consistency Improves
  Chain-of-Thought Reasoning* (arXiv 2203.11171). Adapted to clicks
  by 50-px tolerance clustering.
- ``SemanticEntropy``: Kuhn et al. 2023 / Farquhar et al. *Nature* 2024.
  Cluster-frequency form (the v1-v5 implementation; matches the
  closed-source-feasible variant).
- ``SemanticEntropyWeighted``: Kuhn 2023 / Farquhar 2024 canonical
  form. Sequence-probability-weighted entropy. Open-source-only.
- ``MCSE``: Malinin & Gales 2021. Mean per-sample NLL. Reuses cached
  ``samples_mean_seq_logprobs``. Open-source-only.
- ``MCNSE``: length-normalised MCSE. Open-source-only.
- ``LexicalSimilarity``: Fomicheva et al. 2020 / lm-polygraph
  ``LexicalSimilarity``. Mean pairwise ``1 - SequenceMatcher.ratio``.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from argus_uq.estimators.base import Estimator, FamilyName
from argus_uq.records import coordinate_clusters


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector, clipped for numerical safety."""

    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


class SelfConsistency(Estimator):
    """Fraction of samples that disagree with the modal cluster.

    Returns ``1 - |C_modal| / N`` where ``C_modal`` is the largest
    cluster of samples within ``tolerance_px`` of each other. Returns
    ``0`` when all samples agree, approaches ``1`` as samples scatter.
    """

    name = "self_consistency"
    family: FamilyName = "sampling"
    available_on_closed_source = True

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        clicks = record.get("samples_clicks") or []
        if not clicks:
            return 1.0
        labels = coordinate_clusters(clicks, self.tolerance_px)
        counts = Counter(labels)
        most_common = max(counts.values())
        n = len(labels)
        return float(1.0 - most_common / n)


class SemanticEntropy(Estimator):
    """Cluster-frequency Shannon entropy over sample clicks.

    Shannon entropy over cluster-size proportions ``|C_m| / N``. Does
    not weight clusters by sequence probability. Matches the canonical
    form Farquhar et al. *Nature* 2024 use when logprobs are not
    available; this is the closed-source-feasible variant of
    semantic entropy.
    """

    name = "semantic_entropy"
    family: FamilyName = "sampling"
    available_on_closed_source = True

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        clicks = record.get("samples_clicks") or []
        if not clicks:
            return 0.0
        labels = coordinate_clusters(clicks, self.tolerance_px)
        counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
        probs = counts / counts.sum()
        return _entropy(probs)


class SemanticEntropyWeighted(Estimator):
    """Sequence-probability-weighted semantic entropy (open-source-only).

    AAAI 2026 tutorial slide 70 / Kuhn et al. 2023 / Farquhar 2024.
    For each cluster ``m``, ``P_hat_m = sum_{i in m} P(y^i | x)`` is
    the sum of sequence probabilities of the samples in cluster ``m``.
    The uncertainty is ``-sum_m P_hat_m * log(P_hat_m)`` after
    renormalising the cluster masses.

    Sequence probabilities come from ``samples_mean_seq_logprobs``;
    the per-sample sequence probability is ``exp(mean_seq_logprob)``.

    Falls back to cluster-size entropy when sample logprobs are
    missing or all ``-inf``.

    Open-source only - closed APIs don't expose reliable sample
    logprobs for the latest snapshots.
    """

    name = "semantic_entropy_weighted"
    family: FamilyName = "sampling"
    available_on_closed_source = False

    def __init__(self, tolerance_px: float = 50.0) -> None:
        self.tolerance_px = tolerance_px

    def score_item(self, record: dict[str, Any]) -> float:
        clicks = record.get("samples_clicks") or []
        if not clicks:
            return 0.0
        labels = coordinate_clusters(clicks, self.tolerance_px)
        if not labels:
            return 0.0

        seq_logprobs = list(record.get("samples_mean_seq_logprobs") or [])
        if (
            not seq_logprobs
            or len(seq_logprobs) != len(clicks)
            or all(not np.isfinite(lp) for lp in seq_logprobs)
        ):
            counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
            return _entropy(counts / counts.sum())

        finite = np.asarray(seq_logprobs, dtype=np.float64)
        finite_mask = np.isfinite(finite)
        if not finite_mask.any():
            counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
            return _entropy(counts / counts.sum())
        shifted = finite[finite_mask] - np.max(finite[finite_mask])
        weights = np.zeros_like(finite)
        weights[finite_mask] = np.exp(shifted)

        cluster_mass: dict[int, float] = {}
        for label, w in zip(labels, weights, strict=False):
            cluster_mass[label] = cluster_mass.get(label, 0.0) + float(w)
        masses = np.asarray(list(cluster_mass.values()), dtype=np.float64)
        if masses.sum() <= 0:
            counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
            return _entropy(counts / counts.sum())
        return _entropy(masses / masses.sum())


class MCSE(Estimator):
    """Monte-Carlo Sequence Entropy: mean per-sample NLL (open-source-only).

    AAAI 2026 tutorial slide 68 (Malinin & Gales 2021):
    ``U_MCSE = -(1/N) * sum_i log P(y^i | x)``. Reuses
    ``samples_mean_seq_logprobs`` saved during capture.

    Open-source only - closed APIs don't expose sample logprobs.
    """

    name = "mcse"
    family: FamilyName = "sampling"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        seq_lp = list(record.get("samples_mean_seq_logprobs") or [])
        finite = [lp for lp in seq_lp if np.isfinite(lp)]
        if not finite:
            return float("inf")
        return float(-np.mean(finite))


class MCNSE(Estimator):
    """Monte-Carlo length-normalised Sequence Entropy (open-source-only).

    For our pipeline, ``samples_mean_seq_logprobs`` is already
    length-normalised, so MCNSE equals MCSE in implementation. Both
    are reported separately so the paper can cite the two
    Malinin & Gales 2021 variants.

    Open-source only - same reason as :class:`MCSE`.
    """

    name = "mcnse"
    family: FamilyName = "sampling"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        seq_lp = list(record.get("samples_mean_seq_logprobs") or [])
        finite = [lp for lp in seq_lp if np.isfinite(lp)]
        if not finite:
            return float("inf")
        return float(-np.mean(finite))


class LexicalSimilarity(Estimator):
    """Mean pairwise ``1 - SequenceMatcher.ratio`` over sample texts.

    Fomicheva et al. 2020 / lm-polygraph ``LexicalSimilarity``. Returns
    ``0`` for identical samples; approaches ``1`` for maximally
    diverse samples.
    """

    name = "lexical_similarity"
    family: FamilyName = "sampling"
    available_on_closed_source = True

    def score_item(self, record: dict[str, Any]) -> float:
        texts = record.get("samples_texts") or []
        if len(texts) < 2:
            return 0.0
        sims: list[float] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(SequenceMatcher(None, texts[i], texts[j]).ratio())
        return float(1.0 - np.mean(sims))
