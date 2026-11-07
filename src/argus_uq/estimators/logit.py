"""Logit-family UQ methods (5 methods).

All methods here operate solely on per-token logprobs from greedy
decoding. They are the cheapest estimators in the panel and act as
baselines.

References:

- ``MSP``: Hendrycks & Gimpel 2017, *A Baseline for Detecting
  Misclassified and Out-of-Distribution Examples in Neural Networks*
  (arXiv 1610.02136). Per-token mean form matches lm-polygraph
  ``MaximumSequenceProbability``.
- ``Perplexity``: canonical log form ``-mean(logprob)``. Matches
  lm-polygraph ``MeanLogProbability`` (estimators/mean_logprobability.py).
- ``PerplexityExp``: legacy ``exp(-mean(logprob))`` form retained for
  back-compat with v1-v5 results. Rank-equivalent to ``Perplexity``.
- ``SequenceProbability``: raw NLL ``-sum(logprob)``. Cited as **SP**
  in the AAAI 2026 tutorial reference table.
- ``MeanTokenEntropy``: mean per-step vocabulary entropy. Matches
  lm-polygraph ``MeanTokenEntropy``.

Closed-source availability: NONE. Closed APIs do not expose per-token
logprobs reliably for the latest snapshot models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.estimators.base import Estimator, FamilyName


class MSP(Estimator):
    """Max Softmax Probability uncertainty.

    Returns ``1 - mean_t(P_top_t)`` where ``P_top_t = exp(logprob_t)``
    is the per-step probability of the chosen top-1 token. Per-token
    aggregation by mean matches lm-polygraph's whitebox MSP default.
    """

    name = "msp"
    family: FamilyName = "logit"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        logprobs = record.get("greedy_logprobs") or []
        if not logprobs:
            return 1.0
        probs = np.exp(np.asarray(logprobs, dtype=np.float64))
        return float(1.0 - probs.mean())


class Perplexity(Estimator):
    """Perplexity - canonical log form ``-mean(logprob)``.

    Higher = more uncertain. Rank-invariant with :class:`PerplexityExp`
    (so AUROC / PRR / Spearman / RCC / AUARC are identical).
    """

    name = "perplexity"
    family: FamilyName = "logit"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        logprobs = record.get("greedy_logprobs") or []
        if not logprobs:
            return float("inf")
        return float(-np.mean(logprobs))


class PerplexityExp(Estimator):
    """Legacy perplexity in the exponential form ``exp(-mean(logprob))``.

    Retained for back-compat with v1-v5 results. Rank-invariant with
    :class:`Perplexity`.
    """

    name = "perplexity_exp"
    family: FamilyName = "logit"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        logprobs = record.get("greedy_logprobs") or []
        if not logprobs:
            return float("inf")
        return float(np.exp(-np.mean(logprobs)))


class SequenceProbability(Estimator):
    """Sequence Probability - raw NLL ``-sum(logprob)`` of the greedy.

    AAAI 2026 tutorial slide 48 cites this as **SP**. Differs from
    :class:`Perplexity` in *not* length-normalising; longer sequences
    naturally accumulate more uncertainty.
    """

    name = "sequence_probability"
    family: FamilyName = "logit"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        logprobs = record.get("greedy_logprobs") or []
        if not logprobs:
            return float("inf")
        return float(-np.sum(logprobs))


class MeanTokenEntropy(Estimator):
    """Mean of per-step vocabulary entropies (nats).

    Matches lm-polygraph ``MeanTokenEntropy``.
    """

    name = "mean_token_entropy"
    family: FamilyName = "logit"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        entropies = record.get("greedy_entropies") or []
        if not entropies:
            return 0.0
        return float(np.mean(entropies))
