"""Prompt-family UQ methods (3 methods).

Each method operates on a scalar confidence elicited from the model
itself via a follow-up prompt on the same image. The elicitation is
done upstream by the capture phase; the method just converts that
scalar into an uncertainty.

References:

- ``PTrueMethod``: Kadavath et al. 2022, *Language Models (Mostly)
  Know What They Know* (arXiv 2207.05221), section 4 + Appendix.
  Asks the model whether its own greedy answer is correct (A=correct,
  B=incorrect) and reports ``1 - P(A)``. **Open-source only** -
  closed APIs don't expose the per-token probability of "True" / "A"
  reliably for the latest snapshots.

- ``Verbalized1S``: Tian et al. 2023, *Just Ask for Calibration*
  (arXiv 2305.14975). Asks the model for a 0-100 integer confidence
  in a single turn; reports ``1 - score / 100``. Available on both
  panels.

- ``Verbalized2S``: Xiong et al. 2024, *Can LLMs Express Their
  Uncertainty?* (arXiv 2306.13063). Two-stage protocol: model first
  emits an answer, then in a second turn emits a 0-100 confidence
  in that answer. Available on both panels.
"""

from __future__ import annotations

from typing import Any

from argus_uq.estimators.base import Estimator, FamilyName


class PTrueMethod(Estimator):
    """Uncertainty from a P(True) self-evaluation probe.

    Kadavath et al. 2022. The probe asks the model whether its greedy
    answer is correct; ``record['p_true']`` is ``P(model says
    "True")``. Returns ``1 - P(True)``.

    Open-source only - closed APIs don't expose reliable per-token
    probabilities for the latest snapshots.
    """

    name = "p_true"
    family: FamilyName = "prompt"
    available_on_closed_source = False

    def score_item(self, record: dict[str, Any]) -> float:
        p = record.get("p_true")
        if p is None:
            return 1.0
        return float(1.0 - p)


class Verbalized1S(Estimator):
    """Uncertainty from a single-stage verbalized confidence probe.

    Tian et al. 2023. The probe elicits a 0-100 integer confidence in
    one turn; ``record['verbalized_1s']`` is the parsed value
    normalised to ``[0, 1]``. Returns ``1 - confidence``.

    Available on both panels.
    """

    name = "verbalized_1s"
    family: FamilyName = "prompt"
    available_on_closed_source = True

    def score_item(self, record: dict[str, Any]) -> float:
        v = record.get("verbalized_1s")
        if v is None:
            return 1.0
        return float(1.0 - v)


class Verbalized2S(Estimator):
    """Uncertainty from a two-stage verbalized confidence probe.

    Xiong et al. 2024. Two-turn protocol; ``record['verbalized_2s']``
    is the parsed second-stage confidence in ``[0, 1]``. Returns
    ``1 - confidence``.

    Available on both panels.
    """

    name = "verbalized_2s"
    family: FamilyName = "prompt"
    available_on_closed_source = True

    def score_item(self, record: dict[str, Any]) -> float:
        v = record.get("verbalized_2s")
        if v is None:
            return 1.0
        return float(1.0 - v)
