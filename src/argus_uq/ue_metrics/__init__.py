"""argus_uq.ue_metrics - UQ-evaluation metrics.

Six headline metrics + their isotonic-recalibrated companions, plus
selective-accuracy variants and a generic bootstrap helper.

Headline (paper-canonical):

- ``auroc_incorrect`` - AUROC for predicting incorrectness
- ``prr_normalized_at_05`` - oracle-normalized PRR at 50% rejection
- ``ause`` - Area Under the Sparsification Error curve (log distance)
- ``rcc_auc`` (a.k.a. ``aurc``) - Risk-Coverage AUC; lower = better
- ``ece_isotonic`` - ECE with isotonic-PCC calibrated confidences
- ``brier_isotonic`` - Brier with isotonic-PCC calibrated confidences

Companions: ``prr``, ``prr_normalized``, ``prr_at_05``, ``auarc``,
``ece``, ``mce``, ``mce_isotonic``, ``brier``, ``pr_auc_incorrect``,
``pr_auc_correct``, ``spearman_corr``,
``selective_accuracy_at_coverage``, ``bootstrap_ci``.
"""

from argus_uq.ue_metrics.aurc import auarc, aurc, rcc_auc
from argus_uq.ue_metrics.auroc import auroc_incorrect, pr_auc_correct, pr_auc_incorrect
from argus_uq.ue_metrics.ause import ause
from argus_uq.ue_metrics.bootstrap import bootstrap_ci
from argus_uq.ue_metrics.brier import brier
from argus_uq.ue_metrics.ece import ece, mce
from argus_uq.ue_metrics.isotonic import (
    IsotonicCalibrator,
    brier_isotonic,
    ece_isotonic,
    mce_isotonic,
)
from argus_uq.ue_metrics.prr import (
    prr,
    prr_at_05,
    prr_normalized,
    prr_normalized_at_05,
)
from argus_uq.ue_metrics.selective import selective_accuracy_at_coverage
from argus_uq.ue_metrics.spearman import spearman_corr

__all__ = [
    # Headline
    "auroc_incorrect",
    "prr_normalized_at_05",
    "ause",
    "rcc_auc",
    "aurc",
    "ece_isotonic",
    "brier_isotonic",
    # Companions
    "prr",
    "prr_normalized",
    "prr_at_05",
    "auarc",
    "ece",
    "mce",
    "mce_isotonic",
    "brier",
    "pr_auc_incorrect",
    "pr_auc_correct",
    "spearman_corr",
    "selective_accuracy_at_coverage",
    "bootstrap_ci",
    "IsotonicCalibrator",
]


# Single-split metric registry (callable as ``f(correctness, uncertainty)``).
ALL_METRICS = {
    "prr": prr,
    "prr_normalized": prr_normalized,
    "prr_at_0.5": prr_at_05,
    "prr_normalized_at_0.5": prr_normalized_at_05,
    "auroc_incorrect": auroc_incorrect,
    "pr_auc_incorrect": pr_auc_incorrect,
    "pr_auc_correct": pr_auc_correct,
    "rcc_auc": rcc_auc,
    "aurc": aurc,
    "auarc": auarc,
    "ece": ece,
    "mce": mce,
    "brier": brier,
    "spearman": spearman_corr,
    "sel_acc@0.5": lambda c, u: selective_accuracy_at_coverage(c, u, 0.5),
    "sel_acc@0.75": lambda c, u: selective_accuracy_at_coverage(c, u, 0.75),
    "sel_acc@0.9": lambda c, u: selective_accuracy_at_coverage(c, u, 0.9),
}


# Calibration-aware metrics (require both calib and test splits).
ISOTONIC_METRICS = {
    "ece_isotonic": ece_isotonic,
    "mce_isotonic": mce_isotonic,
    "brier_isotonic": brier_isotonic,
}
