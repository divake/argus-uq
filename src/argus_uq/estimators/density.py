"""Density-family UQ methods (5 methods; all trainable; open-source only).

These methods consume per-token last-layer hidden states and require
fitting on a calibration split before scoring.

References:

- ``Mahalanobis``: Lee et al. 2018, *A Simple Unified Framework for
  Detecting Out-of-Distribution Samples and Adversarial Attacks*
  (NeurIPS 2018). Mahalanobis distance to the positive-class
  (correct) centroid with shared LDA-style within-class covariance.

- ``MahalanobisRMD``: Ren et al. 2023, *Out-of-Distribution Detection
  and Selective Generation for Conditional Language Models*
  (arXiv 2209.15558). Subtracts a background-Gaussian distance,
  isolating the correctness-specific direction.

- ``MahalanobisRDE``: Robust Density Estimation. KernelPCA (RBF, 100
  dims) + MinCovDet (Rousseeuw 1984) on the positive class. Mirrors
  lm-polygraph ``RDESeq``.

- ``SemanticEntropyProbe`` (SEP): Kossen et al. 2024, *Semantic
  Entropy Probes* (arXiv 2406.15927), simplified to a single
  logistic regressor on pooled hidden states.

- ``SAPLMA``: Azaria & Mitchell 2023, *The Internal State of an LLM
  Knows When It's Lying* (EMNLP 2023). MLP probe (256-128-64-1 with
  ReLU) on pooled last-layer hidden states predicting a binary
  correctness label.

Closed-source availability: NONE. Closed APIs do not expose hidden
states.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.estimators.base import Estimator, FamilyName
from argus_uq.records import pool_hidden_states, safe_stack_hidden


# Extended jitter grid matching lm-polygraph's full range.
_JITTER_GRID = [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]


def _stack_pooled(features: list[np.ndarray]) -> np.ndarray:
    """Stack a list of pooled feature vectors into a 2-D matrix."""

    return np.stack(
        [
            pool_hidden_states([f]) if f.ndim == 1 else f.mean(axis=0)
            for f in features
        ],
        axis=0,
    )


def _fit_precision_with_jitter(
    cov: np.ndarray,
    jitter_grid: list[float] = _JITTER_GRID,
    relative_tol: float = 1e-4,
) -> np.ndarray:
    """Pick the smallest jitter that gives a numerically PSD covariance.

    Strengthens the lm-polygraph absolute-tolerance check to a
    *relative* tolerance:
    ``min_eig >= relative_tol * max(trace(cov)/d, 1e-12)``. This
    requires every direction to be well-conditioned relative to the
    covariance's own scale, not an absolute floor.
    """

    d = cov.shape[0]
    eye = np.eye(d)
    trace_per_dim = float(np.trace(cov)) / max(d, 1)
    threshold = relative_tol * max(trace_per_dim, 1e-12)
    for eps in jitter_grid:
        reg = cov + eps * eye
        try:
            eigenvalues = np.linalg.eigvalsh(reg)
        except np.linalg.LinAlgError:
            continue
        if float(eigenvalues.min()) >= threshold:
            return np.linalg.pinv(reg)
    return np.linalg.pinv(cov + jitter_grid[-1] * eye)


def _features_from_records(records: list[dict[str, Any]]) -> list[np.ndarray]:
    """Extract per-item pooled hidden-state features from records."""

    out: list[np.ndarray] = []
    for r in records:
        hs = r.get("greedy_last_hidden") or []
        out.append(pool_hidden_states(hs))
    return out


class Mahalanobis(Estimator):
    """Mahalanobis distance to the correct-prediction centroid.

    Adapted from Lee et al. 2018 for the correctness-prediction
    setting. Fit estimates the mean of pooled hidden states on
    correct calibration items and a shared within-class covariance
    (LDA-style). Score = Mahalanobis distance from the test pooled
    hidden state to the correct centroid.

    Open-source only - hidden states required.
    """

    name = "mahalanobis"
    family: FamilyName = "density"
    trainable = True
    available_on_closed_source = False

    def __init__(self) -> None:
        self.mean_pos: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, calib_records: list[dict[str, Any]], calib_labels: np.ndarray) -> None:
        features = _features_from_records(calib_records)
        feats = _stack_pooled(features)
        labels = np.asarray(calib_labels)
        d = feats.shape[1]

        if labels.sum() < 2:
            self.mean_pos = feats.mean(axis=0)
            cov = np.cov(feats, rowvar=False) if feats.shape[0] > 1 else np.eye(d)
            self.precision = _fit_precision_with_jitter(cov)
            return

        self.mean_pos = feats[labels == 1].mean(axis=0)
        centered_parts: list[np.ndarray] = [feats[labels == 1] - self.mean_pos]
        if (labels == 0).sum() >= 2:
            mean_neg = feats[labels == 0].mean(axis=0)
            centered_parts.append(feats[labels == 0] - mean_neg)
        centered = np.concatenate(centered_parts, axis=0)
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        self.precision = _fit_precision_with_jitter(cov)

    def score_item(self, record: dict[str, Any]) -> float:
        if self.mean_pos is None or self.precision is None:
            raise RuntimeError(f"{self.name} must be fit before it is called")
        feat = pool_hidden_states(record.get("greedy_last_hidden") or [])
        if feat.size == 0:
            return float("inf")
        delta = feat - self.mean_pos
        return float(delta @ self.precision @ delta)


class MahalanobisRMD(Estimator):
    """Relative Mahalanobis Distance (Ren et al. 2023).

    Score = ``d(x, mean_pos) - d(x, mean_bg)`` where ``mean_bg`` is
    the pooled mean over all calibration items. The "background" term
    subtracts out prompt/image-complexity variance, isolating the
    correctness-specific direction.

    Open-source only.
    """

    name = "mahalanobis_rmd"
    family: FamilyName = "density"
    trainable = True
    available_on_closed_source = False

    def __init__(self) -> None:
        self.mean_pos: np.ndarray | None = None
        self.mean_bg: np.ndarray | None = None
        self.precision_pos: np.ndarray | None = None
        self.precision_bg: np.ndarray | None = None

    def fit(self, calib_records: list[dict[str, Any]], calib_labels: np.ndarray) -> None:
        features = _features_from_records(calib_records)
        feats = _stack_pooled(features)
        labels = np.asarray(calib_labels)
        d = feats.shape[1]

        # Background = pooled Gaussian over all calibration items.
        self.mean_bg = feats.mean(axis=0)
        cov_bg = np.cov(feats, rowvar=False) if feats.shape[0] > 1 else np.eye(d)
        self.precision_bg = _fit_precision_with_jitter(cov_bg)

        if labels.sum() < 2:
            self.mean_pos = self.mean_bg.copy()
            self.precision_pos = self.precision_bg.copy()
            return

        self.mean_pos = feats[labels == 1].mean(axis=0)
        centered_parts: list[np.ndarray] = [feats[labels == 1] - self.mean_pos]
        if (labels == 0).sum() >= 2:
            mean_neg = feats[labels == 0].mean(axis=0)
            centered_parts.append(feats[labels == 0] - mean_neg)
        centered = np.concatenate(centered_parts, axis=0)
        cov_pos = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        self.precision_pos = _fit_precision_with_jitter(cov_pos)

    def score_item(self, record: dict[str, Any]) -> float:
        if self.mean_pos is None or self.precision_pos is None:
            raise RuntimeError(f"{self.name} must be fit before it is called")
        feat = pool_hidden_states(record.get("greedy_last_hidden") or [])
        if feat.size == 0:
            return float("inf")
        delta_pos = feat - self.mean_pos
        d_pos = float(delta_pos @ self.precision_pos @ delta_pos)
        delta_bg = feat - self.mean_bg
        d_bg = float(delta_bg @ self.precision_bg @ delta_bg)
        return d_pos - d_bg


class MahalanobisRDE(Estimator):
    """Robust Density Estimation: KernelPCA + MinCovDet (lm-polygraph RDESeq).

    Pipeline:

    1. Fit ``KernelPCA(n_components=100, kernel='rbf')`` on all
       calibration features. Handles ``n_pos << d`` regime.
    2. Fit ``MinCovDet`` (Rousseeuw 1984) on reduced positive-class
       features. Robust to outliers.
    3. Score = MCD's ``mahalanobis()`` distance at test.

    Falls back gracefully if positive class is too small.

    Open-source only.
    """

    name = "mahalanobis_rde"
    family: FamilyName = "density"
    trainable = True
    available_on_closed_source = False

    def __init__(self, n_components: int = 100) -> None:
        self.n_components = n_components
        self.kpca: Any | None = None
        self.mcd: Any | None = None

    def fit(self, calib_records: list[dict[str, Any]], calib_labels: np.ndarray) -> None:
        from sklearn.covariance import MinCovDet
        from sklearn.decomposition import KernelPCA

        features = _features_from_records(calib_records)
        feats = _stack_pooled(features)
        labels = np.asarray(calib_labels)

        n_pos = int(labels.sum())
        if n_pos < 5:
            self.kpca = None
            self.mcd = None
            return

        n_comp = min(self.n_components, feats.shape[0] - 1, max(2 * n_pos, 10))
        self.kpca = KernelPCA(n_components=n_comp, kernel="rbf", gamma=None)
        reduced = self.kpca.fit_transform(feats)

        pos_reduced = reduced[labels == 1]
        support_fraction = max(0.5, (n_comp + 1) / n_pos) if n_pos > n_comp else None
        self.mcd = MinCovDet(support_fraction=support_fraction, random_state=0)
        try:
            self.mcd.fit(pos_reduced)
        except (ValueError, np.linalg.LinAlgError):
            self.mcd = None

    def score_item(self, record: dict[str, Any]) -> float:
        if self.kpca is None or self.mcd is None:
            return float("inf")
        feat = pool_hidden_states(record.get("greedy_last_hidden") or [])
        if feat.size == 0:
            return float("inf")
        reduced = self.kpca.transform(feat.reshape(1, -1))
        return float(self.mcd.mahalanobis(reduced)[0])


class SemanticEntropyProbe(Estimator):
    """Logistic probe over pooled hidden states predicting P(correct).

    Inspired by Kossen et al. 2024, simplified to a single logistic
    regressor. Returns ``1 - P(correct)``.

    Open-source only.
    """

    name = "sep"
    family: FamilyName = "density"
    trainable = True
    available_on_closed_source = False

    def __init__(self, l2: float = 1.0, max_iter: int = 1000) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, calib_records: list[dict[str, Any]], calib_labels: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        features = _features_from_records(calib_records)
        feats = _stack_pooled(features)
        labels = np.asarray(calib_labels)
        if len(set(labels.tolist())) < 2:
            self.weights = np.zeros(feats.shape[1], dtype=np.float32)
            self.bias = float(labels.mean())
            return
        clf = LogisticRegression(C=1.0 / self.l2, max_iter=self.max_iter)
        clf.fit(feats, labels)
        self.weights = clf.coef_[0].astype(np.float32)
        self.bias = float(clf.intercept_[0])

    def score_item(self, record: dict[str, Any]) -> float:
        if self.weights is None:
            raise RuntimeError(f"{self.name} must be fit before it is called")
        feat = pool_hidden_states(record.get("greedy_last_hidden") or [])
        if feat.size == 0:
            return 1.0
        logit = float(feat @ self.weights + self.bias)
        p_correct = 1.0 / (1.0 + float(np.exp(-logit)))
        return float(1.0 - p_correct)


class SAPLMA(Estimator):
    """SAPLMA - MLP probe on pooled last-layer hidden states.

    Azaria & Mitchell 2023, *The Internal State of an LLM Knows When
    It's Lying* (EMNLP 2023). Sklearn ``MLPClassifier`` with hidden
    layer sizes ``(256, 128, 64)``, ReLU activation, sigmoid head.
    Returns ``1 - P(correct)``.

    Open-source only.
    """

    name = "saplma"
    family: FamilyName = "density"
    trainable = True
    available_on_closed_source = False

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        max_iter: int = 500,
        random_state: int = 0,
        l2: float = 1e-4,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.l2 = l2
        self.mlp: Any | None = None
        self._fallback_prob = 0.5

    def fit(self, calib_records: list[dict[str, Any]], calib_labels: np.ndarray) -> None:
        from sklearn.neural_network import MLPClassifier

        features = _features_from_records(calib_records)
        feats = _stack_pooled(features)
        labels = np.asarray(calib_labels)

        if len(set(labels.tolist())) < 2:
            self.mlp = None
            self._fallback_prob = float(labels.mean())
            return

        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_sizes,
            activation="relu",
            solver="adam",
            alpha=self.l2,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False,
            batch_size=min(64, len(feats)),
        )
        self.mlp.fit(feats, labels)

    def score_item(self, record: dict[str, Any]) -> float:
        feat = pool_hidden_states(record.get("greedy_last_hidden") or [])
        if feat.size == 0:
            return 1.0
        if self.mlp is None:
            return float(1.0 - self._fallback_prob)
        p_correct = float(self.mlp.predict_proba(feat.reshape(1, -1))[0, 1])
        return float(1.0 - p_correct)


# Suppress stack-empty warnings in sklearn for tiny calibration sets
_ = safe_stack_hidden  # keep import; used in tests
