"""Estimator abstract base class.

A UQ method (estimator) consumes a per-item record and emits a scalar
uncertainty (higher = more uncertain). The record schema is documented
in :mod:`argus_uq.records`.

Some estimators (density / probe family) require fitting on a calibration
split before scoring a test split; these set ``trainable = True`` and
implement :meth:`fit`.

Design adapted from the lm-polygraph Estimator pattern, simplified to a
per-item interface that maps directly to the released per-item parquet
schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np


# Closed enumeration of UQ-method families.
FamilyName = Literal[
    "logit",
    "sampling",
    "consistency",
    "density",
    "attention",
    "prompt",
    "vlm_native",
]


# Aggregation level. "item" means one float per input record; "cell"
# is reserved for future use.
EstimatorLevel = Literal["item", "cell"]


class Estimator(ABC):
    """Abstract base class for a UQ method.

    Subclasses must:

    1. Set ``name`` (lowercase snake_case; matches per-item parquet
       column header).
    2. Set ``family`` (one of the seven FamilyName values).
    3. Implement :meth:`score_item` to return a scalar uncertainty.

    Optional class attributes:

    - ``available_on_closed_source`` (default ``True``) - if ``False``,
      the closed-source scoring path skips this method.
    - ``trainable`` (default ``False``) - if ``True``, the manager calls
      :meth:`fit` on the calibration split before scoring.
    """

    name: str = ""
    family: FamilyName = "logit"
    level: EstimatorLevel = "item"
    available_on_closed_source: bool = True
    trainable: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            return  # tolerate intermediate abstract classes
        from argus_uq.factory import FactoryEstimator

        FactoryEstimator.register(cls)

    def fit(
        self,
        calib_records: list[dict[str, Any]],
        calib_labels: np.ndarray,
    ) -> None:
        """Fit any trainable parameters on the calibration split.

        Default is a no-op. Trainable estimators (``SAPLMA``, ``SEP``,
        ``Mahalanobis*``) override.
        """

        if self.trainable:
            raise NotImplementedError(
                f"Estimator {self.name!r} declares trainable=True but does not implement fit()."
            )

    @abstractmethod
    def score_item(self, record: dict[str, Any]) -> float:
        """Score a single per-item record.

        Parameters
        ----------
        record:
            A dict with the per-item fields described in
            :mod:`argus_uq.records`. Missing optional fields are
            ``None``; estimators that need them must handle ``None``
            gracefully.

        Returns
        -------
        float:
            Scalar uncertainty. Higher = more uncertain. Convention:
            methods that naturally return confidences must invert
            (e.g. ``1 - max_softmax``).
        """

    def score_batch(self, records: list[dict[str, Any]]) -> np.ndarray:
        """Score a list of per-item records.

        Default: per-item loop. Subclasses may override for vectorised
        scoring (e.g. attention family is faster on a batched fp32 cast).
        """

        return np.array([self.score_item(r) for r in records], dtype=np.float64)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, family={self.family!r})"
