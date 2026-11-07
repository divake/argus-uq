"""StatCalculator abstract base class.

Adapted from lm-polygraph (stat_calculators/stat_calculator.py:9-83). Each
calculator declares which signals it produces and which it depends on.
The manager builds a DAG over calculators, topologically sorts it, and
runs each calculator exactly once per item batch. Estimators then read
the cached signals.

This pattern saves an order of magnitude on a 27-method panel: greedy
inference, sampling, hidden-state extraction, and attention capture all
run once and serve all methods that need them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class StatCalculator(ABC):
    """Abstract base for a one-pass signal extractor.

    A StatCalculator's job is to produce one or more named signals (e.g.
    ``greedy.text``, ``greedy.click``, ``greedy.logprobs``,
    ``greedy.last_hidden``) from a model + a batch of input items, and
    return them in a dict. Estimators consume the dict.

    Subclasses must:

    1. Implement ``meta_info()`` (staticmethod) returning a tuple
       ``(provides, depends)`` of two string lists.
    2. Implement ``__call__(deps, items, model, **kwargs)`` returning a
       dict whose keys are the names listed in ``provides``.
    """

    @staticmethod
    @abstractmethod
    def meta_info() -> tuple[list[str], list[str]]:
        """Static metadata for the dependency graph.

        Returns
        -------
        provides:
            List of stat names this calculator emits.
        depends:
            List of stat names this calculator consumes (produced by
            other StatCalculators upstream). Empty list for "raw"
            calculators that only depend on the model + items.
        """

    @abstractmethod
    def __call__(
        self,
        deps: dict[str, np.ndarray],
        items: list[Any],
        model: Any,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Compute the signals.

        Parameters
        ----------
        deps:
            Dict of upstream stats (matching the names in
            ``meta_info()[1]``).
        items:
            List of input items (e.g. ``GroundingItem`` instances).
        model:
            The agent (a ``Model`` subclass) - whitebox or blackbox.
        kwargs:
            Run-level configuration (sampling temperature, n_samples,
            max_new_tokens, ...).

        Returns
        -------
        dict[str, np.ndarray]:
            Keyed by names in ``meta_info()[0]``.
        """

    def __repr__(self) -> str:
        provides, depends = self.meta_info()
        return f"{type(self).__name__}(provides={provides!r}, depends={depends!r})"
