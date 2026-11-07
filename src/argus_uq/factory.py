"""FactoryEstimator - name -> Estimator class registry.

Modelled after lm-polygraph's factory_estimator
(utils/factory_estimator.py:64-78). Estimators self-register on subclass
creation via Estimator.__init_subclass__ (see estimators/base.py).

Use:
    >>> from argus_uq.factory import FactoryEstimator
    >>> FactoryEstimator.list_names()
    ['ccp', 'cocoa', 'cocoa_1mca', 'focus', 'hedge', ...]
    >>> est = FactoryEstimator.build("saplma")
    >>> est = FactoryEstimator.build("saplma", hidden_dims=[256, 128, 64])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argus_uq.estimators.base import Estimator


class FactoryEstimator:
    """Registry of Estimator classes, keyed by Estimator.name."""

    _registry: dict[str, type[Estimator]] = {}

    @classmethod
    def register(cls, est_cls: type[Estimator]) -> None:
        """Register an Estimator subclass by its `name`."""

        if not est_cls.name:
            raise ValueError(f"Estimator class {est_cls.__name__} has empty `name`.")
        if est_cls.name in cls._registry:
            raise ValueError(
                f"Duplicate Estimator name {est_cls.name!r}: "
                f"{cls._registry[est_cls.name].__name__} vs {est_cls.__name__}"
            )
        cls._registry[est_cls.name] = est_cls

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Estimator:
        """Instantiate the registered Estimator with the given name."""

        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Unknown estimator {name!r}. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_names(cls) -> list[str]:
        """All registered estimator names, sorted."""

        return sorted(cls._registry)

    @classmethod
    def list_by_family(cls) -> dict[str, list[str]]:
        """Estimator names grouped by family."""

        out: dict[str, list[str]] = {}
        for name, est_cls in cls._registry.items():
            out.setdefault(est_cls.family, []).append(name)
        for family in out:
            out[family].sort()
        return out
