"""Smoke tests: every public symbol is importable.

These run on every PR. They do not require GPU, network, or vendor
credentials. If any of these break, the package is not even importable.
"""

from __future__ import annotations


def test_top_level_imports():
    import argus_uq

    assert argus_uq.__version__
    assert callable(argus_uq.load)
    assert callable(argus_uq.score)
    assert callable(argus_uq.conformal_disk)
    assert hasattr(argus_uq, "Cell")


def test_subpackages_import():
    import argus_uq.analysis  # noqa: F401
    import argus_uq.cell  # noqa: F401
    import argus_uq.cli  # noqa: F401
    import argus_uq.conformal  # noqa: F401
    import argus_uq.data  # noqa: F401
    import argus_uq.envs  # noqa: F401
    import argus_uq.estimators  # noqa: F401
    import argus_uq.factory  # noqa: F401
    import argus_uq.manager  # noqa: F401
    import argus_uq.models  # noqa: F401
    import argus_uq.models.closed_source  # noqa: F401
    import argus_uq.stat_calculators  # noqa: F401
    import argus_uq.ue_metrics  # noqa: F401
    import argus_uq.utils  # noqa: F401


def test_estimator_base_class_importable():
    from argus_uq.estimators.base import Estimator, EstimatorLevel, FamilyName  # noqa: F401


def test_stat_calculator_base_class_importable():
    from argus_uq.stat_calculators.base import StatCalculator  # noqa: F401


def test_factory_registry_populated():
    """Importing argus_uq.estimators triggers auto-registration of all 27 methods."""

    import argus_uq.estimators  # noqa: F401
    from argus_uq.factory import FactoryEstimator

    assert len(FactoryEstimator.list_names()) == 27


def test_listing_1_imports_succeed_at_module_level():
    """Listing 1 itself uses load/score/conformal_disk; just import it."""

    import argus_uq

    assert argus_uq.load is not None
    assert argus_uq.score is not None
    assert argus_uq.conformal_disk is not None
