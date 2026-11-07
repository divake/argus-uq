"""argus_uq.stat_calculators - one-pass signal extraction layer.

A StatCalculator runs the (expensive) inference once per item and emits
a dict of named signals (greedy.logprobs, samples.clicks, etc.). The
manager topologically orders calculators by their (provides, depends)
graph so each signal is computed exactly once.

Adapted from lm-polygraph's stat_calculators design (see
``NOTICES.md``).
"""

from argus_uq.stat_calculators.base import StatCalculator

__all__ = ["StatCalculator"]
