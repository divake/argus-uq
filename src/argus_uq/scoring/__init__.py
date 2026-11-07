"""argus_uq.scoring - per-method scoring + multi-seed aggregation.

Two main entry points:
- :func:`score_cached` - read cached per-method scores from a Cell's
  per-item parquet, compute multi-seed metrics, return ScoreResult.
  This is what :func:`argus_uq.score` calls.
- :class:`MultiSeedScorer` - run the full N-seed CP loop for a single
  cell, producing the canonical paper-numbers row of
  ``derived/master_long.csv``.
"""

from argus_uq.scoring.cached import score_cached
from argus_uq.scoring.multi_seed import (
    MultiSeedScorer,
    SEED_LIST_CANONICAL,
    multi_seed_metrics,
)

__all__ = [
    "score_cached",
    "MultiSeedScorer",
    "SEED_LIST_CANONICAL",
    "multi_seed_metrics",
]
