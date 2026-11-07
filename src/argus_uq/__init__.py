"""argus-uq: uncertainty quantification benchmark for VLM GUI grounding agents.

This is the public API surface:

>>> import argus_uq
>>> cell = argus_uq.load("PTxOSG")
>>> result = argus_uq.score(cell, method="saplma", seeds=50)
>>> disk = argus_uq.conformal_disk(cell, plug_in="saplma", alpha=0.10,
...                                 variant="normalized")

Anything else is internal and may change between versions.
"""

from argus_uq._version import __version__
from argus_uq.api import conformal_disk, load, score
from argus_uq.cell import Cell

__all__ = [
    "__version__",
    "load",
    "score",
    "conformal_disk",
    "Cell",
]
