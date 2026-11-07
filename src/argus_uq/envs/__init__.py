"""argus_uq.envs - dataset wrappers.

Four single-step grounding datasets:

- ``ScreenSpotV2`` (V2; 1,272 items; Apache 2.0)
- ``ScreenSpotPro`` (SP; 1,581 items; Apache 2.0)
- ``OSWorldG`` (OSG; 564 non-refusal items; research)
- ``UIVision`` (UIV; 501-item subset; MIT)

Each env discovers its data via env-var override or HuggingFace cache
fallback, and yields :class:`GroundingItem` instances with metadata
suitable for slicing / stratification.
"""

from argus_uq.envs.base import Env, GroundingItem
from argus_uq.envs.osworld_g import OSWorldG
from argus_uq.envs.screenspot_pro import ScreenSpotPro
from argus_uq.envs.screenspot_v2 import ScreenSpotV2
from argus_uq.envs.ui_vision import UIVision

__all__ = [
    "Env",
    "GroundingItem",
    "ScreenSpotV2",
    "ScreenSpotPro",
    "OSWorldG",
    "UIVision",
]
