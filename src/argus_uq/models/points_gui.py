"""POINTS-GUI-G-8B agent (whitebox; Qwen3 LLM head).

POINTS-GUI-G-8B is a grounding-specialist VLM from Tencent that uses
a Qwen3 LLM head and a native 0-1000 normalised bbox parser. The
inference loop differs from Qwen2.5-VL in (a) the prompt template
(POINTS uses a point-form grounding directive) and (b) the click
parsing (returns ``(x_norm, y_norm)`` in ``[0, 1000]``, rescaled to
original image space at inference time).
"""

from __future__ import annotations

from argus_uq.models.whitebox import WhiteboxModel


class POINTSGUI(WhiteboxModel):
    """POINTS-GUI-G-8B grounding agent."""

    model_id = "tencent/POINTS-GUI-G"
    cell_tag = "PT"
