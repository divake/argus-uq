"""UI-TARS-1.5-7B agent (whitebox; Qwen2.5-VL fine-tune).

UI-TARS is a grounding-fine-tuned variant of Qwen2.5-VL-7B from
ByteDance-Seed. Inference path is identical to Qwen2.5-VL except for
the prompt template (UI-TARS uses a slightly different action format
and EOS token).
"""

from __future__ import annotations

from argus_uq.models.whitebox import WhiteboxModel


class UITARS(WhiteboxModel):
    """UI-TARS-1.5-7B grounding agent."""

    model_id = "ByteDance-Seed/UI-TARS-1.5-7B"
    cell_tag = "UI"
