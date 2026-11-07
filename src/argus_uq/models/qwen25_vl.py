"""Qwen2.5-VL agent (whitebox; HF transformers).

Wraps Qwen/Qwen2.5-VL-7B-Instruct (and the 72B-AWQ variant). Uses the
canonical Qwen2.5-VL grounding tool spec from the HF model card.

The full inference implementation - tool-spec prompt building,
smart_resize coordinate rescaling, per-token logprob/entropy/last-hidden
capture, stochastic sample decoding - is not part of this release. This
module wires the configuration.
"""

from __future__ import annotations

from typing import Any

from argus_uq.models.whitebox import WhiteboxModel


class Qwen25VL(WhiteboxModel):
    """Qwen2.5-VL-7B-Instruct grounding agent."""

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    cell_tag = "Q7"


class Qwen25VL72BAWQ(WhiteboxModel):
    """Qwen2.5-VL-72B-Instruct-AWQ (4-bit quantised; 72B effective)."""

    model_id = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    cell_tag = "Q72"
    load_in_4bit = True

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("dtype", "float16")  # AWQ requires fp16 activations
        super().__init__(**kwargs)
