"""argus_uq.models - agent abstractions.

Whitebox (HF transformers) and closed-source (vendor APIs) agents share
the same ``Model`` ABC, with three generation methods (greedy / samples
/ ask_text) and lazy weight / API loading.

Whitebox agents:
- ``Qwen25VL`` - Qwen2.5-VL-7B-Instruct (cell tag Q7)
- ``Qwen25VL72BAWQ`` - Qwen2.5-VL-72B-AWQ (Q72)
- ``UITARS`` - UI-TARS-1.5-7B (UI)
- ``POINTSGUI`` - POINTS-GUI-G-8B (PT)
- ``OSAtlas`` - OS-Atlas-Base-7B (kept for reference; not in paper)

Closed-source agents (vendor-isolated):
- ``OpenAIGPT`` - GPT-5.4 (OAI)
- ``AnthropicSonnet`` - Claude Sonnet 4.6 (Son)
- ``GoogleGemini`` - Gemini 3.1 Pro (Gem)
"""

from argus_uq.models.base import (
    GreedyOutput,
    Model,
    SamplesOutput,
    StepSignals,
)
from argus_uq.models.closed_source.anthropic import AnthropicSonnet
from argus_uq.models.closed_source.google import GoogleGemini
from argus_uq.models.closed_source.openai import OpenAIGPT
from argus_uq.models.os_atlas import OSAtlas
from argus_uq.models.points_gui import POINTSGUI
from argus_uq.models.qwen25_vl import Qwen25VL, Qwen25VL72BAWQ
from argus_uq.models.ui_tars import UITARS
from argus_uq.models.whitebox import WhiteboxModel

__all__ = [
    "Model",
    "WhiteboxModel",
    "GreedyOutput",
    "SamplesOutput",
    "StepSignals",
    # Whitebox
    "Qwen25VL",
    "Qwen25VL72BAWQ",
    "UITARS",
    "POINTSGUI",
    "OSAtlas",
    # Closed-source
    "OpenAIGPT",
    "AnthropicSonnet",
    "GoogleGemini",
]
