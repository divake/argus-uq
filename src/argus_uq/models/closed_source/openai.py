"""OpenAI GPT-5.4 grounding agent.

Vendor-isolated: imports ONLY from ``argus_uq.models.closed_source.base``
and ``openai`` SDK. Does not import any other vendor module.

The vendor SDK is a soft dependency; install via
``pip install argus-uq[closed]``.
"""

from __future__ import annotations

import os
from typing import Any

from argus_uq.models.base import GreedyOutput, SamplesOutput
from argus_uq.models.closed_source.base import (
    BlackboxModel,
    parse_normalized_bbox_to_pixel_click,
    parse_verbalized_confidence,
)


class OpenAIGPT(BlackboxModel):
    """OpenAI GPT-5.4 grounding agent.

    Snapshot used in the paper: ``gpt-5.4-2026-03-05`` with
    reasoning effort = ``medium``.
    """

    model_id = "gpt-5.4-2026-03-05"
    vendor = "openai"
    cell_tag = "OAI"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str = "medium",
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, api_key=api_key, timeout=timeout, **kwargs)
        self.reasoning_effort = reasoning_effort

    def _connect(self) -> Any:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAI agent requires the [closed] extra: pip install argus-uq[closed]"
            ) from exc
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set; pass api_key=... or set the env var."
            )
        return openai.OpenAI(api_key=api_key, timeout=self.timeout)

    def generate_greedy(
        self, image: Any, instruction: str, *, max_new_tokens: int = 32
    ) -> GreedyOutput:
        raise NotImplementedError(
            "OpenAIGPT.generate_greedy is not implemented in this release."
        )

    def generate_samples(
        self,
        image: Any,
        instruction: str,
        *,
        n_samples: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 32,
    ) -> SamplesOutput:
        raise NotImplementedError(
            "OpenAIGPT.generate_samples is not implemented in this release."
        )

    def ask_text(
        self,
        image: Any,
        followup_text: str,
        *,
        max_new_tokens: int = 32,
        return_token_probs_for: list[str] | None = None,
    ) -> tuple[str, dict[str, float] | None]:
        raise NotImplementedError(
            "OpenAIGPT.ask_text is not implemented in this release."
        )


# Re-exports of shared helpers so users can parse records without
# importing the vendor base directly.
__all__ = [
    "OpenAIGPT",
    "parse_normalized_bbox_to_pixel_click",
    "parse_verbalized_confidence",
]
