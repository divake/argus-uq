"""Google Gemini 3.1 Pro grounding agent.

Vendor-isolated: imports ONLY from ``argus_uq.models.closed_source.base``
and ``google.genai`` SDK. Does not import any other vendor module.
"""

from __future__ import annotations

import os
from typing import Any

from argus_uq.models.base import GreedyOutput, SamplesOutput
from argus_uq.models.closed_source.base import BlackboxModel


class GoogleGemini(BlackboxModel):
    """Google Gemini 3.1 Pro grounding agent.

    Snapshot used in the paper: ``gemini-3.1-pro-preview`` with
    ``thinking_level=medium``.
    """

    model_id = "gemini-3.1-pro-preview"
    vendor = "google"
    cell_tag = "Gem"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        thinking_level: str = "medium",
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, api_key=api_key, timeout=timeout, **kwargs)
        self.thinking_level = thinking_level

    def _connect(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Google agent requires the [closed] extra: pip install argus-uq[closed]"
            ) from exc
        api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set; pass api_key=... or set the env var."
            )
        return genai.Client(api_key=api_key)

    def generate_greedy(
        self, image: Any, instruction: str, *, max_new_tokens: int = 32
    ) -> GreedyOutput:
        raise NotImplementedError(
            "GoogleGemini.generate_greedy is not implemented in this release."
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
            "GoogleGemini.generate_samples is not implemented in this release."
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
            "GoogleGemini.ask_text is not implemented in this release."
        )
