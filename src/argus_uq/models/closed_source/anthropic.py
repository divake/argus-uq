"""Anthropic Claude Sonnet 4.6 grounding agent.

Vendor-isolated: imports ONLY from ``argus_uq.models.closed_source.base``
and ``anthropic`` SDK. Does not import any other vendor module.
"""

from __future__ import annotations

import os
from typing import Any

from argus_uq.models.base import GreedyOutput, SamplesOutput
from argus_uq.models.closed_source.base import BlackboxModel


class AnthropicSonnet(BlackboxModel):
    """Anthropic Claude Sonnet 4.6 grounding agent.

    Snapshot used in the paper: ``claude-sonnet-4-6`` with
    ``thinking.budget_tokens=10000``.
    """

    model_id = "claude-sonnet-4-6"
    vendor = "anthropic"
    cell_tag = "Son"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        thinking_budget_tokens: int = 10000,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, api_key=api_key, timeout=timeout, **kwargs)
        self.thinking_budget_tokens = thinking_budget_tokens

    def _connect(self) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Anthropic agent requires the [closed] extra: pip install argus-uq[closed]"
            ) from exc
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; pass api_key=... or set the env var."
            )
        return anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

    def generate_greedy(
        self, image: Any, instruction: str, *, max_new_tokens: int = 32
    ) -> GreedyOutput:
        raise NotImplementedError(
            "AnthropicSonnet.generate_greedy is not implemented in this release."
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
            "AnthropicSonnet.generate_samples is not implemented in this release."
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
            "AnthropicSonnet.ask_text is not implemented in this release."
        )
