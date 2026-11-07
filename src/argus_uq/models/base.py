"""Model abstractions for whitebox HF transformers and closed-source APIs.

A ``Model`` produces three kinds of outputs from a (image, instruction)
pair:

1. **Greedy** decode (deterministic, single click + per-token logprobs +
   per-token last-layer hidden states + attention if available).
2. **N stochastic samples** at temperature > 0 (clicks + sample-level
   mean logprobs + sample texts).
3. **Follow-up text probe** (chat with image but no tool spec; used for
   P(True) and verbalised confidence elicitation).

Adapted from lm-polygraph's WhiteboxModel / BlackboxModel split.
The model is lazy-loaded: instantiating a subclass only stores
configuration; the actual weights / API client are loaded on
:meth:`load`. This keeps imports cheap and lets unit tests instantiate
every model class without GPU or API credentials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GreedyOutput:
    """Output of a deterministic greedy decode."""

    raw_text: str
    click_original: tuple[int, int] | None  # parsed click in original-image space
    click_model: tuple[int, int] | None  # parsed click in model-resized image space
    resized_hw: tuple[int, int]  # (h, w) image dims fed to the model
    original_hw: tuple[int, int]  # (h, w) original image dims
    signals: "StepSignals" = field(default_factory=lambda: StepSignals())


@dataclass
class StepSignals:
    """Per-token signals captured during greedy decoding."""

    logprobs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    topk_logits: list[Any] = field(default_factory=list)
    topk_ids: list[Any] = field(default_factory=list)
    last_hidden_states: list[Any] = field(default_factory=list)
    attention: dict[str, Any] | None = None


@dataclass
class SamplesOutput:
    """Output of N stochastic samples."""

    texts: list[str] = field(default_factory=list)
    click_originals: list[tuple[int, int] | None] = field(default_factory=list)
    mean_seq_logprobs: list[float] = field(default_factory=list)


class Model(ABC):
    """Abstract base for any agent that grounds (image, instruction) -> click.

    Subclasses set ``model_id`` (HF repo id or vendor snapshot string)
    and implement the three generation methods. Instantiation is cheap
    (no weight load / no API call). Call :meth:`load` to materialise.
    """

    model_id: str = ""
    family: str = "whitebox"  # "whitebox" or "closed_source"

    def __init__(self, *, model_id: str | None = None, **kwargs: Any) -> None:
        if model_id is not None:
            self.model_id = model_id
        self._loaded = False
        self._kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        """Materialise the model (load weights / connect to API)."""

    @abstractmethod
    def generate_greedy(
        self, image: Any, instruction: str, *, max_new_tokens: int = 32
    ) -> GreedyOutput:
        """Produce a deterministic greedy click."""

    @abstractmethod
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
        """Produce N stochastic samples."""

    @abstractmethod
    def ask_text(
        self,
        image: Any,
        followup_text: str,
        *,
        max_new_tokens: int = 32,
        return_token_probs_for: list[str] | None = None,
    ) -> tuple[str, dict[str, float] | None]:
        """Plain chat probe (no tool spec).

        Returns ``(text, token_probs)``. ``token_probs`` maps each
        target token in ``return_token_probs_for`` to ``P(token | image,
        followup_text)``; ``None`` when not requested or not available.
        """

    def unload(self) -> None:
        """Release weights / close API client (default: no-op)."""

        self._loaded = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_id={self.model_id!r}, "
            f"family={self.family!r}, loaded={self._loaded})"
        )
