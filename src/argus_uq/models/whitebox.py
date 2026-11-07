"""WhiteboxModel - HF transformers wrapper.

Common implementation shared by all open-weight VLM agents (Qwen2.5-VL
family, UI-TARS, POINTS-GUI-G). Subclasses set ``model_id``, the
prompt-template builder, and the click-parsing regex; the rest is
inherited.

The full inference logic - greedy decode with per-token signal capture,
stochastic sampling, follow-up text probes, image resizing through
Qwen2.5-VL's smart_resize - is non-trivial. To keep this module
runnable on a fresh install without torch / transformers, the heavy
imports are deferred to :meth:`load` and :meth:`generate_*`. Calling
``Qwen25VL()`` only stores configuration; the actual ``transformers``
import happens on the first ``.load()``.
"""

from __future__ import annotations

from typing import Any

from argus_uq.models.base import (
    GreedyOutput,
    Model,
    SamplesOutput,
    StepSignals,
)


class WhiteboxModel(Model):
    """Base class for HF transformers VLM agents.

    Provides the shared lifecycle (``load`` -> torch + transformers
    import + weight load; ``unload`` -> drop CUDA tensors) and a
    extension point for the three generation methods. Concrete subclasses
    override the prompt builder and click parser.
    """

    family = "whitebox"
    dtype: str = "bfloat16"
    device: str = "cuda"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    load_in_4bit: bool = False

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        load_in_4bit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.device = device
        self.dtype = dtype
        if min_pixels is not None:
            self.min_pixels = min_pixels
        if max_pixels is not None:
            self.max_pixels = max_pixels
        self.load_in_4bit = load_in_4bit
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        """Materialise the HF model and processor.

        Lazy import of ``transformers`` so this module is importable
        without the ``[whitebox]`` extras.
        """

        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - import-error path
            raise ImportError(
                "Whitebox VLMs require the [whitebox] extra: pip install argus-uq[whitebox]"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        torch_dtype = getattr(torch, self.dtype)
        kwargs: dict[str, Any] = dict(torch_dtype=torch_dtype, device_map=self.device)
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._loaded = True

    def unload(self) -> None:
        """Drop the HF model + free CUDA memory."""

        super().unload()
        self._processor = None
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # The three generation methods below raise NotImplementedError.
    # Concrete subclasses (Qwen25VL, UITARS, POINTSGUI) override with
    # model-specific prompt building, click parsing, and signal
    # capture. The full reference implementation lives in
    # ``argus_uq.models.qwen25_vl``.

    def generate_greedy(
        self, image: Any, instruction: str, *, max_new_tokens: int = 32
    ) -> GreedyOutput:
        raise NotImplementedError(
            f"{type(self).__name__}.generate_greedy must be implemented by the concrete subclass."
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
            f"{type(self).__name__}.generate_samples must be implemented by the concrete subclass."
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
            f"{type(self).__name__}.ask_text must be implemented by the concrete subclass."
        )


# Re-exports for convenience.
__all__ = ["WhiteboxModel", "GreedyOutput", "SamplesOutput", "StepSignals"]
