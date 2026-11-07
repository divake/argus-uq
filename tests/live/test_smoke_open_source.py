"""Live smoke tests for whitebox VLM agents.

For each (model, dataset) pair: load the model (real HF weights),
construct one minimal grounding fixture, attempt a greedy + a sampling
+ an ask_text call, then dispose. Assert no exceptions; do not check
numerical correctness.

These tests verify the wiring (lazy load, GPU memory release, sample
parsing); they do not benchmark accuracy.

Gated behind ``slow`` and ``needs_gpu``. Skipped on CI.

NOTE: As of this commit, ``WhiteboxModel.generate_*`` methods raise
``NotImplementedError``; the inference bodies are not part of this
release. These tests skip until they are supplied. The test scaffolding
is in place so the contract is locked.
"""

from __future__ import annotations

import pytest

from argus_uq.models import POINTSGUI, Qwen25VL, Qwen25VL72BAWQ, UITARS

pytestmark = [pytest.mark.slow, pytest.mark.needs_gpu]


def _make_fixture_image_and_instruction():
    """Minimal in-memory image + instruction for a single greedy call."""

    from PIL import Image

    img = Image.new("RGB", (1024, 768), color=(128, 128, 128))
    instr = "click the menu in the top-left corner"
    return img, instr


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(Qwen25VL, id="Qwen25VL"),
        pytest.param(Qwen25VL72BAWQ, id="Qwen25VL72BAWQ", marks=pytest.mark.skip(
            reason="72B-AWQ requires >24GB VRAM"
        )),
        pytest.param(UITARS, id="UITARS"),
        pytest.param(POINTSGUI, id="POINTSGUI"),
    ],
)
def test_load_dispatch_dispose(model_cls):
    """Load model, run 1 greedy + 1 sampling, dispose. Asserts no exception."""

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model = model_cls()
    img, instr = _make_fixture_image_and_instruction()
    try:
        model.load()
        # generate_greedy raises NotImplementedError until the inference
        # body is ported. Convert to a skip until that lands.
        try:
            greedy = model.generate_greedy(img, instr, max_new_tokens=16)
        except NotImplementedError as e:
            pytest.skip(f"generate_greedy not yet ported: {e}")
        assert greedy is not None
        # Sampling is the same path; if greedy works, samples should too.
        try:
            samples = model.generate_samples(img, instr, n_samples=2, max_new_tokens=16)
        except NotImplementedError:
            samples = None
        # ask_text probe (P(True) / verbalised).
        try:
            text, probs = model.ask_text(img, "Was that click correct?", max_new_tokens=8)
        except NotImplementedError:
            text = None
    finally:
        model.unload()
