"""Live smoke tests for closed-source vendor agents.

For each vendor: connect to the API (real key required), issue 1
greedy call on a fixture image, parse, disconnect. Asserts no
exceptions; does not check numerical correctness.

Gated behind ``slow`` and ``needs_api``. Skipped on CI.

Estimated cost: 3 vendors x 1 call x ~$0.01 = a few cents total.
"""

from __future__ import annotations

import os

import pytest

from argus_uq.models import AnthropicSonnet, GoogleGemini, OpenAIGPT

pytestmark = [pytest.mark.slow, pytest.mark.needs_api]


def _make_fixture_image_and_instruction():
    from PIL import Image

    img = Image.new("RGB", (1024, 768), color=(128, 128, 128))
    instr = "click the menu in the top-left corner"
    return img, instr


@pytest.mark.parametrize(
    "model_cls,env_var",
    [
        pytest.param(OpenAIGPT, "OPENAI_API_KEY", id="OpenAIGPT"),
        pytest.param(AnthropicSonnet, "ANTHROPIC_API_KEY", id="AnthropicSonnet"),
        pytest.param(GoogleGemini, "GOOGLE_API_KEY", id="GoogleGemini"),
    ],
)
def test_connect_dispatch(model_cls, env_var):
    """Connect, issue 1 greedy call, parse, disconnect. Asserts no exception."""

    if env_var not in os.environ:
        pytest.skip(f"{env_var} not set")

    pytest.importorskip(model_cls.vendor)

    model = model_cls()
    img, instr = _make_fixture_image_and_instruction()
    try:
        model.load()
        try:
            greedy = model.generate_greedy(img, instr, max_new_tokens=16)
        except NotImplementedError as e:
            pytest.skip(f"generate_greedy not yet ported for {model_cls.vendor}: {e}")
        assert greedy is not None
    finally:
        model.unload()
