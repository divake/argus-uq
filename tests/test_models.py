"""Unit tests for model abstractions (no real weight load)."""

from __future__ import annotations

import pytest

from argus_uq.models import (
    AnthropicSonnet,
    GoogleGemini,
    GreedyOutput,
    Model,
    OpenAIGPT,
    OSAtlas,
    POINTSGUI,
    Qwen25VL,
    Qwen25VL72BAWQ,
    SamplesOutput,
    StepSignals,
    UITARS,
    WhiteboxModel,
)


@pytest.mark.parametrize(
    "cls,expected_id,expected_tag",
    [
        (Qwen25VL, "Qwen/Qwen2.5-VL-7B-Instruct", "Q7"),
        (Qwen25VL72BAWQ, "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "Q72"),
        (UITARS, "ByteDance-Seed/UI-TARS-1.5-7B", "UI"),
        (POINTSGUI, "tencent/POINTS-GUI-G", "PT"),
        (OSAtlas, "OS-Copilot/OS-Atlas-Base-7B", "OA"),
    ],
)
def test_whitebox_class_attributes(cls, expected_id, expected_tag):
    """Each whitebox VLM class exposes the right model_id and cell_tag."""

    assert cls.model_id == expected_id
    assert cls.cell_tag == expected_tag
    assert cls.family == "whitebox"


@pytest.mark.parametrize(
    "cls,expected_id,expected_vendor,expected_tag",
    [
        (OpenAIGPT, "gpt-5.4-2026-03-05", "openai", "OAI"),
        (AnthropicSonnet, "claude-sonnet-4-6", "anthropic", "Son"),
        (GoogleGemini, "gemini-3.1-pro-preview", "google", "Gem"),
    ],
)
def test_closed_source_class_attributes(cls, expected_id, expected_vendor, expected_tag):
    """Each closed-source class has model_id, vendor, cell_tag set."""

    assert cls.model_id == expected_id
    assert cls.vendor == expected_vendor
    assert cls.cell_tag == expected_tag
    assert cls.family == "closed_source"


def test_whitebox_instantiates_without_weight_load():
    """Whitebox classes are lazy: no transformers import on __init__."""

    m = Qwen25VL()
    assert m.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert m._loaded is False


def test_closed_source_instantiates_without_api_call():
    """Closed-source classes are lazy: no SDK import on __init__."""

    m = OpenAIGPT(api_key="dummy")
    assert m._loaded is False
    assert m.api_key == "dummy"


def test_whitebox_overrides_model_id_via_kwarg():
    m = Qwen25VL(model_id="custom-id")
    assert m.model_id == "custom-id"


def test_closed_source_load_without_key_raises():
    """Connecting without an API key surfaces a clear error."""

    import os

    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        m = OpenAIGPT()
        with pytest.raises((RuntimeError, ImportError)):
            m.load()
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def test_all_classes_implement_abstract_methods():
    """Every concrete subclass implements (or inherits) the three generation methods."""

    for cls in [Qwen25VL, Qwen25VL72BAWQ, UITARS, POINTSGUI, OSAtlas]:
        m = cls()
        assert callable(m.generate_greedy)
        assert callable(m.generate_samples)
        assert callable(m.ask_text)


def test_dataclasses_default_construction():
    """GreedyOutput / SamplesOutput / StepSignals construct with defaults."""

    g = GreedyOutput(
        raw_text="",
        click_original=None,
        click_model=None,
        resized_hw=(0, 0),
        original_hw=(0, 0),
    )
    assert isinstance(g.signals, StepSignals)
    s = SamplesOutput()
    assert s.texts == []
    assert s.click_originals == []


def test_model_repr_includes_id_and_loaded_state():
    m = Qwen25VL()
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in repr(m)
    assert "loaded=False" in repr(m)


def test_whitebox_is_subclass_of_model():
    assert issubclass(WhiteboxModel, Model)
    assert issubclass(Qwen25VL, Model)
