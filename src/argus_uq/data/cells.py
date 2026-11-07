"""Cell catalog and metadata.

Lists the canonical 16 open-source + 12 closed-source cells with model
+ dataset + license metadata. Used by :func:`argus_uq.load` to validate
cell IDs and resolve metadata at load time.
"""

from __future__ import annotations

from argus_uq.cell import CellMetadata


# ----------------------------------------------------------------------
# Open-source cells (16)
# ----------------------------------------------------------------------

_DATASETS = {
    "V2": ("ScreenSpot-v2", "Apache-2.0"),
    "SP": ("ScreenSpot-Pro", "MIT"),
    "OSG": ("OSWorld-G non-refusal subset", "research"),
    "UIV": ("UI-Vision Element Grounding 501-item subset", "MIT"),
}

_OPEN_MODELS = {
    "Q7": ("Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "Q72": ("Qwen2.5-VL-72B-Instruct-AWQ", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"),
    "UI": ("UI-TARS-1.5-7B", "ByteDance-Seed/UI-TARS-1.5-7B"),
    "PT": ("POINTS-GUI-G-8B", "tencent/POINTS-GUI-G"),
}

_CLOSED_VENDORS = {
    "OAI": ("OpenAI GPT-5.4", "gpt-5.4-2026-03-05"),
    "Son": ("Anthropic Claude Sonnet 4.6", "claude-sonnet-4-6"),
    "Gem": ("Google Gemini 3.1 Pro", "gemini-3.1-pro-preview"),
}


def _open_cell_id(model_tag: str, dataset_tag: str) -> str:
    """Cell-id naming: ``Q7xV2``, ``PxUIV`` (historic for PT x UIV)."""

    if model_tag == "PT" and dataset_tag == "UIV":
        return "PxUIV"
    return f"{model_tag}x{dataset_tag}"


def _closed_cell_id(vendor_tag: str, dataset_tag: str) -> str:
    return f"{vendor_tag}x{dataset_tag}"


# Approximate per-cell test_acc (multi-seed mean); used for Cell.repr
# cosmetics.
_OPEN_TEST_ACC = {
    "Q7xV2": 0.883,
    "Q7xSP": 0.274,
    "Q7xOSG": 0.341,
    "Q7xUIV": 0.40,
    "Q72xV2": 0.700,
    "Q72xSP": 0.447,
    "Q72xOSG": 0.535,
    "Q72xUIV": 0.45,
    "UIxV2": 0.878,
    "UIxSP": 0.407,
    "UIxOSG": 0.513,
    "UIxUIV": 0.50,
    "PTxV2": 0.955,
    "PTxSP": 0.583,
    "PTxOSG": 0.659,
    "PxUIV": 0.531,
}

_CLOSED_TEST_ACC = {
    "OAIxV2": 0.870,
    "OAIxSP": 0.367,
    "OAIxOSG": 0.667,
    "OAIxUIV": 0.485,
    "SonxV2": 0.540,
    "SonxSP": 0.341,
    "SonxOSG": 0.363,
    "SonxUIV": 0.369,
    "GemxV2": 0.447,
    "GemxSP": 0.313,
    "GemxOSG": 0.311,
    "GemxUIV": 0.444,
}


def build_catalog() -> dict[str, CellMetadata]:
    """Return the full {cell_id -> CellMetadata} catalog (28 entries)."""

    catalog: dict[str, CellMetadata] = {}

    for m_tag, (m_name, m_hf) in _OPEN_MODELS.items():
        for d_tag, (d_name, d_lic) in _DATASETS.items():
            cell_id = _open_cell_id(m_tag, d_tag)
            catalog[cell_id] = CellMetadata(
                cell_id=cell_id,
                tier="open_source",
                model_tag=m_tag,
                model_name=m_name,
                model_hf_id=m_hf,
                dataset_tag=d_tag,
                dataset_name=d_name,
                n_items=0,  # filled at load() time
                n_test=0,
                n_calib=0,
                test_acc=_OPEN_TEST_ACC.get(cell_id, 0.0),
                license_code=d_lic,
            )

    for v_tag, (v_name, v_id) in _CLOSED_VENDORS.items():
        for d_tag, (d_name, d_lic) in _DATASETS.items():
            cell_id = _closed_cell_id(v_tag, d_tag)
            catalog[cell_id] = CellMetadata(
                cell_id=cell_id,
                tier="closed_source",
                model_tag=v_tag,
                model_name=v_name,
                model_hf_id=None,
                dataset_tag=d_tag,
                dataset_name=d_name,
                n_items=0,
                n_test=0,
                n_calib=0,
                test_acc=_CLOSED_TEST_ACC.get(cell_id, 0.0),
                license_code=d_lic,
            )

    return catalog


CATALOG: dict[str, CellMetadata] = build_catalog()
"""Read-only mapping from cell_id to CellMetadata."""


OPEN_CELLS: list[str] = sorted(c for c, m in CATALOG.items() if m.tier == "open_source")
CLOSED_CELLS: list[str] = sorted(c for c, m in CATALOG.items() if m.tier == "closed_source")
ALL_CELLS: list[str] = sorted(CATALOG)
