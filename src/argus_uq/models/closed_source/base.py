"""BlackboxModel - closed-source API agent base.

Vendor isolation rule: each vendor module imports ONLY from this module
and Python stdlib + the vendor's own SDK. No cross-vendor imports.
``tests/test_vendor_isolation.py`` enforces this.

The CallRecord schema below is the on-disk format for
the released closed-source records (``data_release/closed_source_records/``).
Each record represents one API call (greedy / sample_i / hedge_text_i /
hedge_image_i / p_true / verbalized_1s / verbalized_2s_stage{1,2}) on
one item.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from argus_uq.models.base import GreedyOutput, Model, SamplesOutput


CALL_TYPES = (
    "greedy",
    "sample_0",
    "sample_1",
    "sample_2",
    "sample_3",
    "sample_4",
    "hedge_text_0",
    "hedge_text_1",
    "hedge_text_2",
    "hedge_image_bright",
    "hedge_image_contrast",
    "hedge_image_autocontrast",
    "p_true",
    "verbalized_1s",
    "verbalized_2s_stage1",
    "verbalized_2s_stage2",
)


@dataclass
class CallRecord:
    """One API call's request + response + parsed result.

    API keys are not part of the released records.
    """

    item_id: str
    call_type: str  # one of CALL_TYPES
    vendor: str  # "openai" | "anthropic" | "google"
    model_id: str
    instruction: str
    image_sha256: str  # opaque image hash for joining; raw image not in record
    image_w: int
    image_h: int
    request_params: dict[str, Any] = field(default_factory=dict)
    raw_response_text: str = ""
    parsed_click: tuple[int, int] | None = None
    parsed_confidence: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    error: str | None = None


# ----------------------------------------------------------------------
# Shared parsing helpers
# ----------------------------------------------------------------------

# Regex for [[x0, y0, x1, y1]] normalised to [0, 1] (Anthropic format).
NORMALIZED_BBOX_RE = re.compile(
    r"\[\[\s*([01]?\.?\d*)\s*,\s*([01]?\.?\d*)\s*,\s*([01]?\.?\d*)\s*,\s*([01]?\.?\d*)\s*\]\]"
)

# Regex for verbalised confidence numbers (with optional %).
CONFIDENCE_RE = re.compile(r"\b([0-9]{1,3}(?:\.[0-9]+)?)\s*%?\b")


def parse_normalized_bbox_to_pixel_click(
    text: str, image_w: int, image_h: int
) -> tuple[int, int] | None:
    """Parse ``[[x0, y0, x1, y1]]`` (normalised [0, 1]) to pixel center."""

    m = NORMALIZED_BBOX_RE.search(text)
    if m is None:
        return None
    try:
        x0, y0, x1, y1 = (float(g) for g in m.groups())
    except ValueError:
        return None
    cx = int(round(((x0 + x1) / 2.0) * image_w))
    cy = int(round(((y0 + y1) / 2.0) * image_h))
    return cx, cy


def parse_verbalized_confidence(text: str) -> float | None:
    """Parse a 0-100 (or 0-1) confidence number from text; returns [0, 1]."""

    m = CONFIDENCE_RE.search(text)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


class BlackboxModel(Model):
    """Closed-source API agent base.

    Subclasses (one per vendor) instantiate the SDK on :meth:`load`,
    issue a single API call per ``score_item`` invocation, and parse
    the response into the standard ``GreedyOutput`` / ``SamplesOutput``
    contract.

    Closed-source agents do NOT expose token logprobs / hidden states /
    attention. The 8-method closed-source panel uses only the methods
    where this is OK.
    """

    family = "closed_source"
    vendor: str = ""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.api_key = api_key
        self.timeout = timeout
        self._client: Any = None

    @abstractmethod
    def _connect(self) -> Any:
        """Instantiate the vendor SDK client. Returns the client."""

    def load(self) -> None:
        """Connect to the vendor API."""

        if self._loaded:
            return
        self._client = self._connect()
        self._loaded = True

    def unload(self) -> None:
        super().unload()
        self._client = None
