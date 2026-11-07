"""Per-item record schema.

Estimators consume per-item records produced by the capture phase. The
schema below is the contract every record must satisfy. Missing optional
fields are ``None``; estimators that need them must handle ``None``
gracefully.

Records are stored in two formats:

1. **Parquet rows** (``data_release/per_item_full/<CELL>.parquet``) have
   one column per per-method UQ score plus the metadata fields below.
   This is the lightweight format used by ``argus_uq.load(cell_id)``.

2. **Raw torch dicts** (``<CELL>/raw/*.pt``, not part of this release) have
   the full GPU-side capture payload: per-token logprobs, last-layer
   hidden states, attention tensors, sample texts, probe scalars. This
   is the heavy format used by the recapture pipeline.

The ``Record`` TypedDict below documents the dict shape that estimators
see at score time.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class Record(TypedDict, total=False):
    """Per-item record. All fields are optional; estimators document
    which ones they require."""

    # ---- identity ----
    item_id: str
    correct: int  # 0/1 ground truth
    dist: float  # pixels to gold bbox (0 if click inside bbox)
    dist_norm: float  # dist / image diagonal
    image_w: int
    image_h: int
    gold_x1: int
    gold_y1: int
    gold_x2: int
    gold_y2: int

    # ---- greedy generation (consumed by logit + density + attention) ----
    greedy_text: str
    greedy_click: tuple[int, int] | None
    greedy_logprobs: list[float]
    greedy_entropies: list[float]
    greedy_last_hidden: list[np.ndarray]  # per-token last-layer hidden states

    # ---- attention capture (consumed by attention family) ----
    # The attention dict has the v6_attention layout:
    # {"attn_step_layer_head_key": np.ndarray of shape (T, L, H, K),
    #  "T_gen": int, "L": int, "H": int, "prompt_len": int,
    #  "image_mask": np.ndarray bool of shape (K,),
    #  "logprobs": np.ndarray of shape (T,)}
    attention: dict[str, Any] | None

    # ---- N samples (consumed by sampling + consistency families) ----
    samples_texts: list[str]
    samples_clicks: list[tuple[int, int] | None]
    samples_mean_seq_logprobs: list[float]

    # ---- prompt probes (consumed by prompt family) ----
    p_true: float | None
    verbalized_1s: float | None
    verbalized_2s: float | None

    # ---- HEDGE / IMG-HEDGE (consumed by vlm_native family) ----
    # List of (paraphrase_id_or_perturbation_name, click_xy_or_None).
    hedge_samples: list[tuple[str, tuple[int, int] | None]]
    imghedge_samples: list[tuple[str, tuple[int, int] | None]]


def safe_stack_hidden(signals_list: list[np.ndarray]) -> np.ndarray:
    """Stack a list of same-dim 1-D arrays into a 2-D array ``(T, D)``.

    Returns an empty ``(0, 0)`` array if the input is empty.
    """

    if not signals_list:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(signals_list, axis=0).astype(np.float32)


def pool_hidden_states(hs: list[np.ndarray]) -> np.ndarray:
    """Mean-pool per-token last-layer hidden states to a single vector."""

    stacked = safe_stack_hidden(hs)
    if stacked.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return stacked.mean(axis=0)


def coordinate_clusters(
    clicks: list[tuple[int, int] | None],
    tolerance_px: float,
) -> list[int]:
    """Assign each click to a cluster id by Euclidean tolerance.

    Two clicks share a cluster if they are both parsable and within
    ``tolerance_px`` Euclidean pixels. Unparsable clicks each get their
    own singleton cluster.
    """

    labels: list[int] = [-1] * len(clicks)
    next_id = 0
    for i, ci in enumerate(clicks):
        if labels[i] != -1:
            continue
        if ci is None:
            labels[i] = next_id
            next_id += 1
            continue
        labels[i] = next_id
        for j in range(i + 1, len(clicks)):
            cj = clicks[j]
            if cj is None or labels[j] != -1:
                continue
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            if (dx * dx + dy * dy) ** 0.5 <= tolerance_px:
                labels[j] = next_id
        next_id += 1
    return labels
