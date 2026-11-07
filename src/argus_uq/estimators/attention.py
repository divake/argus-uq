"""Attention-family UQ methods (3 methods; open-source only).

The canonical attention panel is **three** methods. The other four
attention-trace methods that historically lived in this module
(``attention_rollout``, ``llm_check``, ``attention_entropy``,
``lookback_lens_unsupervised``) and the v5 short-form ``rauq_proxy``
were broken signals (near-chance or anti-AUROC on every cell scored)
and are removed. **Do not re-introduce.**

References:

- ``focus``: Zhang et al. 2023, *FOCUS: Hallucination Detection via
  Attention Pattern Analysis*. NLL with attention-propagated penalty
  from prior tokens.

- ``rauq_proxy_full``: Vazhentsev et al. 2023, *RAUQ: Recurrent
  Attention-Based UQ for LLMs*. Online unsupervised head pick (max
  mean subdiag on middle-third layers) + recurrent confidence over
  layers.

- ``uqac``: 2025 attention-aggregation method. Top-K low-entropy
  heads per layer + max-pool aggregate + chain-marginalised NLL
  penalty.

All three consume an attention payload with the v6_attention layout:

    record['attention'] = {
        'attn_step_layer_head_key': np.ndarray (T, L, H, K),
        'T_gen': int, 'L': int, 'H': int, 'prompt_len': int,
        'image_mask': np.ndarray bool (K,),
        'logprobs': np.ndarray (T,),
    }

Closed-source availability: NONE. Closed APIs do not expose attention
tensors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus_uq.estimators.base import Estimator, FamilyName


def _safe_attn(rec: dict[str, Any]) -> np.ndarray:
    """Return attention as fp32 with NaN -> 0."""

    a = rec["attn_step_layer_head_key"]
    a = np.where(np.isnan(a), 0.0, a).astype(np.float32, copy=False)
    return a


def _unpack(rec: dict[str, Any]) -> dict[str, Any]:
    a = rec["attn_step_layer_head_key"]
    return dict(
        attn=a,
        T_gen=int(rec["T_gen"]),
        L=int(rec["L"]),
        H=int(rec["H"]),
        prompt_len=int(rec["prompt_len"]),
        image_mask=np.asarray(rec["image_mask"], dtype=bool),
        logprobs=np.asarray(rec["logprobs"], dtype=np.float32),
    )


def _focus(rec: dict[str, Any], alpha: float = 0.9, *, attn_fp32: np.ndarray | None = None) -> float:
    """FOCUS: NLL with attention-propagated penalty from prior tokens."""

    d = _unpack(rec)
    T, P = d["T_gen"], d["prompt_len"]
    if T == 0:
        return 0.0
    if attn_fp32 is None:
        attn_fp32 = _safe_attn(rec)
    att_pool = attn_fp32.max(axis=(1, 2))
    base_u = -d["logprobs"]
    hat_u = base_u.copy()
    for i in range(1, T):
        cols = att_pool[i, P : P + i]
        s = cols.sum()
        if s <= 0:
            continue
        w = cols / s
        hat_u[i] = base_u[i] + alpha * float(np.dot(w, hat_u[:i]))
    return float(hat_u.mean())


def _rauq_proxy_full(
    rec: dict[str, Any],
    alpha: float = 0.5,
    *,
    attn_fp32: np.ndarray | None = None,
) -> float:
    """RAUQ adaptation: online unsup head pick + recurrent confidence."""

    d = _unpack(rec)
    T, L, _H, P = d["T_gen"], d["L"], d["H"], d["prompt_len"]
    if T == 0:
        return 0.5
    if attn_fp32 is None:
        attn_fp32 = _safe_attn(rec)
    p_y = np.exp(np.clip(d["logprobs"], -30.0, 0.0))
    K = attn_fp32.shape[-1]
    rows = np.arange(T)
    prev_cols = np.maximum(np.minimum(P + rows - 1, K - 1), 0)
    sub = attn_fp32[rows, :, :, prev_cols]
    sub[0] = 1.0
    L_lo, L_hi = L // 3, (2 * L) // 3 + 1
    sub_mid = sub[:, L_lo:L_hi, :]
    if T > 1:
        head_score = sub_mid[1:].mean(axis=0)
    else:
        head_score = sub_mid[0]
    h_pick = head_score.argmax(axis=-1)
    Lp = sub_mid.shape[1]
    sub_pick = sub_mid[:, np.arange(Lp), h_pick]
    c = np.zeros((T, Lp), dtype=np.float32)
    c[0] = p_y[0]
    for t in range(1, T):
        c[t] = alpha * p_y[t] + (1 - alpha) * sub_pick[t] * c[t - 1]
    layer_means = c.mean(axis=0)
    return float(1.0 - layer_means.max()) if Lp > 0 else 0.5


def _uqac(
    rec: dict[str, Any],
    k_heads: int = 5,
    theta: float = 0.7,
    *,
    attn_fp32: np.ndarray | None = None,
) -> float:
    """UQAC simplified: top-K low-entropy heads + max-pool + chain marginalisation."""

    d = _unpack(rec)
    T, L, H, P = d["T_gen"], d["L"], d["H"], d["prompt_len"]
    if T == 0:
        return 0.0
    if attn_fp32 is None:
        attn_fp32 = _safe_attn(rec)
    s = attn_fp32.sum(axis=-1, keepdims=True)
    p = attn_fp32 / np.maximum(s, 1e-12)
    safe = np.maximum(p, 1e-12)
    ent = -(p * np.log(safe)).sum(axis=-1)
    head_ent_mean = ent.mean(axis=0)
    K_eff = min(k_heads, H)
    top_heads = np.argsort(head_ent_mean, axis=1)[:, :K_eff]
    L_idx = np.arange(L).reshape(L, 1)
    sel = attn_fp32[:, L_idx, top_heads, :]
    agg = sel.max(axis=(1, 2))
    base_u = -d["logprobs"]
    chain_u = base_u.copy()
    for i in range(1, T):
        cols = agg[i, P : P + i]
        s_ = cols.sum()
        if s_ <= 0:
            continue
        w = cols / s_
        chain_u[i] = base_u[i] + theta * float(np.dot(w, chain_u[:i]))
    return float(chain_u.mean())


class Focus(Estimator):
    """FOCUS attention-propagated NLL penalty (Zhang et al. 2023)."""

    name = "focus"
    family: FamilyName = "attention"
    available_on_closed_source = False

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha

    def score_item(self, record: dict[str, Any]) -> float:
        attn = record.get("attention")
        if attn is None or "attn_step_layer_head_key" not in attn:
            return 0.0
        try:
            return _focus(attn, alpha=self.alpha)
        except (KeyError, ValueError, IndexError):
            return float("nan")


class RAUQProxyFull(Estimator):
    """RAUQ: online head pick + recurrent confidence (Vazhentsev et al. 2023)."""

    name = "rauq_proxy_full"
    family: FamilyName = "attention"
    available_on_closed_source = False

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def score_item(self, record: dict[str, Any]) -> float:
        attn = record.get("attention")
        if attn is None or "attn_step_layer_head_key" not in attn:
            return 0.5
        try:
            return _rauq_proxy_full(attn, alpha=self.alpha)
        except (KeyError, ValueError, IndexError):
            return float("nan")


class UQAC(Estimator):
    """UQAC: top-K low-entropy heads + chain marginalisation (UQAC 2025)."""

    name = "uqac"
    family: FamilyName = "attention"
    available_on_closed_source = False

    def __init__(self, k_heads: int = 5, theta: float = 0.7) -> None:
        self.k_heads = k_heads
        self.theta = theta

    def score_item(self, record: dict[str, Any]) -> float:
        attn = record.get("attention")
        if attn is None or "attn_step_layer_head_key" not in attn:
            return 0.0
        try:
            return _uqac(attn, k_heads=self.k_heads, theta=self.theta)
        except (KeyError, ValueError, IndexError):
            return float("nan")


def attn_all_scores(record_attention: dict[str, Any]) -> dict[str, float]:
    """Compute all 3 attention-method scores for a single attention payload.

    Shares a single fp32-cast attention tensor across methods (saves
    3x bf16->fp32 work). Used by the multi-seed scoring pass.
    """

    attn_fp32 = _safe_attn(record_attention)
    out: dict[str, float] = {}
    for name, fn in {
        "focus": _focus,
        "rauq_proxy_full": _rauq_proxy_full,
        "uqac": _uqac,
    }.items():
        try:
            out[name] = float(fn(record_attention, attn_fp32=attn_fp32))
        except Exception:
            out[name] = float("nan")
    return out
