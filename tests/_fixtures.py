"""Synthetic fixtures for unit tests.

Reusable per-item records and small calibration corpora that exercise
every code path in the estimator + metric layers without loading any
real model.
"""

from __future__ import annotations

import numpy as np


def make_record(
    *,
    n_tokens: int = 8,
    correct: int = 1,
    greedy_click: tuple[int, int] | None = (100, 200),
    n_samples: int = 5,
    samples_clicks: list[tuple[int, int] | None] | None = None,
    samples_texts: list[str] | None = None,
    samples_logprobs: list[float] | None = None,
    p_true: float | None = 0.7,
    verbalized_1s: float | None = 0.8,
    verbalized_2s: float | None = 0.75,
    hidden_dim: int = 32,
    with_attention: bool = True,
    n_paraphrases: int = 3,
    n_perturbations: int = 3,
    rng_seed: int = 0,
) -> dict:
    """Build a complete synthetic per-item record.

    Defaults give a "well-behaved" record where every estimator can
    score without hitting an edge-case branch.
    """

    rng = np.random.default_rng(rng_seed)

    # Greedy logprobs and entropies.
    greedy_logprobs = list(rng.uniform(-2.0, -0.05, size=n_tokens))
    greedy_entropies = list(rng.uniform(0.1, 1.5, size=n_tokens))

    # Per-token last-layer hidden states.
    greedy_last_hidden = [
        rng.normal(0.0, 1.0, size=hidden_dim).astype(np.float32) for _ in range(n_tokens)
    ]

    # Default sample clicks form 2 clusters (3 within 50 px of greedy, 2 elsewhere).
    # When greedy_click is None we anchor at (100, 200) so the fixture still has
    # a coherent set of sample clicks that exercises the cluster code path.
    anchor = greedy_click if greedy_click is not None else (100, 200)
    if samples_clicks is None:
        samples_clicks = [
            anchor,
            (anchor[0] + 5, anchor[1] - 3),
            (anchor[0] - 10, anchor[1] + 4),
            (anchor[0] + 200, anchor[1] + 100),
            (anchor[0] - 250, anchor[1] - 150),
        ][:n_samples]
    if samples_texts is None:
        samples_texts = [f"click ({c[0]}, {c[1]})" if c else "no parse" for c in samples_clicks]
    if samples_logprobs is None:
        samples_logprobs = list(rng.uniform(-3.0, -0.1, size=len(samples_clicks)))

    # HEDGE / IMG-HEDGE samples (anchored to greedy_click when present).
    hedge_samples = [
        (f"para_{i}", (anchor[0] + i * 5, anchor[1] + i * 3))
        for i in range(n_paraphrases)
    ]
    imghedge_samples = [
        (f"perturb_{i}", (anchor[0] - i * 4, anchor[1] + i * 7))
        for i in range(n_perturbations)
    ]

    # Attention payload (v6_attention layout).
    attention = None
    if with_attention:
        T, L, H, prompt_len = n_tokens, 4, 4, 16
        K = prompt_len + n_tokens
        attn = rng.uniform(0.0, 1.0, size=(T, L, H, K)).astype(np.float32)
        # Normalize across last axis so it looks like attention.
        attn = attn / np.maximum(attn.sum(axis=-1, keepdims=True), 1e-9)
        attention = {
            "attn_step_layer_head_key": attn,
            "T_gen": T,
            "L": L,
            "H": H,
            "prompt_len": prompt_len,
            "image_mask": np.zeros(K, dtype=bool),
            "logprobs": np.asarray(greedy_logprobs, dtype=np.float32),
        }

    return {
        "item_id": f"fixture_{rng_seed}",
        "correct": int(correct),
        "dist": float(0.0 if correct else 100.0),
        "dist_norm": float(0.0 if correct else 0.1),
        "image_w": 1920,
        "image_h": 1080,
        "gold_x1": 50,
        "gold_y1": 150,
        "gold_x2": 150,
        "gold_y2": 250,
        "greedy_text": "click(100, 200)",
        "greedy_click": greedy_click,
        "greedy_logprobs": greedy_logprobs,
        "greedy_entropies": greedy_entropies,
        "greedy_last_hidden": greedy_last_hidden,
        "attention": attention,
        "samples_texts": samples_texts,
        "samples_clicks": samples_clicks,
        "samples_mean_seq_logprobs": samples_logprobs,
        "p_true": p_true,
        "verbalized_1s": verbalized_1s,
        "verbalized_2s": verbalized_2s,
        "hedge_samples": hedge_samples,
        "imghedge_samples": imghedge_samples,
    }


def make_corpus(n_items: int = 50, p_correct: float = 0.6, rng_seed: int = 0) -> list[dict]:
    """Build a synthetic corpus of records with varied correctness."""

    rng = np.random.default_rng(rng_seed)
    records: list[dict] = []
    for i in range(n_items):
        correct = int(rng.random() < p_correct)
        # If correct, samples cluster tightly around greedy. If incorrect,
        # samples are scattered.
        scatter = 5.0 if correct else 200.0
        clicks = [
            (
                int(rng.normal(100, scatter)),
                int(rng.normal(200, scatter)),
            )
            for _ in range(5)
        ]
        records.append(
            make_record(
                correct=correct,
                samples_clicks=clicks,
                rng_seed=rng_seed + i,
            )
        )
    return records


def labels_from_records(records: list[dict]) -> np.ndarray:
    """Extract correctness labels."""

    return np.asarray([int(r["correct"]) for r in records], dtype=np.int64)
