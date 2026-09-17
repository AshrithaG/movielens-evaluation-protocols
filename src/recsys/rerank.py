"""The decision layer: relevance against diversity, as an explicit trade-off.

Maximal marginal relevance. Items are chosen one at a time; each pick maximizes

    lambda * relevance(item) - (1 - lambda) * max similarity to items already picked

so lambda = 1 is the ranker's order and lower values buy diversity with
relevance. Sweeping lambda gives the trade-off curve, which is the artifact
that belongs in a product discussion: not "diversity is good" but "this much
diversity costs this much accuracy".
"""

from __future__ import annotations

import numpy as np


def mmr(
    items: np.ndarray,
    relevance: np.ndarray,
    item_genres: np.ndarray,
    k: int,
    lam: float,
) -> list[int]:
    """Pick ``k`` items from ``items`` by maximal marginal relevance.

    Args:
        items: Candidate ids, any order.
        relevance: Ranker scores for ``items``.
        item_genres: Row normalized genre vectors for every catalog item.
        k: List length.
        lam: Weight on relevance, in [0, 1].
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    items = np.asarray(items, dtype=int)
    rel = np.asarray(relevance, dtype=float)
    if items.size == 0:
        return []
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.zeros_like(rel)

    vectors = item_genres[items]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 0, norms, 1.0)
    sim = vectors @ vectors.T

    chosen: list[int] = []
    remaining = list(range(items.size))
    max_sim = np.zeros(items.size)
    while remaining and len(chosen) < k:
        values = lam * rel[remaining] - (1.0 - lam) * max_sim[remaining]
        pick = remaining[int(np.argmax(values))]
        chosen.append(pick)
        remaining.remove(pick)
        max_sim = np.maximum(max_sim, sim[pick])
    return [int(items[i]) for i in chosen]
