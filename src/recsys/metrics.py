"""Ranking quality and the properties of a list that accuracy ignores.

Accuracy metrics say whether the right items were found. Coverage, novelty and
diversity say what kind of list it was. A recommender can raise recall by
showing everyone the same popular items, and these are the numbers that expose
it, which is why they are always reported together.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of relevant items found in the top ``k``."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary relevance nDCG at ``k``."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def hit_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 when any relevant item is in the top ``k``."""
    return float(bool(set(ranked[:k]) & relevant))


def catalog_coverage(lists: Sequence[Sequence[int]], n_items: int, k: int) -> float:
    """Share of the catalog that appears in at least one top ``k`` list."""
    shown = {item for ranked in lists for item in ranked[:k]}
    return len(shown) / n_items if n_items else 0.0


def novelty(lists: Sequence[Sequence[int]], popularity: np.ndarray, k: int) -> float:
    """Mean self information, ``-log2 p(item)``, of recommended items.

    Higher means less popular items. Popularity is the share of training users
    who interacted with the item.
    """
    values = [
        -math.log2(max(float(popularity[item]), 1e-12))
        for ranked in lists
        for item in ranked[:k]
    ]
    return float(np.mean(values)) if values else 0.0


def intra_list_diversity(
    ranked: Sequence[int], genres: Sequence[Sequence[str]], k: int
) -> float:
    """Mean pairwise Jaccard distance between item genre sets in the top ``k``."""
    head = list(ranked[:k])
    if len(head) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(head)):
        a = set(genres[head[i]])
        for j in range(i + 1, len(head)):
            b = set(genres[head[j]])
            union = a | b
            total += 1.0 - (len(a & b) / len(union) if union else 1.0)
            pairs += 1
    return total / pairs


def evaluate(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, set[int]],
    *,
    k: int,
    n_items: int,
    popularity: np.ndarray,
    genres: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Every metric, averaged over users that have relevant items."""
    users = [u for u in recommendations if relevant.get(u)]
    lists = [recommendations[u] for u in users]
    return {
        "users": float(len(users)),
        f"recall@{k}": float(np.mean([recall_at_k(recommendations[u], relevant[u], k) for u in users])) if users else 0.0,
        f"ndcg@{k}": float(np.mean([ndcg_at_k(recommendations[u], relevant[u], k) for u in users])) if users else 0.0,
        f"hit@{k}": float(np.mean([hit_at_k(recommendations[u], relevant[u], k) for u in users])) if users else 0.0,
        "coverage": catalog_coverage(lists, n_items, k),
        "novelty": novelty(lists, popularity, k),
        "diversity": float(np.mean([intra_list_diversity(r, genres, k) for r in lists])) if lists else 0.0,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates, per probability bin."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)
