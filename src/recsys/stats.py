"""Paired intervals over users for comparing two recommenders."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` with a percentile interval, resampling users.

    Both models are scored on the same users, so pairing removes the large
    between-user variance in how predictable someone is.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("per user scores must line up")
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)
