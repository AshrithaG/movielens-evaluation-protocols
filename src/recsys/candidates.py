"""Candidate generation: three models behind one interface.

Each model scores the whole catalog for a user given their history. The
catalog here is a few thousand items, so full scoring is cheap and exact, which
keeps retrieval recall a property of the model rather than of an approximate
index. At production scale the same scores would be served from an ANN index;
the interface would not change.

``Popularity``   what everyone watches. The baseline personalization must beat.
``ItemKNN``      items co-watched with the user's history. Strong, explainable.
``TwoTower``     a learned user tower over history and an item tower over id and
                 genres, trained with in-batch softmax. The model family that
                 production retrieval systems use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import sparse


class CandidateModel(Protocol):
    """Scores every item for a user history."""

    name: str

    def scores(self, history: np.ndarray) -> np.ndarray:
        """Score per item id, higher is better."""
        ...


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> list[int]:
    """Best ``k`` item ids, never returning an excluded item.

    Ties break by item id so results do not depend on k or on argpartition's
    internal order.
    """
    s = np.asarray(scores, dtype=float).copy()
    if exclude is not None and len(exclude):
        s[np.asarray(exclude, dtype=int)] = -np.inf
    valid = np.flatnonzero(np.isfinite(s))
    order = valid[np.lexsort((valid, -s[valid]))]
    return [int(i) for i in order[:k]]


@dataclass
class Popularity:
    """Items ranked by how many training users interacted with them."""

    name: str = "popularity"
    _scores: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def fit(self, users: np.ndarray, items: np.ndarray, n_items: int) -> "Popularity":
        """Count distinct users per item."""
        pairs = np.unique(np.stack([users, items], axis=1), axis=0)
        self._scores = np.bincount(pairs[:, 1], minlength=n_items).astype(float)
        return self

    def scores(self, history: np.ndarray) -> np.ndarray:
        """The same scores for every user, by design."""
        return self._scores

    @property
    def share(self) -> np.ndarray:
        """Popularity as a share of users, used by the novelty metric."""
        total = self._scores.max() if self._scores.size else 1.0
        return self._scores / max(total, 1.0)


@dataclass
class ItemKNN:
    """Cosine item-item similarity over the user-item matrix.

    Only the ``neighbours`` most similar items are kept per item, which removes
    the long tail of near zero similarities that otherwise add noise to every
    user's scores. Recent history counts more, with weight halving every
    ``half_life`` items back.
    """

    neighbours: int = 100
    half_life: float = 20.0
    recent: int = 50
    name: str = "item_knn"
    _sim: sparse.csr_matrix | None = field(default=None, repr=False)

    def fit(self, users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> "ItemKNN":
        """Build the truncated similarity matrix."""
        matrix = sparse.csr_matrix(
            (np.ones(len(users)), (users, items)), shape=(n_users, n_items)
        )
        matrix.data[:] = 1.0
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        normalized = matrix.multiply(1.0 / norms).tocsc()
        sim = (normalized.T @ normalized).tocsr()
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        rows, cols, vals = [], [], []
        for i in range(n_items):
            start, end = sim.indptr[i], sim.indptr[i + 1]
            if end == start:
                continue
            idx, data = sim.indices[start:end], sim.data[start:end]
            if len(data) > self.neighbours:
                keep = np.argpartition(-data, self.neighbours)[: self.neighbours]
                idx, data = idx[keep], data[keep]
            rows.extend([i] * len(idx))
            cols.extend(idx.tolist())
            vals.extend(data.tolist())
        self._sim = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))
        return self

    def scores(self, history: np.ndarray) -> np.ndarray:
        """Recency weighted sum of neighbour similarities."""
        if self._sim is None:
            raise RuntimeError("fit the model first")
        recent = np.asarray(history[-self.recent :], dtype=int)
        if recent.size == 0:
            return np.zeros(self._sim.shape[0])
        ages = np.arange(len(recent))[::-1]
        weights = 0.5 ** (ages / self.half_life)
        return np.asarray(self._sim[recent].T @ weights).ravel()


@dataclass
class TwoTower:
    """A user tower over history and an item tower over id and genres.

    User embedding: recency weighted mean of history item embeddings, passed
    through a small MLP. Item embedding: id embedding plus mean genre
    embedding, so an item with few interactions still borrows from its genres.
    Trained to predict the next interaction from the history before it, with
    every other positive in the batch as a negative (in-batch softmax) and a
    log popularity correction so frequent items are not over penalized as
    negatives.

    Args:
        dim: Embedding size.
        max_history: History items the user tower sees.
        epochs: Passes over the training sequences.
        batch_size: Also the number of in-batch negatives.
        seed: Controls initialization and sampling.
    """

    dim: int = 64
    max_history: int = 50
    epochs: int = 6
    batch_size: int = 1024
    lr: float = 2e-3
    seed: int = 0
    pop_correction: bool = True
    name: str = "two_tower"
    _item_matrix: np.ndarray | None = field(default=None, repr=False)
    _user_mlp: object | None = field(default=None, repr=False)
    _item_table: np.ndarray | None = field(default=None, repr=False)

    def fit(
        self,
        sequences: dict[int, np.ndarray],
        genres: Sequence[Sequence[str]],
        n_items: int,
        popularity: np.ndarray,
    ) -> "TwoTower":
        """Train on (history prefix, next item) examples from ``sequences``."""
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        genre_names = sorted({g for gs in genres for g in gs})
        genre_index = {g: i for i, g in enumerate(genre_names)}
        item_genres = torch.zeros(n_items, len(genre_names))
        for item, gs in enumerate(genres):
            for g in gs:
                item_genres[item, genre_index[g]] = 1.0
        item_genres = item_genres / item_genres.sum(1, keepdim=True).clamp(min=1.0)

        pad = n_items
        item_emb = nn.Embedding(n_items + 1, self.dim, padding_idx=pad)
        genre_proj = nn.Linear(len(genre_names), self.dim, bias=False)
        user_mlp = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim))
        params = [*item_emb.parameters(), *genre_proj.parameters(), *user_mlp.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr)
        log_pop = torch.log(torch.as_tensor(popularity, dtype=torch.float32).clamp(min=1.0))

        examples: list[tuple[np.ndarray, int]] = []
        for seq in sequences.values():
            for t in range(1, len(seq)):
                examples.append((seq[max(0, t - self.max_history) : t], int(seq[t])))

        def item_vectors(ids: torch.Tensor) -> torch.Tensor:
            return item_emb(ids) + genre_proj(item_genres[ids])

        for _ in range(self.epochs):
            order = rng.permutation(len(examples))
            for start in range(0, len(order), self.batch_size):
                batch = [examples[i] for i in order[start : start + self.batch_size]]
                hist = torch.full((len(batch), self.max_history), pad, dtype=torch.long)
                for row, (h, _) in enumerate(batch):
                    hist[row, -len(h) :] = torch.tensor(h, dtype=torch.long)
                targets = torch.as_tensor([t for _, t in batch])

                user = self._pool(item_emb(hist), hist != pad)
                user = torch.nn.functional.normalize(user_mlp(user), dim=-1)
                items = torch.nn.functional.normalize(item_vectors(targets), dim=-1)
                logits = user @ items.T / 0.05
                if self.pop_correction:
                    # logQ correction: in-batch negatives are sampled in
                    # proportion to popularity, so without this popular items
                    # are over penalized as negatives.
                    logits = logits - log_pop[targets][None, :]
                # Repeated targets in one batch are not negatives of each other.
                same = targets[:, None] == targets[None, :]
                logits = logits.masked_fill(same & ~torch.eye(len(batch), dtype=torch.bool), -1e9)
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch)))
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            all_items = torch.arange(n_items)
            self._item_matrix = torch.nn.functional.normalize(item_vectors(all_items), dim=-1).numpy()
            self._item_table = item_emb.weight.detach().numpy()
        self._user_mlp = user_mlp.eval()
        return self

    def _pool(self, embedded, mask):
        import torch

        weights = mask.float() * torch.linspace(0.5, 1.0, mask.shape[1])[None, :]
        return (embedded * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp(min=1e-6)

    def scores(self, history: np.ndarray) -> np.ndarray:
        """Cosine similarity between the user vector and every item."""
        import torch

        if self._item_matrix is None or self._item_table is None or self._user_mlp is None:
            raise RuntimeError("fit the model first")
        recent = np.asarray(history[-self.max_history :], dtype=int)
        if recent.size == 0:
            return np.zeros(self._item_matrix.shape[0])
        with torch.no_grad():
            emb = torch.as_tensor(self._item_table[recent])[None]
            mask = torch.ones(1, len(recent), dtype=torch.bool)
            padded_mask = torch.zeros(1, self.max_history, dtype=torch.bool)
            padded_emb = torch.zeros(1, self.max_history, emb.shape[-1])
            padded_mask[0, -len(recent) :] = mask[0]
            padded_emb[0, -len(recent) :] = emb[0]
            user = self._pool(padded_emb, padded_mask)
            user = torch.nn.functional.normalize(self._user_mlp(user), dim=-1).numpy()[0]
        return self._item_matrix @ user


@dataclass
class RecentPopularity:
    """Popularity within the last ``window_days`` before the cutoff.

    All time popularity is a weak baseline under a time split, because what was
    popular two years before the cutoff is not what people watch after it. This
    is the baseline a personalized model has to beat to justify itself.
    """

    window_days: float = 30.0
    name: str = ""
    _scores: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def __post_init__(self) -> None:
        """Name after the window."""
        self.name = self.name or f"recent_pop_{self.window_days:g}d"

    def fit(self, items: np.ndarray, timestamps: np.ndarray, n_items: int) -> "RecentPopularity":
        """Count interactions in the final window of the fitting period."""
        cutoff = float(np.max(timestamps)) - self.window_days * 86_400
        recent = items[timestamps >= cutoff]
        self._scores = np.bincount(recent, minlength=n_items).astype(float)
        return self

    def scores(self, history: np.ndarray) -> np.ndarray:
        """The same scores for every user."""
        return self._scores


@dataclass
class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019).

    A closed form item-item model: ridge regression predicting each item from
    all others, with the diagonal forced to zero so an item cannot predict
    itself. One matrix inverse over the catalog, no training loop, and a strong
    result on MovieLens, which makes it the right bar for any learned model.

    Args:
        l2: Ridge penalty. Tuned on validation.
    """

    l2: float = 500.0
    name: str = ""
    _weights: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Name after the penalty."""
        self.name = self.name or f"ease_{self.l2:g}"

    def fit(self, users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> "EASE":
        """Solve for the item-item weight matrix."""
        x = sparse.csr_matrix((np.ones(len(users)), (users, items)), shape=(n_users, n_items))
        x.data[:] = 1.0
        gram = np.asarray((x.T @ x).todense(), dtype=np.float64)
        gram[np.diag_indices(n_items)] += self.l2
        p = np.linalg.inv(gram)
        weights = -p / np.diag(p)[None, :]
        weights[np.diag_indices(n_items)] = 0.0
        self._weights = weights
        return self

    def scores(self, history: np.ndarray) -> np.ndarray:
        """Sum of learned weights from the items in the history."""
        if self._weights is None:
            raise RuntimeError("fit the model first")
        idx = np.unique(np.asarray(history, dtype=int))
        if idx.size == 0:
            return np.zeros(self._weights.shape[0])
        return self._weights[idx].sum(axis=0)
