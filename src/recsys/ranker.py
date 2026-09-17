"""Second stage: merge candidates from every generator and rank them with LambdaMART.

The stages are trained on different periods, and that is the whole leakage
story. Candidate generators are fitted on period A. The ranker is trained on
labels from period B, which the generators never saw, so it learns how far to
trust each generator on data that generator did not memorize. At test time the
generators are refitted on A and B together and the ranker is applied to period
C.

Also here: a pointwise model of "will this user interact with this item in the
next period", used for calibration. A ranking only needs the order right. A
decision layer that thresholds, mixes objectives or routes users needs the
number itself to mean what it says.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .candidates import CandidateModel, top_k

#: Feature names of the pooled layout, kept for callers that predate layouts.
FEATURES: tuple[str, ...] = (
    "gen_score",
    "gen_rank",
    "gen_hits",
    "pop_all",
    "pop_recent",
    "genre_affinity",
    "history_len",
    "user_age",
    "user_gender",
)


@dataclass
class CandidateSet:
    """Candidates for one user with a feature row per candidate."""

    user: int
    items: np.ndarray
    features: np.ndarray


@dataclass
class CandidateMerger:
    """Union of top ``depth`` candidates from several generators, with features.

    Two feature layouts:

    ``pooled``         best standardized score, best rank and proposer count
                       across generators. A fixed schema, but it discards which
                       generator proposed an item.
    ``per_generator``  a standardized score and a rank for every generator,
                       plus the shared item and user features. The ranker can
                       learn how far to trust each generator.

    The first version shipped pooled only. A LambdaMART ranker over pooled
    features lost to EASE alone by 6.8 nDCG points, because it could not tell an
    EASE proposal from a popularity proposal.
    """

    generators: Sequence[CandidateModel]
    depth: int = 100
    pop_all: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pop_recent: np.ndarray = field(default_factory=lambda: np.zeros(0))
    item_genres: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    user_meta: Mapping[int, tuple[float, float]] = field(default_factory=dict)
    layout: str = "pooled"

    def __post_init__(self) -> None:
        """Validate the layout."""
        if self.layout not in {"pooled", "per_generator"}:
            raise ValueError(f"unknown feature layout {self.layout!r}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Column names of ``CandidateSet.features``, in order."""
        shared = ("pop_all", "pop_recent", "genre_affinity", "history_len", "user_age", "user_gender")
        if self.layout == "pooled":
            return ("gen_score", "gen_rank", "gen_hits", *shared)
        per = tuple(f"{g.name}_{kind}" for g in self.generators for kind in ("z", "rank"))
        return (*per, "gen_hits", *shared)

    def build(self, user: int, history: np.ndarray) -> CandidateSet:
        """Candidates and features for ``user``."""
        z_by_gen: list[dict[int, float]] = []
        rank_by_gen: list[dict[int, float]] = []
        hits: dict[int, int] = {}
        for gen in self.generators:
            scores = np.asarray(gen.scores(history), dtype=float)
            finite = scores[np.isfinite(scores)]
            mu, sd = (float(finite.mean()), float(finite.std())) if finite.size else (0.0, 1.0)
            ranked = top_k(scores, self.depth, exclude=history)
            z_by_gen.append({item: (scores[item] - mu) / (sd if sd > 0 else 1.0) for item in ranked})
            rank_by_gen.append({item: float(r) for r, item in enumerate(ranked, start=1)})
            for item in ranked:
                hits[item] = hits.get(item, 0) + 1

        items = np.array(sorted(hits), dtype=int)
        n_shared = 6
        if items.size == 0:
            return CandidateSet(user, items, np.zeros((0, len(self.feature_names))))

        profile = (
            self.item_genres[np.asarray(history, dtype=int)].mean(axis=0)
            if len(history)
            else np.zeros(self.item_genres.shape[1])
        )
        age, gender = self.user_meta.get(user, (0.0, 0.0))
        shared = np.column_stack(
            [
                self.pop_all[items],
                self.pop_recent[items],
                self.item_genres[items] @ profile,
                np.full(items.size, float(len(history))),
                np.full(items.size, age),
                np.full(items.size, gender),
            ]
        )
        assert shared.shape[1] == n_shared
        # A generator that did not propose an item gets a rank past its depth and
        # the lowest standardized score it produced, so absence is ordered below
        # every presence rather than treated as missing.
        missing_rank = float(self.depth + 1)
        hit_col = np.array([hits[i] for i in items], dtype=float)
        if self.layout == "pooled":
            best_z = np.array([max(z.get(i, -np.inf) for z in z_by_gen) for i in items])
            best_rank = np.array([min(r.get(i, missing_rank) for r in rank_by_gen) for i in items])
            feats = np.column_stack([best_z, best_rank, hit_col, shared])
        else:
            cols = []
            for z, r in zip(z_by_gen, rank_by_gen, strict=True):
                floor = min(z.values()) if z else 0.0
                cols.append(np.array([z.get(i, floor) for i in items]))
                cols.append(np.array([r.get(i, missing_rank) for i in items]))
            feats = np.column_stack([*cols, hit_col, shared])
        return CandidateSet(user, items, feats)


@dataclass
class LambdaMARTRanker:
    """LightGBM lambdarank over merged candidates."""

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    seed: int = 0
    columns: tuple[int, ...] | None = None
    #: Per column: +1 score may only rise with the feature, -1 only fall, 0 free.
    #: A better generator rank must never lower an item's score; without this a
    #: ranker given only EASE's rank scored 1.8 nDCG points below EASE itself.
    monotone: tuple[int, ...] | None = None
    _model: object | None = field(default=None, repr=False)

    def _x(self, features: np.ndarray) -> np.ndarray:
        return features if self.columns is None else features[:, list(self.columns)]

    def fit(self, sets: Sequence[CandidateSet], relevant: Mapping[int, set[int]]) -> "LambdaMARTRanker":
        """Train on users whose candidates contain at least one relevant item."""
        import lightgbm

        xs, ys, groups = [], [], []
        for cs in sets:
            labels = np.array([int(i in relevant.get(cs.user, set())) for i in cs.items])
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            xs.append(self._x(cs.features))
            ys.append(labels)
            groups.append(len(labels))
        if not groups:
            raise ValueError("no candidate set contains a relevant item")
        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
        )
        if self.monotone is not None:
            model.set_params(monotone_constraints=list(self.monotone), monotone_constraints_method="advanced")
        model.fit(np.vstack(xs), np.concatenate(ys), group=groups)
        self._model = model
        return self

    def rank(self, cs: CandidateSet) -> tuple[np.ndarray, np.ndarray]:
        """Candidates ordered best first, with their scores."""
        if self._model is None:
            raise RuntimeError("fit the ranker first")
        if cs.items.size == 0:
            return cs.items, np.zeros(0)
        scores = np.asarray(self._model.predict(self._x(cs.features)), dtype=float)  # type: ignore[attr-defined]
        order = np.lexsort((cs.items, -scores))
        return cs.items[order], scores[order]


@dataclass
class InteractionModel:
    """Pointwise probability that a user interacts with a candidate next period.

    Two outputs: the raw classifier probability, and the same probability after
    isotonic regression fitted on a held-out calibration set. Reporting both is
    the point, because gradient boosted classifiers trained on heavily
    imbalanced candidate sets are routinely confident and wrong about their own
    probabilities.
    """

    seed: int = 0
    _clf: object | None = field(default=None, repr=False)
    _iso: object | None = field(default=None, repr=False)

    def fit(
        self,
        train_sets: Sequence[CandidateSet],
        calib_sets: Sequence[CandidateSet],
        relevant: Mapping[int, set[int]],
    ) -> "InteractionModel":
        """Fit the classifier on ``train_sets`` and the calibrator on ``calib_sets``."""
        import lightgbm
        from sklearn.isotonic import IsotonicRegression

        x, y = _stack(train_sets, relevant)
        clf = lightgbm.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=self.seed, verbose=-1,
        )
        clf.fit(x, y)
        xc, yc = _stack(calib_sets, relevant)
        raw = clf.predict_proba(xc)[:, 1]
        self._iso = IsotonicRegression(out_of_bounds="clip").fit(raw, yc)
        self._clf = clf
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw and calibrated probabilities."""
        if self._clf is None or self._iso is None:
            raise RuntimeError("fit the model first")
        raw = self._clf.predict_proba(x)[:, 1]  # type: ignore[attr-defined]
        return raw, np.asarray(self._iso.predict(raw), dtype=float)  # type: ignore[attr-defined]


def _stack(sets: Sequence[CandidateSet], relevant: Mapping[int, set[int]]) -> tuple[np.ndarray, np.ndarray]:
    xs = [cs.features for cs in sets if cs.items.size]
    ys = [np.array([int(i in relevant.get(cs.user, set())) for i in cs.items]) for cs in sets if cs.items.size]
    if not xs:
        raise ValueError("no candidates")
    return np.vstack(xs), np.concatenate(ys)
