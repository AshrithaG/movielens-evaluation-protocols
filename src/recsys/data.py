"""MovieLens-1M as implicit feedback, split by time.

Why a global time split. Random splits let a model train on a user's
interactions from after the moment it is asked to predict, which inflates every
metric and rewards memorizing the future. A global cutoff means training sees
only the past and evaluation only the future, which is the situation a deployed
recommender is actually in.

Why implicit feedback. A rating says the user chose to watch the movie. Ranking
what a user will watch next is the production task; predicting the star value
is not. Ratings of 4 and above are also kept as a stricter "liked" signal for
the relevance labels.

The data is not redistributable under its license, so it is downloaded locally
and never committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ml-1m"


@dataclass(frozen=True)
class Movies:
    """Titles and genres, indexed by internal item id."""

    titles: tuple[str, ...]
    genres: tuple[tuple[str, ...], ...]
    external_ids: tuple[int, ...]


@dataclass
class Split:
    """Interactions split at two time cutoffs.

    Attributes:
        train: Interactions before the validation cutoff.
        valid: Interactions between the two cutoffs.
        test: Interactions after the test cutoff.
        n_users: Number of internal user ids.
        n_items: Number of internal item ids.
        movies: Item metadata.
        cutoffs: The two timestamps.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    movies: Movies
    cutoffs: tuple[int, int]
    user_features: pd.DataFrame = field(default_factory=pd.DataFrame)

    def history(self, frame: pd.DataFrame) -> dict[int, np.ndarray]:
        """Each user's items in ``frame``, oldest first."""
        ordered = frame.sort_values(["user", "timestamp"], kind="stable")
        return {int(u): g["item"].to_numpy() for u, g in ordered.groupby("user", sort=False)}


def _read(name: str, columns: list[str], data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        data_dir / name, sep="::", engine="python", names=columns, encoding="latin-1"
    )


def load_split(
    data_dir: Path | None = None,
    valid_quantile: float = 0.8,
    test_quantile: float = 0.9,
    liked_threshold: int = 4,
) -> Split:
    """Load MovieLens-1M and split it at global time quantiles.

    Items and users are reindexed to dense ids using the training period only,
    so an item that first appears after the cutoff is unknown at training time,
    exactly as it would be in production. Such items are dropped from
    evaluation rather than scored as misses the model had no way to avoid.

    Args:
        data_dir: Folder holding the extracted ``ml-1m`` files.
        valid_quantile: Time quantile where validation starts.
        test_quantile: Time quantile where test starts.
        liked_threshold: Rating at or above which an interaction is a like.
    """
    root = data_dir or DATA_DIR
    ratings = _read("ratings.dat", ["user_ext", "item_ext", "rating", "timestamp"], root)
    movies = _read("movies.dat", ["item_ext", "title", "genres"], root)
    users = _read("users.dat", ["user_ext", "gender", "age", "occupation", "zip"], root)

    t_valid, t_test = (int(q) for q in ratings["timestamp"].quantile([valid_quantile, test_quantile]))
    train_raw = ratings[ratings["timestamp"] < t_valid]

    user_ids = {u: i for i, u in enumerate(sorted(train_raw["user_ext"].unique()))}
    item_ids = {m: i for i, m in enumerate(sorted(train_raw["item_ext"].unique()))}

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        known = frame[frame["user_ext"].isin(user_ids) & frame["item_ext"].isin(item_ids)]
        return pd.DataFrame(
            {
                "user": known["user_ext"].map(user_ids).astype(int),
                "item": known["item_ext"].map(item_ids).astype(int),
                "rating": known["rating"].astype(int),
                "liked": (known["rating"] >= liked_threshold).astype(int),
                "timestamp": known["timestamp"].astype(int),
            }
        ).reset_index(drop=True)

    by_item = movies.set_index("item_ext")
    ordered_ext = sorted(item_ids, key=item_ids.get)
    meta = Movies(
        titles=tuple(str(by_item.loc[m, "title"]) for m in ordered_ext),
        genres=tuple(tuple(str(by_item.loc[m, "genres"]).split("|")) for m in ordered_ext),
        external_ids=tuple(int(m) for m in ordered_ext),
    )
    features = users[users["user_ext"].isin(user_ids)].assign(
        user=lambda f: f["user_ext"].map(user_ids)
    ).set_index("user").sort_index()

    return Split(
        train=encode(train_raw),
        valid=encode(ratings[(ratings["timestamp"] >= t_valid) & (ratings["timestamp"] < t_test)]),
        test=encode(ratings[ratings["timestamp"] >= t_test]),
        n_users=len(user_ids),
        n_items=len(item_ids),
        movies=meta,
        cutoffs=(t_valid, t_test),
        user_features=features[["gender", "age", "occupation"]],
    )


def targets(frame: pd.DataFrame, liked_only: bool = False) -> dict[int, set[int]]:
    """Per user, the items they interacted with in ``frame``."""
    rows = frame[frame["liked"] == 1] if liked_only else frame
    return {int(u): set(g["item"].tolist()) for u, g in rows.groupby("user")}
