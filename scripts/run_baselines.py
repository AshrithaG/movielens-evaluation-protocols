"""Candidate generators on MovieLens-1M, on validation and on test.

    python scripts/run_baselines.py

Validation: fit on the training period, evaluate on the validation period.
Test: refit on everything before the test cutoff, evaluate on the test period.
Items a user already interacted with are never recommended.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.candidates import ItemKNN, Popularity, TwoTower, top_k  # noqa: E402
from recsys.data import load_split, targets  # noqa: E402
from recsys.metrics import evaluate  # noqa: E402

K = 20


def run_period(past: pd.DataFrame, future: pd.DataFrame, split, started: float, label: str) -> dict:
    """Fit every model on ``past`` and evaluate on ``future``."""
    users, items = past["user"].to_numpy(), past["item"].to_numpy()
    histories = split.history(past)
    relevant = targets(future)
    eval_users = [u for u in relevant if u in histories]

    pop = Popularity().fit(users, items, split.n_items)
    share = pop.share
    models = [
        pop,
        ItemKNN().fit(users, items, split.n_users, split.n_items),
        TwoTower().fit(histories, split.movies.genres, split.n_items, pop.scores(np.array([]))),
    ]
    print(f"[{time.time() - started:6.1f}s] {label}: models fitted, {len(eval_users)} users", flush=True)

    out = {}
    for model in models:
        recs = {u: top_k(model.scores(histories[u]), K, exclude=histories[u]) for u in eval_users}
        out[model.name] = evaluate(
            recs, relevant, k=K, n_items=split.n_items, popularity=share, genres=split.movies.genres
        )
        row = out[model.name]
        print(
            f"   {model.name:<12} recall@{K} {row[f'recall@{K}']:.4f}  ndcg@{K} {row[f'ndcg@{K}']:.4f}  "
            f"hit@{K} {row[f'hit@{K}']:.4f}  coverage {row['coverage']:.3f}  novelty {row['novelty']:.2f}  "
            f"diversity {row['diversity']:.3f}",
            flush=True,
        )
    return out


def main() -> None:
    """Validation then test."""
    started = time.time()
    split = load_split()
    print(
        f"train {len(split.train)}, valid {len(split.valid)}, test {len(split.test)} interactions; "
        f"{split.n_users} users, {split.n_items} items",
        flush=True,
    )
    results = {
        "validation": run_period(split.train, split.valid, split, started, "validation"),
        "test": run_period(pd.concat([split.train, split.valid]), split.test, split, started, "test"),
    }
    out = REPO_ROOT / "results-recsys"
    out.mkdir(exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
