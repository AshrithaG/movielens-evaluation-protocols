"""Is the protocol flip caused by the target definition or by leakage?

    python scripts/protocol_decomposition.py

A 2x2 over the two things leave-last-one-out changes at once:

                        target = next item      target = future set
  clean training        time split, first       time split, all test
                        test interaction        interactions (790 days)
  leaky training        leave last one out      leave last 20% out, other
                                                users' future in training

If the two-tower model wins in the "next item" column regardless of leakage,
the flip is the task definition. If it wins in the "leaky" row regardless of
target, it is leakage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.candidates import EASE, Popularity, TwoTower, top_k  # noqa: E402
from recsys.data import load_split, targets  # noqa: E402
from recsys.metrics import ndcg_at_k  # noqa: E402
from recsys.stats import paired_bootstrap  # noqa: E402

K = 20


def main() -> None:
    """Score popularity, EASE and the two-tower model in every cell."""
    split = load_split()
    past = pd.concat([split.train, split.valid])
    test_sorted = split.test.sort_values(["user", "timestamp"], kind="stable")
    first_test = test_sorted.groupby("user").head(1)

    everything = pd.concat([split.train, split.valid, split.test]).sort_values(["user", "timestamp"], kind="stable")
    last_one = everything.groupby("user").tail(1)
    rank_from_end = everything.groupby("user").cumcount(ascending=False)
    counts = everything.groupby("user")["item"].transform("size")
    last_share = everything[rank_from_end < np.maximum(1, (0.2 * counts).astype(int))]

    cells = {
        ("clean", "next_item"): (past, {int(r.user): {int(r.item)} for r in first_test.itertuples()}),
        ("clean", "future_set"): (past, targets(split.test)),
        ("leaky", "next_item"): (everything.drop(last_one.index), {int(r.user): {int(r.item)} for r in last_one.itertuples()}),
        ("leaky", "future_set"): (everything.drop(last_share.index), targets(last_share)),
    }

    # The clean cells can only score users active after the cutoff (heavy users);
    # the leaky cells score everyone. Reporting the leaky cells again on exactly the
    # clean users separates leakage from a change in who is being evaluated.
    clean_users = {int(u) for u in split.test["user"].unique()} & set(split.history(past))
    report = {}
    for (training, target), (fit_on, relevant) in list(cells.items()):
        _run_cell(split, training, target, fit_on, relevant, None, report)
        if training == "leaky":
            _run_cell(split, training, target, fit_on, relevant, clean_users, report)

    out = REPO_ROOT / "results-recsys"
    out.mkdir(exist_ok=True)
    (out / "protocol_decomposition.json").write_text(json.dumps(report, indent=2))


def _run_cell(split, training, target, fit_on, relevant, restrict, report) -> None:
    """Fit and score one cell, optionally on a fixed user subset."""
    hist = split.history(fit_on)
    users = sorted(u for u in relevant if u in hist and (restrict is None or u in restrict))
    u_arr, i_arr = fit_on["user"].to_numpy(), fit_on["item"].to_numpy()
    pop = Popularity().fit(u_arr, i_arr, split.n_items)
    models = {
        "popularity": pop,
        "ease": EASE(l2=5000.0).fit(u_arr, i_arr, split.n_users, split.n_items),
        "two_tower": TwoTower(epochs=4).fit(hist, split.movies.genres, split.n_items, pop.scores(np.array([]))),
    }
    scores = {
        n: np.array([ndcg_at_k(top_k(m.scores(hist[u]), K, exclude=hist[u]), relevant[u], K) for u in users])
        for n, m in models.items()
    }
    key = f"{training}/{target}" + ("/matched_users" if restrict is not None else "")
    report[key] = {
        "users": len(users),
        "mean_targets_per_user": float(np.mean([len(relevant[u]) for u in users])),
        "models": {n: {"ndcg": float(s.mean()), "vs_popularity": paired_bootstrap(s, scores["popularity"]),
                       "vs_ease": paired_bootstrap(s, scores["ease"])} for n, s in scores.items()},
    }
    winner = max(scores, key=lambda n: scores[n].mean())
    print(f"\n{key}  ({len(users)} users, {report[key]['mean_targets_per_user']:.1f} targets per user)  winner: {winner}")
    for n in scores:
        d = report[key]["models"][n]["vs_popularity"]
        print(f"  {n:<12} ndcg@{K} {scores[n].mean():.4f}  vs popularity {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")


if __name__ == "__main__":
    main()
