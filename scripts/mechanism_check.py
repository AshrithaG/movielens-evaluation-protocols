"""Why does the protocol flip the two-tower model from last to first?

    python scripts/mechanism_check.py

Hypothesis: the logQ popularity correction. It penalizes popular items as
in-batch negatives. Under leave last one out the held out item is a single
next item, often from the tail; under a time split the future is dominated by
popular titles. Train the two-tower with and without the correction and score
both under three evaluations:

  time split, full test period (790 days)
  time split, first 30 days of the test period (matched to the 27 day validation)
  leave last one out

EASE and popularity are included as fixed references.
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
    """Fit each model per protocol and report nDCG with paired intervals against popularity."""
    split = load_split()
    past = pd.concat([split.train, split.valid])
    horizon_end = split.cutoffs[1] + 30 * 86_400
    protocols = {
        "time_full": (past, targets(split.test)),
        "time_30d": (past, targets(split.test[split.test["timestamp"] < horizon_end])),
    }
    everything = pd.concat([split.train, split.valid, split.test]).sort_values(["user", "timestamp"], kind="stable")
    last = everything.groupby("user").tail(1)
    protocols["leave_last_one_out"] = (everything.drop(last.index), {int(r.user): {int(r.item)} for r in last.itertuples()})

    report: dict[str, dict] = {}
    for protocol, (fit_on, relevant) in protocols.items():
        hist = split.history(fit_on)
        users = sorted(u for u in relevant if u in hist)
        u_arr, i_arr = fit_on["user"].to_numpy(), fit_on["item"].to_numpy()
        pop = Popularity().fit(u_arr, i_arr, split.n_items)
        models = {
            "popularity": pop,
            "ease": EASE(l2=5000.0).fit(u_arr, i_arr, split.n_users, split.n_items),
            "two_tower_logq": TwoTower(epochs=4, pop_correction=True).fit(hist, split.movies.genres, split.n_items, pop.scores(np.array([]))),
            "two_tower_no_logq": TwoTower(epochs=4, pop_correction=False).fit(hist, split.movies.genres, split.n_items, pop.scores(np.array([]))),
        }
        scores = {
            name: np.array([ndcg_at_k(top_k(m.scores(hist[u]), K, exclude=hist[u]), relevant[u], K) for u in users])
            for name, m in models.items()
        }
        rows = {}
        # Share of recommended items that are in the catalog's 100 most popular.
        head = set(top_k(pop.scores(np.array([])), 100))
        for name, m in models.items():
            diff = paired_bootstrap(scores[name], scores["popularity"])
            recs = [top_k(m.scores(hist[u]), K, exclude=hist[u]) for u in users[:500]]
            head_share = float(np.mean([np.mean([i in head for i in r]) for r in recs]))
            rows[name] = {"ndcg": float(scores[name].mean()), "vs_popularity": diff, "top100_share": head_share}
        target_head = float(np.mean([np.mean([i in head for i in relevant[u]]) for u in users]))
        report[protocol] = {"users": len(users), "target_top100_share": target_head, "models": rows}
        print(f"\n{protocol}  ({len(users)} users; share of TARGET items in top-100 popular: {target_head:.2f})")
        for name, row in rows.items():
            d = row["vs_popularity"]
            print(f"  {name:<20} ndcg@{K} {row['ndcg']:.4f}  vs popularity {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]  "
                  f"recommended in top-100: {row['top100_share']:.2f}")

    out = REPO_ROOT / "results-recsys"
    out.mkdir(exist_ok=True)
    (out / "mechanism_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
