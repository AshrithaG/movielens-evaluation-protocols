"""For whom does each model win? The EASE against two-tower gap, by user.

    python scripts/history_length.py

Under leave-last-one-out on all users, the two-tower model wins overall; on the
users a time split can evaluate, EASE wins. This breaks the full population down
two ways to find out which users produce the flip:

  by history length (how much the model knows about the user)
  by whether the user is still active after the time split's test cutoff
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
from recsys.data import load_split  # noqa: E402
from recsys.metrics import ndcg_at_k  # noqa: E402
from recsys.stats import paired_bootstrap  # noqa: E402

K = 20
BINS = (20, 35, 60, 100, 200, 400, 10_000)


def main() -> None:
    """Fit on leave-last-one-out training data and break scores down by user."""
    split = load_split()
    everything = pd.concat([split.train, split.valid, split.test]).sort_values(["user", "timestamp"], kind="stable")
    last = everything.groupby("user").tail(1)
    rest = everything.drop(last.index)
    hist = split.history(rest)
    relevant = {int(r.user): {int(r.item)} for r in last.itertuples()}
    users = np.array(sorted(u for u in relevant if u in hist))

    u_arr, i_arr = rest["user"].to_numpy(), rest["item"].to_numpy()
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
    lengths = np.array([len(hist[u]) for u in users])
    active_after_cutoff = np.isin(users, split.test["user"].unique())

    def row(mask: np.ndarray, label: str) -> dict:
        tt_vs_ease = paired_bootstrap(scores["two_tower"][mask], scores["ease"][mask])
        ease_vs_pop = paired_bootstrap(scores["ease"][mask], scores["popularity"][mask])
        tt_vs_pop = paired_bootstrap(scores["two_tower"][mask], scores["popularity"][mask])
        out = {
            "group": label,
            "users": int(mask.sum()),
            "median_history": float(np.median(lengths[mask])),
            **{f"ndcg_{n}": float(s[mask].mean()) for n, s in scores.items()},
            "two_tower_minus_ease": tt_vs_ease,
            "ease_minus_popularity": ease_vs_pop,
            "two_tower_minus_popularity": tt_vs_pop,
        }
        print(
            f"  {label:<26} n={out['users']:<5} median history {out['median_history']:>5.0f}  "
            f"pop {out['ndcg_popularity']:.4f}  ease {out['ndcg_ease']:.4f}  two_tower {out['ndcg_two_tower']:.4f}  "
            f"two_tower - ease {tt_vs_ease[0]:+.4f} [{tt_vs_ease[1]:+.4f}, {tt_vs_ease[2]:+.4f}]"
        )
        return out

    report: dict[str, list] = {"by_history_length": [], "by_activity": []}
    print("by history length (leave-last-one-out, all users)")
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
        mask = (lengths >= lo) & (lengths < hi)
        if mask.sum() >= 30:
            report["by_history_length"].append(row(mask, f"{lo} to {hi - 1}" if hi < 10_000 else f"{lo}+"))

    print("\nby activity after the time split's test cutoff")
    report["by_activity"].append(row(active_after_cutoff, "still active (time split users)"))
    report["by_activity"].append(row(~active_after_cutoff, "left before the cutoff"))
    for lo, hi in ((20, 100), (100, 10_000)):
        for active, name in ((True, "active"), (False, "left")):
            mask = (lengths >= lo) & (lengths < hi) & (active_after_cutoff == active)
            if mask.sum() >= 30:
                report["by_activity"].append(row(mask, f"{name}, history {lo}-{hi - 1 if hi < 10_000 else '+'}"))

    out = REPO_ROOT / "results-recsys"
    out.mkdir(exist_ok=True)
    (out / "history_length.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
