"""Tune every candidate model on validation, then compare on test, under two split protocols.

    python scripts/tune_and_compare.py

Time split: fit on the past, evaluate on the future (the production situation).
Leave last one out: each user's final interaction is held out and everything
else, including other users' later interactions, is training data. It is the
most common protocol in papers, and it leaks the future.

The same tuned hyperparameters are used under both, so any change in which
model wins is the protocol, not the tuning.
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

from recsys.candidates import EASE, ItemKNN, Popularity, RecentPopularity, TwoTower, top_k  # noqa: E402
from recsys.data import load_split, targets  # noqa: E402
from recsys.metrics import evaluate, ndcg_at_k  # noqa: E402
from recsys.stats import paired_bootstrap  # noqa: E402

K = 20


def build(name: str, params: dict, past: pd.DataFrame, split, histories):
    """Construct and fit one model configuration on ``past``."""
    users, items = past["user"].to_numpy(), past["item"].to_numpy()
    if name == "popularity":
        return Popularity().fit(users, items, split.n_items)
    if name == "recent_popularity":
        return RecentPopularity(**params).fit(items, past["timestamp"].to_numpy(), split.n_items)
    if name == "item_knn":
        return ItemKNN(**params).fit(users, items, split.n_users, split.n_items)
    if name == "ease":
        return EASE(**params).fit(users, items, split.n_users, split.n_items)
    if name == "two_tower":
        pop = Popularity().fit(users, items, split.n_items).scores(np.array([]))
        return TwoTower(**params).fit(histories, split.movies.genres, split.n_items, pop)
    raise ValueError(name)


GRID: dict[str, list[dict]] = {
    "popularity": [{}],
    "recent_popularity": [{"window_days": d} for d in (7, 30, 90, 365)],
    "item_knn": [{"neighbours": n, "half_life": h} for n in (20, 100, 400) for h in (5.0, 20.0, 1e9)],
    "ease": [{"l2": v} for v in (50.0, 200.0, 500.0, 1500.0, 5000.0)],
    "two_tower": [{"epochs": e} for e in (4, 10, 20)],
}


def per_user_ndcg(model, histories, relevant, users) -> np.ndarray:
    """nDCG@K for each user in ``users``, in order."""
    return np.array(
        [ndcg_at_k(top_k(model.scores(histories[u]), K, exclude=histories[u]), relevant[u], K) for u in users]
    )


def main() -> None:
    """Sweep on validation, report test under both protocols."""
    started = time.time()
    split = load_split()
    share = Popularity().fit(split.train["user"].to_numpy(), split.train["item"].to_numpy(), split.n_items).share

    val_hist = split.history(split.train)
    val_rel = targets(split.valid)
    val_users = sorted(u for u in val_rel if u in val_hist)

    selected: dict[str, dict] = {}
    sweep: dict[str, list] = {}
    for name, configs in GRID.items():
        rows = []
        for params in configs:
            model = build(name, params, split.train, split, val_hist)
            score = float(per_user_ndcg(model, val_hist, val_rel, val_users).mean())
            rows.append({"params": params, "val_ndcg": score})
        best = max(rows, key=lambda r: r["val_ndcg"])
        selected[name] = best["params"]
        sweep[name] = rows
        print(f"[{time.time() - started:6.1f}s] {name:<18} selected {best['params']}  val ndcg@{K} {best['val_ndcg']:.4f}", flush=True)

    report: dict[str, object] = {"selected": selected, "validation_sweep": sweep}

    # Time split test.
    past = pd.concat([split.train, split.valid])
    test_hist = split.history(past)
    test_rel = targets(split.test)
    test_users = sorted(u for u in test_rel if u in test_hist)
    time_scores, time_metrics = {}, {}
    for name, params in selected.items():
        model = build(name, params, past, split, test_hist)
        time_scores[name] = per_user_ndcg(model, test_hist, test_rel, test_users)
        recs = {u: top_k(model.scores(test_hist[u]), K, exclude=test_hist[u]) for u in test_users}
        time_metrics[name] = evaluate(recs, test_rel, k=K, n_items=split.n_items, popularity=share, genres=split.movies.genres)
    print(f"[{time.time() - started:6.1f}s] time split test, {len(test_users)} users", flush=True)

    # Leave last one out, on the same interactions.
    everything = pd.concat([split.train, split.valid, split.test]).sort_values(["user", "timestamp"], kind="stable")
    last = everything.groupby("user").tail(1)
    rest = everything.drop(last.index)
    lloo_hist = split.history(rest)
    lloo_rel = {int(r.user): {int(r.item)} for r in last.itertuples()}
    lloo_users = sorted(u for u in lloo_rel if u in lloo_hist)
    lloo_scores, lloo_metrics = {}, {}
    for name, params in selected.items():
        model = build(name, params, rest, split, lloo_hist)
        lloo_scores[name] = per_user_ndcg(model, lloo_hist, lloo_rel, lloo_users)
        recs = {u: top_k(model.scores(lloo_hist[u]), K, exclude=lloo_hist[u]) for u in lloo_users}
        lloo_metrics[name] = evaluate(recs, lloo_rel, k=K, n_items=split.n_items, popularity=share, genres=split.movies.genres)
    print(f"[{time.time() - started:6.1f}s] leave last one out, {len(lloo_users)} users", flush=True)

    def table(scores: dict[str, np.ndarray], metrics: dict[str, dict], label: str) -> dict:
        ranked = sorted(scores, key=lambda n: -scores[n].mean())
        base = scores["popularity"]
        rows = {}
        print(f"\n{label}")
        for name in ranked:
            diff, lo, hi = paired_bootstrap(scores[name], base) if name != "popularity" else (0.0, 0.0, 0.0)
            m = metrics[name]
            rows[name] = {**m, "ndcg_minus_popularity": diff, "ci_lo": lo, "ci_hi": hi}
            print(
                f"  {name:<18} ndcg@{K} {scores[name].mean():.4f}  vs popularity {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                f"hit@{K} {m[f'hit@{K}']:.3f}  coverage {m['coverage']:.3f}  novelty {m['novelty']:.2f}"
            )
        return {"ranking": ranked, "rows": rows}

    report["time_split"] = table(time_scores, time_metrics, "TIME SPLIT (fit on the past, test on the future)")
    report["leave_last_one_out"] = table(lloo_scores, lloo_metrics, "LEAVE LAST ONE OUT (future leaks into training)")
    out = REPO_ROOT / "results-recsys"
    out.mkdir(exist_ok=True)
    (out / "tune_and_compare.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
