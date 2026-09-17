"""Two-stage recommender on MovieLens-1M: merged candidates, LambdaMART, calibration, MMR.

    python scripts/run_two_stage.py

Ranker and calibrator training data come from the training period only. Each
user's last 20% of training-period interactions become labels, and the
candidate generators are fitted on everything else in that period. An earlier
version used a global cutoff inside the training period instead; because most
MovieLens users rated their whole history in a single sitting, only 192 users
spanned that cutoff, far too few to train a ranker. The per user holdout reuses
other users' later interactions within the training period, which never
reaches the test period, so the test evaluation stays clean.

Users are split in half: one half trains the ranker, the other fits the
isotonic calibrator, so calibration is measured on users the classifier did not
see. The test period is scored over its full span and over the first 30 days,
the horizon that matches validation.

Nothing about the test period is seen before the final evaluation. Generator hyperparameters
come from the validation sweep in tune_and_compare.py.
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
from recsys.metrics import evaluate, expected_calibration_error, ndcg_at_k  # noqa: E402
from recsys.ranker import CandidateMerger, InteractionModel, LambdaMARTRanker  # noqa: E402
from recsys.rerank import mmr  # noqa: E402
from recsys.stats import paired_bootstrap  # noqa: E402

K = 20


def log(msg: str, started: float) -> None:
    """Print with elapsed seconds."""
    print(f"[{time.time() - started:7.1f}s] {msg}", flush=True)


def selected_params() -> dict[str, dict]:
    """Hyperparameters chosen on validation, with defaults if the sweep has not run."""
    path = REPO_ROOT / "results-recsys" / "tune_and_compare.json"
    defaults = {"item_knn": {}, "ease": {"l2": 500.0}, "two_tower": {}, "recent_popularity": {"window_days": 30}}
    if not path.exists():
        return defaults
    chosen = json.loads(path.read_text())["selected"]
    return {name: chosen.get(name, params) for name, params in defaults.items()}


def fit_generators(past: pd.DataFrame, split, params: dict[str, dict]):
    """Every candidate generator plus popularity features, fitted on ``past``."""
    users, items = past["user"].to_numpy(), past["item"].to_numpy()
    histories = split.history(past)
    pop = Popularity().fit(users, items, split.n_items)
    recent = RecentPopularity(**params["recent_popularity"]).fit(items, past["timestamp"].to_numpy(), split.n_items)
    generators = [
        EASE(**params["ease"]).fit(users, items, split.n_users, split.n_items),
        ItemKNN(**params["item_knn"]).fit(users, items, split.n_users, split.n_items),
        TwoTower(**params["two_tower"]).fit(histories, split.movies.genres, split.n_items, pop.scores(np.array([]))),
        recent,
    ]
    recent_share = recent.scores(np.array([])) / max(recent.scores(np.array([])).max(), 1.0)
    return generators, pop.share, recent_share, histories


def main() -> None:
    """Train on a per user holdout, calibrate on held-out users, evaluate on test."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ranker-users",
        choices=["all", "multi_session", "single_session"],
        default="all",
        help="which training users the ranker and calibrator learn from",
    )
    ap.add_argument("--layout", choices=["pooled", "per_generator"], default="pooled")
    ap.add_argument(
        "--sanity-ease-rank-only",
        action="store_true",
        help="ranker sees only EASE's rank; it must reproduce EASE or the pipeline is wrong",
    )
    ap.add_argument("--monotone", action="store_true", help="constrain rank features to be non-increasing and score features non-decreasing")
    ap.add_argument("--out", default="two_stage.json")
    args = ap.parse_args()
    started = time.time()
    split = load_split()
    params = selected_params()
    log(f"generator params {params}", started)

    genre_names = sorted({g for gs in split.movies.genres for g in gs})
    item_genres = np.array([[1.0 if g in gs else 0.0 for g in genre_names] for gs in split.movies.genres])
    item_genres = item_genres / item_genres.sum(1, keepdims=True).clip(min=1.0)
    meta = {
        int(u): (float(r["age"]), 1.0 if r["gender"] == "M" else 0.0)
        for u, r in split.user_features.iterrows()
    }

    def merger_for(past: pd.DataFrame):
        gens, pop_all, pop_recent, hist = fit_generators(past, split, params)
        return CandidateMerger(gens, depth=100, pop_all=pop_all, pop_recent=pop_recent,
                               item_genres=item_genres, user_meta=meta, layout=args.layout), gens, hist

    # Per user holdout inside the training period, for ranker and calibrator training.
    train_sorted = split.train.sort_values(["user", "timestamp"], kind="stable")
    from_end = train_sorted.groupby("user").cumcount(ascending=False)
    size = train_sorted.groupby("user")["item"].transform("size")
    held = train_sorted[from_end < np.maximum(1, (0.2 * size).astype(int))]
    fit_part = train_sorted.drop(held.index)
    merger_a, _, hist_a = merger_for(fit_part)
    rel_held = targets(held)
    users_a = sorted(u for u in rel_held if u in hist_a)
    # Most MovieLens users rated their whole history in one sitting, so their
    # held-out "future" is the next page of the rating interface. Restricting
    # the ranker to users with multi-day activity tests whether that artifact
    # is what the ranker learns.
    span_days = train_sorted.groupby("user")["timestamp"].agg(lambda s: (s.max() - s.min()) / 86_400)
    if args.ranker_users == "multi_session":
        users_a = [u for u in users_a if span_days.get(u, 0.0) >= 1.0]
    elif args.ranker_users == "single_session":
        users_a = [u for u in users_a if span_days.get(u, 0.0) < 1.0]
    log(f"ranker users: {args.ranker_users}, {len(users_a)} available", started)
    rng = np.random.default_rng(0)
    rng.shuffle(users_a)
    half = len(users_a) // 2
    ranker_users, calib_users = users_a[:half], users_a[half:]
    ranker_sets = [merger_a.build(u, hist_a[u]) for u in ranker_users]
    calib_sets = [merger_a.build(u, hist_a[u]) for u in calib_users]
    columns = None
    if args.sanity_ease_rank_only:
        if args.layout != "per_generator":
            raise SystemExit("--sanity-ease-rank-only needs --layout per_generator")
        ease_name = next(g.name for g in merger_a.generators if g.name.startswith("ease"))
        columns = (merger_a.feature_names.index(f"{ease_name}_rank"),)
    monotone = None
    if args.monotone:
        names = merger_a.feature_names if columns is None else tuple(merger_a.feature_names[c] for c in columns)
        monotone = tuple(-1 if n.endswith("rank") else 1 if n.endswith("_z") or n == "gen_score" else 0 for n in names)
    ranker = LambdaMARTRanker(columns=columns, monotone=monotone).fit(ranker_sets, rel_held)
    log(f"ranker trained on {len(ranker_sets)} users, calibrator on {len(calib_sets)} (per user holdout)", started)
    interaction = InteractionModel().fit(ranker_sets, calib_sets, rel_held)

    # Test: generators on everything before the test cutoff.
    past = pd.concat([split.train, split.valid])
    merger_c, gens_c, hist_c = merger_for(past)
    horizon_end = split.cutoffs[1] + 30 * 86_400
    horizons = {
        "full_test_period": targets(split.test),
        "first_30_days": targets(split.test[split.test["timestamp"] < horizon_end]),
    }
    popularity = Popularity().fit(past["user"].to_numpy(), past["item"].to_numpy(), split.n_items)
    share = popularity.share
    results: dict[str, object] = {}

    for horizon, rel_c in horizons.items():
        test_users = sorted(u for u in rel_c if u in hist_c)
        test_sets = {u: merger_c.build(u, hist_c[u]) for u in test_users}
        log(f"{horizon}: candidates for {len(test_users)} users", started)
        cand_recall = float(np.mean([len(set(test_sets[u].items.tolist()) & rel_c[u]) / len(rel_c[u]) for u in test_users]))

        systems: dict[str, dict[int, list[int]]] = {
            "popularity": {u: top_k(popularity.scores(hist_c[u]), K, exclude=hist_c[u]) for u in test_users},
        }
        for gen in gens_c:
            systems[gen.name] = {u: top_k(gen.scores(hist_c[u]), K, exclude=hist_c[u]) for u in test_users}
        ranked = {u: ranker.rank(test_sets[u]) for u in test_users}
        systems["two_stage_ltr"] = {u: ranked[u][0][:K].tolist() for u in test_users}

        per_user = {n: np.array([ndcg_at_k(r[u], rel_c[u], K) for u in test_users]) for n, r in systems.items()}
        best_single = max((n for n in systems if n != "two_stage_ltr"), key=lambda n: per_user[n].mean())
        per_user_record = {n: dict(zip(map(str, test_users), map(float, v), strict=True)) for n, v in per_user.items()}
        section: dict[str, object] = {"users": len(test_users), "candidate_recall_at_depth": cand_recall,
                                      "best_single_generator": best_single, "systems": {}}
        for name, recs in systems.items():
            metrics = evaluate(recs, rel_c, k=K, n_items=split.n_items, popularity=share, genres=split.movies.genres)
            diff = paired_bootstrap(per_user[name], per_user[best_single]) if name != best_single else (0.0, 0.0, 0.0)
            section["systems"][name] = {**metrics, "ndcg_vs_best_single": diff}  # type: ignore[index]
            log(f"  {name:<20} ndcg@{K} {metrics[f'ndcg@{K}']:.4f}  hit@{K} {metrics[f'hit@{K}']:.3f}  "
                f"coverage {metrics['coverage']:.3f}  novelty {metrics['novelty']:.2f}  "
                f"vs {best_single} {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]", started)

        xs = np.vstack([test_sets[u].features for u in test_users if test_sets[u].items.size])
        ys = np.concatenate([np.array([int(i in rel_c[u]) for i in test_sets[u].items]) for u in test_users if test_sets[u].items.size])
        raw, cal = interaction.predict(xs)
        section["calibration"] = {
            "base_rate": float(ys.mean()),
            "ece_raw": expected_calibration_error(raw, ys),
            "ece_isotonic": expected_calibration_error(cal, ys),
            "mean_predicted_raw": float(raw.mean()),
            "mean_predicted_isotonic": float(cal.mean()),
        }
        log(f"  calibration {section['calibration']}", started)

        curve = []
        for lam in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
            recs = {u: mmr(ranked[u][0][:100], ranked[u][1][:100], item_genres, K, lam) for u in test_users}
            m = evaluate(recs, rel_c, k=K, n_items=split.n_items, popularity=share, genres=split.movies.genres)
            curve.append({"lambda": lam, **m})
            log(f"  mmr lambda {lam:.1f}: ndcg@{K} {m[f'ndcg@{K}']:.4f}  diversity {m['diversity']:.3f}  "
                f"coverage {m['coverage']:.3f}  novelty {m['novelty']:.2f}", started)
        section["mmr_tradeoff"] = curve
        section["per_user_ndcg"] = per_user_record
        results[horizon] = section

    out = REPO_ROOT / "results-recsys"
    out.mkdir(exist_ok=True)
    results["ranker_users"] = args.ranker_users
    results["layout"] = args.layout
    results["monotone"] = args.monotone
    results["sanity_ease_rank_only"] = args.sanity_ease_rank_only
    (out / args.out).write_text(json.dumps(results, indent=2))
    log(f"wrote results-recsys/{args.out}", started)


if __name__ == "__main__":
    main()
