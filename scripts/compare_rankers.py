"""Paired comparison of two-stage ranker variants, from their saved per user scores.

    python scripts/compare_rankers.py

Every variant is scored on the same test users, so differences are paired by
user. Reports each ranker against EASE, and the two contrasts the variants were
designed to isolate: feature layout, and which users the ranker learns from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.stats import paired_bootstrap  # noqa: E402

VARIANTS = {
    "sanity: EASE rank only": "ts_sanity.json",
    "pooled, all users": "ts_pooled_all.json",
    "pooled, multi-session users": "ts_pooled_multi.json",
    "pooled, single-session users": "ts_pooled_single.json",
    "per generator, all users": "ts_pergen_all.json",
    "per generator, multi-session users": "ts_pergen_multi.json",
    "sanity: EASE rank only, monotone": "ts_sanity_mono.json",
    "per generator, all users, monotone": "ts_pergen_all_mono.json",
    "per generator, multi-session users, monotone": "ts_pergen_multi_mono.json",
}
CONTRASTS = [
    ("per generator, all users", "pooled, all users", "feature layout"),
    ("per generator, multi-session users", "pooled, multi-session users", "feature layout, multi-session"),
    ("pooled, multi-session users", "pooled, single-session users", "training users"),
    ("per generator, multi-session users", "per generator, all users", "training users, per generator"),
    ("sanity: EASE rank only, monotone", "sanity: EASE rank only", "monotone, sanity"),
    ("per generator, all users, monotone", "per generator, all users", "monotone, per generator"),
    ("per generator, multi-session users, monotone", "per generator, multi-session users", "monotone, multi-session"),
]


def main() -> None:
    """Load every variant present and print the paired table."""
    root = REPO_ROOT / "results-recsys"
    loaded = {name: json.loads((root / f).read_text()) for name, f in VARIANTS.items() if (root / f).exists()}
    report: dict[str, dict] = {}
    for horizon in ("full_test_period", "first_30_days"):
        print(f"\n== {horizon}")
        rows: dict[str, dict] = {}
        any_run = next(iter(loaded.values()))
        users = sorted(any_run[horizon]["per_user_ndcg"]["two_stage_ltr"])
        ease_key = next(k for k in any_run[horizon]["per_user_ndcg"] if k.startswith("ease"))
        ease = np.array([any_run[horizon]["per_user_ndcg"][ease_key][u] for u in users])
        ltr = {n: np.array([d[horizon]["per_user_ndcg"]["two_stage_ltr"][u] for u in users]) for n, d in loaded.items()}
        print(f"  EASE alone                             ndcg@20 {ease.mean():.4f}")
        for name, scores in ltr.items():
            diff = paired_bootstrap(scores, ease)
            rows[name] = {"ndcg": float(scores.mean()), "minus_ease": diff}
            print(f"  {name:<38} ndcg@20 {scores.mean():.4f}  minus EASE {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]")
        contrasts = {}
        for left, right, label in CONTRASTS:
            if left in ltr and right in ltr:
                diff = paired_bootstrap(ltr[left], ltr[right])
                contrasts[f"{label}: {left} minus {right}"] = diff
                print(f"  contrast {label:<32} {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]")
        report[horizon] = {"ease_ndcg": float(ease.mean()), "variants": rows, "contrasts": contrasts}
    (root / "ranker_variants.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
