"""Render the README's result tables from the results files.

    python scripts/make_tables.py > results-recsys/tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results-recsys"


def ci(triple) -> str:
    """Format a (mean, lo, hi) difference in nDCG points."""
    m, lo, hi = (100 * x for x in triple)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def main() -> None:
    """Print every table whose results exist."""
    tc = json.loads((ROOT / "tune_and_compare.json").read_text())
    print("### The protocol decides the winner\n")
    print("nDCG@20 difference from popularity, in points, with 95% paired bootstrap intervals over users.\n")
    print("| Model | Selected on validation | Time split | Leave last one out |")
    print("| --- | --- | --- | --- |")
    names = ["ease", "item_knn", "two_tower", "recent_popularity"]
    labels = {"ease": "EASE", "item_knn": "Item kNN", "two_tower": "Two-tower", "recent_popularity": "Recent popularity"}
    for n in names:
        sel = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in tc["selected"][n].items()) or "none"
        t = tc["time_split"]["rows"][n]
        l = tc["leave_last_one_out"]["rows"][n]
        print(f"| {labels[n]} | {sel} | {ci((t['ndcg_minus_popularity'], t['ci_lo'], t['ci_hi']))} | "
              f"{ci((l['ndcg_minus_popularity'], l['ci_lo'], l['ci_hi']))} |")

    dec = json.loads((ROOT / "protocol_decomposition.json").read_text())
    print("\n### Target, leakage, or population?\n")
    print("| Training | Target | Users | EASE vs popularity | Two-tower vs popularity | Winner |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key, row in dec.items():
        parts = key.split("/")
        who = "time split users only" if key.endswith("matched_users") else ("all users" if parts[0] == "leaky" else "time split users")
        m = row["models"]
        winner = max(m, key=lambda k: m[k]["ndcg"])
        print(f"| {parts[0]} | {parts[1].replace('_', ' ')} | {who} ({row['users']}) | "
              f"{ci(m['ease']['vs_popularity'])} | {ci(m['two_tower']['vs_popularity'])} | {winner} |")

    hl = json.loads((ROOT / "history_length.json").read_text())
    print("\n### Who produces the flip\n")
    print("Two-tower minus EASE, nDCG@20 points, leave last one out.\n")
    print("| Users | n | Median history | Two-tower minus EASE |")
    print("| --- | --- | --- | --- |")
    for row in hl["by_history_length"] + hl["by_activity"]:
        label = row["group"] if "history" in row["group"] or "active" in row["group"] or "left" in row["group"] else f"history {row['group']}"
        print(f"| {label} | {row['users']} | {row['median_history']:.0f} | {ci(row['two_tower_minus_ease'])} |")


if __name__ == "__main__":
    main()


def two_stage_section() -> str:
    """The ranker, calibration and MMR section, generated from saved results."""
    rv = json.loads((ROOT / "ranker_variants.json").read_text())
    base = json.loads((ROOT / "ts_pergen_all_mono.json").read_text())
    lines = ["### Two-stage ranking\n",
             "LambdaMART over candidates merged from all five generators, against EASE alone. nDCG@20; differences in points with paired intervals.\n",
             "| Ranker | Full test period | vs EASE | First 30 days | vs EASE |",
             "| --- | --- | --- | --- | --- |"]
    full, short = rv["full_test_period"], rv["first_30_days"]
    for name, row in full["variants"].items():
        s = short["variants"][name]
        lines.append(f"| {name} | {row['ndcg']:.4f} | {ci(row['minus_ease'])} | {s['ndcg']:.4f} | {ci(s['minus_ease'])} |")
    lines.append(f"| EASE alone | {full['ease_ndcg']:.4f} | | {short['ease_ndcg']:.4f} | |")
    lines += ["\n| Contrast | Full test period | First 30 days |", "| --- | --- | --- |"]
    for key, diff in full["contrasts"].items():
        lines.append(f"| {key} | {ci(diff)} | {ci(short['contrasts'][key])} |")

    lines += ["\n### Calibration of the interaction model\n",
              "| Horizon | Observed rate | Mean predicted, raw | Mean predicted, isotonic | ECE raw | ECE isotonic |",
              "| --- | --- | --- | --- | --- | --- |"]
    for horizon in ("full_test_period", "first_30_days"):
        c = base[horizon]["calibration"]
        lines.append(f"| {horizon.replace('_', ' ')} | {c['base_rate']:.3f} | {c['mean_predicted_raw']:.3f} | "
                     f"{c['mean_predicted_isotonic']:.3f} | {c['ece_raw']:.3f} | {c['ece_isotonic']:.3f} |")

    lines += ["\n### Relevance against diversity (MMR over the monotone ranker, full test period)\n",
              "| Lambda | nDCG@20 | Genre diversity | Catalog coverage | Novelty |",
              "| --- | --- | --- | --- | --- |"]
    for row in base["full_test_period"]["mmr_tradeoff"]:
        lines.append(f"| {row['lambda']:.1f} | {row['ndcg@20']:.4f} | {row['diversity']:.3f} | {row['coverage']:.3f} | {row['novelty']:.2f} |")
    return "\n".join(lines) + "\n"
