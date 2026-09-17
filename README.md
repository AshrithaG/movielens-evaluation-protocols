# When does a recommender's evaluation tell you the truth?

Which recommendation model is best on MovieLens-1M depends almost entirely on
how you split the data. Under the protocol most papers use, a neural two-tower
model is the clear winner. Under a split that trains on the past and tests on
the future, it loses to plain popularity, and only a tuned linear model, EASE,
beats popularity at all.

This project traces that reversal to its cause. It is not leakage and it is not
the target definition. It is who gets evaluated: most MovieLens users rated
their entire history in a single sitting, and the popular protocol scores them
while an honest time split cannot.

## Findings

1. **The protocol reverses the ranking.** Tuned on validation and scored on the
   same interactions, the two-tower model is last of five under a time split
   (6.8 nDCG points below popularity) and first under leave last one out (6.2
   above).

2. **The cause is the evaluated population.** A 2x2 over target definition and
   leakage, plus a control that scores the leaky protocol on exactly the users a
   time split can see, shows EASE winning every cell where the users are held
   fixed. Changing the target never changes the winner; changing the users
   always does.

3. **The flip lives in single-session users.** Users who left before the time
   cutoff have a median activity span of 0.0 days: 81% rated their last two
   movies within 60 seconds. For them the "next item" is the next title on the
   rating page. The two-tower model's advantage appears only there, at every
   history length, and disappears for users with real histories.

4. **This artifact is known; its effect on protocol comparisons is the point
   here.** Woolridge et al. (2021) showed MovieLens sequences are largely
   pseudo-sequences produced by the rating interface, and Fan et al.
   showed MovieLens interactions are driven by the platform's own recommendation
   flow. What this project adds is the decomposition: the widely used protocol
   does not merely leak the future, it silently changes the population, and on
   this dataset that alone decides the model ranking.

5. **The evaluation horizon changes the winner too.** Validation spans 27 days
   and the test period 790. Over the first 30 days of test, 7-day recent
   popularity beats every model; over the full period, EASE does. A model tuned
   on one horizon is tuned for the wrong decision on the other.

6. **A learned ranker over five generators does not beat EASE alone,** and
   needs monotone constraints just to match it. Without them, a ranker given
   only EASE's own ranking loses 1.8 points to EASE.

### The protocol decides the winner

nDCG@20 difference from popularity, in points, with 95% paired bootstrap intervals over users.

| Model | Selected on validation | Time split | Leave last one out |
| --- | --- | --- | --- |
| EASE | l2=5000 | +2.6 [+1.6, +3.6] | +3.9 [+3.5, +4.3] |
| Item kNN | neighbours=400, no recency weighting | -2.3 [-3.3, -1.4] | +2.3 [+1.9, +2.7] |
| Two-tower | epochs=4 | -6.8 [-8.0, -5.7] | +6.2 [+5.6, +6.8] |
| Recent popularity | window_days=7 | -3.1 [-4.3, -1.8] | -1.8 [-2.2, -1.5] |

### Target, leakage, or population?

| Training | Target | Users | EASE vs popularity | Two-tower vs popularity | Winner |
| --- | --- | --- | --- | --- | --- |
| clean | next item | time split users (1008) | +0.6 [-0.3, +1.5] | -1.7 [-2.6, -0.8] | ease |
| clean | future set | time split users (1008) | +2.6 [+1.6, +3.6] | -6.8 [-8.0, -5.7] | ease |
| leaky | next item | all users (5400) | +3.9 [+3.5, +4.3] | +6.2 [+5.6, +6.8] | two_tower |
| leaky | next item | time split users only (1008) | +2.2 [+1.4, +3.1] | +1.4 [+0.3, +2.5] | ease |
| leaky | future set | all users (5361) | +4.0 [+3.6, +4.4] | +7.8 [+7.2, +8.3] | two_tower |
| leaky | future set | time split users only (1007) | +2.3 [+1.5, +3.1] | -1.7 [-2.9, -0.5] | ease |

### Who produces the flip

Two-tower minus EASE, nDCG@20 points, leave last one out.

| Users | n | Median history | Two-tower minus EASE |
| --- | --- | --- | --- |
| history 20 to 34 | 883 | 26 | +2.2 [+0.7, +3.6] |
| history 35 to 59 | 922 | 45 | +2.1 [+0.7, +3.6] |
| history 60 to 99 | 893 | 76 | +3.9 [+2.5, +5.3] |
| history 100 to 199 | 1189 | 137 | +2.1 [+0.8, +3.4] |
| history 200 to 399 | 893 | 276 | +1.4 [+0.1, +2.7] |
| history 400+ | 541 | 555 | +2.2 [+0.6, +3.8] |
| still active (time split users) | 1008 | 237 | -0.8 [-2.0, +0.4] |
| left before the cutoff | 4392 | 76 | +3.0 [+2.3, +3.6] |
| active, history 20-99 | 202 | 64 | +0.9 [-1.5, +3.3] |
| left, history 20-99 | 2496 | 44 | +2.9 [+2.0, +3.8] |
| active, history 100-+ | 804 | 300 | -1.2 [-2.6, +0.1] |
| left, history 100-+ | 1819 | 192 | +3.3 [+2.3, +4.2] |

## The system

**Data.** MovieLens-1M as implicit feedback, split at global time quantiles
(train to the 80th percentile of timestamps, validation to the 90th, test after).
Items and users unseen in training are dropped from evaluation rather than
scored as unavoidable misses. The data license forbids redistribution, so it is
downloaded locally and never committed.

**Candidate generators.**

| Model | What it is |
| --- | --- |
| Popularity | distinct users per item |
| Recent popularity | interactions in the final window before the cutoff |
| Item kNN | cosine item-item similarity, truncated neighbours, optional recency weighting |
| EASE | closed form item-item ridge regression with a zero diagonal (Steck, 2019) |
| Two-tower | user tower over recency weighted history, item tower over id and genres, in-batch softmax with logQ correction |

**Ranking and decision layers.** Candidates from every generator merged, a
LambdaMART ranker, a pointwise interaction model with isotonic calibration, and
maximal marginal relevance re-ranking trading relevance against genre
diversity.

**Metrics.** Recall, nDCG and hit rate at 20, catalog coverage, novelty (mean
self information) and intra-list genre diversity, always reported together.
Every model comparison is a paired bootstrap over users.

## Two-stage ranking, calibration and diversity

A LambdaMART ranker over candidates merged from all five generators, trained on
a per user holdout inside the training period (each user's last 20% of training
interactions are labels), so nothing from the test period reaches it. Half the
users train the ranker and the other half fit the isotonic calibrator.

What held up:

- **Monotone constraints are necessary.** A ranker given nothing but EASE's own
  rank lost 1.8 nDCG points to EASE, because trees fitted to noisy labels learn
  a non-monotone mapping and reshuffle a good order. Constraining rank features
  to be non-increasing and score features non-decreasing recovers EASE to
  within 0.1 points, and improves both real ranker variants by 1.4 and 1.6
  points.
- **Merging generators does not beat EASE on this data.** The best merged
  ranker is 5.2 points below EASE alone over the full test period and 1.2 below
  over the first 30 days. The pipeline itself is verified by the sanity row: it
  can reproduce EASE, it just does not improve on it.
- **Calibration belongs to a horizon.** Over the full test period the model
  predicts a 10.0% interaction rate against 11.2% observed. Over the first 30
  days it predicts 11.4% against 4.2%, and isotonic regression cannot correct a
  base rate that changed with the label definition. A probability is only
  calibrated for the decision window it was fitted to.
- **Some diversity is free, coverage is not.** Moving MMR's relevance weight
  from 1.0 to 0.7 raises genre diversity from 0.763 to 0.816 with no loss of
  nDCG, while catalog coverage falls from 0.312 to 0.298: balancing genres
  inside a list pulls it toward the popular titles of each genre.

What did not hold up is listed under failed hypotheses. Why a merged ranker
still trails EASE is open: the remaining untested difference is that its
training labels are a user's final 20% of the training period while the test
labels are 790 days of future.

### Two-stage ranking

LambdaMART over candidates merged from all five generators, against EASE alone. nDCG@20; differences in points with paired intervals.

| Ranker | Full test period | vs EASE | First 30 days | vs EASE |
| --- | --- | --- | --- | --- |
| sanity: EASE rank only | 0.2437 | -1.8 [-2.3, -1.3] | 0.1069 | -0.7 [-1.3, -0.0] |
| pooled, all users | 0.1937 | -6.8 [-7.8, -5.8] | 0.0876 | -2.6 [-3.7, -1.4] |
| pooled, multi-session users | 0.1998 | -6.2 [-7.1, -5.3] | 0.0869 | -2.7 [-3.7, -1.7] |
| pooled, single-session users | 0.1803 | -8.1 [-9.2, -7.1] | 0.0852 | -2.8 [-4.1, -1.6] |
| per generator, all users | 0.1955 | -6.6 [-7.6, -5.6] | 0.0944 | -1.9 [-3.2, -0.7] |
| per generator, multi-session users | 0.1911 | -7.0 [-8.0, -6.1] | 0.0849 | -2.9 [-4.0, -1.8] |
| sanity: EASE rank only, monotone | 0.2602 | -0.1 [-0.2, -0.1] | 0.1129 | -0.1 [-0.1, +0.0] |
| per generator, all users, monotone | 0.2092 | -5.2 [-6.2, -4.3] | 0.1011 | -1.2 [-2.4, -0.1] |
| per generator, multi-session users, monotone | 0.2067 | -5.5 [-6.4, -4.6] | 0.1001 | -1.3 [-2.4, -0.3] |
| EASE alone | 0.2616 | | 0.1135 | |

| Contrast | Full test period | First 30 days |
| --- | --- | --- |
| feature layout: per generator, all users minus pooled, all users | +0.2 [-0.5, +0.9] | +0.7 [-0.2, +1.6] |
| feature layout, multi-session: per generator, multi-session users minus pooled, multi-session users | -0.9 [-1.6, -0.1] | -0.2 [-1.1, +0.7] |
| training users: pooled, multi-session users minus pooled, single-session users | +1.9 [+1.2, +2.7] | +0.2 [-0.7, +1.1] |
| training users, per generator: per generator, multi-session users minus per generator, all users | -0.4 [-1.2, +0.3] | -0.9 [-1.9, +0.0] |
| monotone, sanity: sanity: EASE rank only, monotone minus sanity: EASE rank only | +1.7 [+1.2, +2.1] | +0.6 [-0.0, +1.2] |
| monotone, per generator: per generator, all users, monotone minus per generator, all users | +1.4 [+0.9, +1.9] | +0.7 [-0.1, +1.5] |
| monotone, multi-session: per generator, multi-session users, monotone minus per generator, multi-session users | +1.6 [+1.0, +2.2] | +1.5 [+0.7, +2.3] |

### Calibration of the interaction model

| Horizon | Observed rate | Mean predicted, raw | Mean predicted, isotonic | ECE raw | ECE isotonic |
| --- | --- | --- | --- | --- | --- |
| full test period | 0.112 | 0.102 | 0.100 | 0.057 | 0.053 |
| first 30 days | 0.042 | 0.117 | 0.114 | 0.075 | 0.072 |

### Relevance against diversity (MMR over the monotone ranker, full test period)

| Lambda | nDCG@20 | Genre diversity | Catalog coverage | Novelty |
| --- | --- | --- | --- | --- |
| 1.0 | 0.2092 | 0.763 | 0.312 | 1.61 |
| 0.9 | 0.2109 | 0.776 | 0.307 | 1.60 |
| 0.8 | 0.2116 | 0.794 | 0.304 | 1.59 |
| 0.7 | 0.2116 | 0.816 | 0.298 | 1.57 |
| 0.5 | 0.2090 | 0.860 | 0.276 | 1.55 |
| 0.3 | 0.2000 | 0.888 | 0.257 | 1.58 |



## Hypotheses that failed

Recorded because each looked right and each would have been a confident wrong
claim.

- **"The logQ popularity correction causes the flip."** Removing it made the
  two-tower model far worse under every protocol. The correction does not
  penalize popular items; it undoes the penalty in-batch sampling already
  imposes.
- **"The future is dominated by popular titles."** Only 15 to 17% of target
  items are in the top 100 under every protocol.
- **"The two-tower model wins for users with short histories."** It beats EASE
  in every history-length bin under leave last one out. History length is not
  the variable; single-session behaviour is.
- **"The flip is leakage."** Held to the same users, leakage lifts every model
  and reorders none.
- **"The ranker loses to EASE because pooled features hide which generator
  proposed an item."** Separate features per generator changed nDCG by +0.2
  points [-0.5, +0.9].
- **"The ranker loses because it learns from single-session users."** Training
  only on multi-day users helped the pooled layout by 1.9 points and changed the
  per-generator layout by -0.4 [-1.2, +0.3]. Not a consistent effect, so not
  claimed.

## Limitations

- One dataset, chosen because it is the one the field benchmarks on. The
  finding is about MovieLens and about protocols applied to it, not a claim that
  two-tower models underperform in general.
- The two-tower model is deliberately modest (64 dimensions, a few epochs, CPU).
  Its selected configuration came from a small validation sweep.
- The time split's test population is 1,008 users, because most users never
  return after the cutoff.

## Reproduce

```bash
pip install -r requirements.txt
# download and unzip https://files.grouplens.org/datasets/movielens/ml-1m.zip into data/
PYTHONPATH=src .venv-ml/bin/pytest -q tests/test_recsys_pkg
.venv-ml/bin/python scripts/tune_and_compare.py        # validation sweep, both protocols
.venv-ml/bin/python scripts/protocol_decomposition.py  # the 2x2 and matched users
.venv-ml/bin/python scripts/history_length.py          # who produces the flip
.venv-ml/bin/python scripts/run_two_stage.py --layout per_generator --monotone --out ts_pergen_all_mono.json
.venv-ml/bin/python scripts/run_two_stage.py --layout per_generator --sanity-ease-rank-only --monotone --out ts_sanity_mono.json
.venv-ml/bin/python scripts/compare_rankers.py         # paired contrasts across ranker variants
.venv-ml/bin/python scripts/make_tables.py             # regenerate the tables
```

## References

- F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets: History and Context. ACM TiiS, 2015.
- Harald Steck. Embarrassingly Shallow Autoencoders for Sparse Data. WWW, 2019.
- Daniel Woolridge, Sean Wilner and Madeleine Glick. Sequence or Pseudo-Sequence? An Analysis of Sequential Recommendation Datasets. PERSPECTIVES workshop at RecSys, 2021. https://ceur-ws.org/Vol-2955/paper8.pdf
- Yu-chen Fan, Yitong Ji, Jie Zhang and Aixin Sun. Our Model Achieves Excellent Performance on MovieLens: What Does It Mean? ACM TOIS. https://doi.org/10.1145/3675163
- Yitong Ji, Aixin Sun, Jie Zhang and Chenliang Li. A Critical Study on Data Leakage in Recommender System Offline Evaluation. ACM TOIS 41(3), 2023. https://arxiv.org/abs/2010.11060
