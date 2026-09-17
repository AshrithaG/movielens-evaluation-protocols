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

