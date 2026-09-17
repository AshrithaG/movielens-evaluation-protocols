### The protocol decides the winner

nDCG@20 difference from popularity, in points, with 95% paired bootstrap intervals over users.

| Model | Selected on validation | Time split | Leave last one out |
| --- | --- | --- | --- |
| EASE | l2=5000 | +2.6 [+1.6, +3.6] | +3.9 [+3.5, +4.3] |
| Item kNN | neighbours=400, half_life=1e+09 | -2.3 [-3.3, -1.4] | +2.3 [+1.9, +2.7] |
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
