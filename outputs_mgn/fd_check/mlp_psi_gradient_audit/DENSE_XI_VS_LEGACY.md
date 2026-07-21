# Dense-xi MLP gradient comparison

The controlled change is the Step 1 xi grid: the legacy run used 16 wake
columns, while `dense_xi_v1` uses 33.  Both MLP runs use the same 4x64 tanh
network, float64, lambda_v=1, 20,000 optimization steps, and seeds 0/1/2.

| metric | legacy 16x64 | dense 33x64 | change |
|---|---:|---:|---:|
| base-CFD epsilon_K floor | 4.8381% | 4.7182% | -0.1199 pp |
| MLP epsilon_K, seed-mean field | 5.6946% | 5.5848% | -0.1098 pp |
| MLP epsilon_K, mean across seeds | 5.7433% | 5.6437% | -0.0996 pp |
| MLP epsilon_K, seed std | 0.0810 pp | 0.1210 pp | +0.0401 pp |
| MLP closure, seed-mean field | 15.3994% | 15.1745% | -0.2249 pp |

Per-seed epsilon_K:

| seed | legacy | dense | change |
|---:|---:|---:|---:|
| 0 | 5.6942% | 5.5832% | -0.1110 pp |
| 1 | 5.8575% | 5.8126% | -0.0449 pp |
| 2 | 5.6783% | 5.5353% | -0.1430 pp |

The dense grid produces a small decrease, but not a material reduction below
the old approximately 5.7% result.  In particular, the seed-mean MLP excess
over the data-gradient floor is 0.8566 pp on the legacy grid and 0.8666 pp on
the dense grid.  The approximately 0.11 pp MLP improvement is therefore
explained by the approximately 0.12 pp movement of the numerical floor, rather
than by a smaller model-to-floor gradient gap.
