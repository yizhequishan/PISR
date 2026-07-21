# P0 derivative-closure experiment

The exact discrete path decomposition checks coordinate/sign bookkeeping. The data-gradient and symbolic-gradient rows check the first-order chain rule with progressively stronger modeling assumptions.

| parameter | K source | rel L2 | correlation | RMSE | sign agreement |
|---|---|---:|---:|---:|---:|
| R | exact_discrete_path | 1.70422e-16 | 1 | 2.97985e-15 | 1.0000 |
| R | baseline_matched_fd | 0.148788 | 0.988987 | 2.60157 | 0.9539 |
| R | baseline_data_gradient | 0.199599 | 0.98039 | 3.49001 | 0.9389 |
| R | symbolic_analytic_gradient | 0.377842 | 0.926984 | 6.60661 | 0.8952 |
| cx | exact_discrete_path | 6.73797e-17 | 1 | 2.12589e-16 | 1.0000 |
| cx | baseline_matched_fd | 0.0610012 | 0.998165 | 0.192464 | 0.9855 |
| cx | baseline_data_gradient | 0.0615871 | 0.998129 | 0.194312 | 0.9855 |
| cx | symbolic_analytic_gradient | 0.353918 | 0.936847 | 1.11664 | 0.9018 |
| cy | exact_discrete_path | 1.60458e-16 | 1 | 9.13802e-16 | 1.0000 |
| cy | baseline_matched_fd | 0.10011 | 0.994984 | 0.570124 | 0.9819 |
| cy | baseline_data_gradient | 0.102494 | 0.994745 | 0.583696 | 0.9819 |
| cy | symbolic_analytic_gradient | 0.435908 | 0.911438 | 2.48248 | 0.8931 |

Interpretation: exact-path closure should be at floating-point precision. A larger data-gradient error measures finite-step/interpolation/nonlinear effects; the additional symbolic-gradient gap measures derivative error in the selected symbolic surrogate.
