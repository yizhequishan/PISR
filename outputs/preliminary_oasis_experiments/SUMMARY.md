# OASIS-SR preliminary experiments: P0-P2

## Overall outcome

All three experiments completed. P0 validates the derivative bookkeeping but exposes a symbolic-gradient gap; P1 establishes an algorithm-independent observation nullspace; P2 demonstrates both controlled and PySR-discovered value/gradient ranking inversions and improves held-out gradients with an analytic derivative loss.

## P0: T = M + K closure

The exact finite-step path decomposition closes to relative L2 <= 1.704e-16, so coordinate transforms, signs, and normalization are internally consistent.

| parameter | baseline CFD-gradient closure | symbolic-gradient closure |
|---|---:|---:|
| R | 0.1996 | 0.3778 |
| cx | 0.0616 | 0.3539 |
| cy | 0.1025 | 0.4359 |

Interpretation: the bookkeeping is correct, while the C=21 symbolic surrogate does not yet preserve the baseline spatial gradient accurately. The radius result also remains affected by finite-step/nonlinear error.

## P1: differential observability

The u-only matrix has rank 3/7 and nullity 4; u+v has rank 6/7 and nullity 1, exactly the constant gauge.

With only three v samples (0.18% of the v grid):

| selection | mean rank (gauge removed) | coefficient error |
|---|---:|---:|
| d_optimal | 6.00 | 0.001136 |
| random | 5.92 | 0.02468 |
| uniform | 4.00 | 0.299 |

This rank result uses only explicit observation matrices and least squares; it is independent of PySR. Existing PySR Pareto fronts show the matching empirical collapse rates (u-fit 89%-92%, psi u-only 60%, psi u+v 33%).

## P2: analytic derivative loss

| selection | complexity | value MSE | gradient MSE | gradient rel L2 |
|---|---:|---:|---:|---:|
| value-only minimum value | 23 | 4.951e-05 | 0.002878 | 0.09627 |
| analytic derivative minimum H1 | 24 | 5.66e-05 | 0.0009098 | 0.05412 |

The derivative-aware selection reduces gradient MSE by 68.4% while increasing value MSE by 14.3%.

At matched complexity C=21, value MSE changes by only 4.7% and gradient MSE falls by 69.9%.

## Decision

Proceed to local parameter-response fields only after treating P0's symbolic-gradient gap as an explicit optimization/evaluation target. P1 supports observability-aware sampling as a real algorithmic mechanism; P2 supports derivative-aware Pareto selection, but the discovered expressions are still complex and should not yet be presented as recovered physics.
