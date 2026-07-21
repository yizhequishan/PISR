# P2 analytic derivative-loss experiment

Both searches used the same operator set and search budget. Candidate derivatives were evaluated analytically on a dense held-out grid.

| selection | complexity | value MSE | gradient MSE | gradient rel L2 |
|---|---:|---:|---:|---:|
| value-only / min value | 5 | 0.00699285 | 0.0788653 | 0.504463 |
| analytic derivative / min H1 | 5 | 0.00699285 | 0.0788653 | 0.504463 |

## Controlled ranking inversion

The field-MSE-preferred candidate has value MSE 0.00126715 and gradient MSE 0.0221913.
The gradient-preferred candidate has the *higher* value MSE 0.0016 but gradient MSE 0.

Across discovered candidates, value-MSE versus gradient-MSE Spearman correlation is 0.3789.

A ranking inversion also exists within the discovered PySR candidates; see `metrics.json` for the two expressions.
