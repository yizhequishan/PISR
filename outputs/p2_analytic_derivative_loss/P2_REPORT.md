# P2 analytic derivative-loss experiment

Both searches used the same operator set and search budget. Candidate derivatives were evaluated analytically on a dense held-out grid.

| selection | complexity | value MSE | gradient MSE | gradient rel L2 |
|---|---:|---:|---:|---:|
| value-only / min value | 23 | 4.95085e-05 | 0.00287848 | 0.0962659 |
| analytic derivative / min H1 | 24 | 5.65978e-05 | 0.000909776 | 0.05412 |

## Controlled ranking inversion

The field-MSE-preferred candidate has value MSE 0.00129842 and gradient MSE 0.0216285.
The gradient-preferred candidate has the *higher* value MSE 0.0016 but gradient MSE 0.

Across discovered candidates, value-MSE versus gradient-MSE Spearman correlation is 0.9048.

A ranking inversion also exists within the discovered PySR candidates; see `metrics.json` for the two expressions.
