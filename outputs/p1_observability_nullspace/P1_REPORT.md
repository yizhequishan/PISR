# P1 observability/nullspace experiment

- u-only observation matrix: rank 3/7, nullity 4.
- u+v observation matrix: rank 6/7, nullity 1; the remaining direction is the constant stream-function gauge.
- Adding the chosen x-only h(xi) changes u by at most 0.000e+00, but changes v with RMS 2.045e-01.
- Minimum-norm coefficient error (excluding gauge): u-only 3.301e-01, u+v 6.078e-15.

These statements follow from the observation matrices and ordinary least squares, so they are independent of PySR's search behavior.

## Existing PySR companion results

| model | formulas | xi-degenerate | rate |
|---|---:|---:|---:|
| u-fit A | 9 | 8 | 0.889 |
| u-fit B | 12 | 11 | 0.917 |
| psi-fit u-only | 15 | 9 | 0.600 |
| psi-fit u+v | 15 | 5 | 0.333 |
